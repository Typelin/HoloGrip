"""HoloGrip glove judgement -> MIDI notes for a music program (Ableton Live).

Uses the frozen field engine of a HoloGrip package (default: ../../11-17的包, or env HOLOGRIP_PACKAGE) unchanged:
only the accepted hits of a running take are turned into MIDI note-on / note-off messages.

  * MIDI output: any existing port (Windows: loopMIDI port for Ableton, or "Microsoft GS Wavetable Synth" to hear
    drums directly), or on macOS/Linux a virtual port named "HoloGrip" that Ableton sees as a MIDI input.
  * Note map: one MIDI note per hand x drum, edited in the GUI (left and right may differ), saved to midi_map.json.
  * Velocity: fixed, or from the glove peak acceleration (g) mapped linearly onto 1..127.
  * Glove mode: connect, zero on the drumhead, start / stop (Raw and predictions are saved like a pretest, under
    <package>/Data/演出/模型實測).
  * Replay test: a recorded Raw CSV is fed to the engine at real speed (no gloves needed); same MIDI output.
  * Monitor: every hit (hand, drum, note, velocity), engine latency (packet arrival -> judgement), process CPU and
    memory.

Run:  python holo_midi_out.py            (needs mido + python-rtmidi; see README_ZH_TW.md)
"""
import csv
import heapq
import json
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get('HOLOGRIP_PACKAGE', HERE.parents[1]/'11-17的包')).resolve()
sys.path.insert(0, str(PKG/'Program'))
CONFIG = HERE/'midi_map.json'
VIRTUAL = 'HoloGrip（虛擬埠）'
IS_WIN = platform.system() == 'Windows'

DRUMS = ['小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
# Same notes as the electronic drum kit MIDI recorded in the field (General MIDI drum map).
DEFAULT_NOTES = {'小鼓': 38, 'Hi-Hat': 42, 'Crash': 49, '高音 Tom': 48, '中音 Tom': 45, 'Ride': 51, '落地 Tom': 43}
DEFAULT = dict(port='', channel=10, velocity_mode='fixed', velocity=100, g_min=1.5, g_max=8.0, note_ms=100,
               notes={h: dict(DEFAULT_NOTES) for h in 'LR'}, mute={d: False for d in DRUMS})
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(n):
    """Ableton naming: note 60 = C3."""
    return f'{NAMES[n % 12]}{n//12-2}'


def load_config():
    cfg = json.loads(json.dumps(DEFAULT))
    try:
        user = json.loads(CONFIG.read_text(encoding='utf-8'))
        for k in DEFAULT:
            if k in user: cfg[k] = user[k]
        for h in 'LR':
            cfg['notes'][h] = {**DEFAULT_NOTES, **{d: int(v) for d, v in user.get('notes', {}).get(h, {}).items() if d in DRUMS}}
        cfg['mute'] = {**DEFAULT['mute'], **{d: bool(v) for d, v in user.get('mute', {}).items() if d in DRUMS}}
    except FileNotFoundError:
        pass
    except Exception as exc:
        print('midi_map.json ignored:', exc)
    return cfg


# --------------------------------------------------------------------------------------------- MIDI output
class MidiOut:
    """Thread-safe note sender; note-offs are scheduled on a background thread."""

    def __init__(self):
        import mido
        try:
            mido.set_backend('mido.backends.rtmidi')
        except Exception:
            pass
        self.mido = mido; self.port = None; self.name = ''; self.lock = threading.Lock()
        self.offs = []; self.cv = threading.Condition(); self.sent = 0
        threading.Thread(target=self._off_loop, daemon=True, name='midi-note-off').start()

    def names(self):
        try:
            out = list(self.mido.get_output_names())
        except Exception:
            out = []
        return ([] if IS_WIN else [VIRTUAL])+out

    def open(self, name):
        self.close()
        with self.lock:
            self.port = self.mido.open_output('HoloGrip', virtual=True) if name == VIRTUAL else self.mido.open_output(name)
            self.name = name

    def close(self):
        with self.lock:
            if self.port is not None:
                try:
                    self.port.reset()
                    self.port.close()
                except Exception:
                    pass
            self.port = None; self.name = ''

    def note(self, note, velocity, channel, length_ms):
        """Send note-on now; note-off after length_ms. Returns the send time (perf_counter)."""
        msg = self.mido.Message('note_on', note=int(note), velocity=int(velocity), channel=int(channel)-1)
        with self.lock:
            if self.port is None: return None
            self.port.send(msg); t = time.perf_counter(); self.sent += 1
        with self.cv:
            heapq.heappush(self.offs, (t+length_ms/1000, int(note), int(channel)-1)); self.cv.notify()
        return t

    def _off_loop(self):
        while True:
            with self.cv:
                while not self.offs: self.cv.wait()
                due, note, ch = self.offs[0]; wait = due-time.perf_counter()
                if wait > 0:
                    self.cv.wait(wait); continue
                heapq.heappop(self.offs)
            with self.lock:
                if self.port is not None:
                    try:
                        self.port.send(self.mido.Message('note_off', note=note, velocity=0, channel=ch))
                    except Exception:
                        pass


# --------------------------------------------------------------------------------------------- hit -> note
class Router:
    """Turns an accepted engine prediction into a MIDI note using the current GUI map; reports to the GUI queue."""

    def __init__(self, midi, events):
        self.midi = midi; self.events = events; self.cfg = load_config(); self.lock = threading.Lock()

    def set_config(self, cfg):
        with self.lock: self.cfg = json.loads(json.dumps(cfg))

    def velocity(self, cfg, peak_g):
        if cfg['velocity_mode'] != 'g' or peak_g is None: return int(cfg['velocity'])
        lo, hi = float(cfg['g_min']), float(cfg['g_max'])
        x = (float(peak_g)-lo)/max(hi-lo, 1e-6)
        return max(1, min(127, int(round(1+126*min(max(x, 0.0), 1.0)))))

    def hit(self, hand, drum, peak_g=None, engine_ms=None, source='手套'):
        with self.lock: cfg = self.cfg
        if cfg['mute'].get(drum): note = vel = None; t = None
        else:
            note = cfg['notes'][hand][drum]; vel = self.velocity(cfg, peak_g)
            t = self.midi.note(note, vel, cfg['channel'], cfg['note_ms'])
        self.events.put(dict(kind='hit', hand=hand, drum=drum, note=note, vel=vel, sent=t is not None,
                             engine_ms=engine_ms, peak_g=peak_g, source=source, at=time.time()))


def install_take_hook(live, router):
    """Wrap seventh_live.Take so that accepted predictions go to the router right after the engine returns them."""
    base = live.Take

    class MidiTake(base):
        def push(self, packet):
            predictions, reason = base.push(self, packet)
            for p in predictions:
                if p.get('accepted'):
                    router.hit(p['hand'], p['drum'], p.get('peak_g'), p.get('processing_latency_ms'))
            return predictions, reason

    live.Take = MidiTake


# --------------------------------------------------------------------------------------------- replay test
class Replay(threading.Thread):
    """Feeds a recorded Raw CSV to a fresh engine at real speed; engine_ms = time from the frame's due time to the
    judgement (the same quantity as processing_latency_ms in the field)."""

    def __init__(self, core, bundle, config, path, router, events, speed=1.0):
        super().__init__(daemon=True, name='replay')
        self.core, self.bundle, self.config, self.path = core, bundle, config, Path(path)
        self.router, self.events, self.speed = router, events, speed; self.stop_flag = threading.Event()

    def run(self):
        core = self.core
        try:
            hands = core.trajectory.read_raw(str(self.path))
            with open(self.path, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
            frames = sorted((s['row_index'], h, s) for h, a in hands.items() for s in a)
            engine = core.Engine(self.bundle, self.config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR)
            t_first = frames[0][2]['t']; start = time.perf_counter()
            self.events.put(dict(kind='status', text=f'重播中：{self.path.name}（{len(frames)} 筆）'))
            for _, h, s in frames:
                if self.stop_flag.is_set(): break
                due = start+(s['t']-t_first)/1000/self.speed; wait = due-time.perf_counter()
                if wait > 0: time.sleep(wait)
                row = source[s['row_index']]
                out, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']),
                                                pitch=float(row['pitch_deg']), roll=s['roll']))
                for p in out:
                    if p.get('accepted'):
                        self.router.hit(p['hand'], p['drum'], p.get('peak_g'), (time.perf_counter()-due)*1000, source='重播')
            self.events.put(dict(kind='status', text='重播結束' if not self.stop_flag.is_set() else '重播已停止'))
        except Exception as exc:
            self.events.put(dict(kind='status', text=f'重播失敗：{type(exc).__name__}: {exc}'))
        finally:
            self.events.put(dict(kind='replay_done'))


# --------------------------------------------------------------------------------------------- resources
def process_memory_mb():
    try:
        if IS_WIN:
            import ctypes
            from ctypes import wintypes as W

            class PMC(ctypes.Structure):
                _fields_ = [('cb', W.DWORD), ('PageFaultCount', W.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                            ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                            ('PeakPagefileUsage', ctypes.c_size_t)]
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL('kernel32'); k32.GetCurrentProcess.restype = W.HANDLE
            fn = k32.K32GetProcessMemoryInfo; fn.argtypes = [W.HANDLE, ctypes.POINTER(PMC), W.DWORD]
            return c.WorkingSetSize/2**20 if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # macOS: bytes; Linux: kB (peak)
        return r/2**20 if platform.system() == 'Darwin' else r/1024
    except Exception:
        return None


# --------------------------------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('HoloGrip → MIDI（Ableton Live）')
        self.geometry('1180x860'); self.minsize(1000, 760)
        self.events = queue.Queue(); self.midi = MidiOut(); self.router = Router(self.midi, self.events)
        self.cfg = self.router.cfg; self.service = None; self.core = None; self.replay = None
        self.lat = deque(maxlen=200); self.hits = 0; self.cpu_prev = (time.perf_counter(), time.process_time())
        self.status = tk.StringVar(value=f'模型包：{PKG.name}。載入中…')
        self._build()
        self.after(50, self._load_engine)
        self.after(100, self._poll)
        self.after(1000, self._resources)
        self.protocol('WM_DELETE_WINDOW', self.close)

    # ---------------- layout
    def _build(self):
        pad = dict(padx=6, pady=4)
        top = ttk.LabelFrame(self, text='MIDI 輸出'); top.pack(fill='x', **pad)
        ttk.Label(top, text='輸出埠').grid(row=0, column=0, sticky='w', **pad)
        self.port_var = tk.StringVar(value=self.cfg.get('port', ''))
        self.port_box = ttk.Combobox(top, textvariable=self.port_var, width=38, state='readonly'); self.port_box.grid(row=0, column=1, **pad)
        ttk.Button(top, text='重新整理', command=self._refresh_ports).grid(row=0, column=2, **pad)
        ttk.Button(top, text='開啟', command=self._open_port).grid(row=0, column=3, **pad)
        self.port_state = ttk.Label(top, text='未開啟', foreground='#b00'); self.port_state.grid(row=0, column=4, sticky='w', **pad)
        ttk.Label(top, text='通道').grid(row=1, column=0, sticky='w', **pad)
        self.ch_var = tk.IntVar(value=self.cfg['channel'])
        ttk.Spinbox(top, from_=1, to=16, textvariable=self.ch_var, width=5, command=self._apply).grid(row=1, column=1, sticky='w', **pad)
        ttk.Label(top, text='音長 ms').grid(row=1, column=2, sticky='e', **pad)
        self.len_var = tk.IntVar(value=self.cfg['note_ms'])
        ttk.Spinbox(top, from_=10, to=2000, increment=10, textvariable=self.len_var, width=6, command=self._apply).grid(row=1, column=3, sticky='w', **pad)
        vf = ttk.Frame(top); vf.grid(row=2, column=0, columnspan=6, sticky='w', **pad)
        ttk.Label(vf, text='力道').pack(side='left')
        self.vm_var = tk.StringVar(value=self.cfg['velocity_mode'])
        ttk.Radiobutton(vf, text='固定', value='fixed', variable=self.vm_var, command=self._apply).pack(side='left', padx=4)
        self.vel_var = tk.IntVar(value=self.cfg['velocity'])
        ttk.Spinbox(vf, from_=1, to=127, textvariable=self.vel_var, width=5, command=self._apply).pack(side='left')
        ttk.Radiobutton(vf, text='跟打擊力道（g）', value='g', variable=self.vm_var, command=self._apply).pack(side='left', padx=(16, 4))
        self.gmin_var = tk.DoubleVar(value=self.cfg['g_min']); self.gmax_var = tk.DoubleVar(value=self.cfg['g_max'])
        ttk.Spinbox(vf, from_=0.5, to=20, increment=0.5, textvariable=self.gmin_var, width=5, command=self._apply).pack(side='left')
        ttk.Label(vf, text='g → 1，').pack(side='left')
        ttk.Spinbox(vf, from_=1, to=30, increment=0.5, textvariable=self.gmax_var, width=5, command=self._apply).pack(side='left')
        ttk.Label(vf, text='g → 127').pack(side='left')

        mid = ttk.Frame(self); mid.pack(fill='both', expand=True, **pad)
        table = ttk.LabelFrame(mid, text='音符對照（左右手可以不同；Ableton 命名：60 = C3）'); table.pack(side='left', fill='y', padx=(0, 6))
        for c, t in enumerate(['鼓', '左手音符', '', '', '右手音符', '', '', '靜音']):
            ttk.Label(table, text=t, font=('', 10, 'bold')).grid(row=0, column=c, **pad)
        self.note_vars = {h: {} for h in 'LR'}; self.name_lbl = {h: {} for h in 'LR'}; self.mute_vars = {}
        for r, d in enumerate(DRUMS, 1):
            ttk.Label(table, text=d).grid(row=r, column=0, sticky='w', **pad)
            for j, h in enumerate('LR'):
                v = tk.IntVar(value=self.cfg['notes'][h][d]); self.note_vars[h][d] = v
                sb = ttk.Spinbox(table, from_=0, to=127, textvariable=v, width=5, command=self._apply)
                sb.grid(row=r, column=1+3*j, **pad); sb.bind('<FocusOut>', lambda e: self._apply()); sb.bind('<Return>', lambda e: self._apply())
                lb = ttk.Label(table, text=note_name(v.get()), width=5); lb.grid(row=r, column=2+3*j); self.name_lbl[h][d] = lb
                ttk.Button(table, text='試打', width=5, command=lambda h=h, d=d: self._test(h, d)).grid(row=r, column=3+3*j, **pad)
            m = tk.BooleanVar(value=self.cfg['mute'][d]); self.mute_vars[d] = m
            ttk.Checkbutton(table, variable=m, command=self._apply).grid(row=r, column=7, **pad)
        bf = ttk.Frame(table); bf.grid(row=len(DRUMS)+1, column=0, columnspan=8, pady=8)
        ttk.Button(bf, text='儲存設定', command=self._save).pack(side='left', padx=4)
        ttk.Button(bf, text='回復預設', command=self._reset).pack(side='left', padx=4)
        ttk.Button(bf, text='全部停音', command=lambda: self.midi.port and self.midi.port.reset()).pack(side='left', padx=4)

        right = ttk.Frame(mid); right.pack(side='left', fill='both', expand=True)
        gl = ttk.LabelFrame(right, text='手套演出'); gl.pack(fill='x')
        udp = os.environ.get('HOLOGRIP_TRANSPORT', '').lower() == 'udp'
        self.gport = [tk.StringVar(value='UDP-L' if udp else 'COM4'), tk.StringVar(value='UDP-R' if udp else 'COM5')]
        self.gbox = []
        for i in range(2):
            cb = ttk.Combobox(gl, textvariable=self.gport[i], width=24); cb.grid(row=0, column=i, **pad); self.gbox.append(cb)
        ttk.Button(gl, text='連線', command=lambda: self._svc('connect', [v.get() for v in self.gport])).grid(row=0, column=2, **pad)
        ttk.Button(gl, text='中斷', command=lambda: self._svc('disconnect')).grid(row=0, column=3, **pad)
        ttk.Button(gl, text='歸零（手掌貼鼓面）', command=lambda: self._svc('zero')).grid(row=1, column=0, **pad)
        ttk.Button(gl, text='開始演出', command=lambda: self._svc('start', dict(session='perform'))).grid(row=1, column=1, **pad)
        ttk.Button(gl, text='停止', command=lambda: self._svc('stop')).grid(row=1, column=2, **pad)
        self.health = ttk.Label(gl, text=''); self.health.grid(row=2, column=0, columnspan=4, sticky='w', **pad)
        ttk.Label(gl, text='開始後每手先打 5 下小鼓（自動錨點），再開始演奏。Raw 與判斷存在 模型包/Data/演出。',
                  foreground='#555').grid(row=3, column=0, columnspan=4, sticky='w', **pad)

        rp = ttk.LabelFrame(right, text='重播測試（不用手套）'); rp.pack(fill='x', pady=6)
        self.rp_var = tk.StringVar(value='')
        ttk.Entry(rp, textvariable=self.rp_var, width=52).grid(row=0, column=0, columnspan=3, **pad)
        ttk.Button(rp, text='選 Raw CSV', command=self._pick).grid(row=0, column=3, **pad)
        ttk.Button(rp, text='開始重播', command=self._replay_start).grid(row=1, column=0, sticky='w', **pad)
        ttk.Button(rp, text='停止重播', command=self._replay_stop).grid(row=1, column=1, sticky='w', **pad)

        mon = ttk.LabelFrame(right, text='即時'); mon.pack(fill='both', expand=True)
        self.big = ttk.Label(mon, text='—', font=('', 28, 'bold')); self.big.pack(anchor='w', padx=8)
        self.stats = ttk.Label(mon, text=''); self.stats.pack(anchor='w', padx=8)
        self.res = ttk.Label(mon, text=''); self.res.pack(anchor='w', padx=8)
        self.log = tk.Listbox(mon, height=14, font=('Consolas' if IS_WIN else 'Menlo', 10)); self.log.pack(fill='both', expand=True, padx=8, pady=6)
        ttk.Label(self, textvariable=self.status, anchor='w', relief='sunken').pack(fill='x', side='bottom')
        self._refresh_ports()

    # ---------------- engine
    def _load_engine(self):
        def work():
            try:
                import seventh_core as core
                import seventh_live as live
                install_take_hook(live, self.router)
                t0 = time.perf_counter()
                svc = live.Service(data_root=PKG/'Data'/'演出')
                self.core, self.service = core, svc
                self.events.put(dict(kind='status', text=f'模型已載入（{time.perf_counter()-t0:.1f} s）：{core.MODEL_NAME}，SHA {core.MODEL_SHA256[:12]}'))
                self.events.put(dict(kind='ports'))
            except Exception as exc:
                self.events.put(dict(kind='status', text=f'模型載入失敗：{type(exc).__name__}: {exc}'))
        threading.Thread(target=work, daemon=True).start()

    def _svc(self, cmd, value=None):
        if not self.service: return self.status.set('模型還在載入')
        def work():
            try:
                self.service.call(cmd, value); self.events.put(dict(kind='status', text=self.service.snapshot(False)['status']))
            except Exception as exc:
                self.events.put(dict(kind='status', text=f'{cmd} 失敗：{exc}'))
        threading.Thread(target=work, daemon=True).start()

    # ---------------- MIDI
    def _refresh_ports(self):
        names = self.midi.names(); self.port_box['values'] = names
        if self.port_var.get() not in names: self.port_var.set(names[0] if names else '')

    def _open_port(self):
        try:
            self.midi.open(self.port_var.get()); self.port_state.config(text=f'已開啟：{self.midi.name}', foreground='#070'); self._apply()
        except Exception as exc:
            self.port_state.config(text='開啟失敗', foreground='#b00'); messagebox.showerror('MIDI', f'{type(exc).__name__}: {exc}')

    def _collect(self):
        cfg = json.loads(json.dumps(self.cfg))
        try:
            cfg.update(port=self.port_var.get(), channel=max(1, min(16, int(self.ch_var.get()))), note_ms=max(10, int(self.len_var.get())),
                       velocity_mode=self.vm_var.get(), velocity=max(1, min(127, int(self.vel_var.get()))),
                       g_min=float(self.gmin_var.get()), g_max=float(self.gmax_var.get()))
            for h in 'LR':
                for d in DRUMS: cfg['notes'][h][d] = max(0, min(127, int(self.note_vars[h][d].get())))
            cfg['mute'] = {d: bool(v.get()) for d, v in self.mute_vars.items()}
        except (tk.TclError, ValueError):
            return None
        return cfg

    def _apply(self):
        cfg = self._collect()
        if cfg is None: return self.status.set('有欄位不是數字，未套用')
        self.cfg = cfg; self.router.set_config(cfg)
        for h in 'LR':
            for d in DRUMS: self.name_lbl[h][d].config(text=note_name(cfg['notes'][h][d]))

    def _save(self):
        self._apply(); CONFIG.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=1), encoding='utf-8')
        self.status.set(f'已儲存：{CONFIG}')

    def _reset(self):
        for h in 'LR':
            for d in DRUMS: self.note_vars[h][d].set(DEFAULT_NOTES[d])
        for d in DRUMS: self.mute_vars[d].set(False)
        self.ch_var.set(10); self.len_var.set(100); self.vm_var.set('fixed'); self.vel_var.set(100)
        self.gmin_var.set(1.5); self.gmax_var.set(8.0); self._apply(); self.status.set('已回復預設（尚未儲存）')

    def _test(self, h, d):
        self._apply()
        if self.midi.port is None: return self.status.set('先開啟 MIDI 輸出埠')
        self.router.hit(h, d, None, None, source='試打')

    # ---------------- replay
    def _pick(self):
        p = filedialog.askopenfilename(initialdir=str(PKG/'Data'), filetypes=[('Raw CSV', '*raw_100hz*.csv'), ('CSV', '*.csv')])
        if p: self.rp_var.set(p)

    def _replay_start(self):
        if not self.service: return self.status.set('模型還在載入')
        if self.replay and self.replay.is_alive(): return self.status.set('重播進行中')
        if not Path(self.rp_var.get()).is_file(): return self.status.set('請先選 Raw CSV')
        self.replay = Replay(self.core, self.service.bundle, self.service.config, self.rp_var.get(), self.router, self.events); self.replay.start()

    def _replay_stop(self):
        if self.replay: self.replay.stop_flag.set()

    # ---------------- polling
    def _poll(self):
        try:
            while True:
                e = self.events.get_nowait()
                if e['kind'] == 'status': self.status.set(e['text'])
                elif e['kind'] == 'ports' and self.service:
                    vals = self.service.ports_provider()
                    for cb in self.gbox: cb['values'] = vals
                elif e['kind'] == 'hit':
                    self.hits += 1
                    if e['engine_ms'] is not None: self.lat.append(float(e['engine_ms']))
                    side = '左' if e['hand'] == 'L' else '右'
                    out = f"{e['note']:3d} {note_name(e['note']):>4s} v{e['vel']:3d}" if e['note'] is not None else '靜音'
                    if e['note'] is not None and not e['sent']: out += '（埠未開）'
                    self.big.config(text=f"{side}手  {e['drum']}  →  {out}")
                    lat = f"{e['engine_ms']:5.1f} ms" if e['engine_ms'] is not None else '   —   '
                    g = f"{e['peak_g']:.1f} g" if e['peak_g'] is not None else ''
                    self.log.insert(0, f"{time.strftime('%H:%M:%S', time.localtime(e['at']))} {e['source']} {side} {e['drum']:<7s} → {out}  判斷 {lat} {g}")
                    if self.log.size() > 300: self.log.delete(300, 'end')
                    if self.lat:
                        a = sorted(self.lat); p = lambda q: a[min(len(a)-1, int(q*len(a)))]
                        self.stats.config(text=f'打擊 {self.hits}　判斷時間（最近 {len(a)} 下）中位 {p(.5):.1f} ms、95% {p(.95):.1f} ms、最大 {a[-1]:.1f} ms'
                                               f'　＋偵測固定等待 80 ms')
                elif e['kind'] == 'replay_done':
                    pass
        except queue.Empty:
            pass
        if self.service:
            try:
                s = self.service.snapshot(False); h = s['health']
                self.health.config(text='　'.join(f"{'左' if k == 'L' else '右'}手 {h[k]['hz']} Hz{'' if h[k]['fresh'] else '（無資料）'}" for k in 'LR')
                                   + ('　錄製中' if s['take'] else ''))
            except Exception:
                pass
        self.after(40, self._poll)

    def _resources(self):
        now, cpu = time.perf_counter(), time.process_time(); w0, c0 = self.cpu_prev; self.cpu_prev = (now, cpu)
        mem = process_memory_mb(); ncpu = os.cpu_count() or 1
        pct = 100*(cpu-c0)/max(now-w0, 1e-6)
        self.res.config(text=f'本程式 CPU {pct:.0f}%（單核為 100%，本機 {ncpu} 執行緒）　記憶體 {mem:.0f} MB' if mem else f'本程式 CPU {pct:.0f}%')
        self.after(1000, self._resources)

    def close(self):
        try:
            if self.replay: self.replay.stop_flag.set()
            if self.service:
                try:
                    if self.service.snapshot(False)['take']: self.service.call('stop', timeout=10)
                except Exception:
                    pass
                self.service.close()
        finally:
            self.midi.close(); self.destroy()


if __name__ == '__main__':
    App().mainloop()
