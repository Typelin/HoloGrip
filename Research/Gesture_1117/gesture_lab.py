"""HoloGrip gesture lab: record labelled glove data and test the rule-based gesture recogniser live or offline.

  * Connect the two gloves (COM ports or UDP-L / UDP-R, same links as the field package).
  * Guided pose calibration (both connected hands at once): press, 2 s to get ready, 3 s hand flat on the table
    (palm down = the ② pose), 3 s hand upright with the thumb-web up (= the ③ / swing pose). Of each 3 s the first
    second is skipped (moving into the pose) and the stillest 1 s of the rest is used. The samples are recorded with
    the labels CAL_FLAT / CAL_UP. Alternative: "用剛錄的②③更新" (median pre-strike pose of recorded strikes).
  * Recording: everything received while connected goes to Data/<time>/gesture_raw.csv; a labelled segment
    ("左①揮" ... "雜訊：走動不控制") sets the label column, segments are listed in segments.json, every decided
    event in events.csv.
  * Live: last gesture per hand, per-segment counts, event log with the measured features, 3-second signal plot.
  * Thresholds editable and saved (gesture_params.json); "檢驗錄音" re-runs the rules on a recorded file and
    prints the label x result table.
  * Optional MIDI out (one note per hand x gesture) through holo_midi_out.MidiOut.

Run:  python gesture_lab.py      (package links from ../../11-17的包/Program or env HOLOGRIP_PACKAGE)
"""
import csv
import json
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get('HOLOGRIP_PACKAGE', HERE.parents[1]/'11-17的包')).resolve()
sys.path[:0] = [str(HERE), str(PKG/'Program'), str(HERE.parent/'MidiOut_1117')]
import gesture_rules as GR

PARAMS = HERE/'gesture_params.json'
DATA = HERE/'Data'
NONE = '（不接）'
FIELDS = ['t_ms', 'hand', 'received_time_ms', 'sensor_time_ms', 'packet_id', 'ax_g', 'ay_g', 'az_g',
          'yaw_deg', 'pitch_deg', 'roll_deg', 'label']
LABELS = [('L1', '左①揮'), ('L2', '左②敲（虎口朝右）'), ('L3', '左③敲（虎口朝上）'),
          ('R1', '右①揮'), ('R2', '右②敲（虎口朝左）'), ('R3', '右③敲（虎口朝上）'),
          ('NOISE', '雜訊：走動、比手勢，不做控制'), ('FREE', '自由測試')]
LABEL_TEXT = dict(LABELS, CAL_FLAT='校正：手平放桌上', CAL_UP='校正：手立起虎口朝上')
CAL_READY_S, CAL_HOLD_S, CAL_SKIP_S, CAL_WIN_S = 2.0, 3.0, 1.0, 1.0
EV_FIELDS = ['label', 'hand', 'kind', 'gesture', 'name', 'reason', 'peak_t', 't0', 't1', 'dur_ms', 'd_yaw', 'd_tilt', 'yaw_share',
             'peak_g', 'peak_dps', 'pose', 'ang_up', 'ang_side']
DEFAULT_NOTES = {'L': {1: 60, 2: 61, 3: 62}, 'R': {1: 64, 2: 65, 3: 66}}


def load_saved():
    try:
        d = json.loads(PARAMS.read_text(encoding='utf-8'))
    except Exception:
        d = {}
    params = dict(GR.DEFAULT_PARAMS, **d.get('params', {}))
    poses = {h: {k: list(map(float, v)) for k, v in d.get('poses', {}).get(h, {}).items()} for h in 'LR'}
    notes = {h: {g: int(d.get('notes', {}).get(h, {}).get(str(g), DEFAULT_NOTES[h][g])) for g in (1, 2, 3)} for h in 'LR'}
    return params, poses, notes


class Recorder:
    """One session folder per connection; raw rows are written by the worker thread."""

    def __init__(self):
        DATA.mkdir(exist_ok=True)
        self.folder = DATA/datetime.now().strftime('%Y%m%d_%H%M%S')
        self.folder.mkdir()
        self.raw_f = (self.folder/'gesture_raw.csv').open('w', encoding='utf-8-sig', newline='')
        self.raw = csv.DictWriter(self.raw_f, fieldnames=FIELDS); self.raw.writeheader()
        self.ev_f = (self.folder/'events.csv').open('w', encoding='utf-8-sig', newline='')
        self.ev = csv.DictWriter(self.ev_f, fieldnames=EV_FIELDS, extrasaction='ignore'); self.ev.writeheader()
        self.segments = []; self.rows = 0; self.last_flush = time.monotonic()

    def write(self, row):
        self.raw.writerow(row); self.rows += 1
        if time.monotonic()-self.last_flush > 1: self.raw_f.flush(); self.ev_f.flush(); self.last_flush = time.monotonic()

    def event(self, e):
        self.ev.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})

    def save_segments(self):
        (self.folder/'segments.json').write_text(json.dumps(self.segments, ensure_ascii=False, indent=1), encoding='utf-8')

    def close(self):
        self.save_segments()
        for f in (self.raw_f, self.ev_f):
            try: f.close()
            except Exception: pass


class Lab:
    """Serial/UDP readers -> one worker: record, recognise, feed the GUI."""

    def __init__(self, gui_q):
        self.q = queue.Queue(maxsize=20000); self.gui_q = gui_q; self.lock = threading.RLock()
        params, poses, self.notes = load_saved()
        self.rules = {h: GR.HandRules(h, params, poses[h]) for h in 'LR'}
        self.conns = []; self.stop = threading.Event(); self.rec = None; self.label = ''; self.seg = None
        self.t_start = time.monotonic(); self.plot = {h: deque(maxlen=300) for h in 'LR'}
        self.rate = {h: deque(maxlen=120) for h in 'LR'}; self.last_acc = {h: deque(maxlen=100) for h in 'LR'}; self.last_eul = {h: deque(maxlen=100) for h in 'LR'}
        self.midi = None; self.drops = 0; self.pre = {h: {2: [], 3: []} for h in 'LR'}
        self.cal_phase = None; self.cal_buf = None
        threading.Thread(target=self.work, daemon=True, name='gesture-worker').start()

    # ----- connection
    def ports(self):
        try:
            from serial.tools import list_ports
            # USB serial first (gloves); Bluetooth serial ports last
            com = [p.device for p in sorted(list_ports.comports(), key=lambda p: ('藍牙' in (p.description or '') or 'Bluetooth' in (p.description or ''), p.device))]
        except Exception:
            com = []
        return com+['UDP-L', 'UDP-R']

    def connect(self, ports):
        if self.conns: raise ValueError('已連線')
        if len(set(ports)) != len(ports): raise ValueError('兩個埠不能一樣')
        import glove_serial_link, glove_udp_link as udp
        from song_collection_server import parse_sensor_line
        self.parse = parse_sensor_line; self.stop.clear()
        for p in ports:
            self.conns.append(udp.open_port(p) if p in udp.PORT_NAMES else glove_serial_link.ReconnectingSerial(p))
        with self.lock:
            self.rec = Recorder()
            for h in 'LR': self.rules[h].reset()
        for c in self.conns: threading.Thread(target=self.read, args=(c,), daemon=True).start()
        return str(self.rec.folder)

    def read(self, conn):
        try:
            while not self.stop.is_set():
                pk = conn.read_packet() if hasattr(conn, 'read_packet') else self.parse(conn.readline().decode('ascii', errors='replace'))
                if pk:
                    try: self.q.put_nowait(pk)
                    except queue.Full: self.drops += 1
        except Exception as exc:
            if not self.stop.is_set(): self.gui_q.put(('status', f'{getattr(conn, "port", "")} 連線中斷：{exc}'))

    def disconnect(self):
        self.stop.set()
        for c in self.conns:
            try: c.close()
            except Exception: pass
        self.conns = []
        with self.lock:
            if self.seg: self.end_segment()
            if self.rec: self.rec.close(); folder = self.rec.folder; self.rec = None; return str(folder)

    # ----- segments
    def begin_segment(self, label, target):
        with self.lock:
            if not self.rec: raise ValueError('先連線')
            if self.seg: self.end_segment()
            self.label = label
            self.seg = dict(label=label, text=LABEL_TEXT[label], target=target, start_t_ms=self.now(),
                            start=datetime.now().isoformat(timespec='seconds'), counts={})
            self.rec.segments.append(self.seg); self.rec.save_segments()

    def end_segment(self):
        with self.lock:
            if self.seg:
                self.seg['end_t_ms'] = self.now(); self.rec.save_segments()
            self.seg = None; self.label = ''

    def now(self):
        return round((time.monotonic()-self.t_start)*1000)

    # ----- calibration
    def calibrate(self, hand, which):
        with self.lock: a = list(self.last_acc[hand]); eul = list(self.last_eul[hand])
        if len(a) < 60: raise ValueError(f'{hand} 資料不足，先連線並靜止 1 秒')
        import numpy as np
        A = np.array(a[-80:]); m = np.linalg.norm(A, axis=1)
        if m.std() > 0.05 or not 0.8 < m.mean() < 1.2: raise ValueError(f'{"左" if hand == "L" else "右"}手沒有靜止（{m.std():.2f} g），請保持姿勢 1 秒再按')
        # gravity direction from the orientation (same quantity the rules compare against while moving)
        v = GR.unit(np.mean([GR.gravity_of(*e) for e in eul[-80:]], 0)).tolist()
        with self.lock: self.rules[hand].poses[which] = v
        poses = self.rules[hand].poses
        return GR.angle_deg(poses['up'], poses['side']) if 'up' in poses and 'side' in poses else None

    # guided calibration: the GUI switches the phases on a timer, the worker collects the samples
    def cal_begin(self):
        with self.lock:
            if not self.rec: raise ValueError('先連線')
            if self.seg: raise ValueError('先結束目前的段落再校正')
            self.cal_buf = {h: {'flat': [], 'up': []} for h in 'LR'}; self.cal_phase = None
        return True

    def cal_set(self, phase):
        """phase: None (getting ready) / 'flat' / 'up'. Each hold phase is also a recorded segment."""
        with self.lock:
            if self.seg: self.end_segment()
            self.cal_phase = phase
            if phase: self.begin_segment('CAL_FLAT' if phase == 'flat' else 'CAL_UP', 0)

    def cal_finish(self):
        """Pose per connected hand from the stillest CAL_WIN_S of each hold. Returns ({hand: angle}, [errors])."""
        import numpy as np
        with self.lock:
            if self.seg: self.end_segment()
            self.cal_phase = None; buf = self.cal_buf; self.cal_buf = None
        res, err = {}, []
        for h in 'LR':
            P = {}
            for phase, which, word in (('flat', 'side', '平放'), ('up', 'up', '虎口朝上')):
                S = buf[h][phase]
                if not S: break
                t0 = S[0][0]; S = [x for x in S if x[0]-t0 >= CAL_SKIP_S*1000]
                n = max(20, round(CAL_WIN_S*len(S)/(CAL_HOLD_S-CAL_SKIP_S)))
                side = '左' if h == 'L' else '右'
                if len(S) < n: err.append(f'{side}手{word}資料不足（{len(S)} 筆）'); break
                best = None
                for i in range(0, len(S)-n+1, 5):
                    m = np.linalg.norm([x[1] for x in S[i:i+n]], axis=1)
                    if best is None or m.std() < best[0]: best = (m.std(), m.mean(), i)
                sd, mean, i = best
                if sd > 0.05 or not 0.8 < mean < 1.2:
                    err.append(f'{side}手{word}時沒有靜止（{sd:.2f} g）'); break
                P[which] = GR.unit(np.mean([GR.gravity_of(*x[2]) for x in S[i:i+n]], 0)).tolist()
            if len(P) == 2:
                with self.lock: self.rules[h].poses.update(P)
                res[h] = GR.angle_deg(P['up'], P['side'])
        return res, err

    def calibrate_from_strikes(self, hand):
        """'side' / 'up' = median pre-strike pose of the strikes recorded in the ② / ③ segments of this session."""
        import numpy as np
        with self.lock: P = self.pre[hand]
        if len(P[2]) < 5 or len(P[3]) < 5: raise ValueError(f'②、③各要至少 5 下敲擊（目前 {len(P[2])}、{len(P[3])}）')
        with self.lock:
            self.rules[hand].poses['side'] = GR.unit(np.median(P[2], 0)).tolist()
            self.rules[hand].poses['up'] = GR.unit(np.median(P[3], 0)).tolist()
        return GR.angle_deg(self.rules[hand].poses['up'], self.rules[hand].poses['side']), len(P[2]), len(P[3])

    # ----- worker
    def work(self):
        while True:
            pk = self.q.get()
            h = pk.hand
            if h not in 'LR': continue
            with self.lock:
                t = round((pk.received_monotonic-self.t_start)*1000)
                if self.rec:
                    self.rec.write(dict(t_ms=t, hand=h, received_time_ms=pk.received_time_ms, sensor_time_ms=pk.sensor_time_ms,
                                        packet_id=pk.packet_id, ax_g=pk.ax, ay_g=pk.ay, az_g=pk.az, yaw_deg=pk.yaw,
                                        pitch_deg=pk.pitch, roll_deg=pk.roll, label=self.label))
                self.last_acc[h].append((pk.ax, pk.ay, pk.az)); self.last_eul[h].append((pk.yaw, pk.pitch, pk.roll)); self.rate[h].append(pk.received_monotonic)
                if self.cal_phase and self.cal_buf is not None:
                    self.cal_buf[h][self.cal_phase].append((t, (pk.ax, pk.ay, pk.az), (pk.yaw, pk.pitch, pk.roll)))
                t0 = time.perf_counter()
                e, sig = self.rules[h].push(float(pk.sensor_time_ms), pk.ax, pk.ay, pk.az, pk.yaw, pk.pitch, pk.roll)
                self.plot[h].append(sig)
                if e:
                    e['label'] = self.label; e['calc_ms'] = (time.perf_counter()-t0)*1000
                    e['lag_ms'] = e['t1']-e['t0']
                    if self.rec: self.rec.event(e)
                    lab_g = int(self.label[1]) if self.label[:1] == h and self.label[1:] in ('2', '3') else 0
                    if lab_g and e.get('kind') == 'strike' and e.get('pre_gvec') and e['d_tilt'] >= self.rules[h].strike_min_deg():
                        self.pre[h][lab_g].append(e['pre_gvec'])
                    if self.seg is not None:
                        k = f'{h}{e["gesture"]}' if e['gesture'] else f'{h}不算'
                        self.seg['counts'][k] = self.seg['counts'].get(k, 0)+1
                    if e['gesture'] and self.midi is not None and self.midi.port is not None:
                        self.midi.note(self.notes[h][e['gesture']], 100, 1, 120)
                    self.gui_q.put(('event', e))

    def hz(self, h):
        with self.lock: r = list(self.rate[h])
        now = time.monotonic(); r = [x for x in r if now-x <= 1]
        return (len(r)-1)/(r[-1]-r[0]) if len(r) > 1 and r[-1] > r[0] else 0.0


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('HoloGrip 手勢錄製與規則判斷測試')
        self.geometry('1320x900'); self.minsize(1100, 780)
        self.gq = queue.Queue(); self.lab = Lab(self.gq)
        self.status = tk.StringVar(value='連接手套後直接選要錄的動作（已載入存好的姿勢校正，不需要重做）。')
        self.counts = Counter(); self._build(); self.after(40, self._poll); self.after(100, self._draw)
        self.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self):
        pad = dict(padx=5, pady=3)
        left = ttk.Frame(self); left.pack(side='left', fill='y', padx=6, pady=6)
        right = ttk.Frame(self); right.pack(side='left', fill='both', expand=True, padx=6, pady=6)

        cf = ttk.LabelFrame(left, text='1. 連線'); cf.pack(fill='x')
        udp = os.environ.get('HOLOGRIP_TRANSPORT', '').lower() == 'udp'
        ports = self.lab.ports(); com = [p for p in ports if not p.startswith('UDP')]
        first = 'UDP-L' if udp else (com[0] if com else 'COM4')
        second = 'UDP-R' if udp else (com[1] if len(com) > 1 else NONE)
        self.pv = [tk.StringVar(value=first), tk.StringVar(value=second)]
        for i in range(2): ttk.Combobox(cf, textvariable=self.pv[i], values=ports+[NONE], width=12).grid(row=0, column=i, **pad)
        ttk.Button(cf, text='連線', command=self._connect).grid(row=0, column=2, **pad)
        ttk.Button(cf, text='中斷', command=self._disconnect).grid(row=0, column=3, **pad)
        self.hz_lbl = ttk.Label(cf, text='左 — Hz　右 — Hz'); self.hz_lbl.grid(row=1, column=0, columnspan=4, sticky='w', **pad)
        self.folder_lbl = ttk.Label(cf, text='', foreground='#555', wraplength=420); self.folder_lbl.grid(row=2, column=0, columnspan=4, sticky='w', **pad)

        kf = ttk.LabelFrame(left, text='2. 姿勢校正（開啟時已載入存檔；校正後要按「儲存」才會保留）'); kf.pack(fill='x', pady=6)
        self.cal_btn = ttk.Button(kf, text='引導校正（兩手同時：準備 2 秒 → 平放 3 秒 → 虎口朝上 3 秒）', command=self._guided_cal)
        self.cal_btn.grid(row=0, column=0, columnspan=4, sticky='w', **pad)
        self.cal_step = tk.Label(kf, text='', font=('', 15, 'bold'), fg='#c25b00', anchor='w', justify='left', wraplength=440)
        self.cal_step.grid(row=1, column=0, columnspan=4, sticky='w', **pad)
        self.cal_lbl = {}
        for r, h in enumerate('LR', 2):
            side = '左' if h == 'L' else '右'
            ttk.Label(kf, text=f'{side}手').grid(row=r, column=0, **pad)
            self.cal_lbl[h] = ttk.Label(kf, text=self._cal_text(h)); self.cal_lbl[h].grid(row=r, column=1, sticky='w', **pad)
            ttk.Button(kf, text='用剛錄的②③更新', command=lambda h=h: self._cal_strikes(h)).grid(row=r, column=2, **pad)

        sf = ttk.LabelFrame(left, text='3. 錄製一段'); sf.pack(fill='x')
        self.lab_var = tk.StringVar(value=LABELS[0][1])
        ttk.Combobox(sf, textvariable=self.lab_var, values=[t for _, t in LABELS], width=30, state='readonly').grid(row=0, column=0, columnspan=2, **pad)
        ttk.Label(sf, text='目標次數').grid(row=1, column=0, sticky='e', **pad)
        self.target = tk.IntVar(value=30); ttk.Spinbox(sf, from_=0, to=500, textvariable=self.target, width=6).grid(row=1, column=1, sticky='w', **pad)
        ttk.Button(sf, text='開始這段', command=self._seg_start).grid(row=2, column=0, **pad)
        ttk.Button(sf, text='結束這段', command=self._seg_end).grid(row=2, column=1, **pad)
        self.seg_lbl = ttk.Label(sf, text='目前沒有錄製段落', font=('', 11, 'bold'), wraplength=420); self.seg_lbl.grid(row=3, column=0, columnspan=2, sticky='w', **pad)
        self.cnt_lbl = ttk.Label(sf, text='', justify='left', font=('Consolas', 10)); self.cnt_lbl.grid(row=4, column=0, columnspan=2, sticky='w', **pad)

        pf = ttk.LabelFrame(left, text='4. 規則門檻'); pf.pack(fill='x', pady=6)
        p = self.lab.rules['L'].p
        spec = [('swing_min_deg', '揮：往外轉到 ≥ °'), ('swing_share', '揮：水平占比 ≥'), ('trig_dps', '揮：開始轉速 > °/s'),
                ('release_dps', '揮：結束轉速 < °/s'), ('strike_g', '敲：加速度峰值 ≥ g'), ('strike_min_deg_L', '敲：左手往下轉 ≥ °'), ('strike_min_deg_R', '敲：右手往下轉 ≥ °'),
                ('strike_share_max', '敲：水平占比 ≤'), ('strike_strong_g', '強敲：峰值 ≥ g'), ('strike_share_max_strong', '強敲：水平占比 ≤'), ('strike_window_ms', '敲：往前看 ms'), ('refractory_ms', '防彈跳：敲→敲 ms'), ('switch_gap_ms', '連發判定：間隔 < ms'), ('switch_confirm', '連發中換手勢需連續 次'),
                ('pose_max_deg', '姿勢：離校正 ≤ °'), ('swing_refractory_ms', '防彈跳：揮→揮 ms'),
                ('swing_rearm_deg', '防彈跳：揮完先往回 ≥ °'), ('strike_to_swing_ms', '防彈跳：敲→揮 ms'),
                ('swing_up_max_deg', '揮：離虎口朝上 ≤ °'), ('swing_to_strike_ms', '防彈跳：揮→敲 ms')]
        self.pvars = {}
        for i, (k, t) in enumerate(spec):
            ttk.Label(pf, text=t).grid(row=i//2, column=(i % 2)*2, sticky='e', **pad)
            v = tk.StringVar(value=str(p[k])); self.pvars[k] = v
            ttk.Entry(pf, textvariable=v, width=7).grid(row=i//2, column=(i % 2)*2+1, sticky='w', **pad)
        r = len(spec)//2+1
        self.dir_var = {}
        for j, h in enumerate('LR'):
            ttk.Label(pf, text=f'{"左" if h == "L" else "右"}手揮的方向（+＝往左）').grid(row=r, column=j*2, sticky='e', **pad)
            self.dir_var[h] = tk.StringVar(value=p[f'swing_dir_{h}'])
            ttk.Combobox(pf, textvariable=self.dir_var[h], values=['+', '-', 'any'], width=5, state='readonly').grid(row=r, column=j*2+1, sticky='w', **pad)
        bf = ttk.Frame(pf); bf.grid(row=r+1, column=0, columnspan=4, pady=4)
        ttk.Button(bf, text='套用', command=self._apply).pack(side='left', padx=3)
        ttk.Button(bf, text='儲存', command=self._save).pack(side='left', padx=3)
        ttk.Button(bf, text='回復預設', command=self._defaults).pack(side='left', padx=3)
        ttk.Button(bf, text='檢驗錄音…', command=self._offline).pack(side='left', padx=3)

        mf = ttk.LabelFrame(left, text='5. MIDI 輸出（選用）'); mf.pack(fill='x')
        self.mport = tk.StringVar(); self.mbox = ttk.Combobox(mf, textvariable=self.mport, width=30, state='readonly'); self.mbox.grid(row=0, column=0, columnspan=3, **pad)
        ttk.Button(mf, text='開啟', command=self._midi_open).grid(row=0, column=3, **pad)
        self.nvars = {}
        for r, h in enumerate('LR', 1):
            ttk.Label(mf, text='左' if h == 'L' else '右').grid(row=r, column=0, **pad)
            for g in (1, 2, 3):
                v = tk.IntVar(value=self.lab.notes[h][g]); self.nvars[h, g] = v
                ttk.Spinbox(mf, from_=0, to=127, textvariable=v, width=4, command=self._notes).grid(row=r, column=g, **pad)
        self._midi_ports()

        bigf = ttk.Frame(right); bigf.pack(fill='x')
        self.big = {}
        for h in 'LR':
            f = tk.Frame(bigf, bd=2, relief='groove', height=90); f.pack(side='left', fill='both', expand=True, padx=4)
            tk.Label(f, text='左手' if h == 'L' else '右手', font=('', 12)).pack(anchor='w', padx=6)
            self.big[h] = tk.Label(f, text='—', font=('', 30, 'bold')); self.big[h].pack(anchor='w', padx=6, pady=4)
        self.canvas = tk.Canvas(right, height=330, bg='white', highlightthickness=1, highlightbackground='#ccc'); self.canvas.pack(fill='x', pady=6)
        ttk.Label(right, text='圖：灰＝轉速、藍＝水平轉速（揮）、橘＝往下轉速（敲）、綠＝加速度（右軸 g）；虛線＝開始門檻與敲擊 g 門檻',
                  foreground='#555').pack(anchor='w')
        cols = ('time', 'label', 'hand', 'result', 'dur', 'yaw', 'tilt', 'share', 'g', 'pose', 'reason')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=14)
        for c, t, w in zip(cols, ['時間', '段落', '手', '判斷', 'ms', '水平°', '往下°', '占比', 'g', '姿勢', '說明'],
                           [70, 90, 40, 130, 50, 60, 60, 50, 50, 70, 260]):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor='w')
        self.tree.pack(fill='both', expand=True)
        ttk.Label(self, textvariable=self.status, anchor='w', relief='sunken').pack(fill='x', side='bottom')

    # ----- actions
    def _try(self, fn, *a):
        try: return fn(*a)
        except Exception as exc:
            self.status.set(f'{type(exc).__name__}: {exc}'); return None

    def _connect(self):
        ports = [v.get().strip() for v in self.pv if v.get().strip() not in ('', NONE)]
        if not ports: return self.status.set('至少選一個埠')
        folder = self._try(self.lab.connect, ports)
        if folder: self.folder_lbl.config(text=f'錄製中：{folder}'); self.status.set('已連線，所有資料都會存檔。直接選動作開始錄。')

    def _disconnect(self):
        folder = self._try(self.lab.disconnect)
        self.status.set(f'已中斷，存檔：{folder}' if folder else '已中斷'); self.seg_lbl.config(text='目前沒有錄製段落')

    def _cal_text(self, h):
        P = self.lab.rules[h].poses
        s = ('上✓' if 'up' in P else '上—')+' '+('側✓' if 'side' in P else '側—')
        if 'up' in P and 'side' in P: s += f'　相差 {GR.angle_deg(P["up"], P["side"]):.0f}°'
        return s

    def _guided_cal(self):
        if not self._try(self.lab.cal_begin): return
        self.cal_btn.state(['disabled']); self.cal_t0 = time.monotonic(); self.cal_cur = 'ready'; self.lab.cal_set(None)
        self.seg_lbl.config(text='校正中'); self._cal_tick()

    def _cal_tick(self):
        el = time.monotonic()-self.cal_t0
        t1 = CAL_READY_S; t2 = t1+CAL_HOLD_S; t3 = t2+CAL_HOLD_S
        if el < t1:
            phase, text = 'ready', f'準備…… {t1-el:.1f} 秒後開始：先把手平放在桌上（掌心朝下）'
        elif el < t2:
            phase, text = 'flat', f'① 手平放在桌上，不要動　{t2-el:.1f}'
        elif el < t3:
            phase, text = 'up', f'② 手立起來、虎口朝上，不要動　{t3-el:.1f}'
        else:
            res, err = self.lab.cal_finish(); self.cal_btn.state(['!disabled']); self.seg_lbl.config(text='目前沒有錄製段落')
            for h in 'LR': self.cal_lbl[h].config(text=self._cal_text(h))
            msg = '　'.join(f'{"左" if h == "L" else "右"}手兩姿勢相差 {a:.0f}°' + ('（太接近，請重做）' if a < 45 else '') for h, a in res.items())
            self.cal_step.config(text=('校正完成：'+msg if res else '校正失敗') + ('\n'+'；'.join(err) if err else ''))
            self.status.set('校正已套用，要保留請按「儲存」' if res else '校正失敗：'+'；'.join(err))
            return
        if phase != self.cal_cur:
            self.cal_cur = phase; self.lab.cal_set(phase); self.bell()
        self.cal_step.config(text=text)
        self.after(100, self._cal_tick)

    def _cal(self, h, which):
        ang = self._try(self.lab.calibrate, h, which)
        self.cal_lbl[h].config(text=self._cal_text(h))
        if ang is not None:
            self.status.set(f'{"左" if h == "L" else "右"}手兩個姿勢相差 {ang:.0f}°' + ('（太接近，請重做）' if ang < 45 else '，可以分'))

    def _cal_strikes(self, h):
        r = self._try(self.lab.calibrate_from_strikes, h)
        self.cal_lbl[h].config(text=self._cal_text(h))
        if r: self.status.set(f'{"左" if h == "L" else "右"}手用敲擊更新校正（②{r[1]} 下、③{r[2]} 下），兩姿勢相差 {r[0]:.0f}°；記得按儲存')

    def _seg_start(self):
        code = next(c for c, t in LABELS if t == self.lab_var.get())
        try:
            self.lab.begin_segment(code, int(self.target.get()))
        except Exception as exc:
            return self.status.set(f'{exc}')
        if True:
            self.counts = Counter(); self.seg_lbl.config(text=f'錄製中：{LABEL_TEXT[code]}（目標 {self.target.get()} 次）'); self._counts()
            self.status.set('動作之間停一下（約 1 秒），每次做完整。')

    def _seg_end(self):
        self.lab.end_segment(); self.seg_lbl.config(text='這段已結束'); self.status.set('段落已存檔')

    def _counts(self):
        seg = self.lab.seg; c = seg['counts'] if seg else {}
        lines = []
        for h in 'LR':
            side = '左' if h == 'L' else '右'
            lines.append(f'{side}: ' + '　'.join(f'{g}{GR.gesture_name(h, g)[:1]} {c.get(f"{h}{g}", 0):2d}' for g in (1, 2, 3)) + f'　不算 {c.get(f"{h}不算", 0)}')
        self.cnt_lbl.config(text='\n'.join(lines))

    def _read_params(self):
        p = {}
        for k, v in self.pvars.items():
            try: p[k] = float(v.get())
            except ValueError: raise ValueError(f'{k} 不是數字')
        for h in 'LR': p[f'swing_dir_{h}'] = self.dir_var[h].get()
        return p

    def _apply(self):
        p = self._try(self._read_params)
        if p is None: return
        with self.lab.lock:
            for h in 'LR': self.lab.rules[h].set_params(p)
        self.status.set('門檻已套用')

    def _save(self):
        self._apply()
        d = dict(params=self.lab.rules['L'].p, poses={h: self.lab.rules[h].poses for h in 'LR'},
                 notes={h: {str(g): self.lab.notes[h][g] for g in (1, 2, 3)} for h in 'LR'})
        PARAMS.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8'); self.status.set(f'已儲存：{PARAMS}')

    def _defaults(self):
        for k, v in self.pvars.items(): v.set(str(GR.DEFAULT_PARAMS[k]))
        for h in 'LR': self.dir_var[h].set(GR.DEFAULT_PARAMS[f'swing_dir_{h}'])
        self._apply()

    def _offline(self):
        path = filedialog.askopenfilename(initialdir=str(DATA), filetypes=[('gesture_raw.csv', 'gesture_raw.csv'), ('CSV', '*.csv')])
        if not path: return
        p = self._try(self._read_params)
        if p is None: return
        rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
        evs = GR.run_file(rows, p, {h: self.lab.rules[h].poses for h in 'LR'})
        self._show_report(Path(path), evs)

    def _show_report(self, path, evs):
        text = report_text(path, evs)
        w = tk.Toplevel(self); w.title(f'檢驗：{path.parent.name}'); w.geometry('900x600')
        t = tk.Text(w, font=('Consolas', 10)); t.pack(fill='both', expand=True); t.insert('1.0', text)

    def _midi_ports(self):
        try:
            from holo_midi_out import MidiOut
            if self.lab.midi is None: self.lab.midi = MidiOut()
            self.mbox['values'] = self.lab.midi.names()
            if self.mbox['values']: self.mport.set(self.mbox['values'][0])
        except Exception as exc:
            self.mbox['values'] = []; self.mport.set(f'MIDI 不可用：{exc}')

    def _midi_open(self):
        if self.lab.midi: self._try(self.lab.midi.open, self.mport.get()); self.status.set(f'MIDI：{self.lab.midi.name or "未開啟"}')

    def _notes(self):
        for (h, g), v in self.nvars.items():
            try: self.lab.notes[h][g] = max(0, min(127, int(v.get())))
            except (tk.TclError, ValueError): pass

    # ----- display
    def _poll(self):
        try:
            while True:
                kind, x = self.gq.get_nowait()
                if kind == 'status': self.status.set(x)
                elif kind == 'event': self._event(x)
        except queue.Empty:
            pass
        self.hz_lbl.config(text=f'左 {self.lab.hz("L"):.0f} Hz　右 {self.lab.hz("R"):.0f} Hz' + (f'　丟包 {self.lab.drops}' if self.lab.drops else ''))
        self.after(40, self._poll)

    def _event(self, e):
        h = e['hand']; ok = bool(e['gesture'])
        self.big[h].config(text=e['name'] if ok else f'（{e["name"]}）', fg='#0a6' if ok else '#999')
        self.after(500, lambda h=h: self.big[h].config(fg='#333'))
        pose = {'up': '上', 'side': '側', '?': '?'}[e['pose']]
        self.tree.insert('', 0, values=(time.strftime('%H:%M:%S'), LABEL_TEXT.get(e['label'], '')[:6], '左' if h == 'L' else '右',
                                        e['name'], f'{e["dur_ms"]:.0f}', f'{e["d_yaw"]:+.0f}', f'{e["d_tilt"]:.0f}',
                                        f'{e["yaw_share"]:.2f}', f'{e["peak_g"]:.1f}', pose, e['reason']))
        kids = self.tree.get_children()
        if len(kids) > 400: self.tree.delete(*kids[400:])
        self._counts()

    def _draw(self):
        c = self.canvas; c.delete('all'); W = max(c.winfo_width(), 600); H = 330; half = H/2
        p = self.lab.rules['L'].p; top_dps = 1200.0; top_g = 6.0
        for i, h in enumerate('LR'):
            y0 = i*half; base = y0+half-14
            with self.lab.lock: s = list(self.lab.plot[h])
            c.create_text(8, y0+10, text='左手' if h == 'L' else '右手', anchor='w', font=('', 10, 'bold'))
            c.create_line(0, base, W, base, fill='#ddd')
            yv = lambda v: base-(min(v, top_dps)/top_dps)*(half-26)
            yg = lambda v: base-(min(v, top_g)/top_g)*(half-26)
            c.create_line(0, yv(p['trig_dps']), W, yv(p['trig_dps']), fill='#999', dash=(4, 3))
            c.create_line(0, yg(p['strike_g']), W, yg(p['strike_g']), fill='#6a6', dash=(2, 4))
            if len(s) > 1:
                n = len(s); x = lambda k: W*(k+300-n)/300
                for key, col, f in (('speed', '#bbb', yv), ('yaw_rate', '#1f6fd1', lambda v: yv(abs(v))),
                                    ('tilt_rate', '#e8820c', yv), ('acc', '#2a9d4a', yg)):
                    pts = []
                    for k, d in enumerate(s): pts += [x(k), f(d[key])]
                    c.create_line(*pts, fill=col, width=1.5 if key != 'speed' else 1)
            c.create_text(W-6, y0+10, text=f'{top_dps:.0f}°/s　{top_g:.0f} g', anchor='e', fill='#888')
        self.after(100, self._draw)

    def close(self):
        self._try(self.lab.disconnect); self.destroy()


def report_text(path, evs):
    """Label x result table; per-label detection rate against the segment target."""
    lines = [f'檔案：{path}', f'判斷事件 {len(evs)} 個（含不算）', '']
    segs = []
    try: segs = json.loads((path.parent/'segments.json').read_text(encoding='utf-8'))
    except Exception: pass
    target = Counter()
    for s in segs: target[s['label']] += int(s.get('target') or 0)
    by = {}
    for e in evs:
        res = f'{e["hand"]}{e["gesture"]}' if e['gesture'] else f'{e["hand"]}不算'
        by.setdefault(e['label'] or '（沒有段落）', Counter())[res] += 1
    cols = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3', 'L不算', 'R不算']
    lines.append(f'{"段落":<22s}' + ''.join(f'{c:>7s}' for c in cols) + '   目標  正確率')
    for lab, cnt in by.items():
        right = cnt.get(lab, 0); tg = target.get(lab, 0)
        acc = f'{right}/{tg}' if tg and lab in [c for c, _ in LABELS[:6]] else ''
        lines.append(f'{LABEL_TEXT.get(lab, lab)[:20]:<22s}' + ''.join(f'{cnt.get(c, 0):7d}' for c in cols) + f'   {tg or "":>4}  {acc}')
    lines += ['', '雜訊段的 L1～R3 都是誤觸發；動作段裡判成別的手勢或「不算」都是錯。', '']
    reasons = Counter(e['reason'].split('（')[0].split(' ')[0] for e in evs if not e['gesture'])
    if reasons: lines.append('不算的原因：' + '、'.join(f'{k} {v}' for k, v in reasons.most_common()))
    return '\n'.join(lines)


if __name__ == '__main__':
    App().mainloop()
