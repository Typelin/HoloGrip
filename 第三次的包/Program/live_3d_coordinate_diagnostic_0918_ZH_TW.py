from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft JhengHei UI', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from song_collection_server import parse_sensor_line
from product_hit_and_zone import (
    LiveRecognizer,
    load_zone_model,
    window_features,
    slice_window,
    ZONE_NAMES,
)
import run_live_hit_and_zone_0821_ZH_TW as live

REF_PATH = ROOT / 'Diagnostic' / 'training_reference_0826.json'
MODEL_PATH = ROOT / 'Model' / 'hologrip_song2_七鼓點模型_真正驗證版_0826.joblib'

COLORS = {
    '小鼓': '#f59e0b', '高音 Tom': '#22c55e', '中音 Tom': '#14b8a6',
    '落地 Tom': '#0ea5e9', 'Hi-Hat': '#a855f7', 'Crash': '#f43f5e', 'Ride': '#6366f1'
}


def direction_xyz(yaw_deg: float, pitch_deg: float, radius: float = 1.0):
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    x = radius * math.cos(p) * math.sin(y)
    z = radius * math.cos(p) * math.cos(y)
    yy = radius * math.sin(p)
    return x, yy, z


def circular_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('HoloGrip 即時 3D 底層座標診斷（方向空間，不是真實 XYZ 位置）')
        self.geometry('1500x900')
        self.minsize(1200, 760)

        self.ref = json.loads(REF_PATH.read_text(encoding='utf-8'))
        self.features = np.asarray(self.ref['features'], dtype=float)
        self.scaled = np.asarray(self.ref['scaled_features'], dtype=float)
        self.labels = list(self.ref['labels'])
        self.nn95 = float(self.ref['nn_thresholds']['p95'])
        self.nn99 = float(self.ref['nn_thresholds']['p99'])
        self.clf = load_zone_model(str(MODEL_PATH))
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)

        self.stop_event = threading.Event()
        self.uiq: queue.Queue = queue.Queue()
        self.serials: list[Any] = []
        self.threads: list[threading.Thread] = []
        self.port_map: dict[str, str] = {}
        self.connected = False
        self.stream = {h: {'port':'','hz':0.0,'n':0,'tick_n':0,'tick_t':0.0,'last':0.0,
                           'raw_yaw':0.0,'cal_yaw':0.0,'pitch':0.0,'roll':0.0,'mag':0.0,
                           'unwrap':0.0,'prev_cal':None} for h in ('L','R')}
        self.recent = {h: deque(maxlen=80) for h in ('L','R')}
        self.hit_xyz = {h: None for h in ('L','R')}
        self.hit_text = {h: '尚無擊打' for h in ('L','R')}
        self.hit_nn = {h: '' for h in ('L','R')}
        self.wrap_hazard = {h: False for h in ('L','R')}
        self._build()
        self._refresh_ports()
        self.after(40, self._drain)
        self.after(100, self._redraw)
        self.protocol('WM_DELETE_WINDOW', self._close)

    def _build(self):
        top = ctk.CTkFrame(self)
        top.pack(fill='x', padx=12, pady=10)
        ctk.CTkLabel(top, text='HoloGrip 即時 3D 底層座標診斷', font=('Microsoft JhengHei UI', 24, 'bold')).pack(side='left', padx=10)
        ctk.CTkLabel(top, text='注意：這是 IMU 方向／姿態空間，不是房間中的絕對 XYZ 位置', text_color='#fbbf24', font=('Microsoft JhengHei UI', 14, 'bold')).pack(side='left', padx=18)

        ctl = ctk.CTkFrame(self)
        ctl.pack(fill='x', padx=12, pady=(0,8))
        self.com1 = ctk.StringVar(value='選擇 COM A'); self.com2 = ctk.StringVar(value='選擇 COM B')
        self.cb1 = ctk.CTkComboBox(ctl, variable=self.com1, width=210); self.cb1.pack(side='left', padx=6, pady=8)
        self.cb2 = ctk.CTkComboBox(ctl, variable=self.com2, width=210); self.cb2.pack(side='left', padx=6, pady=8)
        ctk.CTkButton(ctl, text='掃描', width=80, command=self._refresh_ports).pack(side='left', padx=5)
        ctk.CTkButton(ctl, text='連接', width=90, command=self._connect).pack(side='left', padx=5)
        ctk.CTkButton(ctl, text='斷開', width=90, command=self._disconnect).pack(side='left', padx=5)
        ctk.CTkButton(ctl, text='固定原點歸零', width=140, fg_color='#f59e0b', text_color='#111', command=self._calibrate).pack(side='left', padx=8)
        self.status = ctk.StringVar(value='先連接手套。單手也可以診斷。')
        ctk.CTkLabel(ctl, textvariable=self.status, font=('Microsoft JhengHei UI', 14)).pack(side='left', padx=14)

        body = ctk.CTkFrame(self, fg_color='transparent')
        body.pack(fill='both', expand=True, padx=12, pady=(0,12))
        body.grid_columnconfigure(0, weight=3); body.grid_columnconfigure(1, weight=2); body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,6))
        self.fig = Figure(figsize=(8,7), dpi=100, facecolor='#10131c')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#10131c')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        right = ctk.CTkFrame(body)
        right.grid(row=0,column=1,sticky='nsew',padx=(6,0))
        self.info = {}
        for hand, title in [('R','右手 R'),('L','左手 L')]:
            card = ctk.CTkFrame(right)
            card.pack(fill='x', padx=10, pady=10)
            ctk.CTkLabel(card, text=title, font=('Microsoft JhengHei UI', 21, 'bold')).pack(anchor='w', padx=12, pady=(10,2))
            var = ctk.StringVar(value='未連線')
            self.info[hand] = var
            ctk.CTkLabel(card, textvariable=var, justify='left', anchor='w', font=('Consolas', 15), wraplength=520).pack(fill='x', padx=12, pady=(0,10))

        note = ('怎麼判讀：\n'
                '• 原點歸零後，靜止時 Cal Yaw/Pitch 應接近 0。\n'
                '• 彩色點雲 = 0826 真實 307 次鼓擊的 center_yaw/center_pitch。\n'
                '• 白色射線 = 目前手套方向；大圓點 = 最近一次被 HitDetector 判定的擊打。\n'
                '• WRAP HAZARD = 最近 160ms 跨 ±180°，目前模型的普通 mean/std/range 可能被污染。\n'
                '• NN/OOD 只在有擊打時更新，使用完整 17 維特徵。')
        ctk.CTkLabel(right, text=note, justify='left', anchor='w', font=('Microsoft JhengHei UI', 14), wraplength=530).pack(fill='x', padx=14, pady=10)

    def _refresh_ports(self):
        from serial.tools import list_ports
        vals=[]; self.port_map={}
        for p in list_ports.comports():
            lab=f'{p.device}  {(p.description or "").replace(p.device, "").strip()}'
            vals.append(lab); self.port_map[lab]=p.device; self.port_map[p.device]=p.device
        if not vals: vals=['未找到 COM']
        self.cb1.configure(values=vals); self.cb2.configure(values=vals)
        if vals and vals[0] != '未找到 COM':
            self.com1.set(vals[0]); self.com2.set(vals[1] if len(vals)>1 else '選擇 COM B')

    def _selected(self):
        out=[]
        for s in (self.com1.get(), self.com2.get()):
            d=self.port_map.get(s.strip())
            if d and d not in out: out.append(d)
        return out

    def _connect(self):
        import serial
        ports=self._selected()
        if not ports:
            self.status.set('沒有選到 COM'); return
        self._disconnect()
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)
        self.stop_event=threading.Event(); self.uiq=queue.Queue(); self.serials=[]; self.threads=[]
        try:
            for p in ports:
                s=serial.Serial(port=p, baudrate=460800, timeout=0.2); self.serials.append(s)
            time.sleep(0.8)
            for s,p in zip(self.serials,ports):
                try:s.reset_input_buffer()
                except Exception:pass
                t=threading.Thread(target=self._serial_loop,args=(s,p),daemon=True); t.start(); self.threads.append(t)
            self.connected=True; self.status.set('已連接。等 Hz 穩定後回固定原點按「固定原點歸零」。')
        except Exception as e:
            self._disconnect(); self.status.set(f'連線失敗: {e}')

    def _disconnect(self):
        self.stop_event.set()
        for s in self.serials:
            try:s.close()
            except Exception:pass
        self.serials=[]
        for t in self.threads: t.join(timeout=0.2)
        self.threads=[]; self.connected=False

    def _calibrate(self):
        if not self.connected:
            self.status.set('請先連接'); return
        done=self.recognizer.calibrate()
        for h in done:
            self.recent[h].clear(); self.stream[h]['prev_cal']=None; self.stream[h]['unwrap']=0.0
        self.status.set('歸零完成：' + '、'.join(done) + '。現在慢慢移動，看 3D 射線是否連續。')

    def _serial_loop(self, conn, port):
        local_n=0; tick_t=time.monotonic(); tick_n=0
        while not self.stop_event.is_set() and getattr(conn,'is_open',False):
            try: raw=conn.readline()
            except Exception: break
            if not raw: continue
            pkt=parse_sensor_line(raw.decode('utf-8',errors='ignore'), int(time.time()*1000), time.monotonic())
            if pkt is None: continue
            local_n += 1
            try: hits=self.recognizer.push(pkt)
            except Exception as e:
                self.uiq.put(('err',str(e))); continue
            cal_yaw=cal_pitch=None
            st=self.recognizer.hands.get(pkt.hand)
            if st and pkt.hand in self.recognizer.calibrated_hands:
                cal_yaw,cal_pitch=st.detector.calibrated(pkt)
            now=time.monotonic()
            if now-tick_t>=0.4:
                hz=(local_n-tick_n)/(now-tick_t); tick_t=now; tick_n=local_n
            else:
                hz=None
            self.uiq.put(('packet',{'hand':pkt.hand,'port':port,'raw_yaw':pkt.yaw,'cal_yaw':cal_yaw,'pitch':cal_pitch,'roll':pkt.roll,
                                    'mag':math.sqrt(pkt.ax*pkt.ax+pkt.ay*pkt.ay+pkt.az*pkt.az),'hz':hz,'t_ms':pkt.received_monotonic*1000.0}))
            for hit in hits: self.uiq.put(('hit',hit))

    def _on_packet(self,d):
        h=d['hand']; s=self.stream[h]; s['port']=d['port']; s['raw_yaw']=d['raw_yaw']; s['roll']=d['roll']; s['mag']=d['mag']; s['last']=time.monotonic()
        if d['hz'] is not None: s['hz']=d['hz']
        if d['cal_yaw'] is None: return
        cy=float(d['cal_yaw']); cp=float(d['pitch']); s['cal_yaw']=cy; s['pitch']=cp
        prev=s['prev_cal']
        if prev is None: s['unwrap']=cy
        else: s['unwrap'] += circular_delta_deg(cy, prev)
        s['prev_cal']=cy
        self.recent[h].append((float(d['t_ms']),cy,cp))
        cutoff=float(d['t_ms'])-160.0
        vals=[x for x in self.recent[h] if x[0]>=cutoff]
        if len(vals)>=3:
            ys=[x[1] for x in vals]
            uw=[ys[0]]
            for y in ys[1:]: uw.append(uw[-1]+circular_delta_deg(y,ys[len(uw)-1]))
            naive=max(ys)-min(ys); true_span=max(uw)-min(uw)
            self.wrap_hazard[h]=(naive>180.0 and true_span<120.0)
        else:self.wrap_hazard[h]=False

    def _on_hit(self,hit):
        h=hit['hand']; peak=float(hit['peak_ms'])
        state=self.recognizer.hands[h]
        feat=window_features(slice_window(list(state.samples),peak),peak)
        if feat is not None:
            z=self.clf.named_steps['scaler'].transform([feat])[0]
            dist=np.linalg.norm(self.scaled-z,axis=1); i=int(np.argmin(dist)); nn=float(dist[i])
            band='IN-LIKE' if nn<=self.nn95 else ('BORDERLINE' if nn<=self.nn99 else 'OOD')
            nearest=self.labels[i]
            self.hit_nn[h]=f'17D NN {nn:.2f} {band} | 最近真鼓={nearest}'
            yaw=float(feat[13]); pitch=float(feat[14]); self.hit_xyz[h]=direction_xyz(yaw,pitch,1.14)
        self.hit_text[h]=f"模型={hit['pred_drum']} {hit['pred_proba']*100:.0f}% | peak={hit.get('peak_accel_g',0):.1f}g"

    def _drain(self):
        try:
            while True:
                k,d=self.uiq.get_nowait()
                if k=='packet': self._on_packet(d)
                elif k=='hit': self._on_hit(d)
                elif k=='err': self.status.set('推論錯誤: '+d)
        except queue.Empty: pass
        self.after(30,self._drain)

    def _redraw(self):
        ax=self.ax; ax.cla(); ax.set_facecolor('#10131c')
        # sphere wireframe
        u=np.linspace(0,2*np.pi,28); v=np.linspace(-np.pi/2,np.pi/2,15)
        xs=np.outer(np.cos(v),np.sin(u)); ys=np.outer(np.sin(v),np.ones_like(u)); zs=np.outer(np.cos(v),np.cos(u))
        ax.plot_wireframe(xs,ys,zs,color='#334155',linewidth=0.35,alpha=0.25)
        f=self.features
        for drum in ZONE_NAMES:
            idx=[i for i,x in enumerate(self.labels) if x==drum]
            if not idx: continue
            pts=np.array([direction_xyz(float(f[i,13]),float(f[i,14]),1.0) for i in idx])
            ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=12,alpha=.28,c=COLORS.get(drum,'#aaa'),label=drum)
            cy=float(np.median(f[idx,13])); cp=float(np.median(f[idx,14])); x,y,z=direction_xyz(cy,cp,1.08)
            ax.text(x,y,z,drum,color=COLORS.get(drum,'white'),fontsize=9,fontweight='bold')
        for h,c in [('R','#ffffff'),('L','#7dd3fc')]:
            s=self.stream[h]
            if s['last'] and time.monotonic()-s['last']<1.5 and h in self.recognizer.calibrated_hands:
                x,y,z=direction_xyz(s['cal_yaw'],s['pitch'],1.0)
                ax.plot([0,x],[0,y],[0,z],color=c,linewidth=3)
                ax.scatter([x],[y],[z],s=70,c=c,edgecolors='#111')
            if self.hit_xyz[h] is not None:
                x,y,z=self.hit_xyz[h]; ax.scatter([x],[y],[z],s=180,c=c,marker='o',edgecolors='#fbbf24',linewidths=2)
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.set_zlim(-1.2,1.2)
        ax.set_xlabel('左右 / Yaw',color='#cbd5e1'); ax.set_ylabel('上下 / Pitch',color='#cbd5e1'); ax.set_zlabel('前向',color='#cbd5e1')
        ax.tick_params(colors='#64748b'); ax.grid(False); ax.view_init(elev=22,azim=-55)
        ax.set_title('0826 七鼓方向雲 + 目前手套射線',color='white',fontsize=14)
        # update text cards
        for h in ('R','L'):
            s=self.stream[h]
            if not s['last']:
                self.info[h].set('未連線'); continue
            hazard='!!! WRAP HAZARD !!!' if self.wrap_hazard[h] else 'wrap: OK'
            self.info[h].set(
                f"Port {s['port']} | {s['hz']:.0f} Hz | {s['mag']:.2f} g\n"
                f"Raw Yaw   {s['raw_yaw']:+7.1f}°\n"
                f"Cal Yaw   {s['cal_yaw']:+7.1f}°\n"
                f"Unwrapped {s['unwrap']:+7.1f}°\n"
                f"Pitch     {s['pitch']:+7.1f}°\n"
                f"Roll      {s['roll']:+7.1f}°\n"
                f"{hazard}\n"
                f"{self.hit_text[h]}\n{self.hit_nn[h]}"
            )
        self.canvas.draw_idle(); self.after(120,self._redraw)

    def _close(self):
        self._disconnect(); self.destroy()


def main():
    if '--smoke' in sys.argv:
        assert REF_PATH.is_file()
        assert MODEL_PATH.is_file()
        d = json.loads(REF_PATH.read_text(encoding='utf-8'))
        assert d.get('n') == 307 and len(d.get('features', [])) == 307
        print('LIVE_3D_DIAGNOSTIC_ENTRY_OK')
        return
    ctk.set_appearance_mode('dark')
    ctk.set_default_color_theme('blue')
    App().mainloop()

if __name__=='__main__': main()

