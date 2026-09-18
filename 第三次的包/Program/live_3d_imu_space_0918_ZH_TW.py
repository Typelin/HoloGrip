from __future__ import annotations
import json, math, queue, threading, time
from pathlib import Path
from typing import Any
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
import sys
sys.path.insert(0,str(HERE))
from song_collection_server import parse_sensor_line
from product_hit_and_zone import load_zone_model, LiveRecognizer, FEATURE_NAMES, ZONE_NAMES
import run_live_hit_and_zone_0821_ZH_TW as live

REF=ROOT/'Diagnostic'/'training_reference_0826.json'
FONT='Microsoft JhengHei UI'
BG='#0d1117'; PANEL='#161b22'; TEXT='#f0f6fc'; MUTED='#8b949e'
CLASS_COLORS=['#ff7b72','#ffa657','#d2a8ff','#a5d6ff','#7ee787','#f2cc60','#79c0ff']


def sph(yaw_deg:float,pitch_deg:float,r:float=1.0):
    y=math.radians(yaw_deg); p=math.radians(pitch_deg)
    return [r*math.cos(p)*math.sin(y), -r*math.sin(p), r*math.cos(p)*math.cos(y)]

def rot(v,ay,ax):
    x,y,z=v
    cy,sy=math.cos(ay),math.sin(ay); cx,sx=math.cos(ax),math.sin(ax)
    x,z=x*cy+z*sy,-x*sy+z*cy
    y,z=y*cx-z*sx,y*sx+z*cx
    return x,y,z

def wrap_deg(x):
    return (x+180.0)%360.0-180.0

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('HoloGrip 即時 3D IMU 鼓位空間診斷')
        self.geometry('1280x820'); self.minsize(1080,700); self.configure(fg_color=BG)
        if not REF.is_file(): raise FileNotFoundError(REF)
        d=json.loads(REF.read_text(encoding='utf-8'))
        self.features=d['features']; self.labels=d['labels']; self.hands=d['hands']
        self.i_yaw=d['feature_names'].index('center_yaw'); self.i_pitch=d['feature_names'].index('center_pitch')
        self.ref_points=[]
        for feat,label,hand in zip(self.features,self.labels,self.hands):
            idx = ZONE_NAMES.index(label) if isinstance(label,str) else int(label)
            self.ref_points.append((float(feat[self.i_yaw]),float(feat[self.i_pitch]),idx,hand))
        self.centers={}
        for i,name in enumerate(ZONE_NAMES):
            pts=[p for p in self.ref_points if p[2]==i]
            if pts:
                ys=sorted(p[0] for p in pts); ps=sorted(p[1] for p in pts)
                self.centers[i]=(ys[len(ys)//2],ps[len(ps)//2])
        self.model=load_zone_model(live._resolve_model_path())
        self.rec=LiveRecognizer(self.model,require_calibration=True)
        self.stop=threading.Event(); self.conns=[]; self.threads=[]; self.q=queue.Queue(); self.port_map={}
        self.latest={'L':None,'R':None}; self.last_hit={'L':'—','R':'—'}
        self.cam_y=math.radians(-25); self.cam_x=math.radians(-12); self.drag=None
        self.com_a=tk.StringVar(value='選擇 COM A'); self.com_b=tk.StringVar(value='選擇 COM B')
        self.status=tk.StringVar(value='用途：只檢查座標系與訓練鼓位空間，不是正式 accuracy。')
        self.readout=tk.StringVar(value='尚未連線 / 尚未歸零')
        self._build(); self._refresh_ports(); self.after(33,self._tick); self.protocol('WM_DELETE_WINDOW',self._close)

    def _build(self):
        top=ctk.CTkFrame(self,fg_color=BG); top.pack(fill='x',padx=16,pady=(12,8))
        ctk.CTkLabel(top,text='HoloGrip 3D IMU 鼓位空間',font=(FONT,26,'bold'),text_color=TEXT).pack(side='left')
        ctk.CTkLabel(top,text='舊 Song2 真實 307 擊點雲 + 目前手套方向',font=(FONT,14),text_color=MUTED).pack(side='left',padx=14)
        controls=ctk.CTkFrame(self,fg_color=PANEL); controls.pack(fill='x',padx=16,pady=(0,8))
        self.ca=ctk.CTkComboBox(controls,variable=self.com_a,width=200); self.ca.pack(side='left',padx=8,pady=8)
        self.cb=ctk.CTkComboBox(controls,variable=self.com_b,width=200); self.cb.pack(side='left',padx=4,pady=8)
        ctk.CTkButton(controls,text='掃描',width=78,command=self._refresh_ports).pack(side='left',padx=4)
        ctk.CTkButton(controls,text='連接',width=78,command=self._connect).pack(side='left',padx=4)
        ctk.CTkButton(controls,text='歸零',width=90,fg_color='#f2cc60',text_color='#111',command=self._calibrate).pack(side='left',padx=4)
        ctk.CTkButton(controls,text='斷開',width=78,command=self._disconnect).pack(side='left',padx=4)
        ctk.CTkLabel(controls,textvariable=self.status,font=(FONT,13),text_color=MUTED).pack(side='left',padx=12)
        body=ctk.CTkFrame(self,fg_color=BG); body.pack(fill='both',expand=True,padx=16,pady=(0,12)); body.grid_columnconfigure(0,weight=3); body.grid_columnconfigure(1,weight=1); body.grid_rowconfigure(0,weight=1)
        self.canvas=tk.Canvas(body,bg='#05070a',highlightthickness=0); self.canvas.grid(row=0,column=0,sticky='nsew',padx=(0,8))
        self.canvas.bind('<ButtonPress-1>',self._drag_start); self.canvas.bind('<B1-Motion>',self._drag_move)
        side=ctk.CTkFrame(body,fg_color=PANEL); side.grid(row=0,column=1,sticky='nsew')
        ctk.CTkLabel(side,text='怎麼看',font=(FONT,20,'bold'),text_color=TEXT).pack(anchor='w',padx=14,pady=(14,6))
        expl='• 彩色點雲 = 0826 真實訓練擊打\n• 大字標籤 = 各鼓訓練中心\n• 白色/青色射線 = 目前手套方向\n• 歸零後中心應在 0°,0° 附近\n• 拖曳 3D 畫面可旋轉\n\n若你往同一方向移動，射線卻跳 180° 或反向，才像底層座標問題。'
        ctk.CTkLabel(side,text=expl,justify='left',font=(FONT,14),text_color=MUTED,wraplength=270).pack(anchor='w',padx=14)
        ctk.CTkLabel(side,text='即時數值',font=(FONT,18,'bold'),text_color=TEXT).pack(anchor='w',padx=14,pady=(18,4))
        ctk.CTkLabel(side,textvariable=self.readout,justify='left',font=(FONT,15),text_color=TEXT,wraplength=280).pack(anchor='w',padx=14)
        ctk.CTkLabel(side,text='重要：這個 UI 不判定「實體鼓在哪裡」，它顯示的是模型實際使用的 IMU 姿態空間。',justify='left',font=(FONT,13),text_color='#f2cc60',wraplength=280).pack(anchor='w',padx=14,pady=(18,0))

    def _refresh_ports(self):
        try:
            from serial.tools import list_ports
            ps=list(list_ports.comports()); vals=[]; self.port_map={}
            for p in ps:
                label=f'{p.device}  {p.description or ""}'.strip(); vals.append(label); self.port_map[label]=p.device; self.port_map[p.device]=p.device
            vals=vals or ['未找到 COM']; self.ca.configure(values=vals); self.cb.configure(values=vals)
            if ps: self.com_a.set(vals[0]); self.com_b.set(vals[1] if len(vals)>1 else '選擇 COM B')
        except Exception as e: self.status.set(str(e))
    def _selected(self):
        out=[]
        for s in (self.com_a.get(),self.com_b.get()):
            d=self.port_map.get(s.strip())
            if d and d not in out: out.append(d)
        return out
    def _connect(self):
        import serial
        self._disconnect(); sel=self._selected()
        if not sel: messagebox.showwarning('沒有 COM','請先選 COM'); return
        self.stop=threading.Event(); self.rec=LiveRecognizer(self.model,require_calibration=True)
        try:
            for p in sel:
                c=serial.Serial(p,460800,timeout=.2); self.conns.append(c)
            time.sleep(.5)
            for c,p in zip(self.conns,sel):
                try:c.reset_input_buffer()
                except:pass
                t=threading.Thread(target=self._loop,args=(c,p),daemon=True); self.threads.append(t); t.start()
            self.status.set('已連線。先保持自然原點姿勢，再按「歸零」。')
        except Exception as e: self._disconnect(); messagebox.showerror('連線失敗',str(e))
    def _disconnect(self):
        if hasattr(self,'stop'): self.stop.set()
        for c in getattr(self,'conns',[]):
            try:c.close()
            except:pass
        self.conns=[]; self.threads=[]
    def _calibrate(self):
        done=self.rec.calibrate()
        if not done: messagebox.showwarning('沒有資料','等手套持續有封包再歸零'); return
        self.status.set('歸零完成：'+','.join(done)+'。現在慢慢把手移向你心中的各鼓位置。')
    def _loop(self,conn,port):
        while not self.stop.is_set() and getattr(conn,'is_open',False):
            try: raw=conn.readline()
            except: break
            if not raw: continue
            pkt=parse_sensor_line(raw.decode('utf-8',errors='ignore'),int(time.time()*1000),time.monotonic())
            if pkt is None: continue
            try:
                hits=self.rec.push(pkt)
                if pkt.hand in self.rec.calibrated_hands:
                    st=self.rec.hands[pkt.hand]; cy,cp=st.detector.calibrated(pkt); self.q.put(('ori',(pkt.hand,cy,cp,pkt.roll)))
                for h in hits:self.q.put(('hit',h))
            except Exception as e:self.q.put(('err',str(e)))
    def _tick(self):
        try:
            while True:
                k,v=self.q.get_nowait()
                if k=='ori': self.latest[v[0]]=v[1:]
                elif k=='hit': self.last_hit[v['hand']]=f"{v['pred_drum']} {v['pred_proba']*100:.0f}%"
                elif k=='err': self.status.set(v)
        except queue.Empty: pass
        self._draw(); self.after(33,self._tick)
    def _drag_start(self,e): self.drag=(e.x,e.y,self.cam_y,self.cam_x)
    def _drag_move(self,e):
        if not self.drag:return
        x,y,cy,cx=self.drag; self.cam_y=cy+(e.x-x)*.008; self.cam_x=max(-1.3,min(1.3,cx+(e.y-y)*.008)); self._draw()
    def _proj(self,v,w,h):
        x,y,z=rot(v,self.cam_y,self.cam_x); s=min(w,h)*.36; d=3.2; f=d/(d-z*.55); return w/2+x*s*f,h/2+y*s*f,z
    def _draw(self):
        c=self.canvas; c.delete('all'); w=max(10,c.winfo_width()); h=max(10,c.winfo_height())
        # sphere rings
        for pitch in (-60,-30,0,30,60):
            pts=[]
            for yaw in range(-180,181,5):
                x,y,_=self._proj(sph(yaw,pitch),w,h); pts.extend([x,y])
            if len(pts)>3:c.create_line(*pts,fill='#202832',width=1)
        for yaw in range(-150,181,30):
            pts=[]
            for pitch in range(-80,81,4):
                x,y,_=self._proj(sph(yaw,pitch),w,h);pts.extend([x,y])
            c.create_line(*pts,fill='#151c24',width=1)
        # cloud sorted by depth
        objs=[]
        for yaw,pit,label,hand in self.ref_points:
            v=sph(yaw,pit); x,y,z=self._proj(v,w,h); objs.append((z,x,y,label))
        for z,x,y,label in sorted(objs):
            r=2.2 if z<0 else 3.0; c.create_oval(x-r,y-r,x+r,y+r,fill=CLASS_COLORS[label],outline='')
        # centers
        for i,(yaw,pit) in self.centers.items():
            x,y,z=self._proj(sph(yaw,pit,1.03),w,h); c.create_text(x,y,text=ZONE_NAMES[i],fill=CLASS_COLORS[i],font=(FONT,12,'bold'))
        # current vectors
        lines=[]
        for hand,col in (('L','#58a6ff'),('R','#ffffff')):
            o=self.latest.get(hand)
            if not o: continue
            yaw,pit,roll=o; end=sph(yaw,pit,1.15); x0,y0,_=self._proj([0,0,0],w,h); x1,y1,_=self._proj(end,w,h)
            c.create_line(x0,y0,x1,y1,fill=col,width=5,arrow='last'); c.create_oval(x1-8,y1-8,x1+8,y1+8,outline=col,width=3)
            # angular nearest reference
            best=min(self.ref_points,key=lambda p: math.hypot(wrap_deg(yaw-p[0]),pit-p[1]))
            lines.append(f'{hand}: Yaw {yaw:+.1f}°  Pitch {pit:+.1f}°  Roll {roll:+.1f}°\n    最近姿態鼓：{ZONE_NAMES[best[2]]}  |  最近模型擊打：{self.last_hit[hand]}')
        self.readout.set('\n\n'.join(lines) if lines else '等待歸零後的手套資料…')
        c.create_text(12,12,anchor='nw',text='拖曳旋轉 3D · 點雲=舊訓練資料 · 射線=目前手套',fill='#8b949e',font=(FONT,12))
    def _close(self): self._disconnect(); self.destroy()

if __name__=='__main__':
    if '--smoke' in sys.argv:
        d=json.loads(REF.read_text(encoding='utf-8')); assert len(d['features'])==307; print('LIVE_3D_IMU_ENTRY_OK'); raise SystemExit(0)
    ctk.set_appearance_mode('dark'); ctk.set_default_color_theme('blue'); App().mainloop()


