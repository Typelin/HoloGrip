from __future__ import annotations
import csv,math,queue,sys,threading,time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import serial
from serial.tools import list_ports
import tkinter as tk
from tkinter import ttk

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
PROD=LAB/"HoloGrip_Production_vNext_0918"
sys.path.insert(0,str(APP));sys.path.insert(0,str(LAB));sys.path.insert(0,str(PROD))

from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,WINDOW_HALF_MS
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from production_core import FrameValidator,RefractoryGate

SIDE_MODELS={h:joblib.load(LAB/"song2_pose_side_models"/f"side_{h}.joblib") for h in ("L","R")}
DRUM_MODELS={h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")}
SIDE_IDX=list(range(19,25))  # mean_r6 only
LEFT_BLOCK={3,6}   # floor tom, ride
RIGHT_BLOCK={4,5}  # hi-hat, crash

STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=LAB/"song2_side_guard_demo_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)
raw_f=(SESSION/"raw.csv").open("w",encoding="utf-8",newline="",buffering=1)
hit_f=(SESSION/"hits.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f);hit_w=csv.writer(hit_f)
raw_w.writerow(["wall_time","port","hand","packet_id","ax","ay","az","yaw","pitch","roll","valid","invalid_reason"])
hit_w.writerow(["wall_time","hand","side","side_prob","p_right","raw_drum","raw_prob","guard_drum","guard_prob","blocked","fused_side","fused_score","peak_g"])

class HS:
    def __init__(self):
        self.port="";self.R0=None;self.zero=None;self.cal=[]
        self.buffer=deque(maxlen=500)
        self.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.gate=RefractoryGate(120)
        self.val=FrameValidator()
        self.pending=[]
        self.rx=0;self.hz=0.0
        self.side="—";self.side_prob=0.0
        self.raw="—";self.guard="—";self.blocked=False
states={"L":HS(),"R":HS()}
recent=deque(maxlen=7)  # (monotonic, p_right, confidence, hand)
lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
uiq=queue.Queue()
mode="idle"

def iso(): return datetime.now().astimezone().isoformat(timespec="milliseconds")
def parse(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"): return None
    try:
        return {"hand":f[1],"ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),"yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),"packet_id":int(f[8]),"sensor_ms":int(f[9])}
    except:return None
def ports():
    return sorted([p.device for p in list_ports.comports() if "303A:1001" in (p.hwid or "").upper() or "VID_303A&PID_1001" in (p.hwid or "").upper()])
def reset(st):
    st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS);st.gate=RefractoryGate(120);st.pending.clear();st.buffer.clear()

def fused_side():
    now=time.monotonic()
    vals=[x for x in recent if now-x[0] <= 3.0]
    if not vals:return ("—",0.5,0)
    # weighted by certainty away from 0.5
    w=np.array([max(.1,abs(x[1]-.5)*2) for x in vals],float)
    p=np.array([x[1] for x in vals],float)
    score=float(np.average(p,weights=w))
    if score<=0.42:lab="LEFT"
    elif score>=0.58:lab="RIGHT"
    else:lab="CENTER / 不確定"
    return lab,score,len(vals)

def classify(h,ft,peak_g,wall):
    st=states[h]
    sm=SIDE_MODELS[h]
    sp=sm.predict_proba([ft[SIDE_IDX]])[0]
    cls=int(sm.classes_[int(np.argmax(sp))]);side="LEFT" if cls==0 else "RIGHT"
    p_right=float(sp[list(sm.classes_).index(1)])
    conf=float(np.max(sp))

    dm=DRUM_MODELS[h]
    dp=dm.predict_proba([ft])[0]
    classes=[int(x) for x in dm.classes_]
    raw_j=int(np.argmax(dp));raw_cls=classes[raw_j];raw_name=ZONE_NAMES[raw_cls];raw_prob=float(dp[raw_j])

    recent.append((time.monotonic(),p_right,conf,h))
    fused,score,n=fused_side()

    # Guard only when fused side is decisive.
    allowed=set(classes)
    if fused=="LEFT": allowed-=LEFT_BLOCK
    elif fused=="RIGHT": allowed-=RIGHT_BLOCK

    cand=[(float(dp[j]),classes[j]) for j in range(len(classes)) if classes[j] in allowed]
    final_prob,final_cls=max(cand,key=lambda x:x[0])
    final_name=ZONE_NAMES[final_cls]
    blocked=(final_cls!=raw_cls)

    st.side=side;st.side_prob=conf;st.raw=raw_name;st.guard=final_name;st.blocked=blocked
    hit_w.writerow([wall,h,side,f"{conf:.5f}",f"{p_right:.5f}",raw_name,f"{raw_prob:.5f}",final_name,f"{final_prob:.5f}",int(blocked),fused,f"{score:.5f}",peak_g]);hit_f.flush()
    uiq.put(("hit",h,fused,score,n,raw_name,final_name,blocked))

def worker(port,conn):
    n=0;tick=time.monotonic()
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:continue
            p=parse(b.decode("utf-8","ignore").strip())
            if p is None:continue
            h=p["hand"];st=states[h];now=time.monotonic();nowms=now*1000;wall=iso()
            with lock:
                st.port=port;st.rx+=1;n+=1
                if now-tick>=1:st.hz=n/(now-tick);n=0;tick=now
                valid,reason=st.val.check(p)
                raw_w.writerow([wall,port,h,p["packet_id"],p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],int(valid),reason])
                if st.rx%50==0:raw_f.flush()
                if not valid:
                    reset(st);continue
                if mode=="cal":
                    st.cal.append((p["yaw"],p["pitch"],p["roll"]));continue
                if mode!="live" or st.R0 is None:continue
                rs=make_relative_sample(st.R0,nowms,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
                st.buffer.append(rs)
                pkt=SensorPacket(hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],received_time_ms=int(time.time()*1000),received_monotonic=now)
                ev=st.det.add_packet(pkt,rs["rel_yaw"],rs["rel_pitch"])
                if ev:
                    peak=nowms-PEAK_LAG_MS
                    if st.gate.accept(peak):st.pending.append({"peak_ms":peak,"peak_g":float(ev["peak_accel_g"])})
                remain=[]
                for item in st.pending:
                    if nowms>=item["peak_ms"]+WINDOW_HALF_MS:
                        ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                        if ft is not None:classify(h,ft,item["peak_g"],wall)
                    else:remain.append(item)
                st.pending=remain
    except Exception as e:uiq.put(("status",f"{port} error: {e!r}"))

def connect():
    ps=ports()
    if len(ps)<2:uiq.put(("status",f"只找到 {len(ps)} 顆 XIAO：{ps}"));return
    try:
        for p in ps[:2]:
            s=serial.Serial(p,460800,timeout=.08);serials[p]=s
        time.sleep(.4)
        for p,s in serials.items():
            try:s.reset_input_buffer()
            except:pass
            t=threading.Thread(target=worker,args=(p,s),daemon=True);t.start();threads.append(t)
        uiq.put(("status",f"已連線 {', '.join(ps[:2])}。先按 2 秒歸零。"))
    except Exception as e:uiq.put(("status",f"連線失敗：{e}"))

def start_cal():
    global mode
    mode="cal";recent.clear()
    with lock:
        for st in states.values():st.cal.clear();st.R0=None;reset(st)
    status_var.set("2 秒歸零：雙手放標準起始姿勢，不要動。")
    cal_btn.configure(state="disabled");root.after(2100,finish_cal)
def finish_cal():
    global mode
    ok=True
    with lock:
        for h,st in states.items():
            try:st.R0,st.zero=compute_r0(st.cal);reset(st)
            except Exception as e:ok=False;status_var.set(f"{h} 校正失敗：{e}")
    mode="live" if ok else "idle"
    cal_btn.configure(state="normal")
    if ok:status_var.set("開始測。請在同一個大概方位連打 5–8 下；畫面看『融合大方向』。")

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    try:raw_f.close();hit_f.close()
    except:pass
    root.destroy()

root=tk.Tk();root.title("HoloGrip｜8/12 真鼓方位守門 Demo");root.geometry("1050x650");root.attributes("-topmost",True)
style=ttk.Style();style.configure("T1.TLabel",font=("Microsoft JhengHei UI",22,"bold"));style.configure("BIG.TLabel",font=("Microsoft JhengHei UI",38,"bold"));style.configure("MID.TLabel",font=("Microsoft JhengHei UI",14));style.configure("BTN.TButton",font=("Microsoft JhengHei UI",14,"bold"),padding=10)
ttk.Label(root,text="8/12 真鼓資料訓練｜粗方向守門 Demo",style="T1.TLabel",anchor="center").pack(fill="x",pady=(18,8))
fused_var=tk.StringVar(value="融合大方向：—")
ttk.Label(root,textvariable=fused_var,style="BIG.TLabel",anchor="center").pack(fill="x",pady=10)
status_var=tk.StringVar(value="正在連接雙手套…")
ttk.Label(root,textvariable=status_var,style="MID.TLabel",anchor="center",justify="center",wraplength=990).pack(fill="x",padx=20,pady=7)

frm=ttk.Frame(root);frm.pack(fill="both",expand=True,padx=18,pady=10);frm.columnconfigure(0,weight=1);frm.columnconfigure(1,weight=1)
ui={}
for col,h in enumerate(("L","R")):
    box=ttk.LabelFrame(frm,text=("左手 L" if h=="L" else "右手 R"),padding=12);box.grid(row=0,column=col,sticky="nsew",padx=8)
    side=tk.StringVar(value="單擊方向：—");drum=tk.StringVar(value="鼓類：—");info=tk.StringVar(value="等待資料")
    ttk.Label(box,textvariable=side,style="T1.TLabel",anchor="center").pack(fill="x",pady=(18,8))
    ttk.Label(box,textvariable=drum,style="MID.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    ttk.Label(box,textvariable=info,style="MID.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    ui[h]=(side,drum,info)

btn=ttk.Frame(root);btn.pack(pady=16)
cal_btn=ttk.Button(btn,text="2 秒歸零並開始",style="BTN.TButton",command=start_cal);cal_btn.pack(side="left",padx=8)
ttk.Button(btn,text="關閉",style="BTN.TButton",command=close).pack(side="left",padx=8)

def update():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":status_var.set(e[1])
        elif e[0]=="hit":
            _,h,fused,score,n,raw,final,blocked=e
            fused_var.set(f"融合大方向：{fused}   ({n} hits / score {score:.2f})")
    with lock:
        for h,st in states.items():
            side,drum,info=ui[h]
            side.set(f"單擊方向：{st.side}  {st.side_prob*100:.0f}%")
            drum.set(f"原模型：{st.raw}\n守門後：{st.guard}" + ("  ← 已阻擋跨側" if st.blocked else ""))
            info.set(f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜壞 frame {st.val.invalid}｜120ms suppress {st.gate.suppressed}")
    root.after(100,update)

root.protocol("WM_DELETE_WINDOW",close);root.after(100,update);root.after(250,lambda:threading.Thread(target=connect,daemon=True).start());root.mainloop()
