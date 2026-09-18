from __future__ import annotations
import csv,json,queue,sys,threading,time
from collections import deque,Counter
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
MODELDIR=LAB/"count6_calibration_models_round3_194947"
sys.path.insert(0,str(APP));sys.path.insert(0,str(LAB));sys.path.insert(0,str(PROD))

from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,WINDOW_HALF_MS
from vnext_core import make_relative_sample,feature_from_samples,FEATURE_NAMES
from production_core import FrameValidator,RefractoryGate
from scipy.spatial.transform import Rotation as Rot

POSITIONS=["左上","中上","右上","左中","正中","右中"]
H_EXPECT={0:0,1:1,2:2,3:0,4:1,5:2}
V_EXPECT={0:0,1:0,2:0,3:1,4:1,5:1}
H_NAMES=["LEFT","CENTER","RIGHT"]
V_NAMES=["UP","MID"]
TARGET_PER_HAND=5
COUNT_REFRACTORY_MS=350.0

META=json.loads((MODELDIR/"metadata.json").read_text(encoding="utf-8"))
MODELS={}
for h in ("L","R"):
    MODELS[h]={
        "H":joblib.load(MODELDIR/f"H_{h}.joblib"),
        "V":joblib.load(MODELDIR/f"V_{h}.joblib"),
        "S":joblib.load(MODELDIR/f"S_{h}.joblib"),
        "H_idx":META["hands"][h]["H"]["idx"],
        "V_idx":META["hands"][h]["V"]["idx"],
        "S_idx":META["hands"][h]["S"]["idx"],
    }

STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=LAB/"count6_round4_sessioncal_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)

raw_f=(SESSION/"raw_100hz.csv").open("w",encoding="utf-8",newline="",buffering=1)
pred_f=(SESSION/"predictions.csv").open("w",encoding="utf-8",newline="",buffering=1)
stage_f=(SESSION/"stage_events.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f);pred_w=csv.writer(pred_f);stage_w=csv.writer(stage_f)

raw_w.writerow(["wall_time","mode","stage_index","stage_name","port","hand","packet_id","sensor_ms","ax","ay","az","yaw","pitch","roll","valid","invalid_reason"])
pred_w.writerow([
    "wall_time","stage_index","stage_name","hand","count_after",
    "H_true","H_pred","H_ok","V_true","V_pred","V_ok","S_true","S_pred","S_ok",
    "peak_g",*FEATURE_NAMES
])
stage_w.writerow(["wall_time","event","stage_index","stage_name","L_count","R_count"])

class HS:
    def __init__(self,h):
        self.port=""
        self.R0=Rot.from_euler("ZYX",META["source_r0"][h]["zero"],degrees=True).as_matrix()
        self.buffer=deque(maxlen=500)
        self.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.prod_gate=RefractoryGate(120)
        self.val=FrameValidator()
        self.pending=[]
        self.rx=0;self.hz=0.0;self.count=0;self.last_count_ms=-1e18
        self.lastH="—";self.lastV="—";self.lastS="—"
        self.okH=0;self.okV=0;self.okS=0
states={"L":HS("L"),"R":HS("R")}

lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
uiq=queue.Queue()
mode="wait_stage"
stage_index=0

def iso(): return datetime.now().astimezone().isoformat(timespec="milliseconds")
def parse(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"):return None
    try:return {"hand":f[1],"ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),"yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),"packet_id":int(f[8]),"sensor_ms":int(f[9])}
    except:return None
def ports():
    return sorted([p.device for p in list_ports.comports() if "303A:1001" in (p.hwid or "").upper() or "VID_303A&PID_1001" in (p.hwid or "").upper()])
def stage_name():return POSITIONS[stage_index] if 0<=stage_index<len(POSITIONS) else ""
def reset(st):
    st.buffer.clear();st.pending.clear();st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS);st.prod_gate=RefractoryGate(120)
    st.count=0;st.last_count_ms=-1e18
def log_stage(event):
    stage_w.writerow([iso(),event,stage_index,stage_name(),states["L"].count,states["R"].count]);stage_f.flush()

def classify(h,ft):
    mm=MODELS[h]
    hp=int(mm["H"].predict([ft[mm["H_idx"]]])[0])
    vp=int(mm["V"].predict([ft[mm["V_idx"]]])[0])
    sp=int(mm["S"].predict([ft[mm["S_idx"]]])[0])
    return hp,vp,sp

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
                raw_w.writerow([wall,mode,stage_index,stage_name(),port,h,p["packet_id"],p["sensor_ms"],p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],int(valid),reason])
                if st.rx%50==0:raw_f.flush()
                if not valid:
                    st.buffer.clear();st.pending.clear();st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS);st.prod_gate=RefractoryGate(120)
                    continue
                rs=make_relative_sample(st.R0,nowms,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
                st.buffer.append(rs)
                if mode!="collect":continue
                pkt=SensorPacket(hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],received_time_ms=int(time.time()*1000),received_monotonic=now)
                ev=st.det.add_packet(pkt,rs["rel_yaw"],rs["rel_pitch"])
                if ev:
                    peak=nowms-PEAK_LAG_MS
                    if st.prod_gate.accept(peak):
                        st.pending.append({"peak_ms":peak,"peak_g":float(ev["peak_accel_g"])})
                remain=[]
                for item in st.pending:
                    if nowms<item["peak_ms"]+WINDOW_HALF_MS:
                        remain.append(item);continue
                    if st.count>=TARGET_PER_HAND:continue
                    if item["peak_ms"]-st.last_count_ms<COUNT_REFRACTORY_MS:continue
                    ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                    if ft is None:continue
                    hp,vp,sp=classify(h,ft)
                    st.last_count_ms=item["peak_ms"];st.count+=1
                    he=H_EXPECT[stage_index];ve=V_EXPECT[stage_index];se=stage_index
                    hok=int(hp==he);vok=int(vp==ve);sok=int(sp==se)
                    st.okH+=hok;st.okV+=vok;st.okS+=sok
                    st.lastH=H_NAMES[hp];st.lastV=V_NAMES[vp];st.lastS=POSITIONS[sp]
                    pred_w.writerow([wall,stage_index,stage_name(),h,st.count,H_NAMES[he],H_NAMES[hp],hok,V_NAMES[ve],V_NAMES[vp],vok,POSITIONS[se],POSITIONS[sp],sok,item["peak_g"],*[f"{float(x):.8f}" for x in ft]])
                    pred_f.flush()
                    uiq.put(("count",h,st.count,hp,vp,sp,hok,vok,sok))
                    if states["L"].count>=TARGET_PER_HAND and states["R"].count>=TARGET_PER_HAND:
                        uiq.put(("stage_done",stage_index))
                st.pending=remain
    except Exception as e:uiq.put(("status",f"{port} reader error: {e!r}"))

def connect():
    ps=ports()
    if len(ps)<2:uiq.put(("status",f"只找到 {len(ps)} 顆 XIAO：{ps}"));return
    try:
        for p in ps[:2]:serials[p]=serial.Serial(p,460800,timeout=.08)
        time.sleep(.4)
        for p,s in serials.items():
            try:s.reset_input_buffer()
            except:pass
            t=threading.Thread(target=worker,args=(p,s),daemon=True);t.start();threads.append(t)
        uiq.put(("status",f"已連線 {', '.join(ps[:2])}。Fresh-R0 session 模型已凍結；Round4 資料不會重訓。"))
    except Exception as e:uiq.put(("status",f"COM 失敗：{e}"))

def start_stage():
    global mode
    if mode!="wait_stage":return
    with lock:
        for st in states.values():reset(st)
    mode="collect";start_btn.configure(state="disabled")
    title_var.set(f"Round4：{stage_name()}")
    prompt_var.set(f"請在『{stage_name()}』位置：左手 5 下 + 右手 5 下。")
    log_stage("STAGE_START")
def done_stage(idx):
    global mode
    if mode!="collect" or idx!=stage_index:return
    mode="wait_stage";log_stage("STAGE_COMPLETE")
    if stage_index==5:finish_all();return
    title_var.set(f"{stage_name()} 完成")
    prompt_var.set("移到下一個位置後按『下一位置』。")
    start_btn.configure(text="下一位置",state="normal")
def next_or_start():
    global stage_index
    if mode!="wait_stage":return
    if states["L"].count==TARGET_PER_HAND and states["R"].count==TARGET_PER_HAND:
        stage_index+=1
        with lock:
            for st in states.values():reset(st)
        title_var.set(f"準備：{stage_name()}")
        prompt_var.set(f"移到『{stage_name()}』，準備好後按『開始目前位置』。")
        start_btn.configure(text="開始目前位置",state="normal")
    else:start_stage()

def finish_all():
    global mode
    mode="finished";log_stage("SESSION_COMPLETE")
    raw_f.flush();pred_f.flush();stage_f.flush()
    with (SESSION/"predictions.csv").open(encoding="utf-8",newline="") as f:rr=list(csv.DictReader(f))
    summary={"session":str(SESSION),"source_model_session":META["source_session"],"n":len(rr),"hands":{},"overall":{}}
    for h in ("L","R"):
        q=[r for r in rr if r["hand"]==h]
        summary["hands"][h]={
            "n":len(q),
            "H_acc":sum(int(r["H_ok"]) for r in q)/len(q) if q else 0,
            "V_acc":sum(int(r["V_ok"]) for r in q)/len(q) if q else 0,
            "S_acc":sum(int(r["S_ok"]) for r in q)/len(q) if q else 0,
            "invalid":states[h].val.invalid,
        }
    summary["overall"]={
        "H_acc":sum(int(r["H_ok"]) for r in rr)/len(rr),
        "V_acc":sum(int(r["V_ok"]) for r in rr)/len(rr),
        "S_acc":sum(int(r["S_ok"]) for r in rr)/len(rr),
    }
    (SESSION/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    title_var.set("Round4 完成")
    prompt_var.set(f"水平 {summary['overall']['H_acc']*100:.1f}%｜上下 {summary['overall']['V_acc']*100:.1f}%｜六位置 {summary['overall']['S_acc']*100:.1f}%\n告訴 ChatGPT『Round4 完成』，我會讀完整資料。")
    start_btn.configure(state="disabled")

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    for f in (raw_f,pred_f,stage_f):
        try:f.close()
        except:pass
    root.destroy()

root=tk.Tk();root.title("HoloGrip｜Round4 Session-Cal 六位置盲測｜模型凍結");root.geometry("1050x700");root.attributes("-topmost",True)
style=ttk.Style();style.configure("Title.TLabel",font=("Microsoft JhengHei UI",23,"bold"));style.configure("Big.TLabel",font=("Microsoft JhengHei UI",32,"bold"));style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14));style.configure("Btn.TButton",font=("Microsoft JhengHei UI",13,"bold"),padding=9)
ttk.Label(root,text="Round4 Session-Cal 盲測｜Fresh-R0 空間模型已凍結",style="Title.TLabel",anchor="center").pack(fill="x",pady=(18,6))
title_var=tk.StringVar(value="準備：左上");ttk.Label(root,textvariable=title_var,style="Big.TLabel",anchor="center").pack(fill="x",pady=8)
prompt_var=tk.StringVar(value="這一輪不重新歸零、不重新訓練。沿用 Fresh-R0 與剛才凍結的 session 空間模型。");ttk.Label(root,textvariable=prompt_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=1000).pack(fill="x",pady=8)
status_var=tk.StringVar(value="正在連接…");ttk.Label(root,textvariable=status_var,anchor="center").pack(fill="x",pady=5)

frm=ttk.Frame(root);frm.pack(fill="both",expand=True,padx=18,pady=8);frm.columnconfigure(0,weight=1);frm.columnconfigure(1,weight=1)
ui={}
for col,h in enumerate(("L","R")):
    box=ttk.LabelFrame(frm,text=("左手 L" if h=="L" else "右手 R"),padding=12);box.grid(row=0,column=col,sticky="nsew",padx=8)
    cnt=tk.StringVar(value="0 / 5");pred=tk.StringVar(value="H —｜V —｜位置 —");info=tk.StringVar(value="等待資料")
    ttk.Label(box,textvariable=cnt,style="Big.TLabel",anchor="center").pack(fill="x",pady=(12,8))
    ttk.Label(box,textvariable=pred,style="Mid.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    ttk.Label(box,textvariable=info,anchor="center",justify="center").pack(fill="x",pady=8)
    ui[h]=(cnt,pred,info)

btn=ttk.Frame(root);btn.pack(pady=14)
start_btn=ttk.Button(btn,text="開始目前位置",style="Btn.TButton",command=next_or_start);start_btn.pack(side="left",padx=6)
ttk.Button(btn,text="關閉",style="Btn.TButton",command=close).pack(side="left",padx=6)

def update():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":status_var.set(e[1])
        elif e[0]=="stage_done":done_stage(e[1])
    with lock:
        for h,st in states.items():
            cnt,pred,info=ui[h]
            cnt.set(f"{st.count} / 5")
            pred.set(f"H {st.lastH}｜V {st.lastV}｜位置 {st.lastS}")
            total=max(1, sum(states[x].count for x in ("L","R")) + stage_index*10)
            info.set(f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜壞 frame {st.val.invalid}\n累積正確 H {st.okH}｜V {st.okV}｜六位 {st.okS}")
    root.after(100,update)

root.protocol("WM_DELETE_WINDOW",close);root.after(100,update);root.after(250,lambda:threading.Thread(target=connect,daemon=True).start());root.mainloop()

