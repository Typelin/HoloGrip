from __future__ import annotations
import csv,json,math,queue,sys,threading,time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import serial
from serial.tools import list_ports
import tkinter as tk
from tkinter import ttk

PKG=Path(__file__).resolve().parent
ROOT=PKG.parents[1]
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(PKG));sys.path.insert(0,str(APP))

from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from production_core import FrameValidator,RefractoryGate,adapt_models
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,WINDOW_HALF_MS

MODEL_DIR=PKG/"models"
BASE_MODELS={h:joblib.load(MODEL_DIR/f"base_{h}.joblib") for h in ("L","R")}
ARCHIVE=MODEL_DIR/"base_training_features.npz"

STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=PKG/"school_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)

raw_f=(SESSION/"raw.csv").open("w",encoding="utf-8",newline="",buffering=1)
pred_f=(SESSION/"predictions.csv").open("w",encoding="utf-8",newline="",buffering=1)
cal_f=(SESSION/"fewshot_calibration.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f);pred_w=csv.writer(pred_f);cal_w=csv.writer(cal_f)
raw_w.writerow(["wall_time","mode","port","hand","packet_id","ax","ay","az","yaw","pitch","roll","valid","invalid_reason"])
pred_w.writerow(["wall_time","hand","pred","prob","peak_g"])
cal_w.writerow(["wall_time","hand","label_id","label_name","peak_g"])

class HS:
    def __init__(self):
        self.port=""
        self.R0=None
        self.zero=None
        self.pose_samples=[]
        self.buffer=deque(maxlen=500)
        self.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.gate=RefractoryGate(120)
        self.val=FrameValidator()
        self.pending=[]
        self.rx=0;self.hz=0.0;self.last_pred="—";self.last_prob=0.0
states={"L":HS(),"R":HS()}
lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
uiq=queue.Queue()

mode="idle"       # idle / pose / fewshot / adapting / live
pose_end=0.0
active_models=dict(BASE_MODELS)
fewshot_records=[]
fewshot_plan=[(h,i) for i in range(len(ZONE_NAMES)) for h in ("L","R")]
fewshot_idx=0
ignore_until=0.0

def iso(): return datetime.now().astimezone().isoformat(timespec="milliseconds")

def parse(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"):return None
    try:
        return {"hand":f[1],"ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),
                "yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),
                "packet_id":int(f[8]),"sensor_ms":int(f[9])}
    except:return None

def reset_hand(st):
    st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
    st.gate=RefractoryGate(120)
    st.pending.clear()
    st.buffer.clear()

def xiao_ports():
    out=[]
    for p in list_ports.comports():
        hw=(p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:out.append(p.device)
    return sorted(out)

def current_target():
    if 0<=fewshot_idx<len(fewshot_plan):return fewshot_plan[fewshot_idx]
    return None

def capture_feature(hand,item,now_ms):
    st=states[hand]
    return feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)

def advance_fewshot():
    global fewshot_idx,ignore_until
    fewshot_idx+=1
    ignore_until=time.monotonic()+0.8
    if fewshot_idx>=len(fewshot_plan):
        finish_fewshot()
    else:
        uiq.put(("target",))

def finish_fewshot():
    global mode
    mode="adapting"
    uiq.put(("status",f"已收 {len(fewshot_records)} 個校準 hit，正在重訓左右手模型…"))
    def work():
        global active_models,mode
        models,summary=adapt_models(ARCHIVE,fewshot_records,SESSION/"adapted_models")
        active_models=models
        (SESSION/"adaptation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        mode="idle"
        uiq.put(("adapt_done",summary))
    threading.Thread(target=work,daemon=True).start()

def process_candidate(hand,item,wall):
    global ignore_until
    st=states[hand]
    ft=capture_feature(hand,item,time.monotonic()*1000)
    if ft is None:return
    if mode=="fewshot":
        if time.monotonic()<ignore_until:return
        target=current_target()
        if target is None:return
        th,label_id=target
        if hand!=th:return
        fewshot_records.append({"hand":hand,"label_id":int(label_id),"feature":ft.tolist()})
        cal_w.writerow([wall,hand,label_id,ZONE_NAMES[label_id],item["peak_g"]])
        cal_f.flush()
        uiq.put(("captured",hand,label_id))
        advance_fewshot()
    elif mode=="live":
        m=active_models[hand]
        proba=m.predict_proba([ft])[0]
        j=int(np.argmax(proba));cls=int(m.classes_[j])
        lab=ZONE_NAMES[cls];prob=float(proba[j])
        st.last_pred=lab;st.last_prob=prob
        pred_w.writerow([wall,hand,lab,f"{prob:.6f}",item["peak_g"]]);pred_f.flush()
        uiq.put(("pred",hand,lab,prob))

def serial_worker(port,conn):
    n=0;tick=time.monotonic()
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:continue
            p=parse(b.decode("utf-8","ignore").strip())
            if p is None:continue
            now=time.monotonic();nowms=now*1000;wall=iso();h=p["hand"];st=states[h]
            with lock:
                st.port=port;st.rx+=1;n+=1
                if now-tick>=1.0:st.hz=n/(now-tick);n=0;tick=now
                valid,reason=st.val.check(p)
                raw_w.writerow([wall,mode,port,h,p["packet_id"],p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],int(valid),reason])
                if st.rx%50==0:raw_f.flush()
                if not valid:
                    reset_hand(st)
                    continue
                if mode=="pose":
                    st.pose_samples.append((p["yaw"],p["pitch"],p["roll"]))
                    continue
                if st.R0 is None:continue
                sm=make_relative_sample(st.R0,nowms,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
                st.buffer.append(sm)
                if mode not in ("fewshot","live"):continue
                pkt=SensorPacket(hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
                    packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],received_time_ms=int(time.time()*1000),received_monotonic=now)
                ev=st.det.add_packet(pkt,sm["rel_yaw"],sm["rel_pitch"])
                if ev:
                    peak=nowms-PEAK_LAG_MS
                    if st.gate.accept(peak):
                        st.pending.append({"peak_ms":peak,"peak_g":float(ev["peak_accel_g"])})
                remain=[]
                for item in st.pending:
                    if nowms>=item["peak_ms"]+WINDOW_HALF_MS:
                        process_candidate(h,item,wall)
                    else:remain.append(item)
                st.pending=remain
    except Exception as e:
        uiq.put(("status",f"{port} reader error: {e!r}"))

def connect():
    ports=xiao_ports()
    if len(ports)<2:
        uiq.put(("status",f"只找到 {len(ports)} 顆 XIAO：{ports}"));return
    opened=[]
    try:
        for p in ports[:2]:opened.append((p,serial.Serial(p,460800,timeout=.08)))
        time.sleep(.5)
        for p,s in opened:
            try:s.reset_input_buffer()
            except:pass
            serials[p]=s
            t=threading.Thread(target=serial_worker,args=(p,s),daemon=True);t.start();threads.append(t)
        uiq.put(("status",f"已連線 {', '.join(p for p,_ in opened)}。先做 2 秒姿態歸零。"))
    except Exception as e:
        uiq.put(("status",f"COM 連線失敗：{e}"))

def start_pose():
    global mode,pose_end
    if len(serials)<2:return
    mode="pose";pose_end=time.monotonic()+2.0
    with lock:
        for st in states.values():
            st.pose_samples.clear();st.R0=None;reset_hand(st)
    pose_btn.configure(state="disabled")
    status_var.set("2 秒姿態歸零中：雙手保持標準起始姿勢，不要動。")
    root.after(2100,finish_pose)

def finish_pose():
    global mode
    meta={};ok=True
    with lock:
        for h,st in states.items():
            try:
                st.R0,st.zero=compute_r0(st.pose_samples);reset_hand(st)
                meta[h]={"samples":len(st.pose_samples),"zero":[float(x) for x in st.zero],"port":st.port}
            except Exception as e:
                ok=False;meta[h]={"error":repr(e),"samples":len(st.pose_samples)}
    (SESSION/"pose_calibration.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    mode="idle"
    pose_btn.configure(state="normal")
    if ok:
        status_var.set("姿態歸零完成。下一步做一擊 few-shot 校準。")
        few_btn.configure(state="normal")
    else:status_var.set("姿態歸零失敗："+json.dumps(meta,ensure_ascii=False))

def start_fewshot():
    global mode,fewshot_idx,fewshot_records,active_models,ignore_until
    if any(states[h].R0 is None for h in ("L","R")):return
    mode="fewshot";fewshot_idx=0;fewshot_records=[];active_models=dict(BASE_MODELS);ignore_until=time.monotonic()+.5
    few_btn.configure(state="disabled");live_btn.configure(state="disabled")
    uiq.put(("target",))

def skip_target():
    if mode=="fewshot":advance_fewshot()

def start_live():
    global mode
    if any(states[h].R0 is None for h in ("L","R")):return
    mode="live"
    with lock:
        for st in states.values():reset_hand(st)
    status_var.set("正式即時辨識中。I2C guard + 120ms refractory 已啟用。")

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    for f in (raw_f,pred_f,cal_f):
        try:f.close()
        except:pass
    root.destroy()

root=tk.Tk()
root.title("HoloGrip Production vNext｜學校實戰")
root.geometry("1050x690");root.minsize(980,640);root.attributes("-topmost",True)
style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",22,"bold"))
style.configure("Big.TLabel",font=("Microsoft JhengHei UI",30,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Big.TButton",font=("Microsoft JhengHei UI",13,"bold"),padding=9)

ttk.Label(root,text="HoloGrip Production vNext｜學校實戰",style="Title.TLabel",anchor="center").pack(fill="x",pady=(18,7))
status_var=tk.StringVar(value="正在連接雙手套…")
target_var=tk.StringVar(value="流程：姿態歸零 → 一擊校準 → 正式辨識")
ttk.Label(root,textvariable=status_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=990).pack(fill="x",padx=20,pady=7)
ttk.Label(root,textvariable=target_var,style="Big.TLabel",anchor="center",justify="center").pack(fill="x",padx=20,pady=10)

frm=ttk.Frame(root);frm.pack(fill="both",expand=True,padx=18,pady=8);frm.columnconfigure(0,weight=1);frm.columnconfigure(1,weight=1)
u={}
for col,h in enumerate(("L","R")):
    box=ttk.LabelFrame(frm,text=("左手 L" if h=="L" else "右手 R"),padding=12);box.grid(row=0,column=col,sticky="nsew",padx=8)
    pred=tk.StringVar(value="—");info=tk.StringVar(value="等待資料")
    ttk.Label(box,textvariable=pred,style="Big.TLabel",anchor="center").pack(fill="x",pady=(25,10))
    ttk.Label(box,textvariable=info,style="Mid.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    u[h]=(pred,info)

btn=ttk.Frame(root);btn.pack(pady=15)
pose_btn=ttk.Button(btn,text="1. 2秒姿態歸零",style="Big.TButton",command=start_pose);pose_btn.pack(side="left",padx=5)
few_btn=ttk.Button(btn,text="2. 開始一擊校準",style="Big.TButton",command=start_fewshot,state="disabled");few_btn.pack(side="left",padx=5)
skip_btn=ttk.Button(btn,text="跳過目前 hand×鼓",style="Big.TButton",command=skip_target);skip_btn.pack(side="left",padx=5)
live_btn=ttk.Button(btn,text="3. 正式辨識",style="Big.TButton",command=start_live,state="disabled");live_btn.pack(side="left",padx=5)
ttk.Button(btn,text="關閉",style="Big.TButton",command=close).pack(side="left",padx=5)

def refresh_target():
    t=current_target()
    if t is None:target_var.set("一擊校準完成");return
    h,i=t
    target_var.set(f"請用 {'左手' if h=='L' else '右手'} 打 1 下：{ZONE_NAMES[i]}")

def update_ui():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":status_var.set(e[1])
        elif e[0]=="target":refresh_target()
        elif e[0]=="captured":
            status_var.set(f"已收：{e[1]} / {ZONE_NAMES[e[2]]}。0.8 秒後下一項。")
        elif e[0]=="adapt_done":
            target_var.set("Few-shot 重訓完成")
            status_var.set("Production adapted models 已完成。可以開始正式辨識。")
            live_btn.configure(state="normal")
        elif e[0]=="pred":
            pass
    with lock:
        for h,st in states.items():
            pred,info=u[h];pred.set(st.last_pred)
            info.set(f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜RX {st.rx}\n最後信心 {st.last_prob*100:.1f}%｜壞 frame {st.val.invalid}｜refractory suppress {st.gate.suppressed}")
    root.after(100,update_ui)

root.protocol("WM_DELETE_WINDOW",close)
root.after(100,update_ui)
root.after(250,lambda:threading.Thread(target=connect,daemon=True).start())
root.mainloop()
