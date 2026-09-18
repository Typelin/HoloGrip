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
sys.path.insert(0,str(APP));sys.path.insert(0,str(LAB));sys.path.insert(0,str(PROD))

from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,WINDOW_HALF_MS
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES,ZONE_NAMES
from production_core import FrameValidator,RefractoryGate

# Main seven-drum model: 8/12 only.
DRUM_MODELS={h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")}
# Coarse physical sanity models: also 8/12 only.
H_MODELS={h:joblib.load(LAB/"hierarchical_realdrum_models"/f"H_{h}.joblib") for h in ("L","R")}
V_MODELS={h:joblib.load(LAB/"hierarchical_realdrum_models"/f"V_{h}.joblib") for h in ("L","R")}

HMAP={"Crash":0,"Hi-Hat":0,"小鼓":1,"高音 Tom":1,"中音 Tom":1,"Ride":2,"落地 Tom":2}
VMAP={"Crash":0,"高音 Tom":0,"中音 Tom":0,"Ride":0,"Hi-Hat":1,"小鼓":1,"落地 Tom":1}
HN=["LEFT","CENTER","RIGHT"]
VN=["UP","LOW"]

CAL_SAMPLES=200
STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=LAB/"next_person_realdrum_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)

raw_f=(SESSION/"raw_100hz.csv").open("w",encoding="utf-8",newline="",buffering=1)
hit_f=(SESSION/"hits.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f);hit_w=csv.writer(hit_f)

raw_w.writerow([
    "wall_time","mode","port","hand","packet_id","sensor_ms",
    "ax","ay","az","yaw","pitch","roll","valid","invalid_reason"
])
hit_w.writerow([
    "wall_time","hand","port","peak_ms","peak_g",
    "raw_drum","raw_prob",
    "H_pred","H_prob","V_pred","V_prob",
    "raw_H","raw_V","side_conflict","vertical_disagree",
    "final_display",
    "rel_yaw","rel_pitch","rel_roll",
    *FEATURE_NAMES
])

meta={
    "purpose":"next-person zero-shot real-drum demo",
    "main_model_source":"8/12 P01 real-drum 31D per-hand vNext",
    "coarse_model_source":"8/12 P01 real-drum H/V classifiers",
    "runtime_adaptation":"NONE",
    "per_drum_calibration":"NONE",
    "neutral_calibration_valid_frames_per_hand":CAL_SAMPLES,
    "side_conflict_policy":"If main 7-drum side and coarse H are opposite extremes (LEFT<->RIGHT), suppress drum label and show direction conflict. Do not hard-correct.",
    "evidence":{
        "8_12_to_8_5_same_person_cross_day_main7_acc":0.9271255060728745,
        "8_12_to_8_5_horizontal_acc":0.9392712550607287,
        "8_12_to_8_5_vertical_acc":0.9433198380566802,
        "8_12_to_8_5_main7_opposite_side_errors":"0/204 side events"
    },
}
(SESSION/"session_meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

class HS:
    def __init__(self):
        self.port=""
        self.R0=None
        self.zero=None
        self.cal=[]
        self.buffer=deque(maxlen=500)
        self.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.gate=RefractoryGate(120)
        self.val=FrameValidator()
        self.pending=[]
        self.rx=0;self.hz=0.0
        self.hits=0
        self.last_drum="—"
        self.last_raw="—"
        self.last_prob=0.0
        self.last_h="—";self.last_v="—"
        self.last_hprob=0.0;self.last_vprob=0.0
        self.last_rel=(0.0,0.0,0.0)
        self.last_warning=""
        self.conflicts=0

states={"L":HS(),"R":HS()}
lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
uiq=queue.Queue()
mode="idle"  # idle/cal/live
cal_finish_queued=False

def iso():return datetime.now().astimezone().isoformat(timespec="milliseconds")

def parse(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"):return None
    try:
        return {
            "hand":f[1],
            "ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),
            "yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),
            "packet_id":int(f[8]),"sensor_ms":int(f[9])
        }
    except:return None

def xiao_ports():
    out=[]
    for p in list_ports.comports():
        hw=(p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:out.append(p.device)
    return sorted(out)

def reset_detector(st):
    st.buffer.clear();st.pending.clear()
    st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
    st.gate=RefractoryGate(120)

def connect_all():
    ps=xiao_ports()
    if len(ps)<2:
        uiq.put(("status",f"只找到 {len(ps)} 顆 XIAO：{ps}"));return
    opened=[]
    try:
        for p in ps[:2]:opened.append((p,serial.Serial(p,460800,timeout=.08)))
        time.sleep(.4)
        for p,s in opened:
            try:s.reset_input_buffer()
            except:pass
            serials[p]=s
            t=threading.Thread(target=worker,args=(p,s),daemon=True);t.start();threads.append(t)
        uiq.put(("status",f"已連線 {', '.join(p for p,_ in opened)}。先做 200-frame 0,0 歸零。"))
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except:pass
        uiq.put(("status",f"COM 連線失敗：{e}"))

def classify_hit(h,ft,near,peak_g,wall,peak_ms):
    st=states[h]
    dm=DRUM_MODELS[h]
    prob=dm.predict_proba([ft])[0]
    classes=[int(x) for x in dm.classes_]
    j=int(np.argmax(prob));raw_cls=classes[j]
    raw_name=ZONE_NAMES[raw_cls];raw_prob=float(prob[j])

    hm=H_MODELS[h];hp=hm.predict_proba([ft])[0]
    hj=int(np.argmax(hp));hcls=int(hm.classes_[hj]);hprob=float(hp[hj])

    vm=V_MODELS[h];vp=vm.predict_proba([ft])[0]
    vj=int(np.argmax(vp));vcls=int(vm.classes_[vj]);vprob=float(vp[vj])

    raw_h=HMAP[raw_name];raw_v=VMAP[raw_name]
    opposite=abs(raw_h-hcls)==2
    vdiff=raw_v!=vcls

    if opposite:
        final="方向衝突 / 不確定"
        warning=f"七鼓={HN[raw_h]}，水平模型={HN[hcls]}，拒絕跨側輸出"
        st.conflicts+=1
    else:
        final=raw_name
        warning=("上下模型不同意" if vdiff else "")

    st.hits+=1
    st.last_drum=final;st.last_raw=raw_name;st.last_prob=raw_prob
    st.last_h=HN[hcls];st.last_v=VN[vcls]
    st.last_hprob=hprob;st.last_vprob=vprob
    st.last_rel=(near["rel_yaw"],near["rel_pitch"],near["rel_roll"])
    st.last_warning=warning

    hit_w.writerow([
        wall,h,st.port,f"{peak_ms:.3f}",f"{peak_g:.5f}",
        raw_name,f"{raw_prob:.6f}",
        HN[hcls],f"{hprob:.6f}",VN[vcls],f"{vprob:.6f}",
        HN[raw_h],VN[raw_v],int(opposite),int(vdiff),
        final,
        f"{near['rel_yaw']:.4f}",f"{near['rel_pitch']:.4f}",f"{near['rel_roll']:.4f}",
        *[f"{float(x):.8f}" for x in ft]
    ]);hit_f.flush()
    uiq.put(("hit",h,final,warning))

def worker(port,conn):
    global cal_finish_queued
    n=0;tick=time.monotonic()
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:continue
            p=parse(b.decode("utf-8","ignore").strip())
            if p is None:continue
            h=p["hand"];st=states[h]
            now=time.monotonic();nowms=now*1000.0;wall=iso()

            with lock:
                st.port=port;st.rx+=1;n+=1
                if now-tick>=1.0:
                    st.hz=n/(now-tick);n=0;tick=now

                valid,reason=st.val.check(p)
                raw_w.writerow([wall,mode,port,h,p["packet_id"],p["sensor_ms"],
                                p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],
                                int(valid),reason])
                if st.rx%50==0:raw_f.flush()

                if not valid:
                    reset_detector(st)
                    continue

                if mode=="cal":
                    if len(st.cal)<CAL_SAMPLES:
                        st.cal.append((p["yaw"],p["pitch"],p["roll"]))
                    if (not cal_finish_queued and
                        len(states["L"].cal)>=CAL_SAMPLES and len(states["R"].cal)>=CAL_SAMPLES):
                        cal_finish_queued=True
                        uiq.put(("cal_ready",))
                    continue

                if mode!="live" or st.R0 is None:continue

                rs=make_relative_sample(st.R0,nowms,
                    p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
                st.buffer.append(rs)

                pkt=SensorPacket(
                    hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],
                    yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
                    packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],
                    received_time_ms=int(time.time()*1000),received_monotonic=now)
                ev=st.det.add_packet(pkt,rs["rel_yaw"],rs["rel_pitch"])
                if ev:
                    peak=nowms-PEAK_LAG_MS
                    if st.gate.accept(peak):
                        st.pending.append({"peak_ms":peak,"peak_g":float(ev["peak_accel_g"])})

                remain=[]
                for item in st.pending:
                    if nowms<item["peak_ms"]+WINDOW_HALF_MS:
                        remain.append(item);continue
                    ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                    if ft is None:continue
                    near=min(st.buffer,key=lambda s:abs(s["t"]-item["peak_ms"]))
                    classify_hit(h,ft,near,item["peak_g"],wall,item["peak_ms"])
                st.pending=remain
    except Exception as e:
        uiq.put(("status",f"{port} reader error: {e!r}"))

def start_cal():
    global mode,cal_finish_queued
    if len(serials)<2:return
    mode="cal";cal_finish_queued=False
    with lock:
        for st in states.values():
            st.cal.clear();st.R0=None;st.zero=None
            st.hits=0;st.conflicts=0
            st.last_drum="—";st.last_raw="—";st.last_warning=""
            reset_detector(st)
    cal_btn.configure(state="disabled")
    status_var.set("0,0 歸零中：雙手保持你定義的標準起始姿勢。每手收滿 200 個有效 frame。")
    title_var.set("200-frame 0,0 歸零中")

def finish_cal():
    global mode
    info={};ok=True
    with lock:
        for h,st in states.items():
            try:
                st.R0,st.zero=compute_r0(st.cal[:CAL_SAMPLES])
                reset_detector(st)
                info[h]={"samples":len(st.cal),"zero":[float(x) for x in st.zero],"port":st.port}
            except Exception as e:
                ok=False;info[h]={"error":repr(e),"samples":len(st.cal),"port":st.port}
    (SESSION/"calibration.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    cal_btn.configure(state="normal")
    if not ok:
        mode="idle"
        title_var.set("歸零失敗")
        status_var.set(json.dumps(info,ensure_ascii=False))
        return
    mode="live"
    title_var.set("LIVE：直接打真鼓")
    status_var.set("模型已凍結｜不重訓｜不做每鼓校準。現在直接打七鼓；每擊會即時顯示。")

def close():
    stop.set()
    summary={
        "session":str(SESSION),
        "hits":{h:states[h].hits for h in ("L","R")},
        "side_conflicts":{h:states[h].conflicts for h in ("L","R")},
        "invalid_frames":{h:states[h].val.invalid for h in ("L","R")},
        "invalid_reasons":{h:dict(states[h].val.reasons) for h in ("L","R")},
    }
    try:(SESSION/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    except:pass
    for s in serials.values():
        try:s.close()
        except:pass
    for f in (raw_f,hit_f):
        try:f.close()
        except:pass
    root.destroy()

root=tk.Tk()
root.title("HoloGrip｜下一位測試者｜真七鼓 Zero-Shot Demo")
root.geometry("1120x760");root.minsize(1020,690);root.attributes("-topmost",True)

style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",22,"bold"))
style.configure("Drum.TLabel",font=("Microsoft JhengHei UI",34,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Warn.TLabel",font=("Microsoft JhengHei UI",13,"bold"))
style.configure("Btn.TButton",font=("Microsoft JhengHei UI",13,"bold"),padding=10)

ttk.Label(root,text="HoloGrip｜下一位測試者真七鼓 Zero-Shot",style="Title.TLabel",anchor="center").pack(fill="x",pady=(16,4))
title_var=tk.StringVar(value="等待 0,0 歸零")
ttk.Label(root,textvariable=title_var,style="Title.TLabel",anchor="center").pack(fill="x",pady=6)
status_var=tk.StringVar(value="正在連線 COM4 / COM5…")
ttk.Label(root,textvariable=status_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=1060).pack(fill="x",padx=18,pady=6)

frm=ttk.Frame(root);frm.pack(fill="both",expand=True,padx=18,pady=8)
frm.columnconfigure(0,weight=1);frm.columnconfigure(1,weight=1)
ui={}
for col,h in enumerate(("L","R")):
    box=ttk.LabelFrame(frm,text=("左手 L" if h=="L" else "右手 R"),padding=14)
    box.grid(row=0,column=col,sticky="nsew",padx=8)
    drum=tk.StringVar(value="—")
    raw=tk.StringVar(value="原始七鼓：—")
    coarse=tk.StringVar(value="H —｜V —")
    ang=tk.StringVar(value="相對角：Yaw —｜Pitch —｜Roll —")
    info=tk.StringVar(value="等待資料")
    warn=tk.StringVar(value="")
    ttk.Label(box,textvariable=drum,style="Drum.TLabel",anchor="center").pack(fill="x",pady=(20,8))
    ttk.Label(box,textvariable=raw,style="Mid.TLabel",anchor="center").pack(fill="x",pady=4)
    ttk.Label(box,textvariable=coarse,style="Mid.TLabel",anchor="center").pack(fill="x",pady=4)
    ttk.Label(box,textvariable=ang,anchor="center").pack(fill="x",pady=4)
    ttk.Label(box,textvariable=warn,style="Warn.TLabel",anchor="center",justify="center",wraplength=500).pack(fill="x",pady=8)
    ttk.Label(box,textvariable=info,anchor="center",justify="center").pack(fill="x",pady=8)
    ui[h]=(drum,raw,coarse,ang,warn,info)

btn=ttk.Frame(root);btn.pack(pady=15)
cal_btn=ttk.Button(btn,text="200-frame 0,0 歸零並開始",style="Btn.TButton",command=start_cal)
cal_btn.pack(side="left",padx=7)
ttk.Button(btn,text="關閉",style="Btn.TButton",command=close).pack(side="left",padx=7)

foot=tk.StringVar(value="主答案：8/12 真鼓 31D per-hand MLP｜H/V：物理 sanity check｜新測試者資料不會拿來訓練")
ttk.Label(root,textvariable=foot,anchor="center").pack(fill="x",pady=(0,12))

def update():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":status_var.set(e[1])
        elif e[0]=="cal_ready":finish_cal()
    with lock:
        for h,st in states.items():
            drum,raw,coarse,ang,warn,info=ui[h]
            if mode=="cal":
                drum.set(f"R0 {min(len(st.cal),CAL_SAMPLES)} / {CAL_SAMPLES}")
            else:
                drum.set(st.last_drum)
            raw.set(f"原始七鼓：{st.last_raw}｜信心 {st.last_prob*100:.1f}%")
            coarse.set(f"H {st.last_h} {st.last_hprob*100:.1f}%｜V {st.last_v} {st.last_vprob*100:.1f}%")
            y,p,r=st.last_rel
            ang.set(f"相對角：Yaw {y:+.1f}°｜Pitch {p:+.1f}°｜Roll {r:+.1f}°")
            warn.set(st.last_warning)
            info.set(f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜Hits {st.hits}\n壞 frame {st.val.invalid}｜方向衝突 {st.conflicts}")
    root.after(100,update)

root.protocol("WM_DELETE_WINDOW",close)
root.after(100,update)
root.after(250,lambda:threading.Thread(target=connect_all,daemon=True).start())
root.mainloop()
