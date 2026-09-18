from __future__ import annotations
import csv,json,math,queue,sys,threading,time
from collections import deque,Counter
from datetime import datetime
from pathlib import Path

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
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES
from production_core import FrameValidator,RefractoryGate

POSITIONS=["左上","中上","右上","左中","正中","右中"]
TARGET_PER_HAND=5
CAL_SAMPLES_PER_HAND=200
COUNT_REFRACTORY_MS=350.0

STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=LAB/"count6_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)

raw_f=(SESSION/"raw_100hz.csv").open("w",encoding="utf-8",newline="",buffering=1)
ev_f=(SESSION/"hit_candidates.csv").open("w",encoding="utf-8",newline="",buffering=1)
stage_f=(SESSION/"stage_events.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f);ev_w=csv.writer(ev_f);stage_w=csv.writer(stage_f)

raw_w.writerow([
    "wall_time","mode","stage_index","stage_name","port","hand","packet_id","sensor_ms",
    "ax","ay","az","yaw","pitch","roll","frame_valid","invalid_reason"
])
ev_w.writerow([
    "wall_time","stage_index","stage_name","mode","hand","port",
    "peak_ms","peak_g","production_gate","count_gate","counted",
    "stage_hand_count_after","reason",*FEATURE_NAMES
])
stage_w.writerow(["wall_time","event","stage_index","stage_name","L_count","R_count"])

meta={
    "session":str(SESSION),
    "positions":POSITIONS,
    "target_per_hand":TARGET_PER_HAND,
    "calibration_valid_samples_per_hand":CAL_SAMPLES_PER_HAND,
    "production_refractory_ms":120,
    "collection_count_refractory_ms":COUNT_REFRACTORY_MS,
    "note":"This run is data collection/diagnosis only. No model prediction is used to advance stages."
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
        self.prod_gate=RefractoryGate(120)
        self.val=FrameValidator()
        self.pending=[]
        self.rx=0
        self.hz=0.0
        self.last_count_ms=-1e18
        self.count=0
        self.invalid=0
        self.last_packet_time=0.0

states={"L":HS(),"R":HS()}
lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
uiq=queue.Queue()

mode="idle"   # idle/cal/wait_stage/collect/finished
stage_index=-1

def iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def parse(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"):
        return None
    try:
        return {
            "hand":f[1],
            "ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),
            "yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),
            "packet_id":int(f[8]),"sensor_ms":int(f[9]),
        }
    except Exception:
        return None

def xiao_ports():
    out=[]
    for p in list_ports.comports():
        hw=(p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:
            out.append(p.device)
    return sorted(out)

def reset_detector(st):
    st.det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
    st.prod_gate=RefractoryGate(120)
    st.pending.clear()
    st.buffer.clear()

def stage_name():
    return POSITIONS[stage_index] if 0<=stage_index<len(POSITIONS) else ""

def log_stage(event):
    with lock:
        stage_w.writerow([iso(),event,stage_index,stage_name(),states["L"].count,states["R"].count])
        stage_f.flush()

def serial_worker(port,conn):
    local_n=0;tick=time.monotonic()
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:continue
            p=parse(b.decode("utf-8","ignore").strip())
            if p is None:continue
            h=p["hand"];st=states[h]
            now=time.monotonic();nowms=now*1000.0;wall=iso()

            with lock:
                st.port=port;st.rx+=1;st.last_packet_time=now
                local_n+=1
                if now-tick>=1.0:
                    st.hz=local_n/(now-tick);local_n=0;tick=now

                valid,reason=st.val.check(p)
                st.invalid=st.val.invalid
                raw_w.writerow([
                    wall,mode,stage_index,stage_name(),port,h,p["packet_id"],p["sensor_ms"],
                    p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],
                    int(valid),reason
                ])
                if st.rx%50==0:raw_f.flush()

                if not valid:
                    reset_detector(st)
                    continue

                if mode=="cal":
                    if len(st.cal)<CAL_SAMPLES_PER_HAND:
                        st.cal.append((p["yaw"],p["pitch"],p["roll"]))
                    if len(states["L"].cal)>=CAL_SAMPLES_PER_HAND and len(states["R"].cal)>=CAL_SAMPLES_PER_HAND:
                        uiq.put(("cal_ready",))
                    continue

                if st.R0 is None:
                    continue

                rs=make_relative_sample(
                    st.R0,nowms,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"]
                )
                st.buffer.append(rs)

                # We still run detector in wait_stage so transition motion is fully observable,
                # but only collect mode can increment counts.
                if mode not in ("collect","wait_stage"):
                    continue

                pkt=SensorPacket(
                    hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],
                    yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
                    packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],
                    received_time_ms=int(time.time()*1000),received_monotonic=now
                )
                ev=st.det.add_packet(pkt,rs["rel_yaw"],rs["rel_pitch"])
                if ev:
                    peak=nowms-PEAK_LAG_MS
                    prod_ok=st.prod_gate.accept(peak)
                    st.pending.append({
                        "peak_ms":peak,
                        "peak_g":float(ev["peak_accel_g"]),
                        "prod_ok":prod_ok
                    })

                remain=[]
                for item in st.pending:
                    if nowms < item["peak_ms"]+WINDOW_HALF_MS:
                        remain.append(item);continue

                    ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                    if ft is None:
                        continue

                    prod_ok=bool(item["prod_ok"])
                    count_gate=False
                    counted=False
                    why=""
                    c_after=st.count

                    if not prod_ok:
                        why="production_120ms_suppressed"
                    elif mode!="collect":
                        why="transition_wait"
                    elif st.count>=TARGET_PER_HAND:
                        why="hand_target_already_full"
                    elif item["peak_ms"]-st.last_count_ms < COUNT_REFRACTORY_MS:
                        why="collection_350ms_suppressed"
                    else:
                        count_gate=True
                        counted=True
                        st.last_count_ms=item["peak_ms"]
                        st.count+=1
                        c_after=st.count
                        why="counted"

                    ev_w.writerow([
                        wall,stage_index,stage_name(),mode,h,port,
                        f"{item['peak_ms']:.3f}",f"{item['peak_g']:.5f}",
                        int(prod_ok),int(count_gate),int(counted),c_after,why,
                        *[f"{float(x):.8f}" for x in ft]
                    ])
                    ev_f.flush()

                    if counted:
                        uiq.put(("count",h,st.count))
                        if states["L"].count>=TARGET_PER_HAND and states["R"].count>=TARGET_PER_HAND:
                            uiq.put(("stage_complete",stage_index))
                st.pending=remain
    except Exception as e:
        uiq.put(("status",f"{port} reader error: {e!r}"))

def connect_all():
    ps=xiao_ports()
    if len(ps)<2:
        uiq.put(("status",f"只找到 {len(ps)} 顆 XIAO：{ps}"));return
    opened=[]
    try:
        for p in ps[:2]:
            s=serial.Serial(p,460800,timeout=.08);opened.append((p,s))
        time.sleep(.4)
        for p,s in opened:
            try:s.reset_input_buffer()
            except:pass
            serials[p]=s
            t=threading.Thread(target=serial_worker,args=(p,s),daemon=True)
            t.start();threads.append(t)
        uiq.put(("status",f"已連線：{', '.join(p for p,_ in opened)}。先按『開始歸零』。"))
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except:pass
        uiq.put(("status",f"COM 連線失敗：{e}"))

def start_cal():
    global mode,stage_index
    if len(serials)<2:return
    mode="cal";stage_index=-1
    with lock:
        for st in states.values():
            st.cal.clear();st.R0=None;st.zero=None;st.count=0;st.last_count_ms=-1e18
            reset_detector(st)
    cal_btn.configure(state="disabled")
    next_btn.configure(state="disabled")
    status_var.set("歸零中：雙手保持標準起始姿勢。每手收滿 200 個有效 frame 自動完成。")

def finish_cal():
    global mode,stage_index
    if mode!="cal":return
    meta={}
    ok=True
    with lock:
        for h,st in states.items():
            try:
                st.R0,st.zero=compute_r0(st.cal[:CAL_SAMPLES_PER_HAND])
                reset_detector(st)
                meta[h]={"samples":len(st.cal),"zero":[float(x) for x in st.zero],"port":st.port}
            except Exception as e:
                ok=False;meta[h]={"error":repr(e),"samples":len(st.cal),"port":st.port}
    (SESSION/"calibration.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    if not ok:
        mode="idle";cal_btn.configure(state="normal")
        status_var.set("歸零失敗："+json.dumps(meta,ensure_ascii=False));return

    stage_index=0
    mode="wait_stage"
    with lock:
        for st in states.values():
            st.count=0;st.last_count_ms=-1e18;reset_detector(st)
    cal_btn.configure(state="normal")
    next_btn.configure(state="normal")
    title_var.set("準備：左上")
    prompt_var.set("移到『左上』。準備好後按『開始目前位置』。\n每隻手各打 5 下；每下請明確分開。")
    status_var.set("歸零完成。Raw 仍持續收集；目前等待你開始左上。")
    log_stage("CALIBRATION_DONE")

def start_current_stage():
    global mode
    if stage_index<0 or stage_index>=len(POSITIONS):return
    with lock:
        for st in states.values():
            st.count=0;st.last_count_ms=-1e18;reset_detector(st)
    mode="collect"
    next_btn.configure(state="disabled")
    title_var.set(f"正在收集：{stage_name()}")
    prompt_var.set(f"位置：{stage_name()}\n左手 5 下 + 右手 5 下。兩手都滿 5 才會完成。")
    status_var.set("開始打。這輪不顯示模型答案，只收乾淨資料。")
    log_stage("STAGE_START")

def complete_stage(idx):
    global mode
    if mode!="collect" or idx!=stage_index:return
    mode="wait_stage"
    log_stage("STAGE_COMPLETE")
    if stage_index==len(POSITIONS)-1:
        finish_all()
        return
    title_var.set(f"{stage_name()} 完成")
    prompt_var.set(f"這一段 L=5 / R=5 已收滿。\n移到下一位置後，按『下一位置』。")
    next_btn.configure(text="下一位置",state="normal")
    status_var.set("切換期間 Raw 繼續 100 Hz 保存，但 hit 不會算進下一段。")

def go_next():
    global stage_index
    if mode!="wait_stage":return
    # If this position is prepared but has not started yet, start it.
    # A completed position has 5/5, so only then advance to the next position.
    if states["L"].count==0 and states["R"].count==0:
        start_current_stage();return
    stage_index+=1
    if stage_index>=len(POSITIONS):
        finish_all();return
    with lock:
        for st in states.values():
            st.count=0;st.last_count_ms=-1e18;reset_detector(st)
    title_var.set(f"準備：{stage_name()}")
    prompt_var.set(f"移到『{stage_name()}』。準備好後按『開始目前位置』。")
    next_btn.configure(text="開始目前位置",state="normal")
    status_var.set("切換完成後再開始。Raw 一直在收。")

def reset_stage():
    if mode not in ("collect","wait_stage") or stage_index<0:return
    with lock:
        for st in states.values():
            st.count=0;st.last_count_ms=-1e18;reset_detector(st)
    log_stage("STAGE_RESET")
    if mode=="collect":
        status_var.set("目前位置計數已重置為 0/5；重新打。")
    else:
        status_var.set("目前位置計數已重置。")

def finish_all():
    global mode
    mode="finished"
    log_stage("SESSION_COMPLETE")
    raw_f.flush();ev_f.flush();stage_f.flush()
    summary={
        "session":str(SESSION),
        "positions":POSITIONS,
        "target_per_hand":TARGET_PER_HAND,
        "rx":{"L":states["L"].rx,"R":states["R"].rx},
        "invalid":{"L":states["L"].val.invalid,"R":states["R"].val.invalid},
        "invalid_reasons":{"L":dict(states["L"].val.reasons),"R":dict(states["R"].val.reasons)},
    }
    (SESSION/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    title_var.set("六位置採集完成")
    prompt_var.set("全部完成。請直接告訴 ChatGPT『完成』，我會讀 Raw / hit_candidates / stage_events 做完整分析。")
    next_btn.configure(state="disabled")
    status_var.set(f"Session：{SESSION.name}")

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    for f in (raw_f,ev_f,stage_f):
        try:f.close()
        except:pass
    root.destroy()

root=tk.Tk()
root.title("HoloGrip｜六位置 × 每手5下｜純資料採集")
root.geometry("1000x650")
root.minsize(930,600)
root.attributes("-topmost",True)

style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",24,"bold"))
style.configure("Big.TLabel",font=("Microsoft JhengHei UI",34,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Big.TButton",font=("Microsoft JhengHei UI",13,"bold"),padding=9)

ttk.Label(root,text="HoloGrip｜六位置計數式資料採集",style="Title.TLabel",anchor="center").pack(fill="x",pady=(18,6))
title_var=tk.StringVar(value="等待歸零")
ttk.Label(root,textvariable=title_var,style="Big.TLabel",anchor="center").pack(fill="x",pady=8)

prompt_var=tk.StringVar(value="按『開始歸零』。歸零不是計時：左右手各收滿 200 個有效 frame 才完成。")
ttk.Label(root,textvariable=prompt_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=950).pack(fill="x",padx=20,pady=8)

counts=ttk.Frame(root);counts.pack(fill="x",padx=25,pady=14);counts.columnconfigure(0,weight=1);counts.columnconfigure(1,weight=1)
ui={}
for col,h in enumerate(("L","R")):
    box=ttk.LabelFrame(counts,text=("左手 L" if h=="L" else "右手 R"),padding=16)
    box.grid(row=0,column=col,sticky="nsew",padx=10)
    cnt=tk.StringVar(value="0 / 5")
    info=tk.StringVar(value="等待資料")
    ttk.Label(box,textvariable=cnt,style="Big.TLabel",anchor="center").pack(fill="x",pady=(15,8))
    ttk.Label(box,textvariable=info,style="Mid.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    ui[h]=(cnt,info)

status_var=tk.StringVar(value="正在連接兩隻 XIAO…")
ttk.Label(root,textvariable=status_var,anchor="center",justify="center",wraplength=950).pack(fill="x",padx=20,pady=8)

btn=ttk.Frame(root);btn.pack(pady=14)
cal_btn=ttk.Button(btn,text="開始歸零",style="Big.TButton",command=start_cal)
cal_btn.pack(side="left",padx=5)
next_btn=ttk.Button(btn,text="開始目前位置",style="Big.TButton",command=go_next,state="disabled")
next_btn.pack(side="left",padx=5)
ttk.Button(btn,text="重置目前位置",style="Big.TButton",command=reset_stage).pack(side="left",padx=5)
ttk.Button(btn,text="關閉",style="Big.TButton",command=close).pack(side="left",padx=5)

def update_ui():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":
            status_var.set(e[1])
        elif e[0]=="cal_ready":
            finish_cal()
        elif e[0]=="count":
            pass
        elif e[0]=="stage_complete":
            complete_stage(e[1])

    with lock:
        for h,st in states.items():
            cnt,info=ui[h]
            if mode=="cal":
                cnt.set(f"歸零 {min(len(st.cal),CAL_SAMPLES_PER_HAND)} / {CAL_SAMPLES_PER_HAND}")
            else:
                cnt.set(f"{st.count} / {TARGET_PER_HAND}")
            info.set(
                f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜RX {st.rx}\n"
                f"壞 frame {st.val.invalid}｜候選防重 suppress {st.prod_gate.suppressed}"
            )
    root.after(100,update_ui)

root.protocol("WM_DELETE_WINDOW",close)
root.bind("<space>",lambda e: go_next())
root.after(100,update_ui)
root.after(250,lambda:threading.Thread(target=connect_all,daemon=True).start())
root.mainloop()
