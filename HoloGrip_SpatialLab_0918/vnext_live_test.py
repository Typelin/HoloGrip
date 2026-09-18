from __future__ import annotations
import csv, json, math, queue, sys, threading, time
from collections import deque, Counter
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
sys.path.insert(0,str(APP))
sys.path.insert(0,str(LAB))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, ZONE_NAMES

MODELS={
    h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib")
    for h in ("L","R")
}

STAMP=datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION=LAB/"vnext_live_sessions"/STAMP
SESSION.mkdir(parents=True,exist_ok=True)

raw_f=(SESSION/"raw.csv").open("w",encoding="utf-8",newline="",buffering=1)
hit_f=(SESSION/"hits.csv").open("w",encoding="utf-8",newline="",buffering=1)
raw_w=csv.writer(raw_f); hit_w=csv.writer(hit_f)
raw_w.writerow(["wall_time","t_ms","port","hand","packet_id","ax","ay","az","yaw","pitch","roll","rel_yaw","rel_pitch","rel_roll","calibrated"])
hit_w.writerow(["wall_time","hand","port","peak_ms","pred","prob","stability","recent","peak_accel_g"])
log_lock=threading.Lock()

class HState:
    def __init__(self):
        self.port=""
        self.R0=None
        self.zero=None
        self.cal_samples=[]
        self.buffer=deque(maxlen=450)
        self.detector=HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.pending=[]
        self.recent=deque(maxlen=10)
        self.last_pred="—"
        self.last_prob=0.0
        self.hit_count=0
        self.hz=0.0
        self.rx=0
        self.bad=0
        self.last_packet=0.0
        self.peak_g=0.0
        self.last_rel=(0.0,0.0,0.0)

states={"L":HState(),"R":HState()}
state_lock=threading.RLock()
stop=threading.Event()
serials={}
threads=[]
calibrating=False
calibrated=False
cal_start=0.0
cal_duration=2.0
uiq=queue.Queue()

def parse_line(line):
    f=line.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"):
        return None
    try:
        return {
            "hand":f[1],"ax":float(f[2]),"ay":float(f[3]),"az":float(f[4]),
            "yaw":float(f[5]),"pitch":float(f[6]),"roll":float(f[7]),
            "packet_id":int(f[8]),"sensor_ms":int(f[9])
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

def stability_for(st):
    if not st.recent:
        return 0.0
    c=Counter(st.recent)
    return max(c.values())/len(st.recent)

def serial_worker(port, conn):
    global calibrated, calibrating
    n=0; tick=time.monotonic()
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:
                continue
            line=b.decode("utf-8","ignore").strip()
            p=parse_line(line)
            if p is None:
                if line.startswith("D,"):
                    with state_lock:
                        for st in states.values():
                            if st.port==port: st.bad+=1
                continue
            now=time.monotonic()
            now_ms=now*1000.0
            wall=datetime.now().astimezone().isoformat(timespec="milliseconds")
            h=p["hand"]
            pkt=SensorPacket(
                hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],
                yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
                packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],
                received_time_ms=int(time.time()*1000),received_monotonic=now
            )
            with state_lock:
                st=states[h]
                st.port=port
                st.rx+=1; n+=1; st.last_packet=now
                if now-tick>=1.0:
                    st.hz=n/(now-tick); n=0; tick=now

                rel_y=rel_p=rel_r=""
                if calibrating:
                    st.cal_samples.append((p["yaw"],p["pitch"],p["roll"]))
                elif calibrated and st.R0 is not None:
                    sm=make_relative_sample(st.R0,now_ms,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
                    st.buffer.append(sm)
                    st.last_rel=(sm["rel_yaw"],sm["rel_pitch"],sm["rel_roll"])
                    rel_y,rel_p,rel_r=st.last_rel
                    ev=st.detector.add_packet(pkt,sm["rel_yaw"],sm["rel_pitch"])
                    if ev:
                        st.pending.append({"peak_ms":now_ms-PEAK_LAG_MS,"peak_g":ev["peak_accel_g"]})
                    remain=[]
                    for item in st.pending:
                        if now_ms>=item["peak_ms"]+WINDOW_HALF_MS:
                            ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                            if ft is None:
                                continue
                            m=MODELS[h]
                            proba=m.predict_proba([ft])[0]
                            j=int(np.argmax(proba))
                            cls=int(m.classes_[j])
                            label=ZONE_NAMES[cls]
                            prob=float(proba[j])
                            st.recent.append(label)
                            st.last_pred=label
                            st.last_prob=prob
                            st.hit_count+=1
                            st.peak_g=float(item["peak_g"])
                            stab=stability_for(st)
                            with log_lock:
                                hit_w.writerow([wall,h,port,item["peak_ms"],label,f"{prob:.6f}",f"{stab:.4f}","|".join(st.recent),st.peak_g])
                            uiq.put(("hit",h,label,prob,stab,list(st.recent)))
                        else:
                            remain.append(item)
                    st.pending=remain

                with log_lock:
                    raw_w.writerow([wall,f"{now_ms:.3f}",port,h,p["packet_id"],p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],rel_y,rel_p,rel_r,int(calibrated)])
    except Exception as e:
        uiq.put(("error",port,repr(e)))

def connect_all():
    ports=xiao_ports()
    if len(ports)<2:
        uiq.put(("status",f"只找到 {len(ports)} 顆 XIAO：{ports}。請確認兩隻手套都插著。"))
        return
    opened=[]
    try:
        for p in ports[:2]:
            s=serial.Serial(p,460800,timeout=0.08)
            opened.append((p,s))
        time.sleep(0.5)
        for p,s in opened:
            try:s.reset_input_buffer()
            except:pass
            serials[p]=s
            t=threading.Thread(target=serial_worker,args=(p,s),daemon=True)
            t.start();threads.append(t)
        uiq.put(("status",f"已連線：{', '.join(p for p,_ in opened)}。請把雙手放到固定標準姿勢，3 秒後自動校正。"))
        root.after(3000,start_calibration)
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except:pass
        uiq.put(("status",f"COM 連線失敗：{e}"))

def start_calibration():
    global calibrating, calibrated, cal_start
    with state_lock:
        calibrated=False
        calibrating=True
        cal_start=time.monotonic()
        for st in states.values():
            st.cal_samples=[]
            st.R0=None; st.zero=None
            st.buffer.clear(); st.pending.clear(); st.recent.clear()
            st.last_pred="—"; st.last_prob=0; st.hit_count=0
            st.detector=HitDetector(**PRODUCT_DETECTOR_KWARGS)
    status_var.set("校正中 2 秒：雙手保持完全不動…")
    root.after(int(cal_duration*1000)+150,finish_calibration)

def finish_calibration():
    global calibrating, calibrated
    with state_lock:
        calibrating=False
        ok=True; meta={}
        for h,st in states.items():
            try:
                R0,zero=compute_r0(st.cal_samples)
                st.R0=R0;st.zero=zero
                st.buffer.clear();st.pending.clear()
                st.detector=HitDetector(**PRODUCT_DETECTOR_KWARGS)
                meta[h]={"samples":len(st.cal_samples),"zero":[float(x) for x in zero],"port":st.port}
            except Exception as e:
                ok=False
                meta[h]={"error":repr(e),"samples":len(st.cal_samples),"port":st.port}
        calibrated=ok
    (SESSION/"calibration.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    if ok:
        status_var.set("校正完成。現在開始敲：同一個位置／同一種動作連續敲 8–10 次，看結果是否穩定。")
    else:
        status_var.set("校正失敗：" + json.dumps(meta,ensure_ascii=False))

def close_app():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    try:raw_f.close();hit_f.close()
    except:pass
    root.destroy()

root=tk.Tk()
root.title("HoloGrip vNext｜新模型即時穩定性測試")
root.geometry("1040x650")
root.minsize(980,600)
root.attributes("-topmost",True)

style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",22,"bold"))
style.configure("Big.TLabel",font=("Microsoft JhengHei UI",32,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Small.TLabel",font=("Microsoft JhengHei UI",11))
style.configure("Big.TButton",font=("Microsoft JhengHei UI",14,"bold"),padding=9)

ttk.Label(root,text="HoloGrip vNext｜全新模型即時測試",style="Title.TLabel",anchor="center").pack(fill="x",pady=(20,8))
status_var=tk.StringVar(value="正在載入新模型並尋找兩隻 XIAO…")
ttk.Label(root,textvariable=status_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=980).pack(fill="x",padx=20,pady=8)

main=ttk.Frame(root)
main.pack(fill="both",expand=True,padx=18,pady=10)
main.columnconfigure(0,weight=1);main.columnconfigure(1,weight=1)

ui={}
for col,h in enumerate(("L","R")):
    frame=ttk.LabelFrame(main,text=("左手 L" if h=="L" else "右手 R"),padding=14)
    frame.grid(row=0,column=col,sticky="nsew",padx=8)
    frame.columnconfigure(0,weight=1)
    pred=tk.StringVar(value="—")
    info=tk.StringVar(value="等待資料")
    recent=tk.StringVar(value="最近 10 擊：—")
    rel=tk.StringVar(value="相對姿態：—")
    ttk.Label(frame,textvariable=pred,style="Big.TLabel",anchor="center").grid(row=0,column=0,sticky="ew",pady=(20,12))
    ttk.Label(frame,textvariable=info,style="Mid.TLabel",anchor="center",justify="center").grid(row=1,column=0,sticky="ew",pady=8)
    ttk.Label(frame,textvariable=recent,style="Small.TLabel",anchor="center",justify="center",wraplength=430).grid(row=2,column=0,sticky="ew",pady=10)
    ttk.Label(frame,textvariable=rel,style="Small.TLabel",anchor="center").grid(row=3,column=0,sticky="ew",pady=6)
    ui[h]={"pred":pred,"info":info,"recent":recent,"rel":rel}

note=tk.StringVar(value="測法：校正完成後，選一個固定方向／固定揮法連敲 8–10 次。你現在在家不用對應真鼓；我們先看『同動作是否穩定判成同一類』。")
ttk.Label(root,textvariable=note,style="Mid.TLabel",anchor="center",justify="center",wraplength=980).pack(fill="x",padx=20,pady=10)

buttons=ttk.Frame(root);buttons.pack(pady=(0,18))
ttk.Button(buttons,text="重新做 2 秒校正",style="Big.TButton",command=start_calibration).pack(side="left",padx=8)
ttk.Button(buttons,text="關閉",style="Big.TButton",command=close_app).pack(side="left",padx=8)

def update_ui():
    while True:
        try:
            e=uiq.get_nowait()
        except queue.Empty:
            break
        if e[0]=="status": status_var.set(e[1])
        elif e[0]=="error": status_var.set(f"{e[1]} 讀取中斷：{e[2]}")
    with state_lock:
        now=time.monotonic()
        for h,st in states.items():
            stab=stability_for(st)
            ui[h]["pred"].set(st.last_pred)
            age=(now-st.last_packet) if st.last_packet else 999
            ui[h]["info"].set(
                f"{st.port or '未辨識'}｜{st.hz:.1f} Hz｜Hits {st.hit_count}\n"
                f"信心 {st.last_prob*100:.1f}%｜最近穩定度 {stab*100:.0f}%｜packet age {age:.2f}s"
            )
            ui[h]["recent"].set("最近 10 擊：" + (" → ".join(st.recent) if st.recent else "—"))
            if calibrated and st.R0 is not None:
                y,p,r=st.last_rel
                ui[h]["rel"].set(f"相對姿態：Yaw {y:+.1f}° / Pitch {p:+.1f}° / Roll {r:+.1f}°")
            else:
                ui[h]["rel"].set(f"校正樣本：{len(st.cal_samples)}")
    root.after(100,update_ui)

root.protocol("WM_DELETE_WINDOW",close_app)
root.after(100,update_ui)
root.after(250,lambda:threading.Thread(target=connect_all,daemon=True).start())
root.after(500,root.lift)
root.mainloop()
