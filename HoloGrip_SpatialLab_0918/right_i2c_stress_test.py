from __future__ import annotations
import csv, json, math, queue, threading, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports
import tkinter as tk
from tkinter import ttk

LAB = Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = LAB / "right_i2c_stress_sessions" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("靜止基準", 5.0, "右手完全不動。左手也盡量不動。"),
    ("慢速大幅揮動", 10.0, "只動右手：慢速、大幅度左右／上下揮動。"),
    ("快速大幅揮動", 10.0, "只動右手：快速、大幅度連續揮動。"),
    ("猛烈壓力段", 10.0, "只動右手：用剛才最容易出問題的方式猛烈、連續快速動。"),
    ("結束靜止", 5.0, "右手完全停住。這段看資料能否恢復正常。"),
]
TOTAL = sum(x[1] for x in STAGES)

raw_f = (SESSION/"raw.csv").open("w", encoding="utf-8", newline="", buffering=1)
raw_w = csv.writer(raw_f)
raw_w.writerow([
    "wall_time","elapsed_s","stage_index","stage_name","port","hand",
    "packet_id","sensor_ms","ax","ay","az","yaw","pitch","roll",
    "mag_g","valid","invalid_reason"
])

runtime_f = (SESSION/"runtime.log").open("w", encoding="utf-8", buffering=1)

stop = threading.Event()
serials = {}
threads = []
q = queue.Queue()
lock = threading.RLock()

class State:
    def __init__(self):
        self.port=""
        self.hand=""
        self.rx=0
        self.hz=0.0
        self.invalid=0
        self.invalid_reasons=Counter()
        self.packet_gap=0
        self.packet_back=0
        self.last_packet_id=None
        self.last_sig=None
        self.sig_repeat=0
        self.last_packet_time=0.0

states={"L":State(),"R":State()}
running=False
done=False
test_start=None
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
    except:
        return None

def validate(st,p):
    vals=[p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"]]
    if not all(math.isfinite(v) for v in vals):
        return False,"non_finite"

    if all(abs(v)<1e-12 for v in vals):
        return False,"all_zero"

    acc_same=max(abs(p["ax"]-p["ay"]),abs(p["ay"]-p["az"]),abs(p["ax"]-p["az"]))<1e-6
    ang_same=max(abs(p["yaw"]-p["pitch"]),abs(p["pitch"]-p["roll"]),abs(p["yaw"]-p["roll"]))<1e-6
    scale_same=abs(p["yaw"]-p["ax"]*11.25)<0.02
    if acc_same and ang_same and scale_same and abs(p["ax"])>0.01:
        return False,"i2c_same_word"

    mag=math.sqrt(p["ax"]**2+p["ay"]**2+p["az"]**2)
    if mag>28:
        return False,"impossible_accel"

    sig=tuple(round(p[k],4) for k in ("ax","ay","az","yaw","pitch","roll"))
    if sig==st.last_sig:
        st.sig_repeat+=1
    else:
        st.last_sig=sig
        st.sig_repeat=1
    if st.sig_repeat>=5 and mag>2.3:
        return False,"stale_high_plateau"

    return True,""

def current_stage(elapsed):
    acc=0.0
    for i,(name,dur,inst) in enumerate(STAGES):
        if acc <= elapsed < acc+dur:
            return i,name,inst,dur-(elapsed-acc)
        acc+=dur
    return len(STAGES), "完成", "", 0.0

def serial_worker(port,conn):
    n=0
    tick=time.monotonic()
    runtime_f.write(f"[{iso()}] THREAD_START {port}\n")
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b: continue
            line=b.decode("utf-8","ignore").strip()
            p=parse(line)
            if p is None: continue
            h=p["hand"]
            st=states[h]
            now=time.monotonic()
            with lock:
                st.port=port; st.hand=h; st.rx+=1; st.last_packet_time=now
                n+=1
                if now-tick>=1.0:
                    st.hz=n/(now-tick); n=0; tick=now

                if st.last_packet_id is not None:
                    d=p["packet_id"]-st.last_packet_id
                    if d>1: st.packet_gap += d-1
                    elif d<=0: st.packet_back += 1
                st.last_packet_id=p["packet_id"]

                valid,reason=validate(st,p)
                if not valid:
                    st.invalid+=1
                    st.invalid_reasons[reason]+=1

                elapsed=(now-test_start) if (running and test_start is not None) else -1.0
                si,sname,_,_=current_stage(elapsed) if elapsed>=0 else (-1,"等待開始","",0)
                mag=math.sqrt(p["ax"]**2+p["ay"]**2+p["az"]**2)
                raw_w.writerow([
                    iso(),f"{elapsed:.3f}",si,sname,port,h,
                    p["packet_id"],p["sensor_ms"],
                    p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],
                    f"{mag:.5f}",int(valid),reason
                ])
                if st.rx%50==0: raw_f.flush()
    except Exception as e:
        runtime_f.write(f"[{iso()}] ERROR {port} {repr(e)}\n")
        q.put(("error",port,repr(e)))

def xiao_ports():
    out=[]
    for p in list_ports.comports():
        hw=(p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:
            out.append(p.device)
    return sorted(out)

def connect():
    ports=xiao_ports()
    if len(ports)<2:
        q.put(("status",f"只找到 {len(ports)} 顆 XIAO：{ports}"))
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
        q.put(("status",f"已連線 {', '.join(p for p,_ in opened)}。按『開始 40 秒測試』。"))
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except:pass
        q.put(("status",f"連線失敗：{e}"))

def start_test():
    global running,done,test_start,stage_index
    if running or len(serials)<2:return
    with lock:
        for st in states.values():
            st.invalid=0;st.invalid_reasons.clear()
            st.packet_gap=0;st.packet_back=0
            st.last_sig=None;st.sig_repeat=0
    running=True;done=False
    test_start=time.monotonic()
    start_btn.configure(state="disabled")
    tick()

def finish():
    global running,done
    running=False;done=True
    raw_f.flush()
    result={"session":str(SESSION),"duration_s":TOTAL,"hands":{}}
    with lock:
        for h,st in states.items():
            result["hands"][h]={
                "port":st.port,"rx":st.rx,"hz_last":st.hz,
                "invalid":st.invalid,
                "invalid_reasons":dict(st.invalid_reasons),
                "packet_gap":st.packet_gap,
                "packet_back":st.packet_back,
            }
    # Right hand is the primary pass/fail.
    r=result["hands"]["R"]
    right_pass=(90<=r["hz_last"]<=110 and r["invalid"]==0 and r["packet_gap"]==0 and r["packet_back"]==0)
    result["right_pass"]=right_pass
    (SESSION/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

    title_var.set("PASS" if right_pass else "FAIL")
    stage_var.set("右手純硬體壓力測試完成")
    countdown_var.set("完成")
    inst_var.set("右手 PASS：100 Hz / 0 壞 frame / 0 packet gap" if right_pass else
                 f"右手 FAIL：壞 frame {r['invalid']} / gap {r['packet_gap']} / back {r['packet_back']}")
    start_btn.configure(text="重新跑",state="normal")

def tick():
    if not running:return
    e=time.monotonic()-test_start
    if e>=TOTAL:
        finish();return
    i,name,inst,rem=current_stage(e)
    title_var.set(f"步驟 {i+1}/{len(STAGES)}")
    stage_var.set(name)
    countdown_var.set(f"{rem:.1f} 秒")
    inst_var.set(inst)
    root.after(100,tick)

def update_ui():
    while True:
        try:e=q.get_nowait()
        except queue.Empty:break
        if e[0]=="status":status_var.set(e[1])
        elif e[0]=="error":status_var.set(f"{e[1]} 錯誤：{e[2]}")
    with lock:
        for h,st in states.items():
            v=ui[h]
            reasons=", ".join(f"{k}:{n}" for k,n in st.invalid_reasons.items())
            v.set(
                f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜RX {st.rx}\n"
                f"壞 frame {st.invalid}" + (f"（{reasons}）" if reasons else "") +
                f"\npacket gap {st.packet_gap}｜back/dup {st.packet_back}"
            )
    root.after(100,update_ui)

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    try:raw_f.close();runtime_f.close()
    except:pass
    root.destroy()

root=tk.Tk()
root.title("HoloGrip｜右手 I2C 純硬體壓力測試")
root.geometry("900x560")
root.minsize(850,520)
root.attributes("-topmost",True)

style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",24,"bold"))
style.configure("Stage.TLabel",font=("Microsoft JhengHei UI",30,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Big.TButton",font=("Microsoft JhengHei UI",14,"bold"),padding=10)

title_var=tk.StringVar(value="右手 I2C 純硬體壓力測試")
stage_var=tk.StringVar(value="等待開始")
countdown_var=tk.StringVar(value="40 秒")
inst_var=tk.StringVar(value="這輪完全不做模型分類，只測右手資料鏈。")
status_var=tk.StringVar(value="正在連接兩隻手套…")

ttk.Label(root,textvariable=title_var,style="Title.TLabel",anchor="center").pack(fill="x",pady=(20,8))
ttk.Label(root,textvariable=stage_var,style="Stage.TLabel",anchor="center").pack(fill="x",pady=8)
ttk.Label(root,textvariable=countdown_var,style="Title.TLabel",anchor="center").pack(fill="x",pady=4)
ttk.Label(root,textvariable=inst_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=840).pack(fill="x",padx=20,pady=12)

frame=ttk.Frame(root);frame.pack(fill="x",padx=20,pady=8)
frame.columnconfigure(0,weight=1);frame.columnconfigure(1,weight=1)
ui={}
for col,h in enumerate(("L","R")):
    sv=tk.StringVar(value="等待資料")
    ui[h]=sv
    box=ttk.LabelFrame(frame,text=("左手對照 L" if h=="L" else "右手主測 R"),padding=14)
    box.grid(row=0,column=col,sticky="nsew",padx=8)
    ttk.Label(box,textvariable=sv,style="Mid.TLabel",anchor="center",justify="center").pack(fill="x")

ttk.Label(root,textvariable=status_var,anchor="center").pack(fill="x",padx=20,pady=8)

buttons=ttk.Frame(root);buttons.pack(pady=16)
start_btn=ttk.Button(buttons,text="開始 40 秒測試",style="Big.TButton",command=start_test)
start_btn.pack(side="left",padx=8)
ttk.Button(buttons,text="關閉",style="Big.TButton",command=close).pack(side="left",padx=8)

root.protocol("WM_DELETE_WINDOW",close)
root.after(100,update_ui)
root.after(250,lambda:threading.Thread(target=connect,daemon=True).start())
root.after(500,root.lift)
root.mainloop()
