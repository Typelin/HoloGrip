from __future__ import annotations
import csv, json, queue, sys, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import serial
from serial.tools import list_ports
import tkinter as tk
from tkinter import ttk

ROOT = Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB = ROOT / "HoloGrip_SpatialLab_0918"
APP = ROOT / "Apps" / "Song_Collection_COM"
PROD = LAB / "HoloGrip_Production_vNext_0918"
MODELDIR = LAB / "count6_calibration_models_185504"
sys.path.insert(0, str(APP)); sys.path.insert(0, str(LAB)); sys.path.insert(0, str(PROD))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, FEATURE_NAMES
from production_core import FrameValidator, RefractoryGate

POSITIONS = ["左上","中上","右上","左中","正中","右中"]
H_EXPECT = {0:0,1:1,2:2,3:0,4:1,5:2}
V_EXPECT = {0:0,1:0,2:0,3:1,4:1,5:1}
H_NAMES = ["LEFT","CENTER","RIGHT"]
V_NAMES = ["UP","MID"]
TARGET_PER_HAND = 5
CAL_SAMPLES = 200
COUNT_REFRACTORY_MS = 350.0

META = json.loads((MODELDIR / "metadata.json").read_text(encoding="utf-8"))
MODELS = {}
for h in ("L","R"):
    MODELS[h] = {
        "H": joblib.load(MODELDIR / f"H_{h}.joblib"),
        "V": joblib.load(MODELDIR / f"V_{h}.joblib"),
        "S": joblib.load(MODELDIR / f"S_{h}.joblib"),
        "H_idx": META["models"][h]["H_idx"],
        "V_idx": META["models"][h]["V_idx"],
        "S_idx": META["models"][h]["S_idx"],
    }

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = LAB / "count6_round3_fresh_r0_sessions" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

raw_f = (SESSION/"raw_100hz.csv").open("w", encoding="utf-8", newline="", buffering=1)
pred_f = (SESSION/"predictions.csv").open("w", encoding="utf-8", newline="", buffering=1)
stage_f = (SESSION/"stage_events.csv").open("w", encoding="utf-8", newline="", buffering=1)
raw_w = csv.writer(raw_f); pred_w = csv.writer(pred_f); stage_w = csv.writer(stage_f)

raw_w.writerow(["wall_time","mode","stage_index","stage_name","port","hand","packet_id","sensor_ms",
                "ax","ay","az","yaw","pitch","roll","valid","invalid_reason"])
pred_w.writerow(["wall_time","stage_index","stage_name","hand","count_after",
                 "H_true","H_pred","H_ok","V_true","V_pred","V_ok","S_true","S_pred","S_ok",
                 "peak_g", *FEATURE_NAMES])
stage_w.writerow(["wall_time","event","stage_index","stage_name","L_count","R_count","L_invalid_delta","R_invalid_delta"])

class HS:
    def __init__(self):
        self.port = ""
        self.R0 = None
        self.zero = None
        self.cal = []
        self.buffer = deque(maxlen=500)
        self.det = HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.prod_gate = RefractoryGate(120)
        self.val = FrameValidator()
        self.pending = []
        self.rx = 0
        self.hz = 0.0
        self.count = 0
        self.last_count_ms = -1e18
        self.lastH = "—"; self.lastV = "—"; self.lastS = "—"
        self.okH = 0; self.okV = 0; self.okS = 0

states = {"L":HS(), "R":HS()}
lock = threading.RLock()
stop = threading.Event()
serials = {}
threads = []
uiq = queue.Queue()

mode = "idle"   # idle/cal/wait_stage/collect/finished
stage_index = 0
stage_invalid_base = {"L":0,"R":0}
stage_quality = {}
cal_finished = False

def iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def parse(line):
    f = line.strip().split(",")
    if len(f) < 10 or f[0] != "D" or f[1] not in ("L","R"):
        return None
    try:
        return {
            "hand":f[1], "ax":float(f[2]), "ay":float(f[3]), "az":float(f[4]),
            "yaw":float(f[5]), "pitch":float(f[6]), "roll":float(f[7]),
            "packet_id":int(f[8]), "sensor_ms":int(f[9])
        }
    except Exception:
        return None

def xiao_ports():
    return sorted([
        p.device for p in list_ports.comports()
        if "303A:1001" in (p.hwid or "").upper() or "VID_303A&PID_1001" in (p.hwid or "").upper()
    ])

def stage_name():
    return POSITIONS[stage_index] if 0 <= stage_index < len(POSITIONS) else ""

def reset_detector(st, reset_count=False):
    st.buffer.clear()
    st.pending.clear()
    st.det = HitDetector(**PRODUCT_DETECTOR_KWARGS)
    st.prod_gate = RefractoryGate(120)
    if reset_count:
        st.count = 0
        st.last_count_ms = -1e18

def log_stage(event, ld=0, rd=0):
    stage_w.writerow([iso(), event, stage_index, stage_name(),
                      states["L"].count, states["R"].count, ld, rd])
    stage_f.flush()

def classify(h, ft):
    mm = MODELS[h]
    hp = int(mm["H"].predict([ft[mm["H_idx"]]])[0])
    vp = int(mm["V"].predict([ft[mm["V_idx"]]])[0])
    sp = int(mm["S"].predict([ft[mm["S_idx"]]])[0])
    return hp, vp, sp

def worker(port, conn):
    n = 0
    tick = time.monotonic()
    try:
        while not stop.is_set():
            b = conn.readline()
            if not b:
                continue
            p = parse(b.decode("utf-8","ignore").strip())
            if p is None:
                continue
            h = p["hand"]; st = states[h]
            now = time.monotonic(); nowms = now * 1000.0; wall = iso()

            with lock:
                st.port = port; st.rx += 1; n += 1
                if now - tick >= 1.0:
                    st.hz = n / (now - tick); n = 0; tick = now

                valid, reason = st.val.check(p)
                raw_w.writerow([wall,mode,stage_index,stage_name(),port,h,p["packet_id"],p["sensor_ms"],
                                p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],
                                int(valid),reason])
                if st.rx % 50 == 0:
                    raw_f.flush()

                if not valid:
                    reset_detector(st, reset_count=False)
                    continue

                if mode == "cal":
                    if len(st.cal) < CAL_SAMPLES:
                        st.cal.append((p["yaw"],p["pitch"],p["roll"]))
                    if len(states["L"].cal) >= CAL_SAMPLES and len(states["R"].cal) >= CAL_SAMPLES:
                        uiq.put(("cal_ready",))
                    continue

                if st.R0 is None:
                    continue

                rs = make_relative_sample(st.R0, nowms,
                                          p["ax"],p["ay"],p["az"],
                                          p["yaw"],p["pitch"],p["roll"])
                st.buffer.append(rs)

                if mode != "collect":
                    continue

                pkt = SensorPacket(
                    hand=h, ax=p["ax"], ay=p["ay"], az=p["az"],
                    yaw=p["yaw"], pitch=p["pitch"], roll=p["roll"],
                    packet_id=p["packet_id"], sensor_time_ms=p["sensor_ms"],
                    received_time_ms=int(time.time()*1000), received_monotonic=now
                )
                ev = st.det.add_packet(pkt, rs["rel_yaw"], rs["rel_pitch"])
                if ev:
                    peak = nowms - PEAK_LAG_MS
                    if st.prod_gate.accept(peak):
                        st.pending.append({"peak_ms":peak, "peak_g":float(ev["peak_accel_g"])})

                remain = []
                for item in st.pending:
                    if nowms < item["peak_ms"] + WINDOW_HALF_MS:
                        remain.append(item); continue
                    if st.count >= TARGET_PER_HAND:
                        continue
                    if item["peak_ms"] - st.last_count_ms < COUNT_REFRACTORY_MS:
                        continue
                    ft = feature_from_samples(list(st.buffer), item["peak_ms"], WINDOW_HALF_MS)
                    if ft is None:
                        continue

                    hp, vp, sp = classify(h, ft)
                    st.last_count_ms = item["peak_ms"]
                    st.count += 1
                    he = H_EXPECT[stage_index]; ve = V_EXPECT[stage_index]; se = stage_index
                    hok = int(hp == he); vok = int(vp == ve); sok = int(sp == se)
                    st.okH += hok; st.okV += vok; st.okS += sok
                    st.lastH = H_NAMES[hp]; st.lastV = V_NAMES[vp]; st.lastS = POSITIONS[sp]

                    pred_w.writerow([wall,stage_index,stage_name(),h,st.count,
                                     H_NAMES[he],H_NAMES[hp],hok,
                                     V_NAMES[ve],V_NAMES[vp],vok,
                                     POSITIONS[se],POSITIONS[sp],sok,
                                     item["peak_g"], *[f"{float(x):.8f}" for x in ft]])
                    pred_f.flush()
                    uiq.put(("count",h,st.count))

                    if states["L"].count >= TARGET_PER_HAND and states["R"].count >= TARGET_PER_HAND:
                        uiq.put(("stage_done",stage_index))
                st.pending = remain
    except Exception as e:
        uiq.put(("status", f"{port} reader error: {e!r}"))

def connect_all():
    ps = xiao_ports()
    if len(ps) < 2:
        uiq.put(("status", f"只找到 {len(ps)} 颗 XIAO：{ps}"))
        return
    opened = []
    try:
        for p in ps[:2]:
            opened.append((p, serial.Serial(p,460800,timeout=.08)))
        time.sleep(.4)
        for p,s in opened:
            try: s.reset_input_buffer()
            except Exception: pass
            serials[p] = s
            t = threading.Thread(target=worker,args=(p,s),daemon=True)
            t.start(); threads.append(t)
        uiq.put(("status", f"已连接 {', '.join(p for p,_ in opened)}。先做 Fresh-R0 归零。"))
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except Exception:pass
        uiq.put(("status", f"COM 失败：{e}"))

def start_cal():
    global mode, cal_finished, stage_index
    if len(serials) < 2:
        return
    stage_index = 0
    cal_finished = False
    mode = "cal"
    with lock:
        for st in states.values():
            st.cal.clear()
            st.R0 = None
            st.zero = None
            st.count = 0
            st.last_count_ms = -1e18
            reset_detector(st, reset_count=True)
    cal_btn.configure(state="disabled")
    start_btn.configure(state="disabled")
    title_var.set("Fresh-R0 归零中")
    prompt_var.set("双手保持标准起始姿势。每手收满 200 个有效 frame 自动完成。")
    status_var.set("这是本轮新 R0，不沿用上一轮。")

def finish_cal():
    global mode, cal_finished
    if mode != "cal" or cal_finished:
        return
    cal_finished = True
    info = {}
    ok = True
    with lock:
        for h,st in states.items():
            try:
                st.R0, st.zero = compute_r0(st.cal[:CAL_SAMPLES])
                reset_detector(st, reset_count=True)
                info[h] = {"samples":len(st.cal), "zero":[float(x) for x in st.zero], "port":st.port}
            except Exception as e:
                ok = False
                info[h] = {"error":repr(e), "samples":len(st.cal), "port":st.port}
    (SESSION/"fresh_r0_calibration.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")

    if not ok:
        mode = "idle"
        cal_btn.configure(state="normal")
        status_var.set("Fresh-R0 失败：" + json.dumps(info,ensure_ascii=False))
        return

    mode = "wait_stage"
    title_var.set("Fresh-R0 完成｜准备：左上")
    prompt_var.set("移到『左上』，准备好后按『开始目前位置』。")
    status_var.set("第一轮模型仍冻结；这一轮只评估，不重训。")
    start_btn.configure(state="normal")
    cal_btn.configure(state="normal")
    log_stage("FRESH_R0_DONE")

def start_stage():
    global mode, stage_invalid_base
    if mode != "wait_stage":
        return
    with lock:
        for st in states.values():
            reset_detector(st, reset_count=True)
        stage_invalid_base = {"L":states["L"].val.invalid, "R":states["R"].val.invalid}
    mode = "collect"
    title_var.set(f"正在盲测：{stage_name()}")
    prompt_var.set(f"位置『{stage_name()}』：左手 5 下 + 右手 5 下。")
    start_btn.configure(state="disabled")
    log_stage("STAGE_START")

def done_stage(idx):
    global mode
    if mode != "collect" or idx != stage_index:
        return
    ld = states["L"].val.invalid - stage_invalid_base["L"]
    rd = states["R"].val.invalid - stage_invalid_base["R"]
    stage_quality[idx] = {"L_invalid":ld, "R_invalid":rd, "L_clean":ld==0, "R_clean":rd==0}
    mode = "wait_stage"
    log_stage("STAGE_COMPLETE", ld, rd)

    contam = []
    if ld: contam.append(f"L 坏frame {ld}")
    if rd: contam.append(f"R 坏frame {rd}")
    contam_text = ("；本段污染：" + " / ".join(contam)) if contam else "；本段硬件干净"

    if stage_index == len(POSITIONS)-1:
        finish_all()
        return

    title_var.set(f"{stage_name()} 完成")
    prompt_var.set("移到下一位置后按『下一位置』。" + contam_text)
    start_btn.configure(text="下一位置",state="normal")

def next_or_start():
    global stage_index
    if mode != "wait_stage":
        return
    if states["L"].count == TARGET_PER_HAND and states["R"].count == TARGET_PER_HAND:
        stage_index += 1
        with lock:
            for st in states.values():
                reset_detector(st, reset_count=True)
        title_var.set(f"准备：{stage_name()}")
        prompt_var.set(f"移到『{stage_name()}』，准备好后按『开始目前位置』。")
        start_btn.configure(text="开始目前位置",state="normal")
    else:
        start_stage()

def finish_all():
    global mode
    mode = "finished"
    log_stage("SESSION_COMPLETE")
    raw_f.flush(); pred_f.flush(); stage_f.flush()

    with (SESSION/"predictions.csv").open(encoding="utf-8",newline="") as f:
        rr = list(csv.DictReader(f))

    def metrics(rows):
        if not rows:
            return {"n":0,"H_acc":None,"V_acc":None,"S_acc":None}
        return {
            "n":len(rows),
            "H_acc":sum(int(r["H_ok"]) for r in rows)/len(rows),
            "V_acc":sum(int(r["V_ok"]) for r in rows)/len(rows),
            "S_acc":sum(int(r["S_ok"]) for r in rows)/len(rows),
        }

    clean = []
    for r in rr:
        s = int(r["stage_index"]); h = r["hand"]
        q = stage_quality.get(s,{})
        if q.get(f"{h}_clean",False):
            clean.append(r)

    summary = {
        "session":str(SESSION),
        "source_model_session":META["source_session"],
        "fresh_r0":True,
        "raw_metrics":metrics(rr),
        "clean_metrics":metrics(clean),
        "stage_quality":stage_quality,
        "hands":{
            h:{
                "raw":metrics([r for r in rr if r["hand"]==h]),
                "clean":metrics([r for r in clean if r["hand"]==h]),
                "invalid_total":states[h].val.invalid
            } for h in ("L","R")
        }
    }
    (SESSION/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    cm = summary["clean_metrics"]
    if cm["n"]:
        clean_text = f"Clean H {cm['H_acc']*100:.1f}%｜V {cm['V_acc']*100:.1f}%｜六位 {cm['S_acc']*100:.1f}%（n={cm['n']}）"
    else:
        clean_text = "Clean：没有无污染样本"

    title_var.set("Fresh-R0 盲测完成")
    prompt_var.set(clean_text + "\n告诉 ChatGPT『Fresh-R0 完成』，我会读完整数据。")
    start_btn.configure(state="disabled")

def close():
    stop.set()
    for s in serials.values():
        try:s.close()
        except Exception:pass
    for f in (raw_f,pred_f,stage_f):
        try:f.close()
        except Exception:pass
    root.destroy()

root = tk.Tk()
root.title("HoloGrip｜Fresh-R0 六位置盲测｜模型冻结")
root.geometry("1080x720")
root.minsize(1000,650)
root.attributes("-topmost", True)

style = ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",23,"bold"))
style.configure("Big.TLabel",font=("Microsoft JhengHei UI",32,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Btn.TButton",font=("Microsoft JhengHei UI",13,"bold"),padding=9)

ttk.Label(root,text="Fresh-R0 六位置盲测｜第一轮模型冻结",style="Title.TLabel",anchor="center").pack(fill="x",pady=(18,6))
title_var = tk.StringVar(value="等待 Fresh-R0 归零")
ttk.Label(root,textvariable=title_var,style="Big.TLabel",anchor="center").pack(fill="x",pady=8)
prompt_var = tk.StringVar(value="先按『200-frame Fresh-R0 归零』，再开始六位置。")
ttk.Label(root,textvariable=prompt_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=1020).pack(fill="x",pady=8)
status_var = tk.StringVar(value="正在连接 COM4 / COM5…")
ttk.Label(root,textvariable=status_var,anchor="center").pack(fill="x",pady=5)

frm = ttk.Frame(root)
frm.pack(fill="both",expand=True,padx=18,pady=8)
frm.columnconfigure(0,weight=1); frm.columnconfigure(1,weight=1)
ui = {}
for col,h in enumerate(("L","R")):
    box = ttk.LabelFrame(frm,text=("左手 L" if h=="L" else "右手 R"),padding=12)
    box.grid(row=0,column=col,sticky="nsew",padx=8)
    cnt = tk.StringVar(value="0 / 5")
    pred = tk.StringVar(value="H —｜V —｜位置 —")
    info = tk.StringVar(value="等待资料")
    ttk.Label(box,textvariable=cnt,style="Big.TLabel",anchor="center").pack(fill="x",pady=(12,8))
    ttk.Label(box,textvariable=pred,style="Mid.TLabel",anchor="center",justify="center").pack(fill="x",pady=8)
    ttk.Label(box,textvariable=info,anchor="center",justify="center").pack(fill="x",pady=8)
    ui[h] = (cnt,pred,info)

btn = ttk.Frame(root); btn.pack(pady=14)
cal_btn = ttk.Button(btn,text="200-frame Fresh-R0 归零",style="Btn.TButton",command=start_cal)
cal_btn.pack(side="left",padx=6)
start_btn = ttk.Button(btn,text="开始目前位置",style="Btn.TButton",command=next_or_start,state="disabled")
start_btn.pack(side="left",padx=6)
ttk.Button(btn,text="关闭",style="Btn.TButton",command=close).pack(side="left",padx=6)

def update():
    while True:
        try:e = uiq.get_nowait()
        except queue.Empty:break
        if e[0] == "status":
            status_var.set(e[1])
        elif e[0] == "cal_ready":
            finish_cal()
        elif e[0] == "stage_done":
            done_stage(e[1])

    with lock:
        for h,st in states.items():
            cnt,pred,info = ui[h]
            if mode == "cal":
                cnt.set(f"R0 {min(len(st.cal),CAL_SAMPLES)} / {CAL_SAMPLES}")
            else:
                cnt.set(f"{st.count} / {TARGET_PER_HAND}")
            pred.set(f"H {st.lastH}｜V {st.lastV}｜位置 {st.lastS}")
            info.set(f"{st.port or '未连接'}｜{st.hz:.1f} Hz｜坏 frame {st.val.invalid}\nRX {st.rx}")

    root.after(100,update)

root.protocol("WM_DELETE_WINDOW",close)
root.after(100,update)
root.after(250,lambda:threading.Thread(target=connect_all,daemon=True).start())
root.mainloop()
