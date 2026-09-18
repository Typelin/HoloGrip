from __future__ import annotations

import csv
import json
import math
import queue
import sys
import threading
import time
from collections import Counter, deque
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
sys.path.insert(0, str(APP))
sys.path.insert(0, str(LAB))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, ZONE_NAMES

MODEL_DIR = LAB / "vnext_models"
MODELS = {
    h: joblib.load(MODEL_DIR / f"hologrip_vnext_{h}_relative_rotation.joblib")
    for h in ("L", "R")
}
META = json.loads((MODEL_DIR / "vnext_metadata.json").read_text(encoding="utf-8"))

# ---- Protocol --------------------------------------------------------------
# kind: calibration / transition / target / stress / done
PROTOCOL = [
    {"kind":"calibration", "name":"歸零校正", "duration":2.0,
     "instruction":"雙手維持標準起始姿勢，完全不要動。"},
    {"kind":"transition", "name":"切換：準備左上", "duration":5.0,
     "instruction":"5 秒切換。把雙手移到『左上』準備位置，不要急著打。"},
    {"kind":"target", "name":"方位 1／左上", "target":"左上", "duration":8.0,
     "instruction":"左右手交替打『左上』，節奏穩定。"},
    {"kind":"transition", "name":"切換：左上 → 中上", "duration":5.0,
     "instruction":"停止打擊，移到『中上』。Raw 仍持續 100 Hz 收集。"},
    {"kind":"target", "name":"方位 2／中上", "target":"中上", "duration":8.0,
     "instruction":"左右手交替打『中上』，節奏穩定。"},
    {"kind":"transition", "name":"切換：中上 → 右上", "duration":5.0,
     "instruction":"停止打擊，移到『右上』。"},
    {"kind":"target", "name":"方位 3／右上", "target":"右上", "duration":8.0,
     "instruction":"左右手交替打『右上』，節奏穩定。"},
    {"kind":"transition", "name":"切換：右上 → 左中", "duration":5.0,
     "instruction":"停止打擊，移到『左中』。"},
    {"kind":"target", "name":"方位 4／左中", "target":"左中", "duration":8.0,
     "instruction":"左右手交替打『左中』，節奏穩定。"},
    {"kind":"transition", "name":"切換：左中 → 右中", "duration":5.0,
     "instruction":"停止打擊，移到『右中』。"},
    {"kind":"target", "name":"方位 5／右中", "target":"右中", "duration":8.0,
     "instruction":"左右手交替打『右中』，節奏穩定。"},
    {"kind":"transition", "name":"切換：右中 → 右中偏高", "duration":5.0,
     "instruction":"停止打擊，移到『右中偏高』。"},
    {"kind":"target", "name":"方位 6／右中偏高", "target":"右中偏高", "duration":8.0,
     "instruction":"左右手交替打『右中偏高』，節奏穩定。"},
    {"kind":"transition", "name":"切換：準備雙手同打", "duration":5.0,
     "instruction":"停止打擊。回到你最舒服的中間位置，準備雙手同時打。"},
    {"kind":"stress", "name":"壓力測試 A／雙手同時", "target":"中間舒適位置", "duration":10.0,
     "instruction":"雙手『同時』打，每次兩手一起落下，連續測試。"},
    {"kind":"transition", "name":"切換：準備左右交替掃動", "duration":5.0,
     "instruction":"停止打擊。準備做左→右→左的掃動。"},
    {"kind":"stress", "name":"壓力測試 B／左右交替掃動", "target":"左→右→左", "duration":16.0,
     "instruction":"左右手交替打，方位從『左 → 右 → 左』慢慢掃回來。"},
    {"kind":"transition", "name":"結束靜止觀察", "duration":5.0,
     "instruction":"完全停手 5 秒。這段專門檢查是否還會出現幽靈 hit。"},
]

# ---- Session files ---------------------------------------------------------
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = LAB / "protocol_sessions" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

raw_f = (SESSION / "raw_100hz.csv").open("w", encoding="utf-8", newline="", buffering=1)
hit_f = (SESSION / "hits.csv").open("w", encoding="utf-8", newline="", buffering=1)
stage_f = (SESSION / "stages.csv").open("w", encoding="utf-8", newline="", buffering=1)

raw_w = csv.writer(raw_f)
hit_w = csv.writer(hit_f)
stage_w = csv.writer(stage_f)
runtime_f = (SESSION / "runtime.log").open("w", encoding="utf-8", buffering=1)
runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] SESSION_CREATE {SESSION}\n")

raw_w.writerow([
    "wall_time","protocol_elapsed_s","port","hand","packet_id","sensor_ms",
    "ax","ay","az","yaw","pitch","roll",
    "rel_yaw","rel_pitch","rel_roll",
    "stage_index","stage_kind","stage_name","target",
    "frame_valid","invalid_reason","hit_allowed"
])
hit_w.writerow([
    "wall_time","protocol_elapsed_s","hand","port","peak_ms",
    "pred","prob","recent_stability","recent10","peak_accel_g",
    "stage_index","stage_kind","stage_name","target"
])
stage_w.writerow([
    "stage_index","kind","name","target",
    "start_wall_time","start_protocol_elapsed_s",
    "end_wall_time","end_protocol_elapsed_s"
])

# ---- Runtime state ---------------------------------------------------------
class HState:
    def __init__(self):
        self.port = ""
        self.R0 = None
        self.zero = None
        self.cal_samples = []
        self.buffer = deque(maxlen=500)
        self.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.pending = []
        self.recent = deque(maxlen=10)
        self.last_pred = "—"
        self.last_prob = 0.0
        self.hit_count = 0
        self.hz = 0.0
        self.rx = 0
        self.invalid = 0
        self.invalid_reasons = Counter()
        self.last_packet_time = 0.0
        self.last_rel = (0.0, 0.0, 0.0)
        self.last_sig = None
        self.sig_repeat = 0
        self.last_valid_monotonic = 0.0

states = {"L":HState(), "R":HState()}
lock = threading.RLock()
log_lock = threading.Lock()
uiq = queue.Queue()
stop = threading.Event()
serials = {}
threads = []

protocol_running = False
protocol_done = False
protocol_start_monotonic = None
stage_index = -1
stage_started_monotonic = None
stage_start_wall = None
stage_token = 0

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def current_elapsed():
    if protocol_start_monotonic is None:
        return 0.0
    return max(0.0, time.monotonic() - protocol_start_monotonic)

def current_stage_snapshot():
    with lock:
        if 0 <= stage_index < len(PROTOCOL):
            s = PROTOCOL[stage_index]
            return stage_index, s["kind"], s["name"], s.get("target","")
        return -1, "waiting", "等待開始", ""

# ---- Sensor parsing / validation ------------------------------------------
def parse_line(line):
    f = line.strip().split(",")
    if len(f) < 10 or f[0] != "D" or f[1] not in ("L","R"):
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

def validate_frame(st, p):
    vals = [p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"]]
    if not all(math.isfinite(v) for v in vals):
        return False, "non_finite"

    # Exact corruption pattern observed on 2026-09-18:
    # same 16-bit raw word reused as ax/ay/az and roll/pitch/yaw.
    acc_same = max(abs(p["ax"]-p["ay"]), abs(p["ay"]-p["az"]), abs(p["ax"]-p["az"])) < 1e-6
    ang_same = max(abs(p["yaw"]-p["pitch"]), abs(p["pitch"]-p["roll"]), abs(p["yaw"]-p["roll"])) < 1e-6
    scaled_same = abs(p["yaw"] - p["ax"] * 11.25) < 0.02
    if acc_same and ang_same and scaled_same and abs(p["ax"]) > 0.01:
        return False, "i2c_same_word"

    if all(abs(v) < 1e-12 for v in vals):
        return False, "all_zero"

    mag = math.sqrt(p["ax"]**2 + p["ay"]**2 + p["az"]**2)
    if mag > 28.0:
        return False, "impossible_accel_mag"

    sig = tuple(round(p[k], 4) for k in ("ax","ay","az","yaw","pitch","roll"))
    if sig == st.last_sig:
        st.sig_repeat += 1
    else:
        st.last_sig = sig
        st.sig_repeat = 1
    # Exact high plateau repeated for >= 50 ms is not a physical hit pulse.
    if st.sig_repeat >= 5 and mag > 2.3:
        return False, "stale_high_plateau"

    return True, ""

def xiao_ports():
    out=[]
    for p in list_ports.comports():
        hw=(p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:
            out.append(p.device)
    return sorted(out)

def stability(st):
    if not st.recent:
        return 0.0
    c=Counter(st.recent)
    return max(c.values()) / len(st.recent)

def reset_detector(st):
    st.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
    st.pending.clear()
    st.buffer.clear()

# ---- Serial worker ---------------------------------------------------------
def serial_worker(port, conn):
    local_n=0
    local_tick=time.monotonic()
    first_logged=False
    runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] THREAD_START {port}\n")
    try:
        while not stop.is_set():
            b=conn.readline()
            if not b:
                continue
            line=b.decode("utf-8","ignore").strip()
            p=parse_line(line)
            if p is None:
                continue

            now=time.monotonic()
            wall=now_iso()
            hand=p["hand"]
            st=states[hand]
            if not first_logged:
                runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] FIRST_PACKET {port} hand={hand} id={p['packet_id']}\n")
                first_logged=True

            with lock:
                st.port=port
                st.rx += 1
                st.last_packet_time=now
                local_n += 1
                if now-local_tick >= 1.0:
                    st.hz = local_n/(now-local_tick)
                    local_n=0
                    local_tick=now

                valid, reason = validate_frame(st,p)
                if not valid:
                    st.invalid += 1
                    st.invalid_reasons[reason] += 1
                    # Important: clear detector history so corrupt/stale high data
                    # cannot be repeatedly re-used as hits.
                    reset_detector(st)

                si, skind, sname, target = current_stage_snapshot()
                pelapsed=current_elapsed()

                rel_y=rel_p=rel_r=""
                hit_allowed=False

                # Calibration consumes only valid frames.
                if protocol_running and si == 0 and skind == "calibration":
                    if valid:
                        st.cal_samples.append((p["yaw"],p["pitch"],p["roll"]))
                elif protocol_running and st.R0 is not None and valid:
                    sm=make_relative_sample(
                        st.R0, now*1000.0,
                        p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"]
                    )
                    st.buffer.append(sm)
                    st.last_rel=(sm["rel_yaw"],sm["rel_pitch"],sm["rel_roll"])
                    rel_y,rel_p,rel_r=st.last_rel

                    # Transitions are fully recorded but deliberately cannot count as hits.
                    hit_allowed = skind in ("target","stress")

                    pkt=SensorPacket(
                        hand=hand,
                        ax=p["ax"],ay=p["ay"],az=p["az"],
                        yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
                        packet_id=p["packet_id"],sensor_time_ms=p["sensor_ms"],
                        received_time_ms=int(time.time()*1000),received_monotonic=now
                    )

                    if hit_allowed:
                        ev=st.detector.add_packet(pkt,sm["rel_yaw"],sm["rel_pitch"])
                        if ev:
                            st.pending.append({
                                "peak_ms":now*1000.0-PEAK_LAG_MS,
                                "peak_g":float(ev["peak_accel_g"])
                            })
                    else:
                        # Never let movement during the 5 s switching window leak into
                        # the next scoring stage.
                        reset_detector(st)

                    remain=[]
                    for item in st.pending:
                        if now*1000.0 >= item["peak_ms"] + WINDOW_HALF_MS:
                            # Re-read stage: if it changed while waiting for +80 ms,
                            # do not score a hit into the next stage.
                            si2,kind2,name2,target2=current_stage_snapshot()
                            if kind2 not in ("target","stress") or si2 != si:
                                continue
                            ft=feature_from_samples(list(st.buffer),item["peak_ms"],WINDOW_HALF_MS)
                            if ft is None:
                                continue
                            model=MODELS[hand]
                            proba=model.predict_proba([ft])[0]
                            j=int(np.argmax(proba))
                            cls=int(model.classes_[j])
                            label=ZONE_NAMES[cls]
                            prob=float(proba[j])

                            st.recent.append(label)
                            st.last_pred=label
                            st.last_prob=prob
                            st.hit_count+=1
                            stab=stability(st)
                            with log_lock:
                                hit_w.writerow([
                                    wall,f"{pelapsed:.3f}",hand,port,f"{item['peak_ms']:.3f}",
                                    label,f"{prob:.6f}",f"{stab:.4f}","|".join(st.recent),
                                    f"{item['peak_g']:.4f}",
                                    si2,kind2,name2,target2
                                ])
                        else:
                            remain.append(item)
                    st.pending=remain

                with log_lock:
                    raw_w.writerow([
                        wall,f"{pelapsed:.3f}",port,hand,p["packet_id"],p["sensor_ms"],
                        p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"],
                        rel_y,rel_p,rel_r,
                        si,skind,sname,target,
                        int(valid),reason,int(hit_allowed and valid)
                    ])
                    if st.rx % 50 == 0:
                        raw_f.flush()
                        hit_f.flush()
                        stage_f.flush()

    except Exception as e:
        runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] THREAD_ERROR {port} {repr(e)}\n")
        uiq.put(("error",port,repr(e)))

# ---- Protocol engine -------------------------------------------------------
def finalize_calibration():
    meta={}
    ok=True
    with lock:
        for h,st in states.items():
            try:
                R0,zero=compute_r0(st.cal_samples)
                st.R0=R0
                st.zero=zero
                reset_detector(st)
                meta[h]={
                    "samples":len(st.cal_samples),
                    "zero":[float(x) for x in zero],
                    "port":st.port,
                }
            except Exception as e:
                ok=False
                meta[h]={"samples":len(st.cal_samples),"port":st.port,"error":repr(e)}
    (SESSION/"calibration.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return ok,meta

def close_current_stage(end_wall=None,end_elapsed=None):
    global stage_start_wall
    if stage_index < 0 or stage_started_monotonic is None or stage_start_wall is None:
        return
    s=PROTOCOL[stage_index]
    with log_lock:
        stage_w.writerow([
            stage_index,s["kind"],s["name"],s.get("target",""),
            stage_start_wall,
            f"{stage_started_monotonic-protocol_start_monotonic:.3f}",
            end_wall or now_iso(),
            f"{(end_elapsed if end_elapsed is not None else current_elapsed()):.3f}"
        ])

def enter_stage(idx):
    global stage_index, stage_started_monotonic, stage_start_wall, stage_token
    stage_index=idx
    stage_token += 1
    token=stage_token
    if idx >= len(PROTOCOL):
        finish_protocol()
        return
    stage_started_monotonic=time.monotonic()
    stage_start_wall=now_iso()
    s=PROTOCOL[idx]

    with lock:
        # Clear stage-local recent predictions but keep total hit counters.
        for st in states.values():
            st.recent.clear()
            st.pending.clear()
            if s["kind"] == "calibration":
                st.cal_samples.clear()
                st.R0=None
                st.zero=None
            elif s["kind"] == "transition":
                reset_detector(st)

    uiq.put(("stage",idx,s))
    root.after(100,lambda: stage_tick(token))

def stage_tick(token):
    if token != stage_token or not protocol_running:
        return
    s=PROTOCOL[stage_index]
    elapsed=time.monotonic()-stage_started_monotonic
    remain=max(0.0,s["duration"]-elapsed)
    countdown_var.set(f"{remain:0.1f} 秒")
    if elapsed >= s["duration"]:
        endwall=now_iso()
        endelapsed=current_elapsed()
        close_current_stage(endwall,endelapsed)
        if s["kind"]=="calibration":
            ok,meta=finalize_calibration()
            if not ok:
                status_var.set("校正失敗，測試已停止：" + json.dumps(meta,ensure_ascii=False))
                abort_protocol()
                return
        enter_stage(stage_index+1)
    else:
        root.after(100,lambda: stage_tick(token))

def start_protocol():
    global protocol_running, protocol_done, protocol_start_monotonic
    if protocol_running:
        return
    if len(serials)<2:
        status_var.set("兩隻 XIAO 尚未全部連上，不能開始。")
        return
    with lock:
        for st in states.values():
            st.hit_count=0
            st.invalid=0
            st.invalid_reasons.clear()
            st.recent.clear()
            st.last_pred="—"
            st.last_prob=0.0
            st.cal_samples.clear()
            st.R0=None
            st.zero=None
            st.last_sig=None
            st.sig_repeat=0
            reset_detector(st)
    protocol_done=False
    protocol_running=True
    protocol_start_monotonic=time.monotonic()
    start_btn.configure(state="disabled")
    enter_stage(0)

def abort_protocol():
    global protocol_running
    protocol_running=False
    start_btn.configure(state="normal")
    countdown_var.set("停止")

def make_summary():
    try:
        raw_f.flush();hit_f.flush();stage_f.flush()
    except Exception:
        pass
    # Read hits/stages from files after flush.
    try:
        with (SESSION/"hits.csv").open(encoding="utf-8",newline="") as f:
            hitrows=list(csv.DictReader(f))
    except Exception:
        hitrows=[]

    summary={
        "session":str(SESSION),
        "model_version":META.get("version"),
        "model_sha256":META.get("model_sha256"),
        "protocol_total_s":current_elapsed(),
        "hands":{},
        "stages":[],
    }
    with lock:
        for h,st in states.items():
            summary["hands"][h]={
                "port":st.port,"rx":st.rx,"hz_last":st.hz,
                "invalid_frames":st.invalid,
                "invalid_reasons":dict(st.invalid_reasons),
                "hits_total":st.hit_count,
                "zero":None if st.zero is None else [float(x) for x in st.zero],
            }

    for idx,s in enumerate(PROTOCOL):
        if s["kind"] not in ("target","stress"):
            continue
        sr={"stage_index":idx,"name":s["name"],"target":s.get("target",""),"hands":{}}
        for h in ("L","R"):
            hh=[r for r in hitrows if r["hand"]==h and int(r["stage_index"])==idx]
            c=Counter(r["pred"] for r in hh)
            top=c.most_common(1)[0] if c else ("",0)
            sr["hands"][h]={
                "n":len(hh),
                "counts":dict(c),
                "modal_pred":top[0],
                "modal_fraction":(top[1]/len(hh)) if hh else 0.0,
                "median_prob":float(np.median([float(r["prob"]) for r in hh])) if hh else 0.0,
            }
        summary["stages"].append(sr)

    (SESSION/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def finish_protocol():
    global protocol_running, protocol_done, stage_index
    protocol_running=False
    protocol_done=True
    stage_index=len(PROTOCOL)
    summary=make_summary()

    # Human-readable compact result.
    lines=["測試完成。資料已全部保存。",f"Session：{SESSION.name}",""]
    for sr in summary["stages"]:
        L=sr["hands"]["L"];R=sr["hands"]["R"]
        lines.append(
            f"{sr['name']}｜L {L['modal_pred'] or '—'} {L['modal_fraction']*100:.0f}% ({L['n']} hits)"
            f"｜R {R['modal_pred'] or '—'} {R['modal_fraction']*100:.0f}% ({R['n']} hits)"
        )
    lines.append("")
    lines.append(
        f"壞 frame：L {summary['hands']['L']['invalid_frames']}｜R {summary['hands']['R']['invalid_frames']}"
    )
    status_var.set("\n".join(lines))
    title_var.set("測試完成")
    instruction_var.set("請不要重開或覆蓋。直接告訴 ChatGPT『做完了』，我會讀完整 Raw / Hits / Stages / Summary。")
    countdown_var.set("完成")
    start_btn.configure(text="重新跑一輪",state="normal")

def connect_all():
    ports=xiao_ports()
    if len(ports)<2:
        uiq.put(("status",f"只找到 {len(ports)} 顆 XIAO：{ports}"))
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
            runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] CONNECT_OK {p}\n")
            t=threading.Thread(target=serial_worker,args=(p,s),daemon=True)
            t.start()
            threads.append(t)
        uiq.put(("status",f"已連線：{', '.join(p for p,_ in opened)}。按『開始整套測試』後，第一步就是 2 秒歸零。"))
    except Exception as e:
        for _,s in opened:
            try:s.close()
            except:pass
        uiq.put(("status",f"COM 連線失敗：{e}"))

def close_app():
    stop.set()
    for s in serials.values():
        try:s.close()
        except:pass
    runtime_f.write(f"[{datetime.now().astimezone().isoformat()}] APP_CLOSE\n")
    for f in (raw_f,hit_f,stage_f,runtime_f):
        try:f.close()
        except:pass
    root.destroy()

# ---- GUI ------------------------------------------------------------------
root=tk.Tk()
root.title("HoloGrip vNext｜自動六方位＋雙手壓力測試")
root.geometry("1180x760")
root.minsize(1080,700)
root.attributes("-topmost",True)

style=ttk.Style()
style.configure("Title.TLabel",font=("Microsoft JhengHei UI",23,"bold"))
style.configure("Stage.TLabel",font=("Microsoft JhengHei UI",30,"bold"))
style.configure("Countdown.TLabel",font=("Microsoft JhengHei UI",25,"bold"))
style.configure("Mid.TLabel",font=("Microsoft JhengHei UI",14))
style.configure("Small.TLabel",font=("Microsoft JhengHei UI",11))
style.configure("Big.TButton",font=("Microsoft JhengHei UI",14,"bold"),padding=10)

title_var=tk.StringVar(value="HoloGrip vNext｜自動化測試")
stage_var=tk.StringVar(value="等待開始")
countdown_var=tk.StringVar(value="—")
instruction_var=tk.StringVar(value="按『開始整套測試』。第一步會先做 2 秒歸零。")
status_var=tk.StringVar(value="正在連接兩隻 XIAO…")

ttk.Label(root,textvariable=title_var,style="Title.TLabel",anchor="center").pack(fill="x",pady=(18,5))
ttk.Label(root,textvariable=stage_var,style="Stage.TLabel",anchor="center").pack(fill="x",pady=(12,3))
ttk.Label(root,textvariable=countdown_var,style="Countdown.TLabel",anchor="center").pack(fill="x",pady=3)
ttk.Label(root,textvariable=instruction_var,style="Mid.TLabel",anchor="center",justify="center",wraplength=1080).pack(fill="x",padx=20,pady=(8,14))

main=ttk.Frame(root)
main.pack(fill="both",expand=True,padx=18,pady=5)
main.columnconfigure(0,weight=1)
main.columnconfigure(1,weight=1)

ui={}
for col,h in enumerate(("L","R")):
    fr=ttk.LabelFrame(main,text=("左手 L" if h=="L" else "右手 R"),padding=12)
    fr.grid(row=0,column=col,sticky="nsew",padx=8)
    fr.columnconfigure(0,weight=1)
    pred=tk.StringVar(value="—")
    info=tk.StringVar(value="等待資料")
    recent=tk.StringVar(value="最近 10 擊：—")
    rel=tk.StringVar(value="相對姿態：—")
    bad=tk.StringVar(value="壞 frame：0")
    ttk.Label(fr,textvariable=pred,style="Stage.TLabel",anchor="center").grid(row=0,column=0,sticky="ew",pady=(12,8))
    ttk.Label(fr,textvariable=info,style="Mid.TLabel",anchor="center",justify="center").grid(row=1,column=0,sticky="ew",pady=5)
    ttk.Label(fr,textvariable=recent,style="Small.TLabel",anchor="center",justify="center",wraplength=500).grid(row=2,column=0,sticky="ew",pady=7)
    ttk.Label(fr,textvariable=rel,style="Small.TLabel",anchor="center").grid(row=3,column=0,sticky="ew",pady=4)
    ttk.Label(fr,textvariable=bad,style="Small.TLabel",anchor="center").grid(row=4,column=0,sticky="ew",pady=4)
    ui[h]={"pred":pred,"info":info,"recent":recent,"rel":rel,"bad":bad}

ttk.Label(root,textvariable=status_var,style="Small.TLabel",anchor="center",justify="center",wraplength=1120).pack(fill="x",padx=20,pady=9)

buttons=ttk.Frame(root)
buttons.pack(pady=(2,18))
start_btn=ttk.Button(buttons,text="開始整套測試",style="Big.TButton",command=start_protocol)
start_btn.pack(side="left",padx=8)
ttk.Button(buttons,text="關閉",style="Big.TButton",command=close_app).pack(side="left",padx=8)

def update_ui():
    while True:
        try:e=uiq.get_nowait()
        except queue.Empty:break
        if e[0]=="status":
            status_var.set(e[1])
        elif e[0]=="error":
            status_var.set(f"{e[1]} 讀取中斷：{e[2]}")
        elif e[0]=="stage":
            idx,s=e[1],e[2]
            title_var.set(f"步驟 {idx+1}/{len(PROTOCOL)}")
            stage_var.set(s["name"])
            instruction_var.set(s["instruction"])

    with lock:
        for h,st in states.items():
            ui[h]["pred"].set(st.last_pred)
            ui[h]["info"].set(
                f"{st.port or '未辨識'}｜{st.hz:.1f} Hz｜總 Hits {st.hit_count}\n"
                f"最後信心 {st.last_prob*100:.1f}%｜目前段最近穩定度 {stability(st)*100:.0f}%"
            )
            ui[h]["recent"].set("最近 10 擊：" + (" → ".join(st.recent) if st.recent else "—"))
            if st.R0 is not None:
                y,p,r=st.last_rel
                ui[h]["rel"].set(f"相對姿態：Yaw {y:+.1f}° / Pitch {p:+.1f}° / Roll {r:+.1f}°")
            else:
                ui[h]["rel"].set(f"歸零樣本：{len(st.cal_samples)}")
            reasons=", ".join(f"{k}:{v}" for k,v in st.invalid_reasons.items())
            ui[h]["bad"].set(f"壞 frame：{st.invalid}" + (f"（{reasons}）" if reasons else ""))
    root.after(100,update_ui)

root.protocol("WM_DELETE_WINDOW",close_app)
root.after(100,update_ui)
root.after(250,lambda:threading.Thread(target=connect_all,daemon=True).start())
root.after(500,root.lift)
root.mainloop()
