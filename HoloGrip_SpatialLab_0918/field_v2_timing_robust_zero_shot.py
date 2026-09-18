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
PROD = LAB / "HoloGrip_Production_vNext_0918"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROD))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, FEATURE_NAMES, ZONE_NAMES
from production_core import FrameValidator

# ---------------------------------------------------------------------------
# Frozen inference assets
# ---------------------------------------------------------------------------
MODEL_DIR = LAB / "vnext_timing_robust_aug20"
DRUM_MODELS = {
    h: joblib.load(MODEL_DIR / f"hologrip_timing_robust_{h}.joblib")
    for h in ("L", "R")
}
H_MODELS = {
    h: joblib.load(LAB / "hierarchical_realdrum_models" / f"H_{h}.joblib")
    for h in ("L", "R")
}
V_MODELS = {
    h: joblib.load(LAB / "hierarchical_realdrum_models" / f"V_{h}.joblib")
    for h in ("L", "R")
}

HMAP = {"Crash": 0, "Hi-Hat": 0, "小鼓": 1, "高音 Tom": 1, "中音 Tom": 1, "Ride": 2, "落地 Tom": 2}
VMAP = {"Crash": 0, "高音 Tom": 0, "中音 Tom": 0, "Ride": 0, "Hi-Hat": 1, "小鼓": 1, "落地 Tom": 1}
HN = ["LEFT", "CENTER", "RIGHT"]
VN = ["UP", "LOW"]

# ---------------------------------------------------------------------------
# Field-v2 constants, fixed from 2026-09-18 research
# ---------------------------------------------------------------------------
CAL_SAMPLES = 200
MIN_DETECTOR_WARMUP_VALID = 45
BAD_STREAK_RESET = 5

# Weak pre-impact candidate may be replaced by a stronger impact candidate.
PEAK_RESOLVE_MS = 160.0
REPLACE_EARLY_MAX_G = 3.5
REPLACE_RATIO = 1.4
REPLACE_DELTA_G = 0.5

MIN_FEATURE_WINDOW_SAMPLES = 12
SERIAL_BAUD = 460800

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = LAB / "field_v2_sessions" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
raw_f = (SESSION / "raw_100hz.csv").open("w", encoding="utf-8", newline="", buffering=1)
cand_f = (SESSION / "candidates.csv").open("w", encoding="utf-8", newline="", buffering=1)
hit_f = (SESSION / "resolved_hits.csv").open("w", encoding="utf-8", newline="", buffering=1)
health_f = (SESSION / "health.csv").open("w", encoding="utf-8", newline="", buffering=1)
event_f = (SESSION / "events.jsonl").open("w", encoding="utf-8", buffering=1)

raw_w = csv.writer(raw_f)
cand_w = csv.writer(cand_f)
hit_w = csv.writer(hit_f)
health_w = csv.writer(health_f)

raw_w.writerow([
    "wall_time", "elapsed_s", "mode", "marker", "port", "hand",
    "packet_id", "sensor_ms",
    "ax", "ay", "az", "yaw", "pitch", "roll",
    "valid", "invalid_reason", "health", "bad_run", "good_run",
])
cand_w.writerow([
    "wall_time", "elapsed_s", "marker", "hand", "port",
    "candidate_id", "event_type", "peak_ms", "peak_g",
    "related_candidate_id", "gap_ms", "detail",
])
hit_w.writerow([
    "wall_time", "elapsed_s", "marker", "hand", "port",
    "candidate_id", "peak_ms", "peak_g", "window_samples",
    "raw_drum", "raw_prob",
    "H_pred", "H_prob", "V_pred", "V_prob",
    "raw_H", "raw_V", "side_conflict", "vertical_disagree",
    "final_display",
    "rel_yaw", "rel_pitch", "rel_roll",
    *FEATURE_NAMES,
])
health_w.writerow([
    "wall_time", "elapsed_s", "port", "hand", "old_health", "new_health",
    "bad_run", "good_run", "invalid_total", "invalid_reasons",
])

APP_T0 = time.monotonic()
mode = "idle"  # idle / cal / live
current_marker = "自由"
stop = threading.Event()
lock = threading.RLock()
uiq: queue.Queue = queue.Queue()
serials: dict[str, serial.Serial] = {}
threads: list[threading.Thread] = []
candidate_seq = 0
cal_finish_queued = False


def wall_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def elapsed_s() -> float:
    return time.monotonic() - APP_T0


def log_event(kind: str, **data) -> None:
    rec = {"wall_time": wall_iso(), "elapsed_s": round(elapsed_s(), 3), "kind": kind, **data}
    event_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    event_f.flush()


def model_proba(model, ft: np.ndarray):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([ft])[0]
        cls = [int(x) for x in model.classes_]
        j = int(np.argmax(p))
        return cls[j], float(p[j])
    pred = int(model.predict([ft])[0])
    return pred, 1.0


class HandState:
    def __init__(self, hand: str):
        self.hand = hand
        self.port = ""
        self.validator = FrameValidator()
        self.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)

        self.R0 = None
        self.zero = None
        self.cal: list[tuple[float, float, float]] = []
        self.buffer = deque(maxlen=600)

        self.pending: list[dict] = []
        self.bad_run = 0
        self.good_run = 0
        self.needs_warmup = True
        self.health = "WARMUP"

        self.rx = 0
        self.hz = 0.0
        self.hits = 0
        self.candidates = 0
        self.replaced = 0
        self.dropped = 0
        self.side_conflicts = 0

        self.last_drum = "—"
        self.last_raw = "—"
        self.last_prob = 0.0
        self.last_h = "—"
        self.last_hprob = 0.0
        self.last_v = "—"
        self.last_vprob = 0.0
        self.last_rel = (0.0, 0.0, 0.0)
        self.last_warning = ""
        self.last_invalid_reason = ""
        self._health_logged = None

    def reset_detector(self):
        self.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.buffer.clear()
        self.pending.clear()

    def set_health(self, new: str):
        if new == self.health:
            return
        old = self.health
        self.health = new
        health_w.writerow([
            wall_iso(), f"{elapsed_s():.3f}", self.port, self.hand, old, new,
            self.bad_run, self.good_run, self.validator.invalid,
            json.dumps(dict(self.validator.reasons), ensure_ascii=False),
        ])
        health_f.flush()
        log_event("HEALTH", hand=self.hand, port=self.port, old=old, new=new,
                  bad_run=self.bad_run, good_run=self.good_run,
                  invalid_total=self.validator.invalid)


states = {h: HandState(h) for h in ("L", "R")}


def parse_line(line: str):
    f = line.strip().split(",")
    if len(f) < 10 or f[0] != "D" or f[1] not in ("L", "R"):
        return None
    try:
        return {
            "hand": f[1],
            "ax": float(f[2]), "ay": float(f[3]), "az": float(f[4]),
            "yaw": float(f[5]), "pitch": float(f[6]), "roll": float(f[7]),
            "packet_id": int(f[8]), "sensor_ms": int(f[9]),
        }
    except Exception:
        return None


def xiao_ports():
    out = []
    for p in list_ports.comports():
        hw = (p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:
            out.append(p.device)
    return sorted(out)


def next_candidate_id() -> int:
    global candidate_seq
    candidate_seq += 1
    return candidate_seq


def log_candidate(st: HandState, item: dict, event_type: str,
                  related: int | str = "", gap_ms: float | str = "", detail: str = ""):
    cand_w.writerow([
        wall_iso(), f"{elapsed_s():.3f}", current_marker, st.hand, st.port,
        item["id"], event_type, f'{item["peak_ms"]:.3f}', f'{item["peak_g"]:.5f}',
        related, gap_ms if gap_ms == "" else f"{float(gap_ms):.3f}", detail,
    ])
    cand_f.flush()


def add_candidate(st: HandState, peak_ms: float, peak_g: float):
    item = {
        "id": next_candidate_id(),
        "peak_ms": float(peak_ms),
        "peak_g": float(peak_g),
        "suppressed": False,
        "replace_by": None,
    }
    st.candidates += 1
    log_candidate(st, item, "DETECTED")

    # Compare against newest unresolved candidate. If it is the verified
    # weak-prepeak -> strong-impact pattern, suppress the weak one.
    prev = None
    for q in reversed(st.pending):
        if not q["suppressed"]:
            prev = q
            break
    if prev is not None:
        gap = item["peak_ms"] - prev["peak_ms"]
        if (
            0 < gap <= PEAK_RESOLVE_MS
            and prev["peak_g"] <= REPLACE_EARLY_MAX_G
            and item["peak_g"] >= prev["peak_g"] * REPLACE_RATIO
            and item["peak_g"] - prev["peak_g"] >= REPLACE_DELTA_G
        ):
            prev["suppressed"] = True
            prev["replace_by"] = item["id"]
            st.replaced += 1
            log_candidate(
                st, prev, "REPLACED",
                related=item["id"], gap_ms=gap,
                detail=f"weak {prev['peak_g']:.2f}g -> strong {item['peak_g']:.2f}g",
            )
    st.pending.append(item)


def classify_resolved(st: HandState, item: dict):
    if item["suppressed"]:
        return
    if st.R0 is None or mode != "live":
        st.dropped += 1
        log_candidate(st, item, "DROPPED", detail="not_live_or_no_R0")
        return
    if st.needs_warmup or st.health != "OK":
        st.dropped += 1
        log_candidate(st, item, "DROPPED", detail=f"health={st.health}")
        return

    buf = list(st.buffer)
    win = [s for s in buf if item["peak_ms"] - WINDOW_HALF_MS <= s["t"] <= item["peak_ms"] + WINDOW_HALF_MS]
    if len(win) < MIN_FEATURE_WINDOW_SAMPLES:
        st.dropped += 1
        log_candidate(st, item, "DROPPED", detail=f"window_samples={len(win)}")
        return

    ft = feature_from_samples(buf, item["peak_ms"], WINDOW_HALF_MS)
    if ft is None:
        st.dropped += 1
        log_candidate(st, item, "DROPPED", detail="feature_none")
        return

    raw_cls, raw_prob = model_proba(DRUM_MODELS[st.hand], ft)
    raw_name = ZONE_NAMES[raw_cls]

    hcls, hprob = model_proba(H_MODELS[st.hand], ft)
    vcls, vprob = model_proba(V_MODELS[st.hand], ft)

    raw_h = HMAP[raw_name]
    raw_v = VMAP[raw_name]
    opposite = abs(raw_h - hcls) == 2
    vdiff = raw_v != vcls

    if opposite:
        final = "方向衝突 / 不確定"
        warning = f"主模型={raw_name}({HN[raw_h]})，水平模型={HN[hcls]}，拒絕跨側輸出"
        st.side_conflicts += 1
    else:
        final = raw_name
        warning = "上下模型不同意" if vdiff else ""

    near = min(win, key=lambda s: abs(s["t"] - item["peak_ms"]))
    st.hits += 1
    st.last_drum = final
    st.last_raw = raw_name
    st.last_prob = raw_prob
    st.last_h = HN[hcls]
    st.last_hprob = hprob
    st.last_v = VN[vcls]
    st.last_vprob = vprob
    st.last_rel = (near["rel_yaw"], near["rel_pitch"], near["rel_roll"])
    st.last_warning = warning

    hit_w.writerow([
        wall_iso(), f"{elapsed_s():.3f}", current_marker, st.hand, st.port,
        item["id"], f'{item["peak_ms"]:.3f}', f'{item["peak_g"]:.5f}', len(win),
        raw_name, f"{raw_prob:.6f}",
        HN[hcls], f"{hprob:.6f}", VN[vcls], f"{vprob:.6f}",
        HN[raw_h], VN[raw_v], int(opposite), int(vdiff),
        final,
        f'{near["rel_yaw"]:.4f}', f'{near["rel_pitch"]:.4f}', f'{near["rel_roll"]:.4f}',
        *[f"{float(x):.8f}" for x in ft],
    ])
    hit_f.flush()
    log_candidate(st, item, "RESOLVED", detail=f"{raw_name}/{final}")
    uiq.put(("hit", st.hand, final, warning))


def resolve_old_candidates(st: HandState, now_ms: float):
    remain = []
    for item in st.pending:
        # Wait full 160 ms so a stronger impact can replace a weak pre-peak.
        if now_ms < item["peak_ms"] + PEAK_RESOLVE_MS:
            remain.append(item)
            continue
        classify_resolved(st, item)
    st.pending = remain


def update_health_on_invalid(st: HandState, reason: str):
    st.bad_run += 1
    st.good_run = 0
    st.last_invalid_reason = reason
    if st.bad_run >= BAD_STREAK_RESET:
        if not st.needs_warmup:
            st.needs_warmup = True
            st.reset_detector()
        if mode == "cal":
            st.cal.clear()  # require a fresh continuous neutral block after a real fault
        st.set_health("BAD")
    else:
        st.set_health("WARN")


def update_health_on_valid(st: HandState):
    was_bad = st.bad_run > 0
    st.bad_run = 0
    st.good_run += 1

    if st.needs_warmup:
        if st.good_run >= MIN_DETECTOR_WARMUP_VALID:
            st.needs_warmup = False
            st.reset_detector()
            st.set_health("OK")
        else:
            st.set_health("RECOVERING" if was_bad or st.health in ("BAD", "WARN", "RECOVERING") else "WARMUP")
    else:
        st.set_health("OK")


def worker(port: str, conn: serial.Serial):
    global cal_finish_queued
    n = 0
    hz_tick = time.monotonic()

    try:
        while not stop.is_set():
            b = conn.readline()
            if not b:
                continue
            p = parse_line(b.decode("utf-8", "ignore"))
            if p is None:
                continue

            h = p["hand"]
            st = states[h]
            now = time.monotonic()
            now_ms = now * 1000.0
            wall = wall_iso()

            with lock:
                st.port = port
                st.rx += 1
                n += 1
                if now - hz_tick >= 1.0:
                    st.hz = n / (now - hz_tick)
                    n = 0
                    hz_tick = now

                valid, reason = st.validator.check(p)
                if not valid:
                    update_health_on_invalid(st, reason)
                else:
                    update_health_on_valid(st)

                raw_w.writerow([
                    wall, f"{elapsed_s():.3f}", mode, current_marker, port, h,
                    p["packet_id"], p["sensor_ms"],
                    p["ax"], p["ay"], p["az"], p["yaw"], p["pitch"], p["roll"],
                    int(valid), reason, st.health, st.bad_run, st.good_run,
                ])
                if st.rx % 50 == 0:
                    raw_f.flush()

                if not valid:
                    continue

                if mode == "cal":
                    if len(st.cal) < CAL_SAMPLES:
                        st.cal.append((p["yaw"], p["pitch"], p["roll"]))
                    if (
                        not cal_finish_queued
                        and len(states["L"].cal) >= CAL_SAMPLES
                        and len(states["R"].cal) >= CAL_SAMPLES
                    ):
                        cal_finish_queued = True
                        uiq.put(("cal_ready",))
                    continue

                if mode != "live" or st.R0 is None or st.needs_warmup:
                    continue

                rs = make_relative_sample(
                    st.R0, now_ms,
                    p["ax"], p["ay"], p["az"],
                    p["yaw"], p["pitch"], p["roll"],
                )
                st.buffer.append(rs)

                pkt = SensorPacket(
                    hand=h,
                    ax=p["ax"], ay=p["ay"], az=p["az"],
                    yaw=p["yaw"], pitch=p["pitch"], roll=p["roll"],
                    packet_id=p["packet_id"],
                    sensor_time_ms=p["sensor_ms"],
                    received_time_ms=int(time.time() * 1000),
                    received_monotonic=now,
                )
                ev = st.detector.add_packet(pkt, rs["rel_yaw"], rs["rel_pitch"])
                if ev:
                    peak_ms = now_ms - PEAK_LAG_MS
                    add_candidate(st, peak_ms, float(ev["peak_accel_g"]))

                resolve_old_candidates(st, now_ms)

    except Exception as e:
        log_event("SERIAL_ERROR", port=port, error=repr(e))
        uiq.put(("status", f"{port} reader error: {e!r}"))


def connect_all():
    ports = xiao_ports()
    log_event("PORT_SCAN", ports=ports)
    if len(ports) < 2:
        uiq.put(("status", f"只找到 {len(ports)} 顆 XIAO：{ports}"))
        return

    opened = []
    try:
        for p in ports[:2]:
            opened.append((p, serial.Serial(p, SERIAL_BAUD, timeout=0.08)))
        time.sleep(0.4)
        for p, s in opened:
            try:
                s.reset_input_buffer()
            except Exception:
                pass
            serials[p] = s
            t = threading.Thread(target=worker, args=(p, s), daemon=True)
            t.start()
            threads.append(t)
        log_event("CONNECTED", ports=[p for p, _ in opened])
        uiq.put(("status", f"已連線 {', '.join(p for p, _ in opened)}。請先做 200-frame R0 歸零。"))
    except Exception as e:
        for _, s in opened:
            try:
                s.close()
            except Exception:
                pass
        log_event("CONNECT_FAIL", error=repr(e))
        uiq.put(("status", f"COM 連線失敗：{e}"))


def start_cal():
    global mode, cal_finish_queued
    if len(serials) < 2:
        status_var.set("尚未連好兩個 COM。")
        return

    with lock:
        mode = "cal"
        cal_finish_queued = False
        for st in states.values():
            st.cal.clear()
            st.R0 = None
            st.zero = None
            st.hits = 0
            st.candidates = 0
            st.replaced = 0
            st.dropped = 0
            st.side_conflicts = 0
            st.last_drum = "—"
            st.last_raw = "—"
            st.last_warning = ""
            st.pending.clear()
            st.buffer.clear()
            st.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)

    log_event("CALIBRATE_BEGIN")
    cal_btn.configure(state="disabled")
    title_var.set("200-frame R0 歸零中")
    status_var.set("雙手保持標準正前方起始姿勢。若資料鏈連續故障，該手的歸零樣本會自動重收。")


def finish_cal():
    global mode
    info = {}
    ok = True
    with lock:
        for h, st in states.items():
            try:
                st.R0, st.zero = compute_r0(st.cal[:CAL_SAMPLES])
                st.pending.clear()
                st.buffer.clear()
                st.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
                info[h] = {
                    "samples": len(st.cal),
                    "zero": [float(x) for x in st.zero],
                    "port": st.port,
                    "invalid_total": st.validator.invalid,
                }
            except Exception as e:
                ok = False
                info[h] = {"error": repr(e), "samples": len(st.cal), "port": st.port}

    (SESSION / "calibration.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    cal_btn.configure(state="normal")

    if not ok:
        mode = "idle"
        title_var.set("歸零失敗")
        status_var.set(json.dumps(info, ensure_ascii=False))
        log_event("CALIBRATE_FAIL", info=info)
        return

    mode = "live"
    title_var.set("LIVE｜Timing-Robust 真七鼓")
    status_var.set("模型已凍結：8/12 GT + ±20ms temporal augmentation。弱前峰會等待160ms解析；不做每鼓校準、不現場重訓。")
    log_event("CALIBRATE_OK", info=info)


def set_marker():
    global current_marker
    current_marker = marker_combo.get() or "自由"
    marker_now_var.set(f"目前標記：{current_marker}")
    log_event("MARKER", marker=current_marker)


def close():
    stop.set()
    with lock:
        summary = {
            "session": str(SESSION),
            "model_dir": str(MODEL_DIR),
            "hits": {h: states[h].hits for h in ("L", "R")},
            "candidates": {h: states[h].candidates for h in ("L", "R")},
            "replaced_weak_prepeaks": {h: states[h].replaced for h in ("L", "R")},
            "dropped_candidates": {h: states[h].dropped for h in ("L", "R")},
            "side_conflicts": {h: states[h].side_conflicts for h in ("L", "R")},
            "invalid_frames": {h: states[h].validator.invalid for h in ("L", "R")},
            "invalid_reasons": {h: dict(states[h].validator.reasons) for h in ("L", "R")},
        }
    try:
        (SESSION / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    log_event("APP_CLOSE", summary=summary)
    for s in serials.values():
        try:
            s.close()
        except Exception:
            pass
    for f in (raw_f, cand_f, hit_f, health_f, event_f):
        try:
            f.close()
        except Exception:
            pass
    root.destroy()


meta = {
    "purpose": "HoloGrip Field v2 - next-person zero-shot real-drum candidate",
    "created_at": wall_iso(),
    "models": {
        "main": {
            "source": "8/12 P01 true-drum GT",
            "temporal_augmentation_ms": [-20, 0, 20],
            "architecture": "31D -> MLP(64,32) -> 7 drums, per hand",
            "dir": str(MODEL_DIR),
            "cross_date_8_12_to_8_5_accuracy": 0.951417004048583,
            "cross_date_L_accuracy": 0.8888888888888888,
            "cross_date_R_accuracy": 0.9620853080568721,
            "opposite_side_errors": "0/204 side events",
        },
        "coarse_HV": "8/12 P01 H(LEFT/CENTER/RIGHT) and V(UP/LOW), sanity check only",
    },
    "detector": {
        "base": PRODUCT_DETECTOR_KWARGS,
        "peak_lag_ms": PEAK_LAG_MS,
        "feature_window_half_ms": WINDOW_HALF_MS,
        "resolver": {
            "wait_ms": PEAK_RESOLVE_MS,
            "replace_early_if_g_lte": REPLACE_EARLY_MAX_G,
            "later_peak_ratio_gte": REPLACE_RATIO,
            "later_peak_delta_g_gte": REPLACE_DELTA_G,
            "8_12_research_hit_f1_before": 0.9193,
            "8_12_research_hit_f1_after_approx": 0.9395,
        },
    },
    "calibration": {
        "neutral_valid_frames_per_hand": CAL_SAMPLES,
        "per_drum_calibration": "NONE",
        "runtime_retraining": "NONE",
    },
    "health_guard": {
        "validator": "FrameValidator",
        "consecutive_bad_reset": BAD_STREAK_RESET,
        "recovery_valid_frames": MIN_DETECTOR_WARMUP_VALID,
        "firmware_note": "Current firmware may emit zero-valued normal D packets on I2C read failure; host guard rejects them.",
    },
}
(SESSION / "session_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
log_event("APP_START", session=str(SESSION), meta=meta)

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
root = tk.Tk()
root.title("HoloGrip｜Field v2｜Timing-Robust 真七鼓 Zero-Shot")
root.geometry("1240x820")
root.minsize(1120, 740)
root.attributes("-topmost", True)

style = ttk.Style()
style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 22, "bold"))
style.configure("Drum.TLabel", font=("Microsoft JhengHei UI", 34, "bold"))
style.configure("Mid.TLabel", font=("Microsoft JhengHei UI", 14))
style.configure("Health.TLabel", font=("Microsoft JhengHei UI", 14, "bold"))
style.configure("Warn.TLabel", font=("Microsoft JhengHei UI", 12, "bold"))
style.configure("Btn.TButton", font=("Microsoft JhengHei UI", 12, "bold"), padding=9)

ttk.Label(root, text="HoloGrip Field v2｜Timing-Robust 真七鼓", style="Title.TLabel", anchor="center").pack(fill="x", pady=(14, 4))

title_var = tk.StringVar(value="等待 200-frame R0 歸零")
ttk.Label(root, textvariable=title_var, style="Title.TLabel", anchor="center").pack(fill="x", pady=4)

status_var = tk.StringVar(value="正在連線 COM4 / COM5…")
ttk.Label(root, textvariable=status_var, style="Mid.TLabel", anchor="center", justify="center", wraplength=1180).pack(fill="x", padx=18, pady=5)

marker_frame = ttk.Frame(root)
marker_frame.pack(fill="x", padx=20, pady=5)
ttk.Label(marker_frame, text="測試標記（只寫進 log，不影響模型）：").pack(side="left")
marker_combo = ttk.Combobox(
    marker_frame,
    values=["自由", "正前方交叉", "左上 Crash 模擬", "中上", "右上", "左中", "正中", "右中"],
    state="readonly", width=22,
)
marker_combo.set("自由")
marker_combo.pack(side="left", padx=5)
ttk.Button(marker_frame, text="套用標記", command=set_marker).pack(side="left", padx=5)
marker_now_var = tk.StringVar(value="目前標記：自由")
ttk.Label(marker_frame, textvariable=marker_now_var).pack(side="left", padx=12)

frm = ttk.Frame(root)
frm.pack(fill="both", expand=True, padx=18, pady=8)
frm.columnconfigure(0, weight=1)
frm.columnconfigure(1, weight=1)
frm.rowconfigure(0, weight=1)

ui = {}
for col, h in enumerate(("L", "R")):
    box = ttk.LabelFrame(frm, text=("左手 L" if h == "L" else "右手 R"), padding=14)
    box.grid(row=0, column=col, sticky="nsew", padx=8)

    health = tk.StringVar(value="HEALTH：WARMUP")
    drum = tk.StringVar(value="—")
    raw = tk.StringVar(value="主模型：—")
    coarse = tk.StringVar(value="H —｜V —")
    ang = tk.StringVar(value="相對角：Yaw —｜Pitch —｜Roll —")
    warn = tk.StringVar(value="")
    info = tk.StringVar(value="等待資料")

    ttk.Label(box, textvariable=health, style="Health.TLabel", anchor="center").pack(fill="x", pady=(8, 4))
    ttk.Label(box, textvariable=drum, style="Drum.TLabel", anchor="center").pack(fill="x", pady=(10, 8))
    ttk.Label(box, textvariable=raw, style="Mid.TLabel", anchor="center").pack(fill="x", pady=4)
    ttk.Label(box, textvariable=coarse, style="Mid.TLabel", anchor="center").pack(fill="x", pady=4)
    ttk.Label(box, textvariable=ang, anchor="center").pack(fill="x", pady=4)
    ttk.Label(box, textvariable=warn, style="Warn.TLabel", anchor="center", justify="center", wraplength=520).pack(fill="x", pady=8)
    ttk.Label(box, textvariable=info, anchor="center", justify="center").pack(fill="x", pady=8)

    ui[h] = (health, drum, raw, coarse, ang, warn, info)

btn = ttk.Frame(root)
btn.pack(pady=12)
cal_btn = ttk.Button(btn, text="200-frame R0 歸零並開始", style="Btn.TButton", command=start_cal)
cal_btn.pack(side="left", padx=7)
ttk.Button(btn, text="關閉並保存", style="Btn.TButton", command=close).pack(side="left", padx=7)

ttk.Label(
    root,
    text="主模型：8/12 真鼓 GT + ±20ms temporal augmentation｜Peak resolver：弱前峰最多等待160ms｜不重訓",
    anchor="center",
).pack(fill="x", pady=(0, 10))


def ui_update():
    while True:
        try:
            e = uiq.get_nowait()
        except queue.Empty:
            break
        if e[0] == "status":
            status_var.set(e[1])
        elif e[0] == "cal_ready":
            finish_cal()

    with lock:
        for h, st in states.items():
            health, drum, raw, coarse, ang, warn, info = ui[h]

            health.set(f"HEALTH：{st.health}")
            if mode == "cal":
                drum.set(f"R0 {min(len(st.cal), CAL_SAMPLES)} / {CAL_SAMPLES}")
            else:
                drum.set(st.last_drum)

            raw.set(f"主模型：{st.last_raw}｜信心 {st.last_prob * 100:.1f}%")
            coarse.set(f"H {st.last_h} {st.last_hprob * 100:.1f}%｜V {st.last_v} {st.last_vprob * 100:.1f}%")
            y, p, r = st.last_rel
            ang.set(f"相對角：Yaw {y:+.1f}°｜Pitch {p:+.1f}°｜Roll {r:+.1f}°")

            health_warning = ""
            if st.health == "BAD":
                health_warning = f"資料鏈異常：{st.last_invalid_reason}；停止模型輸入"
            elif st.health in ("WARN", "RECOVERING"):
                health_warning = f"資料鏈恢復中：{st.last_invalid_reason}"

            warn.set(health_warning or st.last_warning)
            info.set(
                f"{st.port or '未連線'}｜{st.hz:.1f} Hz｜RX {st.rx}\n"
                f"Hits {st.hits}｜Candidates {st.candidates}｜弱前峰替換 {st.replaced}｜Drop {st.dropped}\n"
                f"壞 frame {st.validator.invalid}｜方向衝突 {st.side_conflicts}"
            )

    root.after(100, ui_update)


root.protocol("WM_DELETE_WINDOW", close)
root.after(100, ui_update)
root.after(250, lambda: threading.Thread(target=connect_all, daemon=True).start())
root.mainloop()
