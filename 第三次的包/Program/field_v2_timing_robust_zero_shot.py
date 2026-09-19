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
from scipy.spatial.transform import Rotation as Rot
from serial.tools import list_ports
import tkinter as tk
from tkinter import ttk

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROGRAM_DIR))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, FEATURE_NAMES, ZONE_NAMES
from production_core import FrameValidator

# ---------------------------------------------------------------------------
# Frozen inference assets
# ---------------------------------------------------------------------------
MODEL_DIR = PACKAGE_ROOT / "Model" / "TimingRobustAug20"
DRUM_MODELS = {
    h: joblib.load(MODEL_DIR / f"hologrip_timing_robust_{h}.joblib")
    for h in ("L", "R")
}
H_MODELS = {
    h: joblib.load(PACKAGE_ROOT / "Model" / "CoarseHV" / f"H_{h}.joblib")
    for h in ("L", "R")
}
V_MODELS = {
    h: joblib.load(PACKAGE_ROOT / "Model" / "CoarseHV" / f"V_{h}.joblib")
    for h in ("L", "R")
}

HMAP = {"Crash": 0, "Hi-Hat": 0, "小鼓": 1, "高音 Tom": 1, "中音 Tom": 1, "Ride": 2, "落地 Tom": 2}
VMAP = {"Crash": 0, "高音 Tom": 0, "中音 Tom": 0, "Ride": 0, "Hi-Hat": 1, "小鼓": 1, "落地 Tom": 1}
HN = ["LEFT", "CENTER", "RIGHT"]
VN = ["UP", "LOW"]

# ---------------------------------------------------------------------------
# Field-v2 constants, fixed from 2026-09-18 research
# ---------------------------------------------------------------------------
CAL_R0_SAMPLES = 200
CAL_STABILITY_SAMPLES = 300
CAL_EDGE_SAMPLES = 50
CAL_ROT_DRIFT_MAX_DEG = 2.0
CAL_GRAVITY_DRIFT_MAX_DEG = 1.5
MIN_DETECTOR_WARMUP_VALID = 300
BAD_STREAK_RESET = 5

# Weak pre-impact candidate may be replaced by a stronger impact candidate.
PEAK_RESOLVE_MS = 160.0
REPLACE_EARLY_MAX_G = 3.5
REPLACE_RATIO = 1.4
REPLACE_DELTA_G = 0.5

MIN_FEATURE_WINDOW_SAMPLES = 12
SERIAL_BAUD = 460800

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = PACKAGE_ROOT / "Data" / "上次模型測試" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
raw_f = (SESSION / "raw_100hz.csv").open("w", encoding="utf-8", newline="", buffering=1)
cand_f = (SESSION / "candidates.csv").open("w", encoding="utf-8", newline="", buffering=1)
hit_f = (SESSION / "resolved_hits.csv").open("w", encoding="utf-8", newline="", buffering=1)
health_f = (SESSION / "health.csv").open("w", encoding="utf-8", newline="", buffering=1)
event_f = (SESSION / "events.jsonl").open("w", encoding="utf-8", buffering=1)
formal_raw_f = (SESSION / "raw_100hz_formal.csv").open("w", encoding="utf-8", newline="", buffering=1)

raw_w = csv.writer(raw_f)
cand_w = csv.writer(cand_f)
hit_w = csv.writer(hit_f)
health_w = csv.writer(health_f)
formal_raw_w = csv.writer(formal_raw_f)

raw_w.writerow([
    "wall_time", "elapsed_s", "test_time_ms", "mode", "marker", "port", "hand",
    "packet_id", "sensor_ms",
    "ax", "ay", "az", "yaw", "pitch", "roll",
    "valid", "invalid_reason", "health", "bad_run", "good_run",
])
cand_w.writerow([
    "wall_time", "elapsed_s", "test_time_ms", "marker", "hand", "port",
    "candidate_id", "event_type", "peak_ms", "peak_g",
    "related_candidate_id", "gap_ms", "detail",
])
hit_w.writerow([
    "wall_time", "elapsed_s", "test_time_ms", "marker", "hand", "port",
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
formal_raw_w.writerow([
    "session_id", "sample_index", "hand", "song_time_ms", "received_time_ms",
    "sensor_time_ms", "packet_id", "ax_g", "ay_g", "az_g", "accel_magnitude_g",
    "yaw_deg", "pitch_deg", "roll_deg", "cal_yaw_deg", "cal_pitch_deg",
])

APP_T0 = time.monotonic()
mode = "idle"  # idle / cal / live
current_marker = "自由演奏"
stop = threading.Event()
lock = threading.RLock()
uiq: queue.Queue = queue.Queue()
serials: dict[str, serial.Serial] = {}
threads: list[threading.Thread] = []
candidate_seq = 0
cal_finish_queued = False
test_running = False
test_completed = False
test_t0 = None
test_start_wall = None
test_stop_wall = None
formal_sample_index = 0
test_start_counts = {}


def wall_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def elapsed_s() -> float:
    return time.monotonic() - APP_T0


def current_test_time_ms(now_monotonic: float | None = None):
    if not test_running or test_t0 is None:
        return ""
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    return max(0.0, (now_monotonic - test_t0) * 1000.0)


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
        self.cal_acc: list[tuple[float, float, float]] = []
        self.cal_ready = False
        self.cal_rot_drift = float("inf")
        self.cal_gravity_drift = float("inf")
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

# Hands that have actually produced at least one packet this session.
# A single glove is a supported configuration: everything below operates on
# connected_hands rather than assuming both gloves are present.
connected_hands: set[str] = set()


def active_hands():
    """Hands currently connected, in a stable order."""
    return [h for h in ("L", "R") if h in connected_hands]


def _angle_between_unit(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def calibration_stability(st: HandState):
    if len(st.cal) < CAL_STABILITY_SAMPLES or len(st.cal_acc) < CAL_STABILITY_SAMPLES:
        return False, float("inf"), float("inf")
    e = np.asarray(st.cal[-CAL_STABILITY_SAMPLES:], float)
    a = np.asarray(st.cal_acc[-CAL_STABILITY_SAMPLES:], float)
    r_first = Rot.from_euler("ZYX", e[:CAL_EDGE_SAMPLES], degrees=True).mean()
    r_last = Rot.from_euler("ZYX", e[-CAL_EDGE_SAMPLES:], degrees=True).mean()
    rot_drift = float(np.degrees((r_first.inv() * r_last).magnitude()))
    a_first = a[:CAL_EDGE_SAMPLES]
    a_last = a[-CAL_EDGE_SAMPLES:]
    u1 = a_first / np.maximum(np.linalg.norm(a_first, axis=1, keepdims=True), 1e-12)
    u2 = a_last / np.maximum(np.linalg.norm(a_last, axis=1, keepdims=True), 1e-12)
    g1 = u1.mean(axis=0); g2 = u2.mean(axis=0)
    gravity_drift = _angle_between_unit(g1, g2)
    stable = (
        rot_drift <= CAL_ROT_DRIFT_MAX_DEG
        and gravity_drift <= CAL_GRAVITY_DRIFT_MAX_DEG
    )
    return stable, rot_drift, gravity_drift


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
        wall_iso(), f"{elapsed_s():.3f}", item.get("test_time_ms", ""), current_marker, st.hand, st.port,
        item["id"], event_type, f'{item["peak_ms"]:.3f}', f'{item["peak_g"]:.5f}',
        related, gap_ms if gap_ms == "" else f"{float(gap_ms):.3f}", detail,
    ])
    cand_f.flush()


def add_candidate(st: HandState, peak_ms: float, peak_g: float):
    item = {
        "id": next_candidate_id(),
        "peak_ms": float(peak_ms),
        "peak_g": float(peak_g),
        "test_time_ms": (f"{peak_ms - test_t0 * 1000.0:.3f}" if test_running and test_t0 is not None else ""),
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
        wall_iso(), f"{elapsed_s():.3f}", item.get("test_time_ms", ""), current_marker, st.hand, st.port,
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
    if mode == "cal":
        st.cal.clear()
        st.cal_acc.clear()
        st.cal_ready = False
        st.cal_rot_drift = float("inf")
        st.cal_gravity_drift = float("inf")
    st.bad_run += 1
    st.good_run = 0
    st.last_invalid_reason = reason
    if st.bad_run >= BAD_STREAK_RESET:
        if not st.needs_warmup:
            st.needs_warmup = True
            st.reset_detector()
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
    global cal_finish_queued, formal_sample_index
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
            connected_hands.add(h)
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

                tt = current_test_time_ms(now)
                raw_w.writerow([
                    wall, f"{elapsed_s():.3f}", (f"{tt:.3f}" if tt != "" else ""), mode, current_marker, port, h,
                    p["packet_id"], p["sensor_ms"],
                    p["ax"], p["ay"], p["az"], p["yaw"], p["pitch"], p["roll"],
                    int(valid), reason, st.health, st.bad_run, st.good_run,
                ])
                if st.rx % 50 == 0:
                    raw_f.flush()

                if not valid:
                    continue

                if mode == "cal":
                    if not st.cal_ready:
                        st.cal.append((p["yaw"], p["pitch"], p["roll"]))
                        st.cal_acc.append((p["ax"], p["ay"], p["az"]))
                        if len(st.cal) > CAL_STABILITY_SAMPLES:
                            st.cal = st.cal[-CAL_STABILITY_SAMPLES:]
                            st.cal_acc = st.cal_acc[-CAL_STABILITY_SAMPLES:]
                        if len(st.cal) >= CAL_STABILITY_SAMPLES:
                            stable, rd, gd = calibration_stability(st)
                            st.cal_rot_drift = rd
                            st.cal_gravity_drift = gd
                            if stable:
                                st.cal_ready = True
                                log_event("CAL_HAND_STABLE", hand=h, rot_drift_deg=rd,
                                          gravity_drift_deg=gd, samples=len(st.cal))
                            elif st.rx % 20 == 0:
                                uiq.put(("cal_drift", h, rd, gd))
                    if (
                        not cal_finish_queued
                        and states["L"].cal_ready
                        and states["R"].cal_ready
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

                if test_running and test_t0 is not None:
                    formal_sample_index += 1
                    song_ms = max(0.0, (now - test_t0) * 1000.0)
                    formal_raw_w.writerow([
                        SESSION.name, formal_sample_index, h, f"{song_ms:.3f}", int(time.time() * 1000),
                        p["sensor_ms"], p["packet_id"],
                        f'{p["ax"]:.6f}', f'{p["ay"]:.6f}', f'{p["az"]:.6f}',
                        f'{math.sqrt(p["ax"]**2 + p["ay"]**2 + p["az"]**2):.6f}',
                        f'{p["yaw"]:.6f}', f'{p["pitch"]:.6f}', f'{p["roll"]:.6f}',
                        f'{rs["rel_yaw"]:.6f}', f'{rs["rel_pitch"]:.6f}',
                    ])
                    if formal_sample_index % 50 == 0:
                        formal_raw_f.flush()

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
    if len(ports) < 1:
        uiq.put(("status", "找不到 XIAO（VID 303A:1001）。插上手套後按『重新連線』。"))
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
        uiq.put(("status", f"已連線 {', '.join(p for p, _ in opened)}。請按『R0 歸零』。"))
    except Exception as e:
        for _, s in opened:
            try:
                s.close()
            except Exception:
                pass
        log_event("CONNECT_FAIL", error=repr(e))
        uiq.put(("status", f"COM 連線失敗：{e}"))


def start_cal():
    global mode, cal_finish_queued, test_running, test_completed, test_t0, formal_sample_index
    if len(serials) < 1:
        status_var.set("尚未連上任何 COM。")
        return
    not_ok = [h for h in active_hands() if states[h].health != "OK"]
    if not not_ok and not active_hands():
        status_var.set("還沒有收到任何手套資料，請確認 USB 與線材。")
        return
    if not_ok:
        status_var.set("先等資料鏈穩定到 HEALTH=OK，再做 R0：" + ", ".join(not_ok))
        return

    with lock:
        mode = "cal"
        cal_finish_queued = False
        test_running = False
        test_completed = False
        test_t0 = None
        formal_sample_index = 0
        for st in states.values():
            st.cal.clear()
            st.cal_acc.clear()
            st.cal_ready = False
            st.cal_rot_drift = float("inf")
            st.cal_gravity_drift = float("inf")
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
    start_test_btn.configure(state="disabled")
    stop_test_btn.configure(state="disabled")
    test_state_var.set("正式測試：尚未開始")
    title_var.set("穩定 R0 歸零中")
    status_var.set(
        "雙手保持標準正前方姿勢不要動。每手先收 3 秒連續乾淨資料；"
        "只有前後姿態漂移 ≤2° 且重力方向漂移 ≤1.5° 才接受 R0。"
    )


def finish_cal():
    global mode
    info = {}
    ok = True
    with lock:
        for h in active_hands():
            st = states[h]
            try:
                st.R0, st.zero = compute_r0(st.cal[-CAL_R0_SAMPLES:])
                st.pending.clear()
                st.buffer.clear()
                st.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
                info[h] = {
                    "samples": len(st.cal),
                    "r0_samples_used": CAL_R0_SAMPLES,
                    "rot_drift_deg": st.cal_rot_drift,
                    "gravity_drift_deg": st.cal_gravity_drift,
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
    title_var.set("PREVIEW｜R0 完成，等待正式測試")
    status_var.set("可先確認兩手 HEALTH=OK。準備好同步 MIDI 後，按『開始正式測試』。")
    start_test_btn.configure(state="normal")
    stop_test_btn.configure(state="disabled")
    log_event("CALIBRATE_OK", info=info)


def start_formal_test():
    global test_running, test_completed, test_t0, test_start_wall, formal_sample_index, test_start_counts
    if mode != "live":
        status_var.set("請先完成 R0 歸零。")
        return
    if test_completed:
        status_var.set("這個程式場次已完成一次正式測試；要再測一次請關閉後重新開 01。")
        return
    bad=[h for h in active_hands() if states[h].health != "OK"]
    if bad:
        status_var.set("不能開始：資料鏈不是 OK：" + ", ".join(bad))
        return
    formal_sample_index = 0
    test_t0 = time.monotonic()
    test_start_wall = wall_iso()
    test_start_counts = {
        "hits": {h: states[h].hits for h in ("L","R")},
        "invalid": {h: states[h].validator.invalid for h in ("L","R")},
    }
    test_running = True
    start_test_btn.configure(state="disabled")
    stop_test_btn.configure(state="normal")
    test_state_var.set("正式測試：RUNNING｜MIDI 0 秒從現在開始")
    title_var.set("FORMAL TEST｜正在測目前凍結模型")
    status_var.set("正式段正在記錄 raw_100hz_formal.csv。打完後一定按『結束正式測試』。")
    log_event("FORMAL_TEST_START", wall=test_start_wall)


def stop_formal_test():
    global test_running, test_completed, test_stop_wall
    if not test_running:
        return
    stop_mono=time.monotonic()
    test_stop_wall=wall_iso()
    duration_ms=(stop_mono-test_t0)*1000.0 if test_t0 is not None else 0.0
    test_running=False
    test_completed=True
    formal_raw_f.flush(); hit_f.flush(); raw_f.flush(); cand_f.flush()
    manifest={
        "session": SESSION.name,
        "formal_raw": str((SESSION/"raw_100hz_formal.csv").resolve()),
        "start_wall": test_start_wall,
        "stop_wall": test_stop_wall,
        "duration_ms": round(duration_ms,3),
        "formal_samples": formal_sample_index,
        "start_counts": test_start_counts,
        "end_counts": {
            "hits": {h: states[h].hits for h in ("L","R")},
            "invalid": {h: states[h].validator.invalid for h in ("L","R")},
        },
        "midi_rule": "Use the MIDI recorded during this exact FORMAL_TEST_START..STOP segment; MIDI is never used during inference.",
    }
    (SESSION/"formal_test_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    start_test_btn.configure(state="disabled")
    stop_test_btn.configure(state="disabled")
    test_state_var.set(f"正式測試：完成｜{duration_ms/1000:.1f}s｜Raw {formal_sample_index} rows")
    title_var.set("正式測試完成｜請保存同一次 MIDI")
    status_var.set("已保存 raw_100hz_formal.csv + resolved_hits.csv。現在可以關閉 01，再進入 02 收第三次新資料。")
    log_event("FORMAL_TEST_STOP", manifest=manifest)


def close():
    if test_running:
        stop_formal_test()
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
    for f in (raw_f, cand_f, hit_f, health_f, event_f, formal_raw_f):
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
        "neutral_stability_frames_per_hand": CAL_STABILITY_SAMPLES,
        "neutral_r0_frames_used_per_hand": CAL_R0_SAMPLES,
        "rotation_drift_max_deg": CAL_ROT_DRIFT_MAX_DEG,
        "gravity_drift_max_deg": CAL_GRAVITY_DRIFT_MAX_DEG,
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

ttk.Label(root, text="HoloGrip｜現場測試", style="Title.TLabel", anchor="center").pack(fill="x", pady=(14, 4))

title_var = tk.StringVar(value="等待連線")
ttk.Label(root, textvariable=title_var, style="Title.TLabel", anchor="center").pack(fill="x", pady=4)

status_var = tk.StringVar(value="正在連線手套…")
ttk.Label(root, textvariable=status_var, style="Mid.TLabel", anchor="center", justify="center", wraplength=1180).pack(fill="x", padx=18, pady=5)

frm = ttk.Frame(root)
frm.pack(fill="both", expand=True, padx=18, pady=8)
frm.columnconfigure(0, weight=1)
frm.columnconfigure(1, weight=1)
frm.rowconfigure(0, weight=1)

ui = {}
boxes = {}
for col, h in enumerate(("L", "R")):
    box = ttk.LabelFrame(frm, text=("左手 L" if h == "L" else "右手 R"), padding=14)
    box.grid(row=0, column=col, sticky="nsew", padx=8)
    boxes[h] = box

    health = tk.StringVar(value="HEALTH：WARMUP")
    drum = tk.StringVar(value="—")
    raw = tk.StringVar(value="")
    coarse = tk.StringVar(value="")
    ang = tk.StringVar(value="")
    warn = tk.StringVar(value="")
    info = tk.StringVar(value="等待資料")

    ttk.Label(box, textvariable=health, style="Health.TLabel", anchor="center").pack(fill="x", pady=(10, 6))
    ttk.Label(box, textvariable=drum, style="Drum.TLabel", anchor="center").pack(fill="x", pady=(18, 14))
    ttk.Label(box, textvariable=raw, style="Mid.TLabel", anchor="center").pack(fill="x", pady=3)
    ttk.Label(box, textvariable=coarse, style="Mid.TLabel", anchor="center").pack(fill="x", pady=3)
    ttk.Label(box, textvariable=ang, anchor="center").pack(fill="x", pady=3)
    ttk.Label(box, textvariable=warn, style="Warn.TLabel", anchor="center", justify="center", wraplength=520).pack(fill="x", pady=8)
    ttk.Label(box, textvariable=info, anchor="center", justify="center").pack(fill="x", pady=8)

    ui[h] = (health, drum, raw, coarse, ang, warn, info)

btn = ttk.Frame(root)
btn.pack(pady=14)
cal_btn = ttk.Button(btn, text="R0 歸零", style="Btn.TButton", command=start_cal)
cal_btn.pack(side="left", padx=7)
start_test_btn = ttk.Button(btn, text="開始測試", style="Btn.TButton", command=start_formal_test, state="disabled")
start_test_btn.pack(side="left", padx=7)
stop_test_btn = ttk.Button(btn, text="結束測試", style="Btn.TButton", command=stop_formal_test, state="disabled")
stop_test_btn.pack(side="left", padx=7)
ttk.Button(btn, text="關閉並保存", style="Btn.TButton", command=close).pack(side="left", padx=7)

test_state_var = tk.StringVar(value="測試：尚未開始")
ttk.Label(root, textvariable=test_state_var, style="Warn.TLabel", anchor="center").pack(fill="x", pady=(0, 12))


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
        elif e[0] == "cal_drift":
            _, hh, rd, gd = e
            status_var.set(
                f"{hh} R0 尚未穩定：姿態漂移 {rd:.2f}°、重力漂移 {gd:.2f}°。"
                " 請保持不動，程式會自動等到穩定。"
            )

    with lock:
        # Only show a panel for a hand that is actually sending data.
        # Before anything connects, show both so the window is not empty.
        act = active_hands()
        shown = act if act else ["L", "R"]
        for h in ("L", "R"):
            if h in shown:
                boxes[h].grid()
            else:
                boxes[h].grid_remove()

        for h in shown:
            st = states[h]
            health, drum, raw, coarse, ang, warn, info = ui[h]

            health.set(f"HEALTH：{st.health}")
            if mode == "cal":
                if st.cal_ready:
                    drum.set("R0 STABLE ✓")
                elif len(st.cal) < CAL_STABILITY_SAMPLES:
                    drum.set(f"R0 穩定窗 {len(st.cal)} / {CAL_STABILITY_SAMPLES}")
                else:
                    drum.set(f"R0 等穩定 {st.cal_rot_drift:.1f}° / {st.cal_gravity_drift:.1f}°")
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

