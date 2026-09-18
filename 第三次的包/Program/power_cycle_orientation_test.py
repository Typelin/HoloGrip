from __future__ import annotations

import csv
import json
import math
import queue
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports
from scipy.spatial.transform import Rotation as Rot
import tkinter as tk
from tkinter import ttk, messagebox

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROGRAM_DIR))
from production_core import FrameValidator

BAUD = 460800
SETTLE_S = 3.0
COLLECT_N = 300
DISCONNECT_S = 1.2
SCAN_S = 0.45
TARGET_VIDPID = ("303A:1001", "VID_303A&PID_1001")

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION = PACKAGE_ROOT / "Data" / "電源循環姿態測試" / STAMP
SESSION.mkdir(parents=True, exist_ok=True)

raw_f = (SESSION / "raw.csv").open("w", encoding="utf-8", newline="", buffering=1)
event_f = (SESSION / "events.jsonl").open("w", encoding="utf-8", buffering=1)
raw_w = csv.writer(raw_f)
raw_w.writerow([
    "wall_time", "elapsed_s", "stage", "collecting", "port", "hand",
    "packet_id", "sensor_ms", "ax", "ay", "az", "yaw", "pitch", "roll",
    "valid", "invalid_reason",
])

APP_T0 = time.monotonic()
lock = threading.RLock()
stop_evt = threading.Event()
uiq: queue.Queue = queue.Queue()

ports_open: dict[str, serial.Serial] = {}
threads: dict[str, threading.Thread] = {}
validators: dict[str, FrameValidator] = {}
port_hand: dict[str, str] = {}
last_packet_by_hand = {"L": 0.0, "R": 0.0}
target_hand = "R"
target_connected = False
ever_target_connected = False
target_port = ""

stage = "IDLE"
collect_name: str | None = None
settle_until = 0.0
stage_samples: dict[str, list[dict]] = {
    "A0": [],
    "B_before": [],
    "B_after": [],
    "A_after_Bboot": [],
    "A_reboot": [],
}
stage_results: dict[str, dict] = {}
disconnect_seen_for = None
reconnect_expected_for = None

STAGE_LABELS = {
    "A0": "A0｜基準姿勢，第一次上電",
    "B_before": "B_before｜同一次上電，由 A 移到 B",
    "B_after": "B_after｜保持 B，斷電後重新上電",
    "A_after_Bboot": "A_after_Bboot｜B 開機後，不斷電移回 A",
    "A_reboot": "A_reboot｜保持 A，再斷電重新上電",
}


def wall_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def elapsed_s():
    return time.monotonic() - APP_T0


def log_event(kind: str, **data):
    event_f.write(json.dumps({
        "wall_time": wall_iso(),
        "elapsed_s": round(elapsed_s(), 3),
        "kind": kind,
        **data,
    }, ensure_ascii=False) + "\n")
    event_f.flush()


def is_xiao(p):
    hw = (p.hwid or "").upper()
    return any(x in hw for x in TARGET_VIDPID)


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


def rot_center(samples: list[dict]):
    e = np.array([[s["yaw"], s["pitch"], s["roll"]] for s in samples], dtype=float)
    rs = Rot.from_euler("ZYX", e, degrees=True)
    center = rs.mean()
    # geodesic spread from mean
    rel = center.inv() * rs
    ang = np.degrees(rel.magnitude())
    return center, ang


def gravity_center(samples: list[dict]):
    a = np.array([[s["ax"], s["ay"], s["az"]] for s in samples], dtype=float)
    n = np.linalg.norm(a, axis=1)
    good = n > 1e-6
    u = a[good] / n[good, None]
    m = u.mean(axis=0)
    m /= np.linalg.norm(m)
    dots = np.clip(u @ m, -1, 1)
    spread = np.degrees(np.arccos(dots))
    return m, spread, np.median(n[good]), np.percentile(n[good], [10, 90])


def angle_vec(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def geodesic_deg(r1: Rot, r2: Rot):
    return float(np.degrees((r1.inv() * r2).magnitude()))


def summarize_stage(name: str):
    ss = stage_samples[name]
    rc, rspread = rot_center(ss)
    gv, gspread, amag_med, amag_p = gravity_center(ss)
    euler = rc.as_euler("ZYX", degrees=True)
    res = {
        "stage": name,
        "label": STAGE_LABELS[name],
        "n": len(ss),
        "mean_rotation_euler_ZYX_deg": [float(x) for x in euler],
        "rotation_spread_deg": {
            "median": float(np.median(rspread)),
            "p90": float(np.percentile(rspread, 90)),
            "max": float(np.max(rspread)),
        },
        "gravity_unit_mean": [float(x) for x in gv],
        "gravity_spread_deg": {
            "median": float(np.median(gspread)),
            "p90": float(np.percentile(gspread, 90)),
        },
        "accel_norm_g": {
            "median": float(amag_med),
            "p10": float(amag_p[0]),
            "p90": float(amag_p[1]),
        },
        "_rot": rc,
        "_gravity": gv,
    }
    stage_results[name] = res
    return res


def public_result(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def finish_analysis():
    for name in stage_samples:
        if len(stage_samples[name]) >= COLLECT_N:
            summarize_stage(name)

    required = ["A0", "B_before", "B_after", "A_after_Bboot", "A_reboot"]
    if any(x not in stage_results for x in required):
        return None

    A0 = stage_results["A0"]
    Bb = stage_results["B_before"]
    Ba = stage_results["B_after"]
    AfromB = stage_results["A_after_Bboot"]
    Ar = stage_results["A_reboot"]

    # Same physical pose across power cycles.
    same_A_raw = geodesic_deg(A0["_rot"], Ar["_rot"])
    same_B_raw = geodesic_deg(Bb["_rot"], Ba["_rot"])
    same_A_grav = angle_vec(A0["_gravity"], Ar["_gravity"])
    same_B_grav = angle_vec(Bb["_gravity"], Ba["_gravity"])

    # A after booting at B: if physical A is reproduced, compare raw frame.
    A_bootpose_raw = geodesic_deg(A0["_rot"], AfromB["_rot"])
    A_bootpose_grav = angle_vec(A0["_gravity"], AfromB["_gravity"])

    # Relative transform test.
    # Boot-at-A session: A0 -> B_before
    T_AB_from_Aboot = A0["_rot"].inv() * Bb["_rot"]
    # Boot-at-B session: B_after -> A_after_Bboot; invert to get A->B
    T_AB_from_Bboot = (Ba["_rot"].inv() * AfromB["_rot"]).inv()
    relative_transform_mismatch = float(np.degrees(
        (T_AB_from_Aboot.inv() * T_AB_from_Bboot).magnitude()
    ))

    relA = T_AB_from_Aboot.as_euler("ZYX", degrees=True)
    relB = T_AB_from_Bboot.as_euler("ZYX", degrees=True)

    # Physical placement quality: gravity cannot verify yaw around gravity,
    # but it catches obvious failure to return to the same tilt.
    physical_A_ok = max(same_A_grav, A_bootpose_grav) <= 5.0
    physical_B_ok = same_B_grav <= 5.0

    if physical_A_ok and physical_B_ok:
        if relative_transform_mismatch <= 3.0 and max(same_A_raw, same_B_raw) <= 5.0:
            verdict = "PASS_STRONG"
            note = "同姿勢跨上電與 A↔B 相對旋轉都高度可重現；上電座標不是目前主要嫌疑。"
        elif relative_transform_mismatch <= 7.0 and max(same_A_raw, same_B_raw) <= 10.0:
            verdict = "PASS_WITH_DRIFT"
            note = "有可見漂移，但相對幾何大致保留；R0 理論上仍可能消掉大部分上電偏移。"
        elif relative_transform_mismatch > 10.0:
            verdict = "FAIL_RELATIVE_FRAME"
            note = "不同上電姿勢下，A↔B 相對旋轉本身都變了；這會直接威脅 R0.T @ R 的跨 session 可重現性。"
        else:
            verdict = "FAIL_POWER_CYCLE_REPEATABILITY"
            note = "同一物理姿勢跨上電的 raw orientation 差異過大；需確認 IMU 初始化/磁航向/融合器重置。"
    else:
        verdict = "AMBIGUOUS_PHYSICAL_POSE"
        note = "A/B 的重力方向本身沒有回到足夠相同的物理傾角，這輪不能把差異全怪給 IMU；建議用桌面/治具重做。"

    result = {
        "session": str(SESSION),
        "target_hand": target_hand,
        "settle_s": SETTLE_S,
        "samples_per_stage": COLLECT_N,
        "stages": {k: public_result(v) for k, v in stage_results.items()},
        "comparisons": {
            "same_A_raw_rotation_diff_deg_A0_vs_A_reboot": same_A_raw,
            "same_B_raw_rotation_diff_deg_B_before_vs_B_after": same_B_raw,
            "same_A_raw_after_boot_at_B_diff_deg": A_bootpose_raw,
            "same_A_gravity_diff_deg_A0_vs_A_reboot": same_A_grav,
            "same_B_gravity_diff_deg_B_before_vs_B_after": same_B_grav,
            "same_A_gravity_after_boot_at_B_diff_deg": A_bootpose_grav,
            "relative_AB_from_Aboot_euler_ZYX_deg": [float(x) for x in relA],
            "relative_AB_from_Bboot_euler_ZYX_deg": [float(x) for x in relB],
            "relative_transform_mismatch_deg": relative_transform_mismatch,
        },
        "physical_pose_check": {
            "A_tilt_repeatable_within_5deg": bool(physical_A_ok),
            "B_tilt_repeatable_within_5deg": bool(physical_B_ok),
            "note": "gravity direction checks tilt only; yaw about gravity still requires the physical fixture/marks to be reproduced.",
        },
        "verdict": verdict,
        "verdict_note": note,
    }

    (SESSION / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def set_instruction(title, body, button_text=None, button_cmd=None):
    uiq.put(("instruction", title, body, button_text, button_cmd))


def start_collect(name: str, settle_s=SETTLE_S):
    global collect_name, settle_until
    with lock:
        stage_samples[name].clear()
        collect_name = name
        settle_until = time.monotonic() + settle_s
    log_event("COLLECT_ARM", stage=name, settle_s=settle_s)
    uiq.put(("collect", name, 0, COLLECT_N, "settling"))


def complete_collect(name: str):
    global collect_name, stage, disconnect_seen_for, reconnect_expected_for
    with lock:
        collect_name = None
    res = summarize_stage(name)
    log_event("COLLECT_DONE", stage=name, result=public_result(res))
    uiq.put(("collect", name, COLLECT_N, COLLECT_N, "done"))

    if name == "A0":
        stage = "MOVE_B_BEFORE"
        set_instruction(
            "步驟 2｜保持通電，移到 B 姿勢",
            "現在不要拔 USB。\n把手套翻到一個明顯不同、但你容易重現的 B 姿勢，保持完全不動。\n擺好後按下面按鈕。",
            "我已擺好 B，開始收 B_before",
            lambda: start_collect("B_before"),
        )
    elif name == "B_before":
        stage = "WAIT_UNPLUG_B"
        disconnect_seen_for = "B"
        reconnect_expected_for = None
        set_instruction(
            "步驟 3｜保持 B 姿勢，拔掉 USB 電源",
            "不要改變 B 的物理姿勢。\n現在把正在測的這隻手套 USB 拔掉。\n程式偵測到斷線後，會叫你在同一個 B 姿勢插回去。",
        )
    elif name == "B_after":
        stage = "MOVE_A_AFTER_BBOOT"
        set_instruction(
            "步驟 4｜不要斷電，從 B 回到原本 A",
            "現在保持 USB 通電。\n把手套移回跟最開始 A0『一模一樣』的實體姿勢。\n最好靠桌角、治具或記號，不要只靠感覺。\n擺好後按下面按鈕。",
            "我已回到 A，開始收 A_after_Bboot",
            lambda: start_collect("A_after_Bboot"),
        )
    elif name == "A_after_Bboot":
        stage = "WAIT_UNPLUG_A"
        disconnect_seen_for = "A"
        reconnect_expected_for = None
        set_instruction(
            "步驟 5｜保持 A 姿勢，拔掉 USB 電源",
            "不要再移動 A。\n保持這個 A 姿勢，現在拔掉 USB。\n偵測到斷線後，程式會叫你保持 A 直接插回。",
        )
    elif name == "A_reboot":
        stage = "DONE"
        result = finish_analysis()
        if result:
            uiq.put(("done", result))


def reader(port: str, conn: serial.Serial):
    global target_connected, ever_target_connected, target_port, settle_until
    validator = FrameValidator()
    validators[port] = validator
    log_event("PORT_OPEN", port=port)
    try:
        while not stop_evt.is_set():
            try:
                b = conn.readline()
            except Exception as e:
                raise OSError(str(e))
            if not b:
                continue
            p = parse_line(b.decode("utf-8", "ignore"))
            if p is None:
                continue

            now = time.monotonic()
            h = p["hand"]
            with lock:
                port_hand[port] = h
                last_packet_by_hand[h] = now
                if h == target_hand:
                    if not target_connected:
                        target_connected = True
                        ever_target_connected = True
                        target_port = port
                        log_event("TARGET_CONNECTED", hand=h, port=port, packet_id=p["packet_id"])
                        uiq.put(("connection", True, port, h))

                    valid, reason = validator.check(p)
                    raw_w.writerow([
                        wall_iso(), f"{elapsed_s():.3f}", stage,
                        int(collect_name is not None), port, h,
                        p["packet_id"], p["sensor_ms"],
                        p["ax"], p["ay"], p["az"], p["yaw"], p["pitch"], p["roll"],
                        int(valid), reason,
                    ])

                    if (not valid) and collect_name is not None:
                        # Scientific integrity: a stage is valid only if all 300 frames
                        # are consecutive and clean. Any toxic frame invalidates the
                        # current partial block and restarts the settle period.
                        if stage_samples[collect_name]:
                            log_event(
                                "COLLECT_RESET_INVALID",
                                stage=collect_name,
                                collected=len(stage_samples[collect_name]),
                                reason=reason,
                            )
                        stage_samples[collect_name].clear()
                        settle_until = now + SETTLE_S
                        uiq.put(("collect_reset", collect_name, reason))

                    if valid and collect_name is not None and now >= settle_until:
                        ss = stage_samples[collect_name]
                        if len(ss) < COLLECT_N:
                            ss.append({
                                **p,
                                "wall_time": wall_iso(),
                                "monotonic": now,
                                "port": port,
                            })
                            n = len(ss)
                            if n % 10 == 0 or n == 1:
                                uiq.put(("collect", collect_name, n, COLLECT_N, "collecting"))
                            if n >= COLLECT_N:
                                uiq.put(("stage_complete", collect_name))
    except Exception as e:
        log_event("PORT_ERROR", port=port, error=repr(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass
        with lock:
            ports_open.pop(port, None)
            threads.pop(port, None)
            validators.pop(port, None)
            h = port_hand.pop(port, None)
        log_event("PORT_CLOSED", port=port, hand=h)


def scanner():
    while not stop_evt.is_set():
        present = {p.device: p for p in list_ports.comports() if is_xiao(p)}
        # Close ports that vanished. Usually reader already errors, but force close if needed.
        with lock:
            opened = list(ports_open.items())
        for dev, conn in opened:
            if dev not in present:
                try:
                    conn.close()
                except Exception:
                    pass

        for dev in present:
            with lock:
                already = dev in ports_open
            if already:
                continue
            try:
                conn = serial.Serial(dev, BAUD, timeout=0.08)
                time.sleep(0.12)
                try:
                    conn.reset_input_buffer()
                except Exception:
                    pass
                with lock:
                    ports_open[dev] = conn
                t = threading.Thread(target=reader, args=(dev, conn), daemon=True)
                with lock:
                    threads[dev] = t
                t.start()
            except Exception:
                pass
        time.sleep(SCAN_S)


def state_monitor():
    global target_connected, stage, disconnect_seen_for, reconnect_expected_for
    while not stop_evt.is_set():
        now = time.monotonic()
        with lock:
            last = last_packet_by_hand.get(target_hand, 0.0)
            was_connected = target_connected
        connected_now = bool(last and now - last < DISCONNECT_S)

        if was_connected and not connected_now:
            with lock:
                target_connected = False
            log_event("TARGET_DISCONNECTED", hand=target_hand)
            uiq.put(("connection", False, "", target_hand))

            if stage == "WAIT_UNPLUG_B":
                stage = "WAIT_REPLUG_B"
                reconnect_expected_for = "B_after"
                set_instruction(
                    "已偵測到斷電｜保持 B 不動，現在插回 USB",
                    "手套保持在同一個 B 姿勢。\n現在插回 USB。\n程式偵測到同一隻手後會先等 3 秒穩定，再自動收 300 frame。",
                )
            elif stage == "WAIT_UNPLUG_A":
                stage = "WAIT_REPLUG_A"
                reconnect_expected_for = "A_reboot"
                set_instruction(
                    "已偵測到斷電｜保持 A 不動，現在插回 USB",
                    "手套保持跟 A0 一模一樣的位置。\n現在插回 USB。\n偵測到後會等 3 秒，再自動收最後 300 frame。",
                )

        if connected_now:
            with lock:
                target_connected = True
            # Reader may already flip target_connected to True on the first packet,
            # so drive the reconnect state machine from packet freshness + stage,
            # not only from the previous boolean value.
            if stage == "WAIT_REPLUG_B" and reconnect_expected_for == "B_after":
                stage = "COLLECT_B_AFTER"
                reconnect_expected_for = None
                log_event("REPLUG_DETECTED", hand=target_hand, target_stage="B_after")
                start_collect("B_after")
            elif stage == "WAIT_REPLUG_A" and reconnect_expected_for == "A_reboot":
                stage = "COLLECT_A_REBOOT"
                reconnect_expected_for = None
                log_event("REPLUG_DETECTED", hand=target_hand, target_stage="A_reboot")
                start_collect("A_reboot")

        time.sleep(0.15)


def choose_target():
    global target_hand, target_connected
    target_hand = target_var.get()
    last = last_packet_by_hand.get(target_hand, 0.0)
    target_connected = bool(last and time.monotonic() - last < DISCONNECT_S)
    target_combo.configure(state="disabled")
    begin_btn.configure(state="disabled")
    stage_label_var.set("等待目標手套連線")
    set_instruction(
        "步驟 1｜建立 A0 基準",
        f"目前測試：{'右手 R' if target_hand == 'R' else '左手 L'}。\n"
        "把手套放到一個『之後能精確回到同一姿勢』的位置，最好靠桌面、桌角或簡單治具。\n"
        "USB 可以先插著。等畫面顯示已連線後，保持完全不動，再按『開始收 A0』。",
        "開始收 A0",
        try_start_A0,
    )
    log_event("TEST_BEGIN", target_hand=target_hand)


def try_start_A0():
    global stage
    if not target_connected:
        messagebox.showwarning("尚未連線", f"目前還沒讀到 {target_hand} 手套。請插上 USB，等連線狀態變成綠色。")
        return
    stage = "COLLECT_A0"
    start_collect("A0")


def close_app():
    stop_evt.set()
    with lock:
        conns = list(ports_open.values())
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
    try:
        raw_f.flush(); raw_f.close()
        event_f.flush(); event_f.close()
    except Exception:
        pass
    root.destroy()


# ---------------- GUI ----------------
root = tk.Tk()
root.title("HoloGrip｜上電姿態重現性測試")
root.geometry("1120x820")
root.minsize(980, 720)
root.attributes("-topmost", True)

style = ttk.Style()
style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 23, "bold"))
style.configure("Step.TLabel", font=("Microsoft JhengHei UI", 17, "bold"))
style.configure("Body.TLabel", font=("Microsoft JhengHei UI", 14))
style.configure("Big.TButton", font=("Microsoft JhengHei UI", 14, "bold"), padding=12)

ttk.Label(root, text="HoloGrip｜IMU 斷電 / 上電姿態重現性", style="Title.TLabel", anchor="center").pack(fill="x", pady=(18, 8))

top = ttk.Frame(root)
top.pack(fill="x", padx=30, pady=8)
ttk.Label(top, text="測試哪一隻：", style="Body.TLabel").pack(side="left")
target_var = tk.StringVar(value="R")
target_combo = ttk.Combobox(top, textvariable=target_var, values=["R", "L"], state="readonly", width=8)
target_combo.pack(side="left", padx=8)
begin_btn = ttk.Button(top, text="開始測試流程", style="Big.TButton", command=choose_target)
begin_btn.pack(side="left", padx=8)

conn_var = tk.StringVar(value="USB：等待 XIAO")
conn_label = tk.Label(root, textvariable=conn_var, font=("Microsoft JhengHei UI", 16, "bold"),
                      bg="#3b2f16", fg="#ffd77d", pady=9)
conn_label.pack(fill="x", padx=30, pady=6)

stage_label_var = tk.StringVar(value="尚未開始")
ttk.Label(root, textvariable=stage_label_var, style="Step.TLabel", anchor="center").pack(fill="x", padx=30, pady=(12, 5))

instruction_var = tk.StringVar(value=(
    "這不是模型測試。\n"
    "它專門檢查：同一個真實物理姿勢，在不同斷電 / 上電姿勢後，IMU 的方向座標是否仍可重現。\n\n"
    "建議先測右手 R，因為右手目前還有 I2C 間歇異常。"
))
ttk.Label(root, textvariable=instruction_var, style="Body.TLabel", justify="left",
          wraplength=1020, anchor="center").pack(fill="x", padx=45, pady=18)

action_frame = ttk.Frame(root)
action_frame.pack(fill="x", padx=30, pady=8)
action_btn = ttk.Button(action_frame, text="—", style="Big.TButton", state="disabled")
action_btn.pack(anchor="center")

progress_var = tk.StringVar(value="收樣：—")
ttk.Label(root, textvariable=progress_var, style="Step.TLabel", anchor="center").pack(fill="x", pady=12)
progress = ttk.Progressbar(root, maximum=COLLECT_N, value=0, length=850)
progress.pack(pady=3)

live_var = tk.StringVar(value="目前尚未收到目標手套資料")
ttk.Label(root, textvariable=live_var, font=("Consolas", 11), anchor="center").pack(fill="x", padx=30, pady=8)

result_var = tk.StringVar(value="")
ttk.Label(root, textvariable=result_var, style="Body.TLabel", justify="left",
          wraplength=1020).pack(fill="x", padx=40, pady=8)

ttk.Button(root, text="關閉", command=close_app).pack(pady=12)


def configure_action(text, cmd):
    if text and cmd:
        action_btn.configure(text=text, command=cmd, state="normal")
    else:
        action_btn.configure(text="等待自動偵測…", state="disabled")


def ui_loop():
    while True:
        try:
            e = uiq.get_nowait()
        except queue.Empty:
            break
        kind = e[0]
        if kind == "instruction":
            _, title, body, bt, cmd = e
            stage_label_var.set(title)
            instruction_var.set(body)
            configure_action(bt, cmd)
        elif kind == "connection":
            _, ok, port, h = e
            if ok:
                conn_var.set(f"USB：已連線 {h}｜{port}")
                conn_label.configure(bg="#173d2a", fg="#8ff0b7")
            else:
                conn_var.set(f"USB：{h} 已斷線，等待重新插入")
                conn_label.configure(bg="#4a2020", fg="#ff9b9b")
        elif kind == "collect":
            _, name, n, total, state = e
            progress.configure(value=n)
            if state == "settling":
                progress_var.set(f"{STAGE_LABELS[name]}｜先等 {SETTLE_S:.0f} 秒穩定…")
                configure_action(None, None)
            elif state == "collecting":
                progress_var.set(f"{STAGE_LABELS[name]}｜{n} / {total}")
                configure_action(None, None)
            else:
                progress_var.set(f"{STAGE_LABELS[name]}｜完成 {n} / {total}")
        elif kind == "collect_reset":
            _, name, reason = e
            progress.configure(value=0)
            progress_var.set(f"{STAGE_LABELS[name]}｜資料異常 {reason}，重新等 {SETTLE_S:.0f} 秒後從 0 / {COLLECT_N} 收")
        elif kind == "stage_complete":
            complete_collect(e[1])
        elif kind == "done":
            r = e[1]
            c = r["comparisons"]
            stage_label_var.set("測試完成")
            instruction_var.set(r["verdict_note"])
            progress_var.set("全部 5 個階段完成")
            result_var.set(
                f"VERDICT = {r['verdict']}\n\n"
                f"同一 A 跨重新上電：{c['same_A_raw_rotation_diff_deg_A0_vs_A_reboot']:.2f}°\n"
                f"同一 B 跨重新上電：{c['same_B_raw_rotation_diff_deg_B_before_vs_B_after']:.2f}°\n"
                f"B 開機後回 A vs 原 A：{c['same_A_raw_after_boot_at_B_diff_deg']:.2f}°\n"
                f"A→B 相對旋轉跨不同開機姿勢 mismatch：{c['relative_transform_mismatch_deg']:.2f}°\n\n"
                f"物理 A 重力方向差：{c['same_A_gravity_diff_deg_A0_vs_A_reboot']:.2f}°\n"
                f"物理 B 重力方向差：{c['same_B_gravity_diff_deg_B_before_vs_B_after']:.2f}°\n\n"
                f"完整資料：{SESSION}"
            )
            configure_action(None, None)

    # live packet preview from latest raw-able target state not stored separately; display connection age.
    now = time.monotonic()
    last = last_packet_by_hand.get(target_hand, 0.0)
    age = now - last if last else float("inf")
    with lock:
        cp = collect_name
        if cp and now < settle_until:
            rem = max(0.0, settle_until - now)
            progress_var.set(f"{STAGE_LABELS[cp]}｜穩定等待 {rem:.1f} 秒，請完全不要動")

    if last:
        live_var.set(f"目標 {target_hand} 最後封包：{age*1000:.0f} ms 前｜Session：{SESSION.name}")

    root.after(100, ui_loop)


meta = {
    "purpose": "power-cycle orientation reproducibility and R0 invariance test",
    "created_at": wall_iso(),
    "protocol": [
        "A0: reference A, collect while powered",
        "B_before: move A->B without power cycle, collect",
        "B_after: keep B, unplug/replug, collect after 3s settle",
        "A_after_Bboot: move B->A without power cycle, collect",
        "A_reboot: keep A, unplug/replug, collect after 3s settle",
    ],
    "samples_per_stage": COLLECT_N,
    "settle_s": SETTLE_S,
    "analysis": [
        "same physical A/B raw rotation repeatability across power cycles",
        "gravity-direction repeatability as a physical-pose sanity check",
        "relative A->B transform mismatch across different boot poses",
    ],
}
(SESSION / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

threading.Thread(target=scanner, daemon=True).start()
threading.Thread(target=state_monitor, daemon=True).start()

root.protocol("WM_DELETE_WINDOW", close_app)
root.after(100, ui_loop)
root.mainloop()
