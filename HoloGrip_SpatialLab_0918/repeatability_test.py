from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import serial
from serial.tools import list_ports

LAB = Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")

root = tk.Tk()
root.title("HoloGrip 重複性測試")
root.geometry("780x440")
root.minsize(780, 440)
root.attributes("-topmost", True)

style = ttk.Style()
style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 22, "bold"))
style.configure("Mid.TLabel", font=("Microsoft JhengHei UI", 14))
style.configure("Big.TButton", font=("Microsoft JhengHei UI", 15, "bold"), padding=10)

title = tk.StringVar(value="HoloGrip IMU 重複性測試")
status = tk.StringVar(value="視窗已就緒。按「開始測試」後才會連接兩隻手套。")
detail = tk.StringVar(value="目的：確認同一個物理姿勢，揮動前後 Yaw / Pitch / Roll 是否真的回得來。")
ports_text = tk.StringVar(value="COM：尚未掃描")

ttk.Label(root, textvariable=title, style="Title.TLabel", anchor="center").pack(fill="x", padx=20, pady=(28, 10))
ttk.Label(root, textvariable=status, style="Mid.TLabel", anchor="center", justify="center", wraplength=730).pack(fill="x", padx=20, pady=8)
ttk.Label(root, textvariable=detail, style="Mid.TLabel", anchor="center", justify="center", wraplength=730).pack(fill="x", padx=20, pady=8)
ttk.Label(root, textvariable=ports_text, font=("Consolas", 12), anchor="center").pack(fill="x", padx=20, pady=8)

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=18)
start_btn = ttk.Button(btn_frame, text="開始測試", style="Big.TButton")
start_btn.pack(side="left", padx=8)
close_btn = ttk.Button(btn_frame, text="關閉", style="Big.TButton", command=root.destroy)
close_btn.pack(side="left", padx=8)

rows: list[dict] = []
lock = threading.Lock()
stop = threading.Event()
threads: list[threading.Thread] = []
serials: list[serial.Serial] = []
started_mono = 0.0
t0_stage = 0.0
out_dir: Path | None = None
running = False

STAGES = [
    (3, "準備", "把兩手放到桌面上方，準備使用完全相同的固定姿勢。"),
    (6, "基準 A：完全固定", "兩隻手掌／手腕放在同一個桌面固定姿勢，不要動。"),
    (16, "正常揮動", "做正常大幅打鼓動作：左右、上方、Crash、Ride 都可以。"),
    (8, "基準 B：回到完全同一姿勢", "回到剛才相同桌面接觸位置與手掌方向，完全固定。"),
]
TOTAL = sum(x[0] for x in STAGES)


def scan_xiao_ports() -> list[str]:
    found = []
    for p in list_ports.comports():
        hw = (p.hwid or "").upper()
        if "303A:1001" in hw or "VID_303A&PID_1001" in hw:
            found.append(p.device)
    return sorted(found)


def parse_line(line: str):
    f = line.strip().split(",")
    if len(f) < 10 or f[0] != "D" or f[1] not in ("L", "R"):
        return None
    try:
        return {
            "hand": f[1],
            "ax": float(f[2]),
            "ay": float(f[3]),
            "az": float(f[4]),
            "yaw": float(f[5]),
            "pitch": float(f[6]),
            "roll": float(f[7]),
            "packet_id": int(f[8]),
            "sensor_ms": int(f[9]),
        }
    except Exception:
        return None


def reader(conn: serial.Serial, port: str):
    while not stop.is_set():
        try:
            b = conn.readline()
        except Exception as e:
            root.after(0, lambda: status.set(f"{port} 讀取失敗：{e}"))
            break
        if not b:
            continue
        pkt = parse_line(b.decode("utf-8", "ignore"))
        if pkt is None:
            continue
        pkt["port"] = port
        pkt["t"] = time.monotonic() - started_mono
        with lock:
            rows.append(pkt)


def circ_mean_deg(vals):
    a = np.asarray(vals, dtype=float)
    r = np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def circ_diff(a, b):
    return ((b - a + 180.0) % 360.0) - 180.0


def stat(rr):
    if not rr:
        return None
    yu = np.rad2deg(np.unwrap(np.deg2rad([r["yaw"] for r in rr])))
    ru = np.rad2deg(np.unwrap(np.deg2rad([r["roll"] for r in rr])))
    return {
        "n": len(rr),
        "yaw": circ_mean_deg([r["yaw"] for r in rr]),
        "pitch": float(np.median([r["pitch"] for r in rr])),
        "roll": circ_mean_deg([r["roll"] for r in rr]),
        "yaw_std": float(np.std(yu)),
        "pitch_std": float(np.std([r["pitch"] for r in rr])),
        "roll_std": float(np.std(ru)),
    }


def finish_test():
    global running
    stop.set()
    for t in threads:
        t.join(timeout=0.5)
    for s in serials:
        try:
            s.close()
        except Exception:
            pass

    offset = t0_stage - started_mono
    # exclude stage transitions
    A = (offset + 3.5, offset + 8.5)
    B = (offset + 25.5, offset + 32.5)

    result = {
        "ports": [s.port for s in serials],
        "rows": len(rows),
        "baseline_A": {},
        "baseline_B": {},
        "delta_B_minus_A": {},
    }

    for h in ("L", "R"):
        with lock:
            ra = [r for r in rows if r["hand"] == h and A[0] <= r["t"] <= A[1]]
            rb = [r for r in rows if r["hand"] == h and B[0] <= r["t"] <= B[1]]
        sa, sb = stat(ra), stat(rb)
        result["baseline_A"][h] = sa
        result["baseline_B"][h] = sb
        if sa and sb:
            result["delta_B_minus_A"][h] = {
                "yaw": circ_diff(sa["yaw"], sb["yaw"]),
                "pitch": sb["pitch"] - sa["pitch"],
                "roll": circ_diff(sa["roll"], sb["roll"]),
            }
        else:
            result["delta_B_minus_A"][h] = None

    assert out_dir is not None
    with (out_dir / "raw.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["t", "port", "hand", "packet_id", "sensor_ms", "ax", "ay", "az", "yaw", "pitch", "roll"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        with lock:
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["測試完成。A → B 同一姿勢差值："]
    for h in ("L", "R"):
        d = result["delta_B_minus_A"].get(h)
        if d:
            lines.append(
                f"{h}：Yaw {d['yaw']:+.2f}°｜Pitch {d['pitch']:+.2f}°｜Roll {d['roll']:+.2f}°"
            )
        else:
            lines.append(f"{h}：資料不足")
    lines.append("")
    lines.append(f"已保存：{out_dir}")

    title.set("測試完成")
    status.set("\n".join(lines))
    detail.set("視窗會保持開啟，不會自動關閉。把結果留著，我可以直接讀檔分析。")
    start_btn.configure(text="重新測試", state="normal")
    running = False


def tick():
    if not running:
        return
    e = time.monotonic() - t0_stage
    if e >= TOTAL:
        finish_test()
        return

    acc = 0
    for dur, name, desc in STAGES:
        if acc <= e < acc + dur:
            remain = int(math.ceil(acc + dur - e))
            title.set(f"{name}｜{remain} 秒")
            status.set(desc)
            detail.set("請照畫面做。這次視窗不會自動關。")
            break
        acc += dur
    root.after(100, tick)


def start_test():
    global running, rows, stop, threads, serials, started_mono, t0_stage, out_dir

    if running:
        return

    ports = scan_xiao_ports()
    ports_text.set("XIAO：" + (", ".join(ports) if ports else "未找到"))

    if len(ports) < 2:
        title.set("尚未開始")
        status.set(
            f"目前只找到 {len(ports)} 顆 XIAO ESP32-C6。\n"
            "請確認兩隻手套都插著，然後再按一次「開始測試」。"
        )
        detail.set("程式不會退出；你可以重插 USB 後直接再試。")
        return

    # Fresh run
    rows = []
    stop = threading.Event()
    threads = []
    serials = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = LAB / "repeatability_sessions" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for port in ports[:2]:
            s = serial.Serial(port, 460800, timeout=0.10)
            serials.append(s)
        time.sleep(0.8)
        for s in serials:
            try:
                s.reset_input_buffer()
            except Exception:
                pass
    except Exception as e:
        for s in serials:
            try:
                s.close()
            except Exception:
                pass
        title.set("連線失敗")
        status.set(f"COM 開啟失敗：{e}")
        detail.set("請關閉 Arduino Serial Monitor / HoloGrip 其他 GUI，再按「開始測試」。")
        return

    started_mono = time.monotonic()
    for s in serials:
        t = threading.Thread(target=reader, args=(s, s.port), daemon=True)
        t.start()
        threads.append(t)

    t0_stage = time.monotonic()
    running = True
    start_btn.configure(state="disabled")
    title.set("準備開始")
    status.set("兩個 COM 已成功開啟。")
    detail.set("請依照倒數指示操作。")
    root.after(100, tick)


start_btn.configure(command=start_test)

# Keep the window visible even before start.
root.after(150, lambda: root.lift())
root.after(250, lambda: root.focus_force())
root.mainloop()

# Close ports if the user closes mid-test.
stop.set()
for s in serials:
    try:
        s.close()
    except Exception:
        pass
