"""HoloGrip song data collection client.

This is a small, dependency-free collection UI for the music department.
It listens to the same 100 Hz UDP stream as ``server.py`` and exposes two
recording modes:

* raw_100hz: every received sensor packet is written, before hit filtering;
* hit_events: only events accepted by the current peak/debounce detector are
  written.

The CSV writer runs on a dedicated thread so disk I/O cannot block UDP
reception. The first sample after pressing Start is song time 0 ms.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import re
import socket
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional


UDP_PORT = 8888
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_OUTPUT_DIRS = {
    "udp": PACKAGE_ROOT / "Data" / "第三次Raw",
    "serial": PACKAGE_ROOT / "Data" / "第三次Raw",
}
HAND_NAMES = {"R": "右手", "L": "左手"}

RAW_FIELDS = [
    "session_id",
    "sample_index",
    "hand",
    "song_time_ms",
    "received_time_ms",
    "sensor_time_ms",
    "packet_id",
    "ax_g",
    "ay_g",
    "az_g",
    "accel_magnitude_g",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "cal_yaw_deg",
    "cal_pitch_deg",
]

HIT_FIELDS = [
    "session_id",
    "hit_index",
    "hand",
    "song_time_ms",
    "received_time_ms",
    "sensor_time_ms",
    "packet_id",
    "feature_yaw_deg",
    "feature_pitch_deg",
    "swing_depth_deg",
    "peak_accel_g",
    "v_score",
    "v_score_long",
    "pitch_diff_deg",
    "vertical_ratio",
    "debounce_ms",
]


@dataclass(frozen=True)
class SensorPacket:
    hand: str
    ax: float
    ay: float
    az: float
    yaw: float
    pitch: float
    roll: float
    packet_id: int
    sensor_time_ms: int
    received_time_ms: int
    received_monotonic: float


def parse_sensor_line(
    line: str,
    received_time_ms: Optional[int] = None,
    received_monotonic: Optional[float] = None,
) -> Optional[SensorPacket]:
    """Parse the firmware's D,HAND,ax,ay,az,yaw,pitch,roll,id,millis line."""
    fields = line.strip().split(",")
    if len(fields) < 8 or fields[0] != "D" or fields[1] not in HAND_NAMES:
        return None
    try:
        values = [float(fields[i]) for i in range(2, 8)]
        packet_id = int(fields[8]) if len(fields) >= 9 else -1
        sensor_time_ms = int(fields[9]) if len(fields) >= 10 else int(time.time() * 1000)
    except (ValueError, IndexError):
        return None

    return SensorPacket(
        hand=fields[1],
        ax=values[0],
        ay=values[1],
        az=values[2],
        yaw=values[3],
        pitch=values[4],
        roll=values[5],
        packet_id=packet_id,
        sensor_time_ms=sensor_time_ms,
        received_time_ms=received_time_ms if received_time_ms is not None else int(time.time() * 1000),
        received_monotonic=received_monotonic if received_monotonic is not None else time.monotonic(),
    )


class HitDetector:
    """The existing local-peak, motion-filter and dynamic debounce detector."""

    def __init__(
        self,
        mag_min: float = 1.7,
        debounce_heavy_s: float = 0.15,
        debounce_light_s: float = 0.25,
        reject_backoff_s: float = 0.10,
        motion_bypass_mag: float = 99.0,
    ) -> None:
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.buffer: deque[tuple[float, ...]] = deque(maxlen=80)
        self.last_hit_monotonic = 0.0
        self.mag_min = mag_min
        self.debounce_heavy_s = debounce_heavy_s
        self.debounce_light_s = debounce_light_s
        self.reject_backoff_s = reject_backoff_s
        self.motion_bypass_mag = motion_bypass_mag
        self.debounce_seconds = debounce_heavy_s
        self.gx, self.gy, self.gz = 0.0, 0.0, 1.0

    def calibrate(self, packet: SensorPacket) -> None:
        self.yaw_offset = (self.yaw_offset + packet.yaw) % 360.0
        self.pitch_offset += packet.pitch

    def reset_collection_context(self) -> None:
        """Start a fresh song timeline without changing the calibration offsets."""
        self.buffer.clear()
        self.last_hit_monotonic = 0.0
        self.debounce_seconds = self.debounce_heavy_s

    def calibrated(self, packet: SensorPacket) -> tuple[float, float]:
        cal_yaw = (packet.yaw - self.yaw_offset + 180.0) % 360.0 - 180.0
        cal_pitch = packet.pitch - self.pitch_offset
        return cal_yaw, cal_pitch

    @staticmethod
    def _magnitude(sample: tuple[float, ...]) -> float:
        return math.sqrt(sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2)

    def add_packet(
        self,
        packet: SensorPacket,
        cal_yaw: float,
        cal_pitch: float,
    ) -> Optional[dict[str, Any]]:
        """Return an event only when the packet passes the existing filters."""
        self.buffer.append(
            (
                packet.ax,
                packet.ay,
                packet.az,
                cal_yaw,
                cal_pitch,
                packet.pitch,
                packet.roll,
                packet.sensor_time_ms,
                packet.packet_id,
                packet.received_monotonic,
                packet.received_time_ms,
            )
        )

        magnitude = self._magnitude(self.buffer[-1])
        if 0.8 < magnitude < 1.2:
            alpha = 0.05
            self.gx = self.gx * (1.0 - alpha) + packet.ax * alpha
            self.gy = self.gy * (1.0 - alpha) + packet.ay * alpha
            self.gz = self.gz * (1.0 - alpha) + packet.az * alpha

        if len(self.buffer) < 45:
            return None

        target = self.buffer[-4]
        peak_magnitude = self._magnitude(target)
        now = packet.received_monotonic
        if peak_magnitude <= self.mag_min or now - self.last_hit_monotonic <= self.debounce_seconds:
            return None

        nearby = [self._magnitude(self.buffer[index]) for index in range(-6, 0)]
        if nearby[2] != max(nearby):
            return None

        buffer_len = len(self.buffer)
        snapshot = [self.buffer[buffer_len - 4 - i] for i in range(40)]
        g_magnitude = math.sqrt(self.gx ** 2 + self.gy ** 2 + self.gz ** 2)
        if g_magnitude > 0.1:
            ux, uy, uz = self.gx / g_magnitude, self.gy / g_magnitude, self.gz / g_magnitude
        else:
            ux, uy, uz = 0.0, 0.0, 1.0

        v_score = 0.0
        v_score_long = 0.0
        sum_dynamic = 0.0
        sum_vertical = 0.0
        for index in range(3, 30):
            sample = snapshot[index]
            s_ax, s_ay, s_az = sample[0], sample[1], sample[2]
            vertical_motion = s_ax * ux + s_ay * uy + s_az * uz - 1.0
            v_score_long += vertical_motion
            if index < 15:
                v_score += vertical_motion
                dyn_x, dyn_y, dyn_z = s_ax - ux, s_ay - uy, s_az - uz
                sum_dynamic += math.sqrt(dyn_x ** 2 + dyn_y ** 2 + dyn_z ** 2)
                sum_vertical += abs(vertical_motion)

        vertical_ratio = sum_vertical / sum_dynamic if sum_dynamic > 0.1 else 1.0
        pitch_diff = snapshot[5][4] - snapshot[15][4]
        is_heavy = v_score < -10.0 or peak_magnitude >= 2.5
        is_raise = v_score > -0.25 or (
            not is_heavy and ((v_score_long > -1.5 and pitch_diff < -0.5) or pitch_diff < -4.0)
        )
        is_horizontal = vertical_ratio < 0.32 or ((v_score >= -8.0) and vertical_ratio < 0.45)
        if (is_raise or is_horizontal) and peak_magnitude < self.motion_bypass_mag:
            # Preserve the old detector's short retry delay after a rejected movement.
            self.last_hit_monotonic = max(0.0, now - self.reject_backoff_s)
            return None

        self.debounce_seconds = self.debounce_heavy_s if is_heavy else self.debounce_light_s
        self.last_hit_monotonic = now

        # Keep the historical feature sample (10 ms before the peak) so the
        # collection remains comparable to the existing KNN training data.
        feature_sample = snapshot[1]
        pitch_values = [sample[4] for sample in snapshot]
        return {
            "event_monotonic": feature_sample[9],
            "received_time_ms": int(feature_sample[10]),
            "sensor_time_ms": int(feature_sample[7]),
            "packet_id": int(feature_sample[8]),
            "feature_yaw_deg": round(feature_sample[3], 4),
            "feature_pitch_deg": round(feature_sample[4], 4),
            "swing_depth_deg": round(max(pitch_values) - min(pitch_values), 4),
            "peak_accel_g": round(peak_magnitude, 4),
            "v_score": round(v_score, 4),
            "v_score_long": round(v_score_long, 4),
            "pitch_diff_deg": round(pitch_diff, 4),
            "vertical_ratio": round(vertical_ratio, 4),
            "debounce_ms": int(round(self.debounce_seconds * 1000.0)),
        }


class AsyncCsvRecorder:
    """One CSV writer thread per recording session."""

    _STOP = object()

    def __init__(self, output_dir: Path, log: Callable[[str], None]) -> None:
        self.output_dir = output_dir
        self.log = log
        self._lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=250_000)
        self._thread: Optional[threading.Thread] = None
        self._accepting = False
        self._mode = ""
        self._session_id = ""
        self._start_monotonic = 0.0
        self._path: Optional[Path] = None
        self.enqueued = 0
        self.written = 0
        self.dropped = 0

    @property
    def active(self) -> bool:
        with self._lock:
            return self._accepting

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def elapsed_ms(self) -> int:
        if not self._start_monotonic:
            return 0
        return max(0, int((time.monotonic() - self._start_monotonic) * 1000.0))

    def start(self, mode: str, session_id: str) -> Path:
        with self._lock:
            if self._accepting:
                raise RuntimeError("recorder is already active")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._mode = mode
            self._session_id = session_id
            self._start_monotonic = time.monotonic()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "raw_100hz" if mode == "raw_100hz" else "hit_events"
            self._path = self.output_dir / f"{session_id}_{suffix}_{stamp}.csv"
            self._queue = queue.Queue(maxsize=250_000)
            self.enqueued = self.written = self.dropped = 0
            self._accepting = True
            self._thread = threading.Thread(target=self._writer, daemon=True, name="csv-writer")
            self._thread.start()
            return self._path

    def enqueue(self, row: dict[str, Any], event_monotonic: float) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            row = dict(row)
            row["session_id"] = self._session_id
            row["song_time_ms"] = max(0, int(round((event_monotonic - self._start_monotonic) * 1000.0)))
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self.dropped += 1
                return False
            self.enqueued += 1
            return True

    def stop(self) -> tuple[Optional[Path], int, int, int]:
        with self._lock:
            if not self._accepting:
                return self._path, self.enqueued, self.written, self.dropped
            self._accepting = False
            thread = self._thread
            self._queue.put(self._STOP)
        if thread is not None:
            thread.join(timeout=20.0)
        return self._path, self.enqueued, self.written, self.dropped

    def _writer(self) -> None:
        fields = RAW_FIELDS if self._mode == "raw_100hz" else HIT_FIELDS
        assert self._path is not None
        try:
            with self._path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                last_flush = time.monotonic()
                while True:
                    item = self._queue.get()
                    if item is self._STOP:
                        stream.flush()
                        break
                    writer.writerow({field: item.get(field, "") for field in fields})
                    self.written += 1
                    if self.written % 250 == 0 or time.monotonic() - last_flush >= 1.0:
                        stream.flush()
                        last_flush = time.monotonic()
        except OSError as exc:
            self.log(f"CSV writer stopped: {exc}")


class SongCollectionApp(tk.Tk):
    def __init__(self, transport: str = "udp") -> None:
        super().__init__()
        if transport not in {"udp", "serial"}:
            raise ValueError(f"unsupported transport: {transport}")
        self.transport = transport
        self.output_dir = COLLECTION_OUTPUT_DIRS[transport]
        self.udp_ready = False
        transport_name = "UDP 無線版" if transport == "udp" else "COM 有線版"
        self.title(f"HoloGrip 歌曲資料收集 - {transport_name}")
        self.geometry("1120x900" if transport == "serial" else "1120x760")
        self.minsize(960, 820 if transport == "serial" else 680)
        self.configure(bg="#10131c")

        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.detectors = {hand: HitDetector() for hand in HAND_NAMES}
        self.hand_locks = {hand: threading.Lock() for hand in HAND_NAMES}
        self.packet_counts = {hand: 0 for hand in HAND_NAMES}
        self.hit_counts = {hand: 0 for hand in HAND_NAMES}
        self.latest = {hand: "等待資料" for hand in HAND_NAMES}
        self.active_hands: set[str] = set()
        self.last_packet_time = {hand: 0.0 for hand in HAND_NAMES}
        self.record_sample_index = 0
        self.record_hit_index = 0

        self.mode_var = tk.StringVar(value="hit_events")
        self.session_var = tk.StringVar(value=f"S{datetime.now().strftime('%Y%m%d')}_P01_song01")
        self.status_var = tk.StringVar(value="等待手套 UDP 封包")
        self.mode_desc_var = tk.StringVar()
        self.file_var = tk.StringVar(value="尚未開始收集")
        self.elapsed_var = tk.StringVar(value="00:00.000")
        self.count_var = tk.StringVar(value="寫入 0 筆 · 丟棄 0 筆")
        self.hand_status_vars = {hand: tk.StringVar(value="等待連線") for hand in HAND_NAMES}
        self.hand_count_vars = {hand: tk.StringVar(value="封包 0 · 打擊 0") for hand in HAND_NAMES}
        self.hand_latest_vars = {hand: tk.StringVar(value="等待資料") for hand in HAND_NAMES}
        self.serial_connections: list[Any] = []

        self._build_ui()
        self._update_mode_description()
        self._start_transport()
        self._refresh_after_id = self.after(100, self._refresh_ui)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        title = tk.Label(
            self,
            text=f"HoloGrip 歌曲資料收集 · {'UDP 無線版' if self.transport == 'udp' else 'COM 有線版'}",
            bg="#10131c",
            fg="#f2f5f7",
            font=("Microsoft JhengHei UI", 22, "bold"),
            anchor="w",
        )
        title.pack(fill="x", padx=22, pady=(18, 2))
        tk.Label(
            self,
            text="按下開始後，立即播放與 MIDI 對應的歌曲；結束後按停止。兩種版本輸出完全相同的 CSV。",
            bg="#10131c",
            fg="#aeb8c2",
            font=("Microsoft JhengHei UI", 11),
            anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 14))

        body = tk.Frame(self, bg="#10131c")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        controls = tk.LabelFrame(
            body,
            text="收集設定",
            bg="#181d28",
            fg="#dce5ec",
            font=("Microsoft JhengHei UI", 12, "bold"),
            padx=14,
            pady=8,
        )
        controls.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 14))

        tk.Label(controls, text="場次／歌曲 ID", bg="#181d28", fg="#aeb8c2", font=("Microsoft JhengHei UI", 10)).pack(anchor="w")
        self.session_entry = tk.Entry(controls, textvariable=self.session_var, width=30, bg="#0f1219", fg="#f2f5f7", insertbackground="#f2f5f7", relief="flat")
        self.session_entry.pack(fill="x", pady=(4, 9))

        if self.transport == "serial":
            self._build_serial_controls(controls)

        tk.Label(controls, text="資料模式", bg="#181d28", fg="#aeb8c2", font=("Microsoft JhengHei UI", 10)).pack(anchor="w")
        self.mode_buttons = []
        for value, label in (("raw_100hz", "100 Hz 原始串流"), ("hit_events", "彈跳後打擊事件")):
            button = tk.Radiobutton(
                controls,
                text=label,
                value=value,
                variable=self.mode_var,
                command=self._update_mode_description,
                bg="#181d28",
                fg="#f2f5f7",
                activebackground="#181d28",
                activeforeground="#65d6c0",
                selectcolor="#273044",
                font=("Microsoft JhengHei UI", 11),
                anchor="w",
            )
            button.pack(fill="x", pady=3)
            self.mode_buttons.append(button)
        tk.Label(controls, textvariable=self.mode_desc_var, justify="left", wraplength=280, bg="#181d28", fg="#aeb8c2", font=("Microsoft JhengHei UI", 9)).pack(fill="x", pady=(4, 8))

        self.calibrate_button = tk.Button(controls, text="雙手歸零", command=self._calibrate_both, bg="#273044", fg="#f2f5f7", activebackground="#35415a", activeforeground="#fff", relief="flat", font=("Microsoft JhengHei UI", 11, "bold"), height=1)
        self.calibrate_button.pack(fill="x", pady=4)
        self.start_button = tk.Button(controls, text="開始歌曲收集", command=self._start_recording, bg="#2d8a78", fg="#fff", activebackground="#3aa991", activeforeground="#fff", relief="flat", font=("Microsoft JhengHei UI", 11, "bold"), height=1)
        self.start_button.pack(fill="x", pady=4)
        self.stop_button = tk.Button(controls, text="停止並關閉 CSV", command=self._stop_recording, bg="#a4514d", fg="#fff", activebackground="#c2635e", activeforeground="#fff", relief="flat", font=("Microsoft JhengHei UI", 11, "bold"), height=1, state="disabled")
        self.stop_button.pack(fill="x", pady=4)

        tk.Label(controls, textvariable=self.file_var, justify="left", wraplength=280, bg="#181d28", fg="#65d6c0", font=("Consolas", 9), anchor="w").pack(fill="x", pady=(8, 2))
        tk.Label(controls, textvariable=self.elapsed_var, bg="#181d28", fg="#f2f5f7", font=("Consolas", 17, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Label(controls, textvariable=self.count_var, bg="#181d28", fg="#aeb8c2", font=("Consolas", 10)).pack(anchor="w", pady=(2, 0))

        status = tk.Label(body, textvariable=self.status_var, bg="#1d2635", fg="#65d6c0", font=("Microsoft JhengHei UI", 11, "bold"), anchor="w", padx=12, pady=8)
        status.grid(row=0, column=1, sticky="ew", pady=(0, 12))

        monitor = tk.Frame(body, bg="#10131c")
        monitor.grid(row=1, column=1, sticky="nsew")
        monitor.columnconfigure(0, weight=1)
        monitor.columnconfigure(1, weight=1)
        for index, hand in enumerate(("R", "L")):
            panel = tk.LabelFrame(monitor, text=HAND_NAMES[hand], bg="#181d28", fg="#89b4fa" if hand == "R" else "#a6e3a1", font=("Microsoft JhengHei UI", 13, "bold"), padx=12, pady=12)
            panel.grid(row=0, column=index, sticky="nsew", padx=(0, 7) if index == 0 else (7, 0))
            tk.Label(panel, textvariable=self.hand_status_vars[hand], bg="#181d28", fg="#aeb8c2", font=("Microsoft JhengHei UI", 11, "bold"), anchor="w").pack(fill="x")
            tk.Label(panel, textvariable=self.hand_count_vars[hand], bg="#181d28", fg="#f2f5f7", font=("Consolas", 11), anchor="w").pack(fill="x", pady=(8, 4))
            tk.Label(panel, textvariable=self.hand_latest_vars[hand], bg="#0f1219", fg="#dce5ec", font=("Consolas", 10), justify="left", anchor="nw", padx=8, pady=8, height=5).pack(fill="both", expand=True)

        log_frame = tk.LabelFrame(body, text="操作紀錄", bg="#181d28", fg="#dce5ec", font=("Microsoft JhengHei UI", 11, "bold"), padx=8, pady=8)
        log_frame.grid(row=2, column=1, sticky="nsew", pady=(12, 0))
        body.rowconfigure(2, weight=1)
        self.log_text = tk.Text(log_frame, height=8, bg="#0f1219", fg="#dce5ec", insertbackground="#f2f5f7", font=("Consolas", 10), relief="flat", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _build_serial_controls(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="有線手套 COM 埠（封包會識別左右手）", bg="#181d28", fg="#aeb8c2", font=("Microsoft JhengHei UI", 9)).pack(anchor="w")
        self.com_a_var = tk.StringVar()
        self.com_b_var = tk.StringVar()
        port_row = tk.Frame(parent, bg="#181d28")
        port_row.pack(fill="x", pady=(4, 3))
        self.com_a_box = ttk.Combobox(port_row, textvariable=self.com_a_var, state="readonly", width=12)
        self.com_a_box.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.com_b_box = ttk.Combobox(port_row, textvariable=self.com_b_var, state="readonly", width=12)
        self.com_b_box.pack(side="left", fill="x", expand=True, padx=(3, 0))
        serial_buttons = tk.Frame(parent, bg="#181d28")
        serial_buttons.pack(fill="x", pady=(3, 8))
        tk.Button(serial_buttons, text="重新整理", command=self._refresh_serial_ports, bg="#273044", fg="#f2f5f7", relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.serial_connect_button = tk.Button(serial_buttons, text="連接 COM", command=self._connect_serial_ports, bg="#355f8a", fg="#fff", relief="flat")
        self.serial_connect_button.pack(side="left", fill="x", expand=True, padx=(3, 0))

    def _update_mode_description(self) -> None:
        if self.mode_var.get() == "raw_100hz":
            self.mode_desc_var.set("每個 UDP 感測封包都保存，不經過彈跳、峰值或動作過濾。適合做事後 AI 標註與重新設計偵測規則。")
        else:
            self.mode_desc_var.set("只保存目前判定為有效的打擊事件。沿用局部峰值、抬手／水平動作過濾與動態 150–250 ms 防彈跳。")

    def _log(self, message: str) -> None:
        self.ui_queue.put(("log", message))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _start_transport(self) -> None:
        if self.transport == "udp":
            self._start_udp()
        else:
            self._refresh_serial_ports()
            self.status_var.set("請選擇左右手 COM 埠，再按「連接 COM」")
            self._append_log("COM 有線版就緒；需要安裝 pyserial。")

    def _refresh_serial_ports(self) -> None:
        if self.transport != "serial":
            return
        try:
            from serial.tools import list_ports
        except ImportError:
            self.status_var.set("COM 有線版需要 pyserial")
            self._append_log("找不到 pyserial，請執行：python -m pip install pyserial")
            return
        ports = [port.device for port in list_ports.comports()]
        self.com_a_box["values"] = ports
        self.com_b_box["values"] = ports
        if ports and not self.com_a_var.get():
            self.com_a_var.set(ports[0])
        if len(ports) > 1 and not self.com_b_var.get():
            self.com_b_var.set(ports[1])
        self._append_log(f"找到 COM 埠：{', '.join(ports) if ports else '無'}")

    def _connect_serial_ports(self) -> None:
        try:
            import serial
        except ImportError:
            messagebox.showerror("缺少套件", "COM 版本需要 pyserial。請執行：python -m pip install pyserial")
            return
        selected = [
            port
            for port in (self.com_a_var.get().strip(), self.com_b_var.get().strip())
            if port and not port.startswith(("選擇", "未找到"))
        ]
        if not selected:
            messagebox.showwarning("尚未選擇", "請先選擇至少一個 COM 埠。")
            return
        if len(set(selected)) != len(selected):
            messagebox.showwarning("COM 埠重複", "左右手不能選同一個 COM 埠。")
            return
        self._close_serial_ports()
        try:
            for port in selected:
                connection = serial.Serial(port=port, baudrate=460800, timeout=0.5)
                self.serial_connections.append(connection)
                threading.Thread(target=self._serial_loop, args=(connection,), daemon=True, name=f"serial-{port}").start()
        except (OSError, serial.SerialException) as exc:
            self._close_serial_ports()
            messagebox.showerror("COM 連線失敗", str(exc))
            return
        self.status_var.set("COM 已連接，等待左右手 D 封包")
        self.serial_connect_button.configure(text="重新連接 COM")
        self._append_log(f"COM 已連接：{', '.join(selected)}（460800 baud）")

    def _serial_loop(self, connection: Any) -> None:
        while not self.stop_event.is_set() and getattr(connection, "is_open", False):
            try:
                raw_line = connection.readline()
            except Exception:
                break
            if not raw_line:
                continue
            received_monotonic = time.monotonic()
            received_time_ms = int(time.time() * 1000)
            packet = parse_sensor_line(raw_line.decode("utf-8", errors="ignore"), received_time_ms, received_monotonic)
            if packet is not None:
                self._process_packet(packet)

    def _close_serial_ports(self) -> None:
        for connection in self.serial_connections:
            try:
                connection.close()
            except Exception:
                pass
        self.serial_connections = []

    def _start_udp(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.server_sock.bind(("0.0.0.0", UDP_PORT))
            self.server_sock.settimeout(0.5)
        except OSError as exc:
            self.status_var.set(f"UDP 連接埠 {UDP_PORT} 無法使用：{exc}")
            self._append_log(f"UDP 啟動失敗：{exc}")
            return
        self.udp_ready = True
        threading.Thread(target=self._udp_loop, daemon=True, name="udp-receiver").start()
        self._append_log(f"UDP 接收端已啟動，Port {UDP_PORT}")

    def _udp_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, _addr = self.server_sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break

            received_monotonic = time.monotonic()
            received_time_ms = int(time.time() * 1000)
            for line in data.decode("utf-8", errors="ignore").splitlines():
                packet = parse_sensor_line(line, received_time_ms, received_monotonic)
                if packet is not None:
                    self._process_packet(packet)

    def _process_packet(self, packet: SensorPacket) -> None:
        with self.hand_locks[packet.hand]:
            detector = self.detectors[packet.hand]
            cal_yaw, cal_pitch = detector.calibrated(packet)
            self.packet_counts[packet.hand] += 1
            self.active_hands.add(packet.hand)
            self.last_packet_time[packet.hand] = packet.received_time_ms / 1000.0
            self.latest[packet.hand] = (
                f"yaw={cal_yaw:7.2f}°  pitch={cal_pitch:7.2f}°\n"
                f"acc={math.sqrt(packet.ax ** 2 + packet.ay ** 2 + packet.az ** 2):.2f} g  sensor={packet.sensor_time_ms} ms"
            )

            recorder = getattr(self, "recorder", None)
            if recorder is not None and recorder.active and recorder.mode == "raw_100hz":
                self.record_sample_index += 1
                recorder.enqueue(
                    {
                        "sample_index": self.record_sample_index,
                        "hand": packet.hand,
                        "received_time_ms": packet.received_time_ms,
                        "sensor_time_ms": packet.sensor_time_ms,
                        "packet_id": packet.packet_id,
                        "ax_g": round(packet.ax, 6),
                        "ay_g": round(packet.ay, 6),
                        "az_g": round(packet.az, 6),
                        "accel_magnitude_g": round(math.sqrt(packet.ax ** 2 + packet.ay ** 2 + packet.az ** 2), 6),
                        "yaw_deg": round(packet.yaw, 6),
                        "pitch_deg": round(packet.pitch, 6),
                        "roll_deg": round(packet.roll, 6),
                        "cal_yaw_deg": round(cal_yaw, 6),
                        "cal_pitch_deg": round(cal_pitch, 6),
                    },
                    packet.received_monotonic,
                )

            event = detector.add_packet(packet, cal_yaw, cal_pitch)
            if event is not None:
                self.hit_counts[packet.hand] += 1
                if recorder is not None and recorder.active and recorder.mode == "hit_events":
                    self.record_hit_index += 1
                    event["hit_index"] = self.record_hit_index
                    event["hand"] = packet.hand
                    recorder.enqueue(event, event["event_monotonic"])

    def _calibrate_both(self) -> None:
        for hand, detector in self.detectors.items():
            # Calibration uses the most recent packet represented in the detector buffer.
            if detector.buffer:
                detector.yaw_offset = (detector.yaw_offset + detector.buffer[-1][3]) % 360.0
                detector.pitch_offset += detector.buffer[-1][4]
                self._append_log(f"{HAND_NAMES[hand]} 已歸零")
        self._append_log("雙手歸零完成")

    @staticmethod
    def _safe_session_id(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", value.strip())
        return value[:80].strip("_") or "song_session"

    def _set_recording_widgets(self, recording: bool) -> None:
        self.start_button.configure(state="disabled" if recording else "normal")
        self.stop_button.configure(state="normal" if recording else "disabled")
        self.calibrate_button.configure(state="disabled" if recording else "normal")
        self.session_entry.configure(state="disabled" if recording else "normal")
        for button in self.mode_buttons:
            button.configure(state="disabled" if recording else "normal")

    def _start_recording(self) -> None:
        if self.transport == "udp" and not self.udp_ready:
            messagebox.showwarning("尚未啟動", "UDP 接收端尚未啟動。")
            return
        if self.transport == "serial" and not self.serial_connections:
            messagebox.showwarning("尚未連接", "請先選擇 COM 埠並按「連接 COM」。")
            return
        session_id = self._safe_session_id(self.session_var.get())
        self.session_var.set(session_id)
        self.record_sample_index = 0
        self.record_hit_index = 0
        for detector in self.detectors.values():
            detector.reset_collection_context()
        self.recorder = AsyncCsvRecorder(self.output_dir, self._log)
        path = self.recorder.start(self.mode_var.get(), session_id)
        self._set_recording_widgets(True)
        self.status_var.set("收集中：按下後請立即播放 MIDI 對應歌曲")
        self.file_var.set(str(path))
        mode_name = "100 Hz 原始串流" if self.mode_var.get() == "raw_100hz" else "彈跳後打擊事件"
        self._append_log(f"開始收集：{mode_name}")
        self._append_log("時間軸已從 0 ms 開始；現在播放歌曲。")

    def _stop_recording(self) -> None:
        recorder = getattr(self, "recorder", None)
        if recorder is None or not recorder.active:
            return
        path, enqueued, written, dropped = recorder.stop()
        self._set_recording_widgets(False)
        self.status_var.set("收集完成，CSV 已關閉")
        self.count_var.set(f"寫入 {written} 筆 · 丟棄 {dropped} 筆")
        self._append_log(f"收集完成：{path}")
        self._append_log(f"佇列 {enqueued} 筆，實際寫入 {written} 筆")
        if dropped:
            self._append_log(f"注意：佇列滿，丟棄 {dropped} 筆資料。")

    def _refresh_ui(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
        except queue.Empty:
            pass

        now = time.time()
        for hand in HAND_NAMES:
            connected = now - self.last_packet_time[hand] < 3.0
            self.hand_status_vars[hand].set("已連線" if connected else "等待連線")
            self.hand_count_vars[hand].set(f"封包 {self.packet_counts[hand]} · 偵測打擊 {self.hit_counts[hand]}")
            self.hand_latest_vars[hand].set(self.latest[hand])

        recorder = getattr(self, "recorder", None)
        if recorder is not None and recorder.active:
            elapsed = recorder.elapsed_ms
            self.elapsed_var.set(f"{elapsed // 60000:02d}:{(elapsed // 1000) % 60:02d}.{elapsed % 1000:03d}")
            self.count_var.set(f"佇列 {recorder.enqueued} · 寫入 {recorder.written} · 丟棄 {recorder.dropped}")
        self._refresh_after_id = self.after(100, self._refresh_ui)

    def _on_close(self) -> None:
        recorder = getattr(self, "recorder", None)
        if recorder is not None and recorder.active:
            self._stop_recording()
        self.stop_event.set()
        try:
            self.after_cancel(self._refresh_after_id)
        except (AttributeError, tk.TclError):
            pass
        try:
            self.server_sock.close()
        except (AttributeError, OSError):
            pass
        self._close_serial_ports()
        self.destroy()


def _self_test() -> None:
    packet = parse_sensor_line("D,R,0.1,0.2,0.95,12.0,-3.0,1.0,42,12345", 1700000000000, 10.0)
    assert packet is not None and packet.hand == "R" and packet.packet_id == 42
    assert packet.sensor_time_ms == 12345
    assert parse_sensor_line("not a packet") is None

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        messages: list[str] = []
        recorder = AsyncCsvRecorder(Path(directory), messages.append)
        path = recorder.start("raw_100hz", "test_session")
        assert recorder.enqueue({"sample_index": 1, "hand": "R", "received_time_ms": 1}, 10.0)
        recorder.stop()
        text = path.read_text(encoding="utf-8-sig")
        assert "song_time_ms" in text and "test_session" in text and "R" in text
    print("song_collection_server self-test: OK")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        transport = "serial" if "--serial" in sys.argv else "udp"
        app = SongCollectionApp(transport=transport)
        app.mainloop()
