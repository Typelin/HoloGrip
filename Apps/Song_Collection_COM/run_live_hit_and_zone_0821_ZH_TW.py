"""Field GUI: ESP COM 100 Hz → hit → (HAND_ID, 7-zone drum). No MIDI."""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any, Optional

import customtkinter as ctk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from product_hit_and_zone import FEATURE_NAMES, ZONE_NAMES, LiveRecognizer, load_zone_model
from song_collection_server import parse_sensor_line

HAND_ZH = {"L": "左手", "R": "右手"}
BG, PANEL, PANEL_ALT, TEXT, MUTED = "#0D0F1A", "#1A1B2E", "#252738", "#FFFFFF", "#A6ADC8"
TEAL, BLUE, AMBER = "#A6E3A1", "#89B4FA", "#F9E2AF"
FONT = "Microsoft JhengHei UI"


def _resolve_model_path() -> str:
    rel = os.path.normpath(
        os.path.join(
            HERE,
            "..",
            "..",
            "Data",
            "Derived",
            "Song_Collection_COM",
            "S20260812_P01_song02_T1127_cleaned",
            "hologrip_song2_七鼓點模型_真正驗證版_0826.joblib",
        )
    )
    return rel


def _device_of(label: str) -> str:
    token = (label or "").strip()
    if not token or token.startswith(("選擇", "未找到")):
        return ""
    return token.split()[0]


class LiveHitZoneApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HoloGrip 現場打擊")
        self.geometry("980x720")
        self.minsize(860, 640)
        self.configure(fg_color=BG)

        model_path = _resolve_model_path()
        if not os.path.isfile(model_path):
            messagebox.showerror("沒有模型", f"找不到七區模型：\n{model_path}")
            self.after(100, self.destroy)
            return

        self.clf = load_zone_model(model_path)
        if self.clf.n_features_in_ != len(FEATURE_NAMES) or list(self.clf.classes_) != list(range(len(ZONE_NAMES))):
            raise ValueError("模型輸入或七鼓點類別不相符，停止啟動")
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)
        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue()
        self.serial_connections: list[Any] = []
        self.serial_threads: list[threading.Thread] = []
        self.port_map: dict[str, str] = {}
        self.hit_count = 0
        self.connected = False
        self._hand_conflict_warned = False
        self._hero_flash_job: Optional[str] = None

        self.stream: dict[str, dict[str, Any]] = {
            hand: {
                "port": "",
                "mag": 0.0,
                "n": 0,
                "hz": 0.0,
                "last_mono": 0.0,
                "tick_n": 0,
                "tick_t": 0.0,
            }
            for hand in ("L", "R")
        }
        self.stream_lock = threading.Lock()

        self.status_var = tk.StringVar(value="插上左右手套 → 掃描 COM → 連接 → 面向鼓組歸零 → 打")
        self.big_hand_var = tk.StringVar(value="—")
        self.big_drum_var = tk.StringVar(value="等待打擊")
        self.big_meta_var = tk.StringVar(value="手別來自手套 HAND_ID，不是另外一個左右手模型")
        self.com_a_var = tk.StringVar(value="選擇 COM A")
        self.com_b_var = tk.StringVar(value="選擇 COM B")
        self.left_var = tk.StringVar(value="左手\n未連線")
        self.right_var = tk.StringVar(value="右手\n未連線")
        self.hand_hit_vars = {h: tk.StringVar(value="等待打擊") for h in ("L", "R")}
        self.hand_meta_vars = {h: tk.StringVar(value="尚未歸零") for h in ("L", "R")}
        self.hand_counts = {h: 0 for h in ("L", "R")}

        self._build()
        self._append(f"模型：{model_path}")
        self._refresh_ports()
        self.after(80, self._drain)
        self.after(400, self._refresh_stream_cards)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color=BG)
        header.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkLabel(header, text="HoloGrip", text_color=TEXT, font=(FONT, 28, "bold")).pack(side="left")
        ctk.CTkLabel(header, text="現場打擊", text_color=MUTED, font=(FONT, 22, "bold")).pack(
            side="left", padx=(10, 0), pady=(4, 0)
        )
        ctk.CTkLabel(
            header,
            text="COM 460800 · 無 MIDI",
            fg_color=PANEL_ALT,
            text_color=AMBER,
            corner_radius=6,
            height=34,
            width=170,
            font=(FONT, 13, "bold"),
        ).pack(side="right")

        top = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        top.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(top, text="左右手各一個 COM，連上後會依封包 HAND_ID 自動分成左／右", text_color=MUTED, font=(FONT, 13)).pack(
            anchor="w", padx=14, pady=(10, 0)
        )
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=10)
        self.com_a = ctk.CTkComboBox(row, variable=self.com_a_var, values=["選擇 COM A"], width=190, font=(FONT, 13))
        self.com_a.pack(side="left", padx=(0, 8))
        self.com_b = ctk.CTkComboBox(row, variable=self.com_b_var, values=["選擇 COM B"], width=190, font=(FONT, 13))
        self.com_b.pack(side="left", padx=(0, 8))
        self.scan_btn = ctk.CTkButton(row, text="掃描", width=72, command=self._refresh_ports)
        self.scan_btn.pack(side="left", padx=3)
        self.connect_btn = ctk.CTkButton(row, text="連接", width=72, fg_color=BLUE, text_color="#111", command=self._connect)
        self.connect_btn.pack(side="left", padx=3)
        self.disconnect_btn = ctk.CTkButton(row, text="斷開", width=72, fg_color=PANEL_ALT, command=self._disconnect)
        self.disconnect_btn.pack(side="left", padx=3)
        self.cal_btn = ctk.CTkButton(row, text="歸零", width=72, fg_color=AMBER, text_color="#1a1a1a", command=self._calibrate)
        self.cal_btn.pack(side="left", padx=3)

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=18, pady=(0, 10))
        cards.grid_columnconfigure(0, weight=1)
        cards.grid_columnconfigure(1, weight=1)
        self.left_card = ctk.CTkLabel(
            cards, textvariable=self.left_var, fg_color=PANEL, text_color=TEXT,
            font=(FONT, 16), corner_radius=12, justify="left", anchor="w", height=96,
        )
        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right_card = ctk.CTkLabel(
            cards, textvariable=self.right_var, fg_color=PANEL, text_color=TEXT,
            font=(FONT, 16), corner_radius=12, justify="left", anchor="w", height=96,
        )
        self.right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for column, hand in enumerate(("L", "R")):
            color = "#64A8FF" if hand == "L" else "#FF7B83"
            ctk.CTkLabel(cards, textvariable=self.hand_hit_vars[hand], font=(FONT, 32, "bold"),
                         text_color=color, height=56).grid(row=1, column=column, sticky="ew")
            ctk.CTkLabel(cards, textvariable=self.hand_meta_vars[hand], font=(FONT, 12),
                         text_color=MUTED).grid(row=2, column=column, sticky="ew")

        self.hero = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16)
        self.hero.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(self.hero, textvariable=self.status_var, text_color=MUTED, font=(FONT, 13), wraplength=760).pack(pady=12)

        log_head = ctk.CTkFrame(self, fg_color="transparent")
        log_head.pack(fill="x", padx=18)
        ctk.CTkLabel(log_head, text="打擊紀錄", text_color=MUTED, font=(FONT, 13)).pack(side="left")
        ctk.CTkButton(log_head, text="清列表", width=80, fg_color=PANEL_ALT, command=self._clear_log).pack(side="right")

        self.log = ctk.CTkTextbox(self, fg_color="#10131c", text_color=TEXT, font=("Consolas", 14))
        self.log.pack(fill="both", expand=True, padx=18, pady=(6, 16))
        self._append("流程：兩隻手套 USB 插上 → 掃描 → 選兩個不同 COM → 連接。")
        self._append("看到左／右約 100 Hz 後，面向鼓組站定，按歸零，再開打。")
        self._append("歸零請重現收集訓練資料時的原點姿勢；同一 COM 請勿同時開採集 UI 或序列監控器。")

    def _append(self, msg: str) -> None:
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.see("end")

    def _clear_log(self) -> None:
        self.log.delete("1.0", "end")
        self.hit_count = 0
        self._append("列表已清。連線與歸零不變。")

    def _refresh_ports(self) -> None:
        try:
            from serial.tools import list_ports
        except ImportError:
            self.status_var.set("需要 pyserial：python -m pip install pyserial")
            return
        labels: list[str] = []
        self.port_map = {}
        for item in list_ports.comports():
            extra = (item.description or "").replace(item.device, "").strip(" ()")
            label = f"{item.device}  {extra}" if extra else item.device
            labels.append(label)
            self.port_map[label] = item.device
            self.port_map[item.device] = item.device
        values = labels or ["未找到 COM"]
        self.com_a.configure(values=values)
        self.com_b.configure(values=values)
        if labels:
            self.com_a_var.set(labels[0])
            self.com_b_var.set(labels[1] if len(labels) > 1 else "選擇 COM B")
        self._append(f"COM：{', '.join(labels) if labels else '無'}")

    def _selected_ports(self) -> list[str]:
        picked = []
        for raw in (self.com_a_var.get(), self.com_b_var.get()):
            device = self.port_map.get(raw.strip(), _device_of(raw))
            if device:
                picked.append(device)
        return picked

    def _set_connected_widgets(self, connected: bool) -> None:
        self.connected = connected
        state = "disabled" if connected else "normal"
        self.com_a.configure(state=state)
        self.com_b.configure(state=state)
        self.scan_btn.configure(state=state)
        self.connect_btn.configure(text="重新連接" if connected else "連接")

    def _connect(self) -> None:
        try:
            import serial
        except ImportError:
            messagebox.showerror("缺少套件", "python -m pip install pyserial")
            return
        selected = self._selected_ports()
        if not selected:
            messagebox.showwarning("尚未選擇", "請選至少一個 COM")
            return
        if len(set(selected)) != len(selected):
            messagebox.showwarning("COM 重複", "兩手不能同一個埠")
            return
        self._close_serial()
        self.stop_event = threading.Event()
        self.ui_queue = queue.Queue()
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)
        self._reset_hit_display()
        self._hand_conflict_warned = False
        with self.stream_lock:
            for hand in ("L", "R"):
                self.stream[hand].update(port="", mag=0.0, n=0, hz=0.0, last_mono=0.0, tick_n=0, tick_t=0.0)
        try:
            for port in selected:
                conn = serial.Serial(port=port, baudrate=460800, timeout=0.2)
                self.serial_connections.append(conn)
            time.sleep(1.0)
            for conn, port in zip(self.serial_connections, selected):
                try:
                    conn.reset_input_buffer()
                except Exception:
                    pass
                worker = threading.Thread(
                    target=self._serial_loop,
                    args=(conn, port, self.stop_event, self.recognizer, self.ui_queue),
                    daemon=True,
                    name=f"serial-{port}",
                )
                self.serial_threads.append(worker)
                worker.start()
        except Exception as exc:
            self._close_serial()
            self._set_connected_widgets(False)
            messagebox.showerror("連線失敗", str(exc))
            return
        self._set_connected_widgets(True)
        self.status_var.set("已連接。等左／右都出現約 100 Hz，面向鼓組按歸零。")
        self._append(f"已連接 {', '.join(selected)}（460800）")

    def _disconnect(self) -> None:
        self._close_serial()
        self._set_connected_widgets(False)
        self._reset_hit_display()
        self.status_var.set("已斷開。可重新掃描再連接。")
        self._append("已斷開 COM")

    def _serial_loop(self, conn: Any, port: str, stop_event: threading.Event,
                     recognizer: LiveRecognizer, events: queue.Queue) -> None:
        last_ui = 0.0
        local_n = 0
        last_hand = ""
        last_mag = 0.0
        while not stop_event.is_set() and getattr(conn, "is_open", False):
            try:
                raw = conn.readline()
            except Exception:
                if not stop_event.is_set():
                    events.put(("dead", port))
                break
            if not raw:
                continue
            packet = parse_sensor_line(
                raw.decode("utf-8", errors="ignore"),
                int(time.time() * 1000),
                time.monotonic(),
            )
            if packet is None:
                continue
            local_n += 1
            last_hand = packet.hand
            last_mag = math.sqrt(packet.ax ** 2 + packet.ay ** 2 + packet.az ** 2)
            try:
                hits = recognizer.push(packet)
            except Exception as exc:
                events.put(("err", f"{port} 推論失敗：{exc}"))
                continue
            for hit in hits:
                events.put(("hit", hit))
            now = time.monotonic()
            if now - last_ui >= 0.12:
                last_ui = now
                events.put(("stream", {
                    "port": port,
                    "hand": last_hand,
                    "mag": last_mag,
                    "n": local_n,
                    "mono": now,
                }))

    def _calibrate(self) -> None:
        if not self.connected or self._hand_conflict_warned:
            messagebox.showwarning("無法歸零", "請先連接並確認兩個手套的 HAND_ID 不同。")
            return
        now = time.monotonic()
        with self.recognizer._lock:
            packets = [s.last_packet for s in self.recognizer.hands.values() if s.last_packet is not None]
        if len(packets) != len(self.serial_connections) or any(now - p.received_monotonic > 1.2 for p in packets):
            messagebox.showwarning("資料未就緒", "請等所有已連接手套都有持續更新的封包，再按歸零。")
            return
        done = self.recognizer.calibrate()
        if not done:
            messagebox.showwarning("還沒有資料", "請先連接，等左或右卡片出現 Hz 再歸零")
            return
        names = "、".join(HAND_ZH[h] for h in done)
        self._reset_hit_display()
        for h in done:
            self.hand_meta_vars[h].set("已歸零；等待打擊")
        self._append(f"歸零完成：{names}。可以打。")
        self.status_var.set(f"已歸零（{names}）。開打。")

    def _on_stream(self, payload: dict[str, Any]) -> None:
        hand = payload.get("hand")
        if hand not in self.stream:
            return
        now = float(payload.get("mono") or time.monotonic())
        n = int(payload.get("n") or 0)
        with self.stream_lock:
            existing = self.stream[hand]["port"]
            if existing and existing != payload["port"] and not self._hand_conflict_warned:
                self._hand_conflict_warned = True
                self._append(
                    f"警告：{HAND_ZH[hand]} 同時出現在 {existing} 與 {payload['port']}。"
                    "兩隻手套可能燒成同一個 HAND_ID。"
                )
                self._disconnect()
                self.status_var.set("HAND_ID 衝突：兩手不可使用同一個 L/R 識別碼。")
                return
            st = self.stream[hand]
            st["port"] = payload["port"]
            st["mag"] = float(payload.get("mag") or 0.0)
            st["n"] = n
            st["last_mono"] = now
            dt = now - st["tick_t"]
            if st["tick_t"] <= 0:
                st["tick_t"] = now
                st["tick_n"] = n
            elif dt >= 0.4:
                st["hz"] = max(0.0, (n - st["tick_n"]) / dt)
                st["tick_t"] = now
                st["tick_n"] = n

    def _refresh_stream_cards(self) -> None:
        now = time.monotonic()
        with self.stream_lock:
            snapshot = {hand: dict(state) for hand, state in self.stream.items()}
        for hand, var, card in (("L", self.left_var, self.left_card), ("R", self.right_var, self.right_card)):
            st = snapshot[hand]
            age = now - st["last_mono"] if st["last_mono"] else 999.0
            if st["last_mono"] and age < 1.2:
                line = (
                    f"{HAND_ZH[hand]}  {st['port'] or '—'}\n"
                    f"{st['hz']:.0f} Hz    {st['mag']:.2f} g"
                )
                card.configure(text_color=TEAL)
            elif self.connected:
                line = f"{HAND_ZH[hand]}\n等封包…"
                card.configure(text_color=AMBER)
            else:
                line = f"{HAND_ZH[hand]}\n未連線"
                card.configure(text_color=MUTED)
            var.set(line)
        self.after(250, self._refresh_stream_cards)

    def _flash_hero(self) -> None:
        if self._hero_flash_job is not None:
            try:
                self.after_cancel(self._hero_flash_job)
            except tk.TclError:
                pass
        self.hero.configure(fg_color="#163326")
        self._hero_flash_job = self.after(220, lambda: self.hero.configure(fg_color=PANEL))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "stream":
                    self._on_stream(payload)
                elif kind == "hit":
                    if payload.get("generation") != self.recognizer.generation or not self.connected:
                        continue
                    self.hit_count += 1
                    hand = HAND_ZH.get(payload["hand"], payload["hand"])
                    hand_id = payload["hand"]
                    drum = payload["pred_drum"]
                    pct = payload["pred_proba"] * 100.0
                    peak = payload.get("peak_accel_g", 0.0)
                    self.big_hand_var.set(hand)
                    self.big_drum_var.set(drum)
                    self.big_meta_var.set(f"把握 {pct:.0f}%   {peak:.1f} g   第 {self.hit_count} 擊")
                    self.hand_counts[hand_id] += 1
                    self.hand_hit_vars[hand_id].set(drum)
                    self.hand_meta_vars[hand_id].set(f"模型信心 {pct:.0f}%  |  {peak:.1f} g  |  第 {self.hand_counts[hand_id]} 擊")
                    self.status_var.set("辨識中；重新歸零前請回到收集資料時的原點姿勢。")
                    self._append(f"{hand} → {drum}   {pct:.0f}%   {peak:.1f}g")
                    self._flash_hero()
                elif kind == "err":
                    self._append(str(payload))
                elif kind == "dead":
                    self._append(f"{payload} 讀取中斷")
                    self._disconnect()
                    self.status_var.set(f"{payload} 讀取中斷，請重新連接與歸零。")
        except queue.Empty:
            pass
        self.after(50, self._drain)

    def _reset_hit_display(self) -> None:
        for h in ("L", "R"):
            self.hand_hit_vars[h].set("等待打擊")
            self.hand_meta_vars[h].set("尚未歸零")
            self.hand_counts[h] = 0

    def _close_serial(self) -> None:
        self.stop_event.set()
        for conn in self.serial_connections:
            try:
                conn.close()
            except Exception:
                pass
        self.serial_connections = []
        for worker in self.serial_threads:
            worker.join(timeout=0.5)
        self.serial_threads = []
        while not self.ui_queue.empty():
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                break

    def _close(self) -> None:
        self.stop_event.set()
        self._close_serial()
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    LiveHitZoneApp().mainloop()
