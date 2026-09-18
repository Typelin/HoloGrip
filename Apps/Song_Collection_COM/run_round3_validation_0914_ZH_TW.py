"""HoloGrip Round 3 validation GUI.

The validation UI is separate from the evidence recorder. This process owns
both COM ports and forwards every packet to the capture module while running
the frozen live recognizer, so no second collector may open the same COMs.
"""
from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from typing import Any, Optional

import customtkinter as ctk

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_live_hit_and_zone_0821_ZH_TW as live
from song_collection_server import parse_sensor_line
from product_hit_and_zone import predict_zone_full, slice_window, window_features
from round3_capture_0914_ZH_TW import (
    EXPECTED_MODEL_SHA256,
    VALIDATION_ROOT,
    ValidationRecorder,
    compute_preflight,
    sha256_file,
)

class Round3ValidationApp(live.LiveHitZoneApp):
    def __init__(self) -> None:
        self.recorder = ValidationRecorder()
        self.phase = "setup"
        self.session_calibrated = False
        self.preflight_active = False
        self.preflight_samples: dict[str, list[dict[str, Any]]] = {"L": [], "R": []}
        self.preflight_report: dict[str, Any] = {}
        self.preflight_after: Optional[str] = None
        self.health_lock = threading.Lock()
        self.health = {h: {"packets": 0, "zero": 0, "gaps": 0, "resets": 0, "last_pid": None} for h in ("L", "R")}
        super().__init__()
        if not self.winfo_exists(): return

        self.model_path = Path(live._resolve_model_path()).resolve()
        self.model_sha = sha256_file(self.model_path)
        if self.model_sha.lower() != EXPECTED_MODEL_SHA256:
            messagebox.showerror(
                "模型雜湊不符",
                "第三次驗證必須使用封版 0826 模型。\n\n"
                f"Expected:\n{EXPECTED_MODEL_SHA256}\n\nActual:\n{self.model_sha}"
            )
            self.after(50, self.destroy); return

        self.title("HoloGrip 第三次正式驗證 · Frozen Model + Raw 100 Hz")
        self.geometry("1040x860")
        self.session_var = tk.StringVar(value=f"R3_{datetime.now().strftime('%Y%m%d')}_P_NEW")
        self.path_var = tk.StringVar(value="尚未建立驗證場次")
        self.preflight_var = tk.StringVar(value="預檢：尚未執行")
        self._build_validation_controls()
        self._append(f"第三次驗證封版模型 SHA-256 OK：{self.model_sha}")
        self._append("重要：這個 GUI 自己保存 Raw 100 Hz；不要再開另一個採集 GUI 搶同一個 COM。")

    def _build_validation_controls(self) -> None:
        box = ctk.CTkFrame(self, fg_color="#202235", corner_radius=12)
        box.pack(fill="x", padx=18, pady=(0, 10), before=self.log)

        row1 = ctk.CTkFrame(box, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(row1, text="驗證場次", font=(live.FONT, 13, "bold")).pack(side="left", padx=(0, 6))
        self.session_entry = ctk.CTkEntry(row1, textvariable=self.session_var, width=210)
        self.session_entry.pack(side="left", padx=4)
        self.start_session_btn = ctk.CTkButton(row1, text="開始場次", width=88, fg_color="#89B4FA", text_color="#111", command=self._start_validation_session)
        self.start_session_btn.pack(side="left", padx=4)
        self.stop_session_btn = ctk.CTkButton(row1, text="停止場次", width=88, fg_color="#F38BA8", text_color="#111", command=self._stop_validation_session)
        self.stop_session_btn.pack(side="left", padx=4)
        ctk.CTkLabel(row1, textvariable=self.path_var, text_color=live.MUTED, font=("Consolas", 10), wraplength=480).pack(side="left", padx=8)

        row2 = ctk.CTkFrame(box, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(row2, text="5秒靜止預檢", width=120, fg_color="#A6E3A1", text_color="#111", command=self._start_preflight).pack(side="left", padx=4)
        ctk.CTkButton(row2, text="七鼓慢打開始", width=120, command=lambda: self._set_phase("slow_test", "SLOW_TEST_START", "七鼓各慢打 5–10 下，不訓練")).pack(side="left", padx=4)
        ctk.CTkButton(row2, text="正式歌曲開始", width=120, fg_color="#CBA6F7", text_color="#111", command=lambda: self._set_phase("song", "SONG_START", "正式 zero-shot 新歌開始")).pack(side="left", padx=4)
        ctk.CTkButton(row2, text="歌曲結束", width=100, fg_color="#585B70", command=lambda: self._set_phase("post_song", "SONG_END", "正式歌曲結束")).pack(side="left", padx=4)
        ctk.CTkLabel(row2, textvariable=self.preflight_var, font=(live.FONT, 12), text_color="#F9E2AF", wraplength=480).pack(side="left", padx=8)
        row3 = ctk.CTkFrame(box, fg_color="transparent")
        row3.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row3, text="慢打目前真值：", font=(live.FONT, 11, "bold")).pack(side="left", padx=(4, 6))
        for drum in ("小鼓", "高音 Tom", "中音 Tom", "落地 Tom", "Hi-Hat", "Crash", "Ride"):
            ctk.CTkButton(row3, text=drum, width=80, height=28, fg_color="#45475A",
                          command=lambda d=drum: self._mark_slow_drum(d)).pack(side="left", padx=2)
        ctk.CTkLabel(box, text="正式驗證順序：連線 → 開始場次 → 原 GUI『歸零』 → 5秒預檢 → 七鼓慢打（每組先按對應鼓名） → 正式新歌 → 歌曲結束 → 停止場次", text_color=live.MUTED, font=(live.FONT, 11)).pack(anchor="w", padx=16, pady=(0, 10))

    def _start_validation_session(self) -> None:
        if not self.connected:
            messagebox.showwarning("尚未連線", "請先連接兩隻手套，再開始驗證場次。")
            return
        if self.recorder.active:
            messagebox.showinfo("場次已在記錄", str(self.recorder.session_dir)); return
        path = self.recorder.start(self.session_var.get(), self.model_path, self.model_sha)
        self.phase = "setup"
        self.session_calibrated = False
        self.path_var.set(str(path))
        self.session_entry.configure(state="disabled")
        self._append(f"驗證場次開始：{path}")
        self._append("Raw 100 Hz / predictions / markers 同步記錄中。")
        self.status_var.set("場次已開始。現在回固定原點按『歸零』，然後做 5 秒靜止預檢。")

    def _stop_validation_session(self) -> None:
        if not self.recorder.active: return
        session_dir = self.recorder.session_dir
        self.recorder.stop()
        assert session_dir is not None
        summary = self._build_session_summary()
        (session_dir / "session_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (session_dir / "session_summary.md").write_text(self._summary_md(summary), encoding="utf-8")
        self.session_entry.configure(state="normal")
        self.phase = "stopped"
        self._append(f"場次停止：Raw {self.recorder.raw_written} 列；Predictions {self.recorder.pred_written} 筆；writer dropped {self.recorder.dropped}。")
        self.status_var.set(f"驗證資料已保存：{session_dir}")

    def _build_session_summary(self) -> dict[str, Any]:
        with self.health_lock:
            health = {h: dict(v) for h, v in self.health.items()}
        with self.stream_lock:
            stream = {h: dict(v) for h, v in self.stream.items()}
        for h in ("L", "R"):
            health[h]["last_hz"] = float(stream[h].get("hz", 0.0))
            health[h]["last_port"] = stream[h].get("port", "")
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": self.recorder.session_id,
            "session_dir": str(self.recorder.session_dir),
            "model_sha256": self.model_sha,
            "raw_rows": self.recorder.raw_written,
            "predictions": self.recorder.pred_written,
            "markers": self.recorder.marker_written,
            "writer_queue_dropped": self.recorder.dropped,
            "stream_health": health,
            "preflight": self.preflight_report,
            "interpretation_order": [
                "1) Raw/COM/packet/漂移是否健康",
                "2) HitDetector 是否抓到電子鼓 MIDI 真實擊打",
                "3) 只在抓到的擊打中評估七鼓 MLP",
            ],
        }

    @staticmethod
    def _summary_md(summary: dict[str, Any]) -> str:
        lines = [
            "# HoloGrip 第三次驗證場次摘要", "",
            f"- Session: `{summary['session_id']}`",
            f"- Frozen model SHA-256: `{summary['model_sha256']}`",
            f"- Raw 100 Hz rows: **{summary['raw_rows']}**",
            f"- Live predictions: **{summary['predictions']}**",
            f"- Writer dropped: **{summary['writer_queue_dropped']}**", "",
            "## Preflight", "",
        ]
        if summary.get("preflight"):
            for h, r in summary["preflight"].items():
                lines.append(f"- {live.HAND_ZH[h]}: **{r.get('status')}**, {r.get('hz',0):.1f} Hz, Yaw drift {r.get('yaw_drift_deg',0):.1f}°, Pitch drift {r.get('pitch_drift_deg',0):.1f}°")
                for reason in r.get("reasons", []): lines.append(f"  - WARN: {reason}")
        else:
            lines.append("- 未執行")
        lines += ["", "## 判讀順序", "", "1. 先排除 Raw/硬體/COM/漂移問題。", "2. 再用 MIDI 對 HitDetector 的漏打/多打。", "3. 最後看已偵測事件的鼓位分類。", ""]
        return "\n".join(lines)

    def _calibrate(self) -> None:
        before_gen = self.recognizer.generation
        super()._calibrate()
        if self.recognizer.calibrated_hands == {"L", "R"} and self.recognizer.generation > before_gen:
            if self.recorder.active:
                self.recorder.marker("CALIBRATE", "固定原點雙手歸零")
                self.session_calibrated = True
            self.preflight_var.set("預檢：尚未執行（歸零完成）")
            self._append("驗證標記：CALIBRATE。接著雙手保持固定原點做 5 秒靜止預檢。")

    def _set_phase(self, phase: str, marker: str, note: str) -> None:
        if not self.recorder.active:
            messagebox.showwarning("先開始場次", "正式驗證必須同步保存 Raw 100 Hz。")
            return
        self.phase = phase
        self.recorder.marker(marker, note)
        self._append(f"Marker {marker}: {note}")
        if marker == "SLOW_TEST_START":
            self.status_var.set("七鼓慢打診斷：每顆 5–10 下。這一段不訓練、不修改 frozen model。")
        elif marker == "SONG_START":
            self.status_var.set("正式 zero-shot 新歌進行中：不要改模型、不要改偵測器參數。")
        elif marker == "SONG_END":
            self.status_var.set("正式歌曲結束。請保存電子鼓 MIDI，再按『停止場次』。")

    def _mark_slow_drum(self, drum: str) -> None:
        if not self.recorder.active:
            messagebox.showwarning("先開始場次", "慢打診斷必須同步保存 Raw 100 Hz。")
            return
        self.phase = "slow_test"
        self.recorder.marker("SLOW_DRUM", drum)
        self.status_var.set(f"慢打真值：{drum}。現在只打這顆 5–10 下，換鼓前再按下一顆鼓名。")
        self._append(f"慢打真值標記：{drum}")

    def _start_preflight(self) -> None:
        if not self.recorder.active:
            messagebox.showwarning("先開始場次", "預檢本身也要被 Raw CSV 記錄。")
            return
        if self.recognizer.calibrated_hands != {"L", "R"} or not self.session_calibrated:
            messagebox.showwarning("尚未完成場次內歸零", "開始場次後，請在固定原點再按一次歸零，讓 CALIBRATE marker 被保存。")
            return
        if self.preflight_active: return
        self.preflight_samples = {"L": [], "R": []}
        self.preflight_active = True
        self.phase = "preflight"
        self.recorder.marker("PREFLIGHT_START", "雙手固定原點靜止 5 秒")
        self.preflight_var.set("預檢：進行中，5 秒內不要揮動或敲鼓…")
        self._append("PREFLIGHT：雙手固定原點靜止 5 秒。")
        self.preflight_after = self.after(5000, self._finish_preflight)

    def _finish_preflight(self) -> None:
        self.preflight_active = False
        self.preflight_report = {h: compute_preflight(self.preflight_samples[h]) for h in ("L", "R")}
        overall = "PASS" if all(r["status"] == "PASS" for r in self.preflight_report.values()) else "WARN"
        self.recorder.marker("PREFLIGHT_END", overall)
        if self.recorder.session_dir:
            (self.recorder.session_dir / "preflight.json").write_text(json.dumps(self.preflight_report, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks = []
        for h in ("L", "R"):
            r = self.preflight_report[h]
            chunks.append(f"{live.HAND_ZH[h]} {r['status']} {r.get('hz',0):.0f}Hz YawΔ{r.get('yaw_drift_deg',0):.1f}° PitchΔ{r.get('pitch_drift_deg',0):.1f}°")
            for reason in r.get("reasons", []): self._append(f"{live.HAND_ZH[h]} PREFLIGHT WARN：{reason}")
        self.preflight_var.set("預檢：" + overall + " | " + " | ".join(chunks))
        self.phase = "ready"
        if overall == "PASS":
            self.status_var.set("預檢 PASS。可以開始七鼓慢打診斷。")
        else:
            self.status_var.set("預檢 WARN。先檢查手套/重插/重新歸零；不要急著把後續錯誤算成模型失敗。")
        self._append("PREFLIGHT 完成：" + overall)

    def _connect(self) -> None:
        with self.health_lock:
            self.health = {h: {"packets": 0, "zero": 0, "gaps": 0, "resets": 0, "last_pid": None} for h in ("L", "R")}
        super()._connect()
        if self.connected and self.recorder.active:
            self.session_calibrated = False
            self.recorder.marker("COM_CONNECT", ",".join(self._selected_ports()))

    def _disconnect(self) -> None:
        if self.recorder.active:
            self.session_calibrated = False
            self.recorder.marker("COM_DISCONNECT", "COM disconnected/reconnect requires new calibration")
        super()._disconnect()

    def _serial_loop(self, conn: Any, port: str, stop_event: threading.Event,
                     recognizer: Any, events: queue.Queue) -> None:
        last_ui = 0.0
        local_n = 0
        last_hand = ""
        last_mag = 0.0
        parse_errors = 0
        while not stop_event.is_set() and getattr(conn, "is_open", False):
            try:
                raw = conn.readline()
            except Exception:
                if not stop_event.is_set(): events.put(("dead", port))
                break
            if not raw: continue
            received_mono = time.monotonic()
            host_ms = int(time.time() * 1000)
            packet = parse_sensor_line(raw.decode("utf-8", errors="ignore"), host_ms, received_mono)
            if packet is None:
                parse_errors += 1
                if parse_errors % 25 == 1: events.put(("err", f"{port} 無法解析封包累計 {parse_errors}"))
                continue

            local_n += 1
            last_hand = packet.hand
            last_mag = math.sqrt(packet.ax**2 + packet.ay**2 + packet.az**2)
            zero_packet = all(abs(v) < 1e-9 for v in (packet.ax, packet.ay, packet.az, packet.yaw, packet.pitch, packet.roll))

            try:
                hits = recognizer.push(packet)
                with recognizer._lock:
                    state = recognizer.hands[packet.hand]
                    calibrated = packet.hand in recognizer.calibrated_hands
                    cal_yaw, cal_pitch = state.detector.calibrated(packet)
            except Exception as exc:
                events.put(("err", f"{port} 推論失敗：{exc}"))
                continue

            with self.health_lock:
                h = self.health[packet.hand]
                h["packets"] += 1
                if zero_packet: h["zero"] += 1
                prev = h["last_pid"]
                if isinstance(prev, int):
                    if packet.packet_id > prev + 1:
                        h["gaps"] += packet.packet_id - prev - 1
                    elif packet.packet_id <= prev:
                        h["resets"] += 1
                h["last_pid"] = packet.packet_id

            if self.preflight_active:
                self.preflight_samples[packet.hand].append({
                    "mono": received_mono, "packet_id": packet.packet_id,
                    "mag": last_mag, "cal_yaw": cal_yaw, "cal_pitch": cal_pitch,
                    "zero": zero_packet,
                })

            if self.recorder.active:
                self.recorder.raw({
                    "session_id": self.recorder.session_id,
                    "session_time_ms": round(self.recorder.ms(received_mono), 3),
                    "host_time_ms": host_ms, "phase": self.phase, "port": port,
                    "hand": packet.hand, "packet_id": packet.packet_id,
                    "sensor_time_ms": packet.sensor_time_ms, "calibrated": int(calibrated),
                    "ax_g": packet.ax, "ay_g": packet.ay, "az_g": packet.az,
                    "accel_magnitude_g": round(last_mag, 6),
                    "yaw_deg": packet.yaw, "pitch_deg": packet.pitch, "roll_deg": packet.roll,
                    "cal_yaw_deg": round(cal_yaw, 6), "cal_pitch_deg": round(cal_pitch, 6),
                    "zero_packet": int(zero_packet),
                })
                for hit in hits:
                    second_drum, second_proba, margin = "", 0.0, 0.0
                    try:
                        with recognizer._lock:
                            samples = list(recognizer.hands[hit["hand"]].samples)
                        feat = window_features(slice_window(samples, hit["peak_ms"]), hit["peak_ms"])
                        if feat is not None:
                            _d, _p, ranked = predict_zone_full(self.clf, feat)
                            if len(ranked) > 1:
                                second_drum, second_proba = ranked[1]
                                margin = float(ranked[0][1] - ranked[1][1])
                    except Exception:
                        pass
                    row = {
                        "session_id": self.recorder.session_id,
                        "phase": self.phase,
                        "peak_session_ms": round(hit["peak_ms"] - self.recorder.start_mono*1000.0, 3),
                        "hand": hit["hand"], "pred_drum": hit["pred_drum"],
                        "pred_proba": round(hit["pred_proba"], 6),
                        "second_drum": second_drum, "second_proba": round(float(second_proba), 6),
                        "margin": round(float(margin), 6),
                        "peak_accel_g": hit.get("peak_accel_g", 0.0),
                        "model_sha256": self.model_sha,
                    }
                    self.recorder.pred(row)

            for hit in hits: events.put(("hit", hit))
            now = time.monotonic()
            if now - last_ui >= 0.12:
                last_ui = now
                events.put(("stream", {"port": port, "hand": last_hand, "mag": last_mag, "n": local_n, "mono": now}))

    def _close(self) -> None:
        if self.recorder.active:
            if not messagebox.askyesno("驗證場次仍在記錄", "先停止並保存場次，再關閉視窗？"):
                return
            self._stop_validation_session()
        if self.preflight_after:
            try: self.after_cancel(self.preflight_after)
            except Exception: pass
        super()._close()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    Round3ValidationApp().mainloop()


if __name__ == "__main__":
    main()
