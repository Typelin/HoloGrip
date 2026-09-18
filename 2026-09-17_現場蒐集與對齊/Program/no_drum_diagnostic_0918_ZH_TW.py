from __future__ import annotations

import json
import math
import queue
import threading
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import customtkinter as ctk
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import run_live_hit_and_zone_0821_ZH_TW as live
from song_collection_server import parse_sensor_line
from product_hit_and_zone import window_features, slice_window

REF_PATH = ROOT / "Diagnostic" / "training_reference_0826.json"


class NoDrumDiagnosticApp(live.LiveHitZoneApp):
    def __init__(self) -> None:
        if not REF_PATH.is_file():
            raise FileNotFoundError(f"找不到訓練 reference: {REF_PATH}")
        self.ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        self.ref_z = np.asarray(self.ref["scaled_features"], dtype=float)
        self.ref_labels = list(self.ref["labels"])
        self.ref_hands = list(self.ref["hands"])
        self.batch: list[dict[str, Any]] = []
        self.target_drum = "未指定"
        super().__init__()
        if not self.winfo_exists():
            return
        self.title("HoloGrip 無鼓自測診斷 · 不是正式正確率")
        self.geometry("1040x840")
        self._build_diag_controls()
        self._append("無鼓診斷只回答『現在動作是否像 0826 訓練分布、重複性如何』，不能代替真鼓 accuracy。")
        self._append("建議拿鼓棒敲枕頭/折毛巾/紙箱等軟目標，製造碰撞與回彈；不要只在空中甩手。")

    def _build_diag_controls(self) -> None:
        box = ctk.CTkFrame(self, fg_color="#202235", corner_radius=12)
        box.pack(fill="x", padx=18, pady=(0, 10), before=self.log)
        ctk.CTkLabel(
            box,
            text="無鼓診斷：選一顆虛擬鼓 → 對同一個軟目標敲 10 下 → 看重複性 / OOD",
            font=(live.FONT, 16, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 6))
        ctk.CTkLabel(
            box,
            text="這不是 accuracy。最好拿鼓棒敲枕頭、折毛巾或紙箱；空揮的加速度/回彈與真鼓差很多。",
            font=(live.FONT, 13), text_color=live.MUTED,
        ).pack(anchor="w", padx=14, pady=(0, 8))
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 6))
        for drum in live.ZONE_NAMES:
            ctk.CTkButton(row, text=drum, width=96, height=34,
                          command=lambda d=drum: self._select_target(d)).pack(side="left", padx=3)
        ctk.CTkButton(row, text="不指定", width=80, height=34, fg_color="#585B70",
                      command=lambda: self._select_target("未指定")).pack(side="left", padx=3)
        row2 = ctk.CTkFrame(box, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(0, 10))
        self.diag_var = ctk.StringVar(value="目標：未指定｜請先連線、歸零，再選一顆虛擬鼓")
        ctk.CTkLabel(row2, textvariable=self.diag_var, font=(live.FONT, 13, "bold"),
                     wraplength=760, justify="left").pack(side="left", padx=4)
        ctk.CTkButton(row2, text="清除本輪", width=100, command=self._clear_batch).pack(side="right", padx=4)

    def _select_target(self, drum: str) -> None:
        self.target_drum = drum
        self.batch.clear()
        if drum == "未指定":
            self.diag_var.set("目標：未指定｜只看分布距離與重複性")
        else:
            s = self.ref["per_class"].get(drum, {})
            self.diag_var.set(
                f"目標：{drum}｜0826 真鼓參考：Yaw 中位 {s.get('center_yaw_median',0):.1f}° "
                f"(P10~P90 {s.get('center_yaw_p10',0):.1f}~{s.get('center_yaw_p90',0):.1f})｜"
                f"Pitch {s.get('center_pitch_median',0):.1f}°｜max g 中位 {s.get('max_mag_median',0):.1f}"
            )
        self._append(f"無鼓診斷目標切換：{drum}。本輪計數歸零。")

    def _clear_batch(self) -> None:
        self.batch.clear()
        self._append("無鼓診斷：本輪已清除。")

    def _diag_for(self, hand: str, feat: list[float]) -> dict[str, Any]:
        x = np.asarray(feat, dtype=float)
        z = self.clf.named_steps["scaler"].transform([x])[0]
        idx = [i for i, h in enumerate(self.ref_hands) if h == hand]
        if not idx:
            idx = list(range(len(self.ref_z)))
        d = np.linalg.norm(self.ref_z[idx] - z, axis=1)
        j_local = int(np.argmin(d))
        j = idx[j_local]
        nn = float(d[j_local])
        p95 = float(self.ref["nn_thresholds"]["p95"])
        p99 = float(self.ref["nn_thresholds"]["p99"])
        status = "IN-LIKE" if nn <= p95 else ("BORDERLINE" if nn <= p99 else "OOD")
        return {
            "nn": nn,
            "ood": status,
            "nearest": self.ref_labels[j],
            "center_yaw": float(x[13]),
            "center_pitch": float(x[14]),
            "max_mag": float(x[0]),
            "jerk_max": float(x[7]),
            "swing_depth": float(x[15]),
        }

    def _serial_loop(self, conn: Any, port: str, stop_event: threading.Event,
                     recognizer: Any, events: queue.Queue) -> None:
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
            packet = parse_sensor_line(raw.decode("utf-8", errors="ignore"), int(time.time()*1000), time.monotonic())
            if packet is None:
                continue
            local_n += 1
            last_hand = packet.hand
            last_mag = math.sqrt(packet.ax**2 + packet.ay**2 + packet.az**2)
            try:
                hits = recognizer.push(packet)
                for hit in hits:
                    with recognizer._lock:
                        samples = list(recognizer.hands[hit["hand"]].samples)
                    feat = window_features(slice_window(samples, hit["peak_ms"]), hit["peak_ms"])
                    if feat is not None:
                        hit = dict(hit)
                        hit["diag"] = self._diag_for(hit["hand"], feat)
                    events.put(("hit", hit))
            except Exception as exc:
                events.put(("err", f"{port} 推論失敗：{exc}"))
                continue
            now = time.monotonic()
            if now - last_ui >= 0.12:
                last_ui = now
                events.put(("stream", {"port": port, "hand": last_hand, "mag": last_mag, "n": local_n, "mono": now}))

    def _handle_diag(self, payload: dict[str, Any]) -> None:
        d = payload.get("diag")
        if not d:
            return
        row = {
            "target": self.target_drum,
            "pred": payload["pred_drum"],
            "proba": float(payload["pred_proba"]),
            **d,
        }
        self.batch.append(row)
        self._append(
            f"[診斷 {len(self.batch):02d}] 目標={self.target_drum}｜pred={row['pred']} {row['proba']*100:.0f}%｜"
            f"Yaw {row['center_yaw']:+.1f}° Pitch {row['center_pitch']:+.1f}°｜max {row['max_mag']:.1f}g｜"
            f"NN {row['nn']:.2f} {row['ood']}｜最近真鼓={row['nearest']}"
        )
        if len(self.batch) % 10 == 0:
            self._summarize_batch()

    def _summarize_batch(self) -> None:
        rows = self.batch[-10:]
        counts = Counter(r["pred"] for r in rows)
        top, topn = counts.most_common(1)[0]
        ood = sum(r["ood"] == "OOD" for r in rows)
        yaw = np.asarray([r["center_yaw"] for r in rows])
        pitch = np.asarray([r["center_pitch"] for r in rows])
        mag = np.asarray([r["max_mag"] for r in rows])
        match = None if self.target_drum == "未指定" else sum(r["pred"] == self.target_drum for r in rows)
        text = (
            f"最近10下：最常預測 {top} {topn}/10｜OOD {ood}/10｜"
            f"Yaw SD {np.std(yaw):.1f}°｜Pitch SD {np.std(pitch):.1f}°｜max g 中位 {np.median(mag):.1f}"
        )
        if match is not None:
            text += f"｜『模擬目標一致』 {match}/10（不是正式 accuracy）"
        self.diag_var.set(text)
        self._append("=== 無鼓10下摘要：" + text + " ===")

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
                    hand = live.HAND_ZH.get(payload["hand"], payload["hand"])
                    hid = payload["hand"]
                    drum = payload["pred_drum"]
                    pct = payload["pred_proba"] * 100.0
                    peak = payload.get("peak_accel_g", 0.0)
                    self.big_hand_var.set(hand)
                    self.big_drum_var.set(drum)
                    self.big_meta_var.set(f"把握 {pct:.0f}%   {peak:.1f} g   第 {self.hit_count} 擊")
                    self.hand_counts[hid] += 1
                    self.hand_hit_vars[hid].set(drum)
                    self.hand_meta_vars[hid].set(f"模型信心 {pct:.0f}%  |  {peak:.1f} g  |  第 {self.hand_counts[hid]} 擊")
                    self.status_var.set("無鼓診斷中；同一目標請盡量保持相同站位、持棒與揮法。")
                    self._handle_diag(payload)
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


def main() -> None:
    if "--smoke-test" in sys.argv:
        ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        clf = live.load_zone_model(live._resolve_model_path())
        assert len(ref["features"]) == 307
        assert clf.n_features_in_ == 17
        assert len(clf.classes_) == 7
        print("NO_DRUM_DIAGNOSTIC_ENTRY_OK")
        return
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    NoDrumDiagnosticApp().mainloop()


if __name__ == "__main__":
    main()
