from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import customtkinter as ctk
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
APP = ROOT / "Apps" / "Song_Collection_COM"
DERIVED = ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
REF_PATH = LAB / "spatial_reference_0826.json"
MODEL_PATH = DERIVED / "hologrip_song2_七鼓點模型_真正驗證版_0826.joblib"
VIDEO_PATH = DERIVED / "S20260812_P01_song02_T1127_video_h264.mp4"

sys.path.insert(0, str(APP))
from product_hit_and_zone import FEATURE_NAMES, LiveRecognizer, load_zone_model, predict_zone, slice_window, window_features
from song_collection_server import parse_sensor_line

FONT = "Microsoft JhengHei UI"
BG = "#080d18"
PANEL = "#111827"
PANEL2 = "#172033"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
R_COLOR = "#fb7185"
L_COLOR = "#60a5fa"

COLORS = {
    "Hi-Hat": "#eab308",
    "Crash": "#f97316",
    "小鼓": "#06b6d4",
    "高音 Tom": "#22c55e",
    "中音 Tom": "#8b5cf6",
    "Ride": "#10b981",
    "落地 Tom": "#0ea5e9",
}

YAW_LEFT = 60.0
YAW_RIGHT = -120.0
PITCH_BOTTOM = -25.0
PITCH_TOP = 65.0

PLOT_L, PLOT_T, PLOT_R, PLOT_B = 70, 55, 990, 645
MAP_W, MAP_H = 1060, 700


def darken(hex_color: str, factor: float = 0.22) -> str:
    c = hex_color.lstrip("#")
    r, g, b = int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SessionLogger:
    def __init__(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = LAB / "live_sessions" / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.started_wall = time.time()
        self.started_mono = time.monotonic()

        self.human = (self.dir / "session.log").open("a", encoding="utf-8", buffering=1)
        self.events = (self.dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.raw_f = (self.dir / "raw_packets.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.hit_f = (self.dir / "hits.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.raw = csv.writer(self.raw_f)
        self.hit = csv.writer(self.hit_f)
        self.raw.writerow([
            "wall_time_iso","elapsed_s","port","hand","packet_id","sensor_time_ms",
            "received_time_ms","ax","ay","az","raw_yaw","raw_pitch","roll",
            "cal_yaw","cal_pitch","mag_g","hz"
        ])
        self.hit.writerow([
            "wall_time_iso","elapsed_s","hand","peak_ms","pred_drum","pred_proba",
            "peak_accel_g","hit_yaw","hit_pitch","nn_distance","nn_band","nearest_train_drum",
            *FEATURE_NAMES
        ])
        meta = {
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": str(MODEL_PATH),
            "model_sha256": sha256(MODEL_PATH),
            "reference": str(REF_PATH),
            "reference_event_count": 307,
            "purpose": "HoloGrip spatial live diagnostic; not formal accuracy scoring",
        }
        (self.dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.event("APP_START", **meta)

    def _iso(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def event(self, kind: str, **data: Any) -> None:
        rec = {
            "time": self._iso(),
            "elapsed_s": round(time.monotonic() - self.started_mono, 6),
            "kind": kind,
            **data,
        }
        with self.lock:
            self.events.write(json.dumps(rec, ensure_ascii=False) + "\n")
            compact = " ".join(f"{k}={v}" for k, v in data.items())
            self.human.write(f"[{rec['time']}] {kind} {compact}\n")

    def raw_packet(self, port: str, pkt: Any, cal_yaw: float | None, cal_pitch: float | None, hz: float) -> None:
        mag = math.sqrt(pkt.ax*pkt.ax + pkt.ay*pkt.ay + pkt.az*pkt.az)
        with self.lock:
            self.raw.writerow([
                self._iso(),
                f"{time.monotonic()-self.started_mono:.6f}",
                port, pkt.hand, pkt.packet_id, pkt.sensor_time_ms, pkt.received_time_ms,
                f"{pkt.ax:.6f}", f"{pkt.ay:.6f}", f"{pkt.az:.6f}",
                f"{pkt.yaw:.6f}", f"{pkt.pitch:.6f}", f"{pkt.roll:.6f}",
                "" if cal_yaw is None else f"{cal_yaw:.6f}",
                "" if cal_pitch is None else f"{cal_pitch:.6f}",
                f"{mag:.6f}", f"{hz:.3f}",
            ])

    def hit_event(
        self,
        hit: dict[str, Any],
        feat: list[float] | np.ndarray | None,
        hy: float | None,
        hp: float | None,
        nn: float | None,
        band: str,
        nearest: str,
    ) -> None:
        vals = list(feat) if feat is not None else [""] * len(FEATURE_NAMES)
        with self.lock:
            self.hit.writerow([
                self._iso(),
                f"{time.monotonic()-self.started_mono:.6f}",
                hit.get("hand",""),
                hit.get("peak_ms",""),
                hit.get("pred_drum",""),
                hit.get("pred_proba",""),
                hit.get("peak_accel_g",""),
                "" if hy is None else f"{hy:.6f}",
                "" if hp is None else f"{hp:.6f}",
                "" if nn is None else f"{nn:.6f}",
                band,
                nearest,
                *vals,
            ])
        self.event(
            "HIT",
            hand=hit.get("hand"),
            pred=hit.get("pred_drum"),
            prob=round(float(hit.get("pred_proba", 0.0)), 4),
            peak_g=round(float(hit.get("peak_accel_g", 0.0)), 3),
            hit_yaw=None if hy is None else round(hy, 3),
            hit_pitch=None if hp is None else round(hp, 3),
            nn=None if nn is None else round(nn, 3),
            band=band,
            nearest=nearest,
        )

    def close(self, summary: dict[str, Any] | None = None) -> None:
        try:
            if summary:
                (self.dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            self.event("APP_CLOSE", summary=summary or {})
        except Exception:
            pass
        for f in (self.raw_f, self.hit_f, self.events, self.human):
            try:
                f.flush()
                f.close()
            except Exception:
                pass


class SpatialCockpitApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HoloGrip 鼓手視角｜角度分界與真實模型診斷")
        self.geometry("1580x930")
        self.minsize(1450, 860)
        self.configure(fg_color=BG)

        self.session_log = SessionLogger()
        self.hit_counts = {"L": 0, "R": 0}
        self.ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        self.order = list(self.ref["drum_order"])
        self.events = self.ref["events"]
        self.classes = self.ref["classes"]
        self.scale = self.ref["scale"]
        self.ref_yaw = np.asarray([r["yaw"] for r in self.events], dtype=float)
        self.ref_pitch = np.asarray([r["pitch"] for r in self.events], dtype=float)
        self.ref_labels = [r["drum"] for r in self.events]
        self.ref_features = np.asarray([r["feature"] for r in self.events], dtype=float)

        self.clf = load_zone_model(str(MODEL_PATH))
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)
        self.scaler = self.clf.named_steps.get("scaler")
        self.ref_scaled = self.scaler.transform(self.ref_features) if self.scaler is not None else self.ref_features

        # Training nearest-neighbour thresholds for full 17D OOD diagnostics.
        dmat = np.linalg.norm(self.ref_scaled[:, None, :] - self.ref_scaled[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        nn = np.min(dmat, axis=1)
        self.nn95 = float(np.percentile(nn, 95))
        self.nn99 = float(np.percentile(nn, 99))

        self.stop_event = threading.Event()
        self.uiq: queue.Queue = queue.Queue()
        self.serials: list[Any] = []
        self.threads: list[threading.Thread] = []
        self.port_map: dict[str, str] = {}
        self.connected = False
        self.last_active_hand = "R"
        self.hand = {
            h: {
                "port": "", "hz": 0.0, "yaw": None, "pitch": None, "roll": 0.0,
                "raw_yaw": None, "raw_pitch": None,
                "mag": 0.0, "last": 0.0, "posture": "—", "posture_share": 0.0,
                "hit": "尚無擊打", "nn": "", "hit_yaw": None, "hit_pitch": None,
            }
            for h in ("R", "L")
        }
        self.port_rx = {}
        self.trail = {"R": [], "L": []}
        self.current_ref_drum = None
        self.video_ctk = None
        self.pad_items = {}
        self._flash_until = 0.0
        self._flash_drum = None
        self.replay_index = 0
        self.replay_running = False

        self._build()
        self._refresh_ports()
        self.after(35, self._drain)
        self.after(80, self._refresh_dynamic)
        self.after(1000, self._heartbeat_log)
        self.after(900, self._auto_connect_xiao)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------- layout ----------
    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color=PANEL)
        header.pack(fill="x", padx=12, pady=(10, 7))
        ctk.CTkLabel(
            header, text="HoloGrip 鼓手視角空間診斷",
            font=(FONT, 25, "bold"), text_color=TEXT
        ).pack(side="left", padx=14, pady=10)
        ctk.CTkLabel(
            header,
            text="來源：8/12 Song2 正面錄影 + 307 筆對齊真值｜已翻轉為鼓手自己的左右",
            font=(FONT, 13, "bold"), text_color="#fbbf24"
        ).pack(side="left", padx=16)

        controls = ctk.CTkFrame(self, fg_color=PANEL)
        controls.pack(fill="x", padx=12, pady=(0, 8))
        self.com_a = ctk.StringVar(value="選擇 COM A")
        self.com_b = ctk.StringVar(value="選擇 COM B")
        self.cb_a = ctk.CTkComboBox(controls, variable=self.com_a, width=215)
        self.cb_b = ctk.CTkComboBox(controls, variable=self.com_b, width=215)
        self.cb_a.pack(side="left", padx=(10, 5), pady=9)
        self.cb_b.pack(side="left", padx=5, pady=9)
        ctk.CTkButton(controls, text="掃描", width=75, command=self._refresh_ports).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="連接", width=80, command=self._connect).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="斷開", width=80, command=self._disconnect).pack(side="left", padx=4)
        ctk.CTkButton(
            controls, text="固定原點歸零", width=145,
            fg_color="#f59e0b", hover_color="#d97706", text_color="#111827",
            command=self._calibrate
        ).pack(side="left", padx=8)
        ctk.CTkButton(controls, text="開啟舊影片", width=110, command=self._open_video).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="重播舊資料", width=110, command=self._start_replay).pack(side="left", padx=4)
        self.status = ctk.StringVar(value="單手也能測。連線後先回原點歸零，再慢慢移手看分界。")
        ctk.CTkLabel(controls, textvariable=self.status, font=(FONT, 13), text_color=TEXT).pack(side="left", padx=12)

        self.rx_status = ctk.StringVar(value="RX：尚未連線")
        ctk.CTkLabel(
            self, textvariable=self.rx_status, font=("Consolas", 12),
            text_color="#93c5fd", anchor="w"
        ).pack(fill="x", padx=22, pady=(0, 6))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.grid_columnconfigure(0, weight=7)
        body.grid_columnconfigure(1, weight=3)
        body.grid_rowconfigure(0, weight=1)

        # Main angle map
        left = ctk.CTkFrame(body, fg_color=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.map_canvas = ctk.CTkCanvas(
            left, width=MAP_W, height=MAP_H, bg="#0b1020",
            highlightthickness=0
        )
        self.map_canvas.pack(padx=8, pady=8, fill="both", expand=False)
        self._draw_static_angle_map()

        # Right column
        right = ctk.CTkFrame(body, fg_color=PANEL)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ctk.CTkLabel(right, text="鼓手座位圖（你坐在下方）", font=(FONT, 17, "bold")).pack(pady=(10, 3))
        self.kit_canvas = ctk.CTkCanvas(right, width=455, height=270, bg="#0b1020", highlightthickness=0)
        self.kit_canvas.pack(padx=10, pady=(0, 8))
        self._draw_kit()

        self.live_title = ctk.StringVar(value="目前姿態：等待歸零")
        ctk.CTkLabel(
            right, textvariable=self.live_title, font=(FONT, 20, "bold"),
            text_color="#f8fafc", wraplength=455
        ).pack(padx=12, pady=(2, 2))

        self.live_detail = ctk.StringVar(value="Yaw —°｜Pitch —°")
        ctk.CTkLabel(
            right, textvariable=self.live_detail, font=("Consolas", 15),
            text_color="#cbd5e1", justify="left", wraplength=455
        ).pack(padx=12, pady=(0, 8))

        self.hit_result = ctk.StringVar(value="真正模型本擊：尚無擊打")
        ctk.CTkLabel(
            right, textvariable=self.hit_result, font=(FONT, 17, "bold"),
            text_color="#fbbf24", justify="left", wraplength=455
        ).pack(padx=12, pady=(0, 7))

        self.ood_result = ctk.StringVar(value="17D 分布：—")
        ctk.CTkLabel(
            right, textvariable=self.ood_result, font=("Consolas", 13),
            text_color=MUTED, justify="left", wraplength=455
        ).pack(padx=12, pady=(0, 8))

        ctk.CTkLabel(
            right, text="舊影片參考（已左右翻轉成鼓手視角）",
            font=(FONT, 14, "bold"), text_color="#cbd5e1"
        ).pack(pady=(4, 3))
        self.video_ref_label = ctk.CTkLabel(
            right, text="移動到某個鼓區後會顯示該鼓的舊影片真實擊打畫面",
            width=455, height=245, fg_color="#0b1020", corner_radius=8,
            font=(FONT, 12), text_color=MUTED
        )
        self.video_ref_label.pack(padx=10, pady=(0, 5))

        self.video_ref_caption = ctk.StringVar(value="")
        ctk.CTkLabel(
            right, textvariable=self.video_ref_caption,
            font=(FONT, 11), text_color=MUTED, wraplength=455
        ).pack(padx=10, pady=(0, 8))

        model_sha = sha256(MODEL_PATH)
        ctk.CTkLabel(
            right,
            text=(
                "判讀方式：\n"
                "• 彩色底區 = 307 筆真實擊打的 Yaw/Pitch 姿態參考分界。\n"
                "• 移動中的圓點 = 目前歸零後手部方向。\n"
                "• 「目前姿態」只看 Yaw/Pitch；不是正式模型答案。\n"
                "• 打下去後「真正模型本擊」才是完整 17 維 frozen MLP 輸出。\n"
                f"• Frozen model SHA: {model_sha[:12]}…"
            ),
            font=(FONT, 11), justify="left", anchor="w", text_color=MUTED,
            wraplength=455
        ).pack(fill="x", padx=12, pady=(2, 10))

    # ---------- reference / mapping ----------
    def _posture_vote(self, yaw: float, pitch: float) -> tuple[str, float]:
        sy = float(self.scale["yaw_scale"])
        sp = float(self.scale["pitch_scale"])
        d = np.hypot((self.ref_yaw - yaw) / sy, (self.ref_pitch - pitch) / sp)
        k = min(11, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        votes = {name: 0.0 for name in self.order}
        for i in idx:
            w = 1.0 / (float(d[int(i)]) + 0.08) ** 2
            votes[self.ref_labels[int(i)]] += w
        total = sum(votes.values()) or 1.0
        best = max(self.order, key=lambda x: votes[x])
        return best, votes[best] / total

    def _xy(self, yaw: float, pitch: float) -> tuple[float, float]:
        x = PLOT_L + (YAW_LEFT - yaw) / (YAW_LEFT - YAW_RIGHT) * (PLOT_R - PLOT_L)
        y = PLOT_B - (pitch - PITCH_BOTTOM) / (PITCH_TOP - PITCH_BOTTOM) * (PLOT_B - PLOT_T)
        return x, y

    def _draw_static_angle_map(self) -> None:
        c = self.map_canvas
        c.delete("all")

        # Empirical 2D regions, 5-degree cells.
        step = 5
        for yaw0 in np.arange(YAW_RIGHT, YAW_LEFT, step):
            for p0 in np.arange(PITCH_BOTTOM, PITCH_TOP, step):
                drum, _ = self._posture_vote(float(yaw0 + step / 2), float(p0 + step / 2))
                x1, y1 = self._xy(float(yaw0), float(p0))
                x2, y2 = self._xy(float(yaw0 + step), float(p0 + step))
                c.create_rectangle(
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2),
                    fill=darken(COLORS[drum], 0.18), outline=""
                )

        # Axes and grid.
        for yaw in range(-120, 61, 10):
            x, _ = self._xy(float(yaw), 0.0)
            c.create_line(x, PLOT_T, x, PLOT_B, fill="#334155", width=1)
            c.create_text(x, PLOT_B + 18, text=f"{yaw:+d}°", fill="#cbd5e1", font=(FONT, 9))
        for pitch in range(-20, 61, 10):
            _, y = self._xy(0.0, float(pitch))
            c.create_line(PLOT_L, y, PLOT_R, y, fill="#334155", width=1)
            c.create_text(PLOT_L - 30, y, text=f"{pitch:+d}°", fill="#cbd5e1", font=(FONT, 9))
        x0, _ = self._xy(0.0, 0.0)
        _, y0 = self._xy(0.0, 0.0)
        c.create_line(x0, PLOT_T, x0, PLOT_B, fill="#f8fafc", width=2)
        c.create_line(PLOT_L, y0, PLOT_R, y0, fill="#64748b", width=1)

        # Reference points and 10-90% boxes.
        for drum in self.order:
            rr = [r for r in self.events if r["drum"] == drum]
            for r in rr:
                x, y = self._xy(float(r["yaw"]), float(r["pitch"]))
                c.create_oval(x-1.5, y-1.5, x+1.5, y+1.5, fill=COLORS[drum], outline="")
            s = self.classes[drum]
            x1, y1 = self._xy(float(s["yaw_p90"]), float(s["pitch_p90"]))
            x2, y2 = self._xy(float(s["yaw_p10"]), float(s["pitch_p10"]))
            c.create_oval(
                min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2),
                outline=COLORS[drum], width=2
            )
            xm, ym = self._xy(float(s["yaw_median"]), float(s["pitch_median"]))
            c.create_oval(xm-8, ym-8, xm+8, ym+8, fill=COLORS[drum], outline="white", width=1)
            c.create_text(
                xm, ym-19, text=drum, fill="white", font=(FONT, 11, "bold"),
                anchor="s"
            )

        c.create_text(
            (PLOT_L + PLOT_R)/2, 18,
            text="鼓手視角角度地圖｜鼓手左 ← +Yaw ｜ 0° ｜ -Yaw → 鼓手右",
            fill="white", font=(FONT, 16, "bold")
        )
        c.create_text(
            PLOT_R - 5, PLOT_T + 8,
            text="↑ 抬高 +Pitch", fill="#fbbf24", font=(FONT, 11, "bold"), anchor="ne"
        )
        c.create_text(
            PLOT_L, PLOT_B + 43,
            text="彩色區域＝姿態參考分界（Yaw/Pitch only）｜真正 17D MLP 只在擊打後顯示",
            fill=MUTED, font=(FONT, 10), anchor="w"
        )

    def _draw_kit(self) -> None:
        c = self.kit_canvas
        c.delete("all")
        c.create_text(228, 14, text="↑ 前方", fill="#fbbf24", font=(FONT, 11, "bold"))
        c.create_oval(195, 232, 260, 290, fill="#334155", outline="#64748b", width=2)
        c.create_text(228, 249, text="鼓手", fill="white", font=(FONT, 11, "bold"))

        minx, maxx = -0.66, 0.66
        minf, maxf = 0.66, 1.31
        for drum in self.order:
            s = self.classes[drum]
            xx = 35 + (float(s["cockpit_x"]) - minx) / (maxx - minx) * 385
            yy = 220 - (float(s["cockpit_forward"]) - minf) / (maxf - minf) * 180
            cymbal = drum in ("Hi-Hat", "Crash", "Ride")
            rw, rh = (53, 17) if cymbal else (43, 28)
            item = c.create_oval(
                xx-rw, yy-rh, xx+rw, yy+rh,
                fill=darken(COLORS[drum], 0.42), outline=COLORS[drum], width=2,
                tags=(f"pad_{drum}",)
            )
            c.create_text(xx, yy, text=drum, fill="white", font=(FONT, 10, "bold"))
            self.pad_items[drum] = item

    # ---------- serial ----------
    def _refresh_ports(self) -> None:
        from serial.tools import list_ports
        vals = []
        self.port_map = {}
        for p in list_ports.comports():
            label = f"{p.device}  {(p.description or '').replace(p.device, '').strip()}"
            vals.append(label)
            self.port_map[label] = p.device
            self.port_map[p.device] = p.device
        if not vals:
            vals = ["未找到 COM"]
        self.cb_a.configure(values=vals)
        self.cb_b.configure(values=vals)
        if vals[0] != "未找到 COM":
            self.com_a.set(vals[0])
            self.com_b.set(vals[1] if len(vals) > 1 else "選擇 COM B")
        try:
            self.session_log.event("PORT_SCAN", values=vals)
        except Exception:
            pass

    def _selected_ports(self) -> list[str]:
        out = []
        for raw in (self.com_a.get(), self.com_b.get()):
            p = self.port_map.get(raw.strip())
            if p and p not in out:
                out.append(p)
        return out

    def _auto_connect_xiao(self) -> None:
        if self.connected:
            return
        try:
            from serial.tools import list_ports
            matches = []
            for p in list_ports.comports():
                hwid = (p.hwid or "").upper()
                if "303A:1001" in hwid or "VID_303A&PID_1001" in hwid:
                    label = next(
                        (k for k, v in self.port_map.items() if v == p.device and k != p.device),
                        p.device,
                    )
                    matches.append((p.device, label))
            if len(matches) >= 2:
                matches.sort(key=lambda x: x[0])
                self.com_a.set(matches[0][1])
                self.com_b.set(matches[1][1])
                self.session_log.event("AUTO_CONNECT_REQUEST", ports=[matches[0][0], matches[1][0]])
                self._connect()
            else:
                self.session_log.event("AUTO_CONNECT_SKIPPED", xiao_count=len(matches))
        except Exception as e:
            self.session_log.event("AUTO_CONNECT_ERROR", error=repr(e))

    def _connect(self) -> None:
        import serial
        ports = self._selected_ports()
        if not ports:
            self.status.set("沒有選到 COM。")
            self.session_log.event("CONNECT_FAIL", reason="no_ports_selected")
            return
        self.session_log.event("CONNECT_BEGIN", ports=ports)
        self._disconnect()
        self.recognizer = LiveRecognizer(self.clf, require_calibration=True)
        self.stop_event = threading.Event()
        self.uiq = queue.Queue()
        self.port_rx = {p: {"raw": 0, "parsed": 0, "last": 0.0, "bad": 0} for p in ports}
        try:
            for port in ports:
                conn = serial.Serial(port=port, baudrate=460800, timeout=0.20)
                self.serials.append(conn)
            time.sleep(0.8)
            for conn, port in zip(self.serials, ports):
                try:
                    conn.reset_input_buffer()
                except Exception:
                    pass
                t = threading.Thread(target=self._serial_loop, args=(conn, port), daemon=True)
                t.start()
                self.threads.append(t)
            self.connected = True
            self.status.set(f"COM 已開啟，正在等第一批資料封包…｜Log: {self.session_log.dir.name}")
            self.session_log.event("CONNECT_OK", ports=ports, log_dir=str(self.session_log.dir))
            self.after(2200, self._connection_watchdog)
        except Exception as e:
            self.session_log.event("CONNECT_FAIL", ports=ports, error=repr(e))
            self._disconnect()
            self.status.set(f"連線失敗：{e}")

    def _disconnect(self) -> None:
        was_connected = self.connected or bool(self.serials)
        if was_connected:
            try:
                self.session_log.event("DISCONNECT_BEGIN", ports=list(self.port_rx))
            except Exception:
                pass
        self.stop_event.set()
        for conn in self.serials:
            try:
                conn.close()
            except Exception:
                pass
        self.serials = []
        for t in self.threads:
            t.join(timeout=0.25)
        self.threads = []
        self.connected = False
        if hasattr(self, "rx_status"):
            self.rx_status.set("RX：尚未連線")
        if was_connected:
            try:
                self.session_log.event("DISCONNECT_DONE")
            except Exception:
                pass

    def _calibrate(self) -> None:
        if not self.connected:
            self.status.set("請先連接。")
            return
        now = time.monotonic()
        fresh = []
        with self.recognizer._lock:
            for hand, state in self.recognizer.hands.items():
                if state.last_packet is not None:
                    fresh.append((hand, now - state.last_packet.received_monotonic))
        if not fresh:
            self.status.set("無法歸零：COM 已開啟，但 recognizer 尚未收到任何有效 D 封包。")
            self.session_log.event("CALIBRATE_FAIL", reason="no_valid_packets")
            return
        if all(age > 1.2 for _, age in fresh):
            ages = " / ".join(f"{h}:{age:.1f}s" for h, age in fresh)
            self.status.set(f"無法歸零：最後封包太舊（{ages}）。請重插/RESET 手套。")
            self.session_log.event("CALIBRATE_FAIL", reason="stale_packets", ages=ages)
            return

        origins = {}
        with self.recognizer._lock:
            for hand, state in self.recognizer.hands.items():
                pkt = state.last_packet
                if pkt is not None:
                    origins[hand] = {
                        "raw_yaw": round(float(pkt.yaw), 4),
                        "raw_pitch": round(float(pkt.pitch), 4),
                        "raw_roll": round(float(pkt.roll), 4),
                        "packet_id": int(pkt.packet_id),
                        "sensor_time_ms": int(pkt.sensor_time_ms),
                    }
        self.session_log.event("CALIBRATE_BEGIN", origins=origins)

        done = self.recognizer.calibrate()
        if not done:
            self.status.set("無法歸零：recognizer 沒有可用手別封包。")
            self.session_log.event("CALIBRATE_FAIL", reason="recognizer_no_packets")
            return
        offsets = {}
        with self.recognizer._lock:
            for h in done:
                det = self.recognizer.hands[h].detector
                offsets[h] = {
                    "yaw_offset": round(float(det.yaw_offset), 4),
                    "pitch_offset": round(float(det.pitch_offset), 4),
                }
        for h in done:
            self.trail[h].clear()
        self.session_log.event("CALIBRATE_OK", hands=done, offsets=offsets)
        self.status.set("歸零完成：" + "、".join(done) + "。慢慢移手，先看圓點是否符合你的實際方向。")

    def _serial_loop(self, conn, port: str) -> None:
        local_n = 0
        tick_n = 0
        tick_t = time.monotonic()
        last_hz = 0.0
        while not self.stop_event.is_set() and getattr(conn, "is_open", False):
            try:
                raw = conn.readline()
            except Exception as e:
                self.uiq.put(("dead", {"port": port, "error": repr(e)}))
                break
            if not raw:
                continue
            self.uiq.put(("rx_raw", {"port": port}))
            pkt = parse_sensor_line(
                raw.decode("utf-8", errors="ignore"),
                int(time.time() * 1000),
                time.monotonic(),
            )
            if pkt is None:
                self.uiq.put(("rx_bad", {"port": port}))
                continue
            local_n += 1
            try:
                hits = self.recognizer.push(pkt)
            except Exception as e:
                self.uiq.put(("err", str(e)))
                continue

            cy = cp = None
            state = self.recognizer.hands.get(pkt.hand)
            if state is not None and pkt.hand in self.recognizer.calibrated_hands:
                cy, cp = state.detector.calibrated(pkt)

            now = time.monotonic()
            if now - tick_t >= 0.5:
                last_hz = (local_n - tick_n) / (now - tick_t)
                tick_t = now
                tick_n = local_n

            try:
                self.session_log.raw_packet(port, pkt, cy, cp, last_hz)
            except Exception as e:
                self.uiq.put(("err", f"log raw failed: {e}"))

            self.uiq.put(("packet", {
                "hand": pkt.hand,
                "port": port,
                "raw_yaw": pkt.yaw,
                "raw_pitch": pkt.pitch,
                "yaw": cy,
                "pitch": cp,
                "roll": pkt.roll,
                "mag": math.sqrt(pkt.ax*pkt.ax + pkt.ay*pkt.ay + pkt.az*pkt.az),
                "hz": last_hz,
            }))
            for hit in hits:
                self.uiq.put(("hit", hit))

    # ---------- live updates ----------
    def _on_packet(self, d: dict[str, Any]) -> None:
        h = d["hand"]
        self.last_active_hand = h
        st = self.hand[h]
        st["port"] = d["port"]
        st["hz"] = float(d["hz"] or 0.0)
        st["roll"] = float(d["roll"])
        st["raw_yaw"] = float(d["raw_yaw"])
        st["raw_pitch"] = float(d["raw_pitch"])
        st["mag"] = float(d["mag"])
        st["last"] = time.monotonic()
        if d["yaw"] is None:
            return
        yaw = float(d["yaw"])
        pitch = float(d["pitch"])
        st["yaw"] = yaw
        st["pitch"] = pitch
        drum, share = self._posture_vote(yaw, pitch)
        st["posture"] = drum
        st["posture_share"] = share
        self.trail[h].append((yaw, pitch))
        if len(self.trail[h]) > 28:
            del self.trail[h][0]

    def _on_hit(self, hit: dict[str, Any]) -> None:
        h = hit["hand"]
        state = self.recognizer.hands[h]
        peak = float(hit["peak_ms"])
        feat = window_features(slice_window(list(state.samples), peak, 80.0), peak)
        nn_text = ""
        hy = hp = None
        nn = None
        band = ""
        nearest = ""
        if feat is not None:
            hy = float(feat[13])
            hp = float(feat[14])
            z = self.scaler.transform([feat])[0] if self.scaler is not None else np.asarray(feat)
            d = np.linalg.norm(self.ref_scaled - z, axis=1)
            nn = float(np.min(d))
            band = "IN-LIKE" if nn <= self.nn95 else ("BORDERLINE" if nn <= self.nn99 else "OOD")
            nearest = self.ref_labels[int(np.argmin(d))]
            nn_text = f"17D NN {nn:.2f} {band}｜最近真實訓練鼓={nearest}"
        self.hand[h]["hit"] = f"{hit['pred_drum']} {hit['pred_proba']*100:.0f}%｜{hit.get('peak_accel_g',0):.1f}g"
        self.hand[h]["nn"] = nn_text
        self.hand[h]["hit_yaw"] = hy
        self.hand[h]["hit_pitch"] = hp
        self.hit_counts[h] += 1
        try:
            self.session_log.hit_event(hit, feat, hy, hp, nn, band, nearest)
        except Exception as e:
            self.session_log.event("HIT_LOG_ERROR", error=repr(e))
        self._flash_drum = hit["pred_drum"]
        self._flash_until = time.monotonic() + 0.8

    def _drain(self) -> None:
        try:
            while True:
                k, d = self.uiq.get_nowait()
                if k == "packet":
                    pr = self.port_rx.get(d["port"])
                    if pr is not None:
                        pr["parsed"] += 1
                        pr["last"] = time.monotonic()
                    self._on_packet(d)
                elif k == "hit":
                    self._on_hit(d)
                elif k == "err":
                    self.status.set("推論錯誤：" + str(d))
                    self.session_log.event("ERROR", message=str(d))
                elif k == "rx_raw":
                    pr = self.port_rx.get(d["port"])
                    if pr is not None:
                        pr["raw"] += 1
                elif k == "rx_bad":
                    pr = self.port_rx.get(d["port"])
                    if pr is not None:
                        pr["bad"] += 1
                elif k == "dead":
                    self.status.set(f"{d['port']} 讀取執行緒中斷：{d['error']}")
                    self.session_log.event("SERIAL_THREAD_DEAD", port=d["port"], error=d["error"])
        except queue.Empty:
            pass
        self.after(35, self._drain)

    def _refresh_dynamic(self) -> None:
        self.map_canvas.delete("dynamic")
        now = time.monotonic()

        # trails + pointers for both hands
        for h, color in (("R", R_COLOR), ("L", L_COLOR)):
            st = self.hand[h]
            pts = self.trail[h]
            if len(pts) >= 2:
                xy = []
                for yaw, pitch in pts:
                    x, y = self._xy(yaw, pitch)
                    xy.extend([x, y])
                self.map_canvas.create_line(*xy, fill=color, width=2, smooth=True, tags="dynamic")
            if st["yaw"] is not None:
                x, y = self._xy(float(st["yaw"]), float(st["pitch"]))
                self.map_canvas.create_oval(x-13, y-13, x+13, y+13, fill=color, outline="white", width=3, tags="dynamic")
                self.map_canvas.create_text(x, y-20, text=h, fill="white", font=(FONT, 11, "bold"), tags="dynamic")

            if st["hit_yaw"] is not None:
                xh, yh = self._xy(float(st["hit_yaw"]), float(st["hit_pitch"]))
                self.map_canvas.create_oval(xh-19, yh-19, xh+19, yh+19, fill="", outline="#ffffff", width=4, tags="dynamic")
                self.map_canvas.create_text(xh, yh+27, text=f"{h} 本擊", fill="white", font=(FONT, 10, "bold"), tags="dynamic")

        # choose active hand for detailed side panel
        h = self.last_active_hand
        st = self.hand[h]
        if st["yaw"] is None:
            other = "L" if h == "R" else "R"
            if self.hand[other]["yaw"] is not None:
                h, st = other, self.hand[other]

        if st["yaw"] is not None:
            yaw = float(st["yaw"])
            pitch = float(st["pitch"])
            side = f"向左 {yaw:.1f}°" if yaw >= 0 else f"向右 {abs(yaw):.1f}°"
            vert = f"抬高 {pitch:.1f}°" if pitch >= 0 else f"降低 {abs(pitch):.1f}°"
            self.live_title.set(f"目前姿態 ≈ {st['posture']}  ({st['posture_share']*100:.0f}% 姿態票)")
            self.live_detail.set(
                f"{h} 手｜Yaw {yaw:+.1f}°（{side}）\n"
                f"Pitch {pitch:+.1f}°（{vert}）｜Roll {st['roll']:+.1f}°\n"
                f"{st['port']}｜{st['hz']:.1f} Hz｜|A| {st['mag']:.2f}g"
            )
            if st["posture"] != self.current_ref_drum:
                self._show_video_ref(st["posture"])
        elif st["raw_yaw"] is not None:
            self.live_title.set("已收到 Raw 封包｜尚未歸零")
            self.live_detail.set(
                f"{h} 手｜Raw Yaw {st['raw_yaw']:+.1f}°｜Raw Pitch {st['raw_pitch']:+.1f}°\n"
                f"{st['port']}｜{st['hz']:.1f} Hz｜|A| {st['mag']:.2f}g"
            )

        self.hit_result.set(
            f"真正模型本擊（{h}）：{st['hit']}" if st["hit"] != "尚無擊打"
            else "真正模型本擊：尚無擊打"
        )
        self.ood_result.set(st["nn"] or "17D 分布：尚無擊打")

        # cockpit highlight
        for drum, item in self.pad_items.items():
            width = 2
            outline = COLORS[drum]
            if st["posture"] == drum:
                width = 5
                outline = "white"
            if self._flash_drum == drum and now < self._flash_until:
                width = 7
                outline = "#fbbf24"
            self.kit_canvas.itemconfigure(item, outline=outline, width=width)

        if self._flash_drum and now >= self._flash_until:
            self._flash_drum = None

        if self.port_rx:
            parts = []
            now2 = time.monotonic()
            for port, pr in self.port_rx.items():
                age = (now2 - pr["last"]) if pr["last"] else None
                age_text = f"{age:.1f}s" if age is not None else "—"
                parts.append(
                    f"{port}: raw={pr['raw']} parsed={pr['parsed']} bad={pr['bad']} last={age_text}"
                )
            self.rx_status.set("RX｜" + "   ".join(parts))

        self.after(80, self._refresh_dynamic)

    def _heartbeat_log(self) -> None:
        try:
            hands = {}
            for h, st in self.hand.items():
                hands[h] = {
                    "port": st["port"],
                    "hz": round(float(st["hz"] or 0.0), 2),
                    "raw_yaw": st["raw_yaw"],
                    "raw_pitch": st["raw_pitch"],
                    "cal_yaw": st["yaw"],
                    "cal_pitch": st["pitch"],
                    "posture": st["posture"],
                    "posture_share": round(float(st["posture_share"] or 0.0), 4),
                    "hit_count": self.hit_counts[h],
                }
            self.session_log.event(
                "HEARTBEAT",
                connected=self.connected,
                rx=self.port_rx,
                hands=hands,
            )
        except Exception:
            pass
        self.after(1000, self._heartbeat_log)

    def _connection_watchdog(self) -> None:
        if not self.connected:
            return
        total_raw = sum(v["raw"] for v in self.port_rx.values())
        total_parsed = sum(v["parsed"] for v in self.port_rx.values())
        if total_raw == 0:
            ports = ", ".join(self.port_rx)
            self.status.set(
                f"{ports} 已成功開啟，但 2 秒內 RX=0。這不是模型問題：請按 ESP32 RESET/EN 或拔插 USB。"
            )
        elif total_parsed == 0:
            self.status.set(
                f"Serial 有收到 {total_raw} 行，但 0 行符合 D,HAND,... 格式；請檢查手套韌體。"
            )
        else:
            self.status.set(
                f"資料流正常：已解析 {total_parsed} 包。現在回固定原點姿勢按「固定原點歸零」。"
            )

    def _show_video_ref(self, drum: str) -> None:
        self.current_ref_drum = drum
        info = self.classes.get(drum, {})
        name = info.get("video_ref_file")
        if not name:
            return
        p = LAB / "video_references" / name
        if not p.is_file():
            return
        try:
            img = Image.open(p).convert("RGB")
            # Preserve 16:9-ish frame while fitting the panel.
            self.video_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(450, 253))
            self.video_ref_label.configure(image=self.video_ctk, text="")
            ms = float(info.get("video_ref_ms", 0.0))
            eid = info.get("video_ref_event_id", "?")
            self.video_ref_caption.set(
                f"{drum}｜舊影片代表擊打 #{eid}｜影片 {ms/1000:.3f}s｜已水平翻轉"
            )
        except Exception as e:
            self.video_ref_label.configure(image=None, text=f"影片參考載入失敗：{e}")

    def _start_replay(self) -> None:
        if self.replay_running:
            self.replay_running = False
            self.status.set("已停止舊資料重播。")
            return
        self.replay_running = True
        self.replay_index = 0
        self.status.set("正在重播 8/12 的 307 筆真實擊打；這不是新的實機測試。")
        self._replay_step()

    def _replay_step(self) -> None:
        if not self.replay_running:
            return
        if self.replay_index >= len(self.events):
            self.replay_running = False
            self.status.set("舊資料重播完成。現在可接手套做實機。")
            return
        r = self.events[self.replay_index]
        self.replay_index += 1
        h = r["hand"] if r["hand"] in ("L", "R") else "R"
        st = self.hand[h]
        yaw = float(r["yaw"])
        pitch = float(r["pitch"])
        posture, share = self._posture_vote(yaw, pitch)
        st["yaw"] = yaw
        st["pitch"] = pitch
        st["posture"] = posture
        st["posture_share"] = share
        st["port"] = "舊資料重播"
        st["hz"] = 100.0
        self.last_active_hand = h
        self.trail[h].append((yaw, pitch))
        if len(self.trail[h]) > 28:
            del self.trail[h][0]

        feat = [float(x) for x in r["feature"]]
        pred, prob = predict_zone(self.clf, feat)
        z = self.scaler.transform([feat])[0] if self.scaler is not None else np.asarray(feat)
        d = np.linalg.norm(self.ref_scaled - z, axis=1)
        # Ignore the exact event itself so the replay NN number remains meaningful.
        same_idx = self.replay_index - 1
        if 0 <= same_idx < len(d):
            d[same_idx] = np.inf
        nn = float(np.min(d))
        band = "IN-LIKE" if nn <= self.nn95 else ("BORDERLINE" if nn <= self.nn99 else "OOD")
        nearest = self.ref_labels[int(np.argmin(d))]
        st["hit"] = f"{pred} {prob*100:.0f}%｜舊 GT={r['drum']}"
        st["nn"] = f"17D NN {nn:.2f} {band}｜最近其他真實訓練鼓={nearest}"
        st["hit_yaw"] = yaw
        st["hit_pitch"] = pitch
        self._flash_drum = pred
        self._flash_until = time.monotonic() + 0.13
        self.after(110, self._replay_step)

    def _open_video(self) -> None:
        try:
            os.startfile(str(VIDEO_PATH))
        except Exception as e:
            self.status.set(f"無法開啟影片：{e}")

    def _close(self) -> None:
        summary = {
            "closed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hit_counts": self.hit_counts,
            "port_rx": self.port_rx,
            "hands": {
                h: {
                    "port": st["port"],
                    "hz": st["hz"],
                    "raw_yaw": st["raw_yaw"],
                    "raw_pitch": st["raw_pitch"],
                    "cal_yaw": st["yaw"],
                    "cal_pitch": st["pitch"],
                    "posture": st["posture"],
                    "hit": st["hit"],
                    "nn": st["nn"],
                }
                for h, st in self.hand.items()
            },
        }
        self._disconnect()
        try:
            self.session_log.close(summary)
        except Exception:
            pass
        self.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    SpatialCockpitApp().mainloop()


if __name__ == "__main__":
    if os.environ.get("HOLOGRIP_SPATIAL_SMOKE") == "1":
        ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        assert ref["event_count"] == 307
        assert MODEL_PATH.is_file()
        print("SPATIAL_COCKPIT_ENTRY_OK")
    else:
        main()
