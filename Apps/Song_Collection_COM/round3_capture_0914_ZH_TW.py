"""Round 3 evidence capture for HoloGrip.

This module has no GUI and does not own serial ports. It only records the
validation evidence stream and computes static preflight health metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import queue
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

EXPECTED_MODEL_SHA256 = "c2f9ce16750887011219a6f46fee5095c2e4f50c8df3f94548b3c5d806b2f660"
VALIDATION_ROOT = PROJECT_ROOT / "Data" / "Validation" / "Round3"

RAW_FIELDS = [
    "session_id", "session_time_ms", "host_time_ms", "phase", "port", "hand",
    "packet_id", "sensor_time_ms", "calibrated", "ax_g", "ay_g", "az_g",
    "accel_magnitude_g", "yaw_deg", "pitch_deg", "roll_deg",
    "cal_yaw_deg", "cal_pitch_deg", "zero_packet",
]
PRED_FIELDS = [
    "session_id", "prediction_id", "phase", "peak_session_ms", "hand",
    "pred_drum", "pred_proba", "second_drum", "second_proba", "margin", "peak_accel_g", "model_sha256",
]
MARKER_FIELDS = ["session_id", "session_time_ms", "host_time_ms", "marker", "note"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_session_id(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return text.strip("_")[:96] or f"R3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _unwrap(values: list[float]) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=float))))


def compute_preflight(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        return {"status": "WARN", "n": len(samples), "reasons": ["樣本不足"]}
    s = sorted(samples, key=lambda x: float(x["mono"]))
    duration = max(1e-6, s[-1]["mono"] - s[0]["mono"])
    hz = (len(s) - 1) / duration
    y = _unwrap([x["cal_yaw"] for x in s])
    p = np.asarray([x["cal_pitch"] for x in s], dtype=float)
    k = max(1, len(s) // 5)
    yaw_drift = float(abs(np.mean(y[-k:]) - np.mean(y[:k])))
    pitch_drift = float(abs(np.mean(p[-k:]) - np.mean(p[:k])))
    yaw_span = float(np.max(y) - np.min(y))
    pitch_span = float(np.max(p) - np.min(p))
    mag_median = float(statistics.median(float(x["mag"]) for x in s))
    zero_ratio = sum(bool(x["zero"]) for x in s) / len(s)
    ids = [int(x["packet_id"]) for x in s if int(x["packet_id"]) >= 0]
    missing = sum(max(0, b - a - 1) for a, b in zip(ids, ids[1:]))
    resets = sum(1 for a, b in zip(ids, ids[1:]) if b <= a)
    loss = missing / max(1, len(ids) + missing)
    reasons = []
    if not 80 <= hz <= 120: reasons.append(f"頻率 {hz:.1f}Hz 不在 80–120Hz")
    if zero_ratio > 0.005: reasons.append(f"全零封包 {zero_ratio*100:.2f}%")
    if loss > 0.02: reasons.append(f"packet gap 約 {loss*100:.2f}%")
    if resets: reasons.append(f"packet_id 重置/倒退 {resets} 次")
    if yaw_drift > 5: reasons.append(f"Yaw 靜止漂移 {yaw_drift:.1f}°")
    if pitch_drift > 4: reasons.append(f"Pitch 靜止漂移 {pitch_drift:.1f}°")
    if not 0.70 <= mag_median <= 1.30: reasons.append(f"靜止加速度 {mag_median:.2f}g 異常")
    return {
        "status": "PASS" if not reasons else "WARN", "n": len(s), "duration_s": duration,
        "hz": hz, "zero_ratio": zero_ratio, "packet_loss_ratio": loss, "packet_id_resets": resets,
        "median_mag_g": mag_median, "yaw_span_deg": yaw_span,
        "yaw_drift_deg": yaw_drift, "pitch_span_deg": pitch_span,
        "pitch_drift_deg": pitch_drift, "reasons": reasons,
    }


class ValidationRecorder:
    STOP = object()
    def __init__(self) -> None:
        self.active = False
        self.session_id = ""
        self.session_dir: Optional[Path] = None
        self.start_mono = 0.0
        self.q: queue.Queue[Any] = queue.Queue(maxsize=250_000)
        self.thread: Optional[threading.Thread] = None
        self.raw_written = self.pred_written = self.marker_written = self.dropped = 0
        self.pred_id = 0
        self.lock = threading.Lock()

    def start(self, name: str, model_path: Path, model_sha: str) -> Path:
        if self.active: raise RuntimeError("session already active")
        self.session_id = safe_session_id(name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = VALIDATION_ROOT / f"{self.session_id}_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.start_mono = time.monotonic()
        self.q = queue.Queue(maxsize=250_000)
        self.raw_written = self.pred_written = self.marker_written = self.dropped = 0
        self.pred_id = 0
        manifest = {
            "session_id": self.session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Round 3 new-player/new-song frozen-model validation",
            "model_path": str(model_path), "model_sha256": model_sha,
            "expected_model_sha256": EXPECTED_MODEL_SHA256, "model_frozen": True,
            "runtime": "Raw 100Hz -> HitDetector -> +/-80ms -> 17 features -> MLP(64,32) -> 7 drums",
            "notes": ["MIDI is not used by live inference.", "This GUI owns both COM ports and records raw 100 Hz."],
        }
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.active = True
        self.thread = threading.Thread(target=self._writer, daemon=True, name="round3-writer")
        self.thread.start()
        self.marker("SESSION_START", "驗證場次開始")
        return self.session_dir

    def ms(self, mono: Optional[float] = None) -> float:
        return max(0.0, ((mono if mono is not None else time.monotonic()) - self.start_mono) * 1000.0)

    def put(self, kind: str, row: dict[str, Any]) -> None:
        if not self.active: return
        try: self.q.put_nowait((kind, row))
        except queue.Full: self.dropped += 1

    def raw(self, row: dict[str, Any]) -> None: self.put("raw", row)

    def pred(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.pred_id += 1
            row = dict(row); row["prediction_id"] = self.pred_id
        self.put("pred", row)

    def marker(self, marker: str, note: str = "") -> None:
        if not self.active: return
        self.put("marker", {"session_id": self.session_id, "session_time_ms": round(self.ms(), 3),
                            "host_time_ms": int(time.time()*1000), "marker": marker, "note": note})

    def stop(self) -> Optional[Path]:
        if not self.active: return self.session_dir
        self.marker("SESSION_STOP", "驗證場次停止")
        self.active = False
        self.q.put(self.STOP)
        if self.thread: self.thread.join(timeout=20)
        return self.session_dir

    def _writer(self) -> None:
        assert self.session_dir is not None
        raw_p = self.session_dir / "raw_100hz.csv"
        pred_p = self.session_dir / "predictions.csv"
        mark_p = self.session_dir / "markers.csv"
        with raw_p.open("w", newline="", encoding="utf-8-sig") as rf, \
             pred_p.open("w", newline="", encoding="utf-8-sig") as pf, \
             mark_p.open("w", newline="", encoding="utf-8-sig") as mf:
            rw = csv.DictWriter(rf, fieldnames=RAW_FIELDS, extrasaction="ignore")
            pw = csv.DictWriter(pf, fieldnames=PRED_FIELDS, extrasaction="ignore")
            mw = csv.DictWriter(mf, fieldnames=MARKER_FIELDS, extrasaction="ignore")
            rw.writeheader(); pw.writeheader(); mw.writeheader()
            pending = 0
            while True:
                item = self.q.get()
                if item is self.STOP:
                    rf.flush(); pf.flush(); mf.flush(); break
                kind, row = item
                if kind == "raw": rw.writerow(row); self.raw_written += 1
                elif kind == "pred": pw.writerow(row); self.pred_written += 1
                elif kind == "marker": mw.writerow(row); self.marker_written += 1
                pending += 1
                if pending >= 200:
                    rf.flush(); pf.flush(); mf.flush(); pending = 0
