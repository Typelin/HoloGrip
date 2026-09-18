"""Product path: IMU-only hit detection + 7-zone classification.

Runtime input: ESP+JY901S 100 Hz stream.
Runtime output: (hand from packet HAND_ID, drum from IMU model).
MIDI / video / GT are teacher labels only.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from song_collection_server import HitDetector, SensorPacket

CLEANED_DIR = r"C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260812_P01_song02_T1127_cleaned"
GT_CSV_PATH = os.path.join(CLEANED_DIR, "Song2_ground_truth_labels_final_0821.csv")
RAW_CSV_PATH = r"C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260812_P01_song02_raw_100hz_20260812_112101.csv"
# This is the model produced by the strict Raw CSV -> hit -> 7-zone validation.
# Keep the verified filename explicit so the field demo cannot silently load an
# older development classifier with a similar feature schema.
MODEL_PATH = os.path.join(CLEANED_DIR, "hologrip_song2_七鼓點模型_真正驗證版_0826.joblib")
META_PATH = os.path.join(CLEANED_DIR, "hologrip_song2_imu_zone_meta.json")
PRED_CSV_PATH = os.path.join(CLEANED_DIR, "Song2_product_hit_zone_predictions_0821.csv")
EVAL_JSON_PATH = os.path.join(CLEANED_DIR, "Song2_product_hit_zone_eval_0821.json")
HTML_PATH = os.path.join(CLEANED_DIR, "HoloGrip_Song2_產品驗證_有打且打哪_0821_ZH_TW.html")

WINDOW_HALF_MS = 80.0
MATCH_TOL_MS = 90.0
PEAK_LAG_MS = 30.0
# Product detector: keep collection UI defaults; only this path uses these.
# Song2 evidence: 1.7g+250ms debounce missed Crash/HH rolls; 4g+ hits were
# killed by raise/horizontal filters. 2.3g cuts ghost extras without losing
# the remaining true hits (those misses were already >4g).
PRODUCT_DETECTOR_KWARGS = {
    "mag_min": 2.3,
    "debounce_heavy_s": 0.10,
    "debounce_light_s": 0.15,
    "motion_bypass_mag": 4.0,
}
ZONE_NAMES = ["小鼓", "高音 Tom", "中音 Tom", "落地 Tom", "Hi-Hat", "Crash", "Ride"]
ZONE_MAP = {name: i for i, name in enumerate(ZONE_NAMES)}
FEATURE_NAMES = [
    "max_mag", "mean_mag", "std_mag", "energy",
    "max_ax", "max_ay", "max_az", "jerk_max",
    "mean_yaw", "mean_pitch", "mean_roll",
    "std_yaw", "std_pitch",
    "center_yaw", "center_pitch", "swing_depth", "yaw_range",
]


def load_gt(path: str = GT_CSV_PATH) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_raw_rows(path: str = RAW_CSV_PATH) -> list[dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: (float(r["song_time_ms"]), r["hand"], int(r["packet_id"])))
    return rows


def rows_to_samples(raw_rows: list[dict[str, str]]) -> dict[str, list[dict[str, float]]]:
    samples = {"L": [], "R": []}
    for row in raw_rows:
        hand = row["hand"]
        if hand not in samples:
            continue
        samples[hand].append({
            "song_time_ms": float(row["song_time_ms"]),
            "ax": float(row["ax_g"]),
            "ay": float(row["ay_g"]),
            "az": float(row["az_g"]),
            "mag": float(row["accel_magnitude_g"]),
            "yaw": float(row["cal_yaw_deg"]),
            "pitch": float(row["cal_pitch_deg"]),
            "roll": float(row["roll_deg"]),
        })
    return samples


def window_features(slc: list[dict[str, float]], center_t: float) -> Optional[list[float]]:
    if not slc:
        return None
    mags = np.array([s["mag"] for s in slc], dtype=float)
    axs = np.array([s["ax"] for s in slc], dtype=float)
    ays = np.array([s["ay"] for s in slc], dtype=float)
    azs = np.array([s["az"] for s in slc], dtype=float)
    yaws = np.array([s["yaw"] for s in slc], dtype=float)
    pitches = np.array([s["pitch"] for s in slc], dtype=float)
    rolls = np.array([s["roll"] for s in slc], dtype=float)
    nearest = min(slc, key=lambda s: abs(s["song_time_ms"] - center_t))
    return [
        float(np.max(mags)),
        float(np.mean(mags)),
        float(np.std(mags)),
        float(np.sum(np.abs(mags - 1.0))),
        float(np.max(np.abs(axs))),
        float(np.max(np.abs(ays))),
        float(np.max(np.abs(azs))),
        float(np.max(np.abs(np.diff(mags)))) if len(mags) > 1 else 0.0,
        float(np.mean(yaws)),
        float(np.mean(pitches)),
        float(np.mean(rolls)),
        float(np.std(yaws)),
        float(np.std(pitches)),
        float(nearest["yaw"]),
        float(nearest["pitch"]),
        float(np.max(pitches) - np.min(pitches)),
        float(np.max(yaws) - np.min(yaws)),
    ]


def slice_window(samples: list[dict[str, float]], center_t: float, half_ms: float = WINDOW_HALF_MS) -> list[dict[str, float]]:
    start_t = center_t - half_ms
    end_t = center_t + half_ms
    return [s for s in samples if start_t <= s["song_time_ms"] <= end_t]


def build_zone_dataset(gt_rows: list[dict[str, str]], samples_by_hand: dict[str, list[dict[str, float]]]):
    X, y = [], []
    for row in gt_rows:
        hand = row["final_hand"]
        drum = row["drum"]
        if hand not in ("L", "R") or drum not in ZONE_MAP:
            continue
        center_t = float(row["csv_center_ms"])
        feat = window_features(slice_window(samples_by_hand[hand], center_t), center_t)
        if feat is None:
            continue
        X.append(feat)
        y.append(ZONE_MAP[drum])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_zone_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
    ])
    clf.fit(X, y)
    return clf


def save_zone_model(clf: Pipeline, path: str = os.path.join(CLEANED_DIR, "hologrip_song2_imu_zone_classifier.joblib"), meta_path: str = META_PATH) -> None:
    joblib.dump(clf, path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": path,
            "window_half_ms": WINDOW_HALF_MS,
            "feature_names": FEATURE_NAMES,
            "zone_names": ZONE_NAMES,
            "teacher": "Song2 GT drum labels; MIDI never used as input",
        }, f, ensure_ascii=False, indent=2)


def load_zone_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_zone_full(clf: Pipeline, feat: list[float]) -> tuple[str, float, list[tuple[str, float]]]:
    proba_vec = clf.predict_proba([feat])[0]
    ranked = sorted(
        ((ZONE_NAMES[i], float(p)) for i, p in enumerate(proba_vec)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0], ranked[0][1], ranked


def predict_zone(clf: Pipeline, feat: list[float]) -> tuple[str, float]:
    drum, proba, _ranked = predict_zone_full(clf, feat)
    return drum, proba


def make_hit_detector(**kwargs: Any) -> HitDetector:
    return HitDetector(**kwargs)


def replay_hits(
    raw_rows: list[dict[str, str]],
    detector_kwargs: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    kwargs = PRODUCT_DETECTOR_KWARGS if detector_kwargs is None else detector_kwargs
    detectors = {"L": HitDetector(**kwargs), "R": HitDetector(**kwargs)}
    detections = []
    for row in raw_rows:
        hand = row["hand"]
        if hand not in detectors:
            continue
        song_ms = float(row["song_time_ms"])
        packet = SensorPacket(
            hand=hand,
            ax=float(row["ax_g"]),
            ay=float(row["ay_g"]),
            az=float(row["az_g"]),
            yaw=float(row["yaw_deg"]),
            pitch=float(row["pitch_deg"]),
            roll=float(row["roll_deg"]),
            packet_id=int(row["packet_id"]),
            sensor_time_ms=int(float(row["sensor_time_ms"])),
            received_time_ms=int(float(row["received_time_ms"])),
            received_monotonic=song_ms / 1000.0,
        )
        ev = detectors[hand].add_packet(packet, float(row["cal_yaw_deg"]), float(row["cal_pitch_deg"]))
        if ev:
            peak_ms = song_ms - PEAK_LAG_MS
            detections.append({
                "hand": hand,
                "trigger_ms": song_ms,
                "peak_ms": peak_ms,
                "peak_accel_g": ev["peak_accel_g"],
            })
    return detections


def classify_detections(detections, samples_by_hand, clf) -> list[dict[str, Any]]:
    out = []
    for i, det in enumerate(detections, start=1):
        center_t = det["peak_ms"]
        feat = window_features(slice_window(samples_by_hand[det["hand"]], center_t), center_t)
        if feat is None:
            drum, proba, ranked = "未知", 0.0, []
        else:
            drum, proba, ranked = predict_zone_full(clf, feat)
        out.append({
            "pred_id": i,
            "hand": det["hand"],
            "peak_ms": round(center_t, 3),
            "trigger_ms": round(det["trigger_ms"], 3),
            "peak_accel_g": det["peak_accel_g"],
            "pred_drum": drum,
            "pred_proba": round(proba, 4),
            "second_drum": ranked[1][0] if len(ranked) > 1 else "",
            "second_proba": round(ranked[1][1], 4) if len(ranked) > 1 else 0.0,
        })
    return out


def evaluate(gt_rows, preds) -> dict[str, Any]:
    gt_single = [
        {
            "event_id": int(g["event_id"]),
            "csv_center_ms": float(g["csv_center_ms"]),
            "hand": g["final_hand"],
            "drum": g["drum"],
        }
        for g in gt_rows
        if g["final_hand"] in ("L", "R")
    ]
    used = set()
    matched = []
    misses = []
    for g in gt_single:
        best_i, best_dt = None, None
        for i, p in enumerate(preds):
            if i in used or p["hand"] != g["hand"]:
                continue
            dt = abs(p["trigger_ms"] - g["csv_center_ms"])
            if dt <= MATCH_TOL_MS and (best_dt is None or dt < best_dt):
                best_dt, best_i = dt, i
        if best_i is None:
            misses.append(g)
            continue
        used.add(best_i)
        p = preds[best_i]
        matched.append({
            **g,
            "pred_drum": p["pred_drum"],
            "pred_id": p["pred_id"],
            "dt_ms": best_dt,
            "zone_ok": p["pred_drum"] == g["drum"],
        })
    extras = [p for i, p in enumerate(preds) if i not in used]
    joint_ok = [m for m in matched if m["zone_ok"]]
    n_gt = len(gt_single)
    return {
        "gt_single": n_gt,
        "predictions": len(preds),
        "matched": len(matched),
        "joint_ok": len(joint_ok),
        "zone_wrong": len(matched) - len(joint_ok),
        "misses": len(misses),
        "extras": len(extras),
        "hit_recall": len(matched) / n_gt if n_gt else 0.0,
        "hit_precision": len(matched) / len(preds) if preds else 0.0,
        "zone_acc_given_hit": (len(joint_ok) / len(matched)) if matched else 0.0,
        "product_recall": len(joint_ok) / n_gt if n_gt else 0.0,
        "miss_by_drum": dict(Counter(g["drum"] for g in misses)),
        "wrong_by_drum": dict(Counter(m["drum"] for m in matched if not m["zone_ok"])),
        "matched_rows": matched,
        "miss_rows": misses,
        "extra_rows": extras,
    }


def write_pred_csv(preds: list[dict[str, Any]], path: str = PRED_CSV_PATH) -> None:
    fields = ["pred_id", "hand", "peak_ms", "trigger_ms", "peak_accel_g", "pred_drum", "pred_proba", "second_drum", "second_proba"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(preds)


def write_eval_json(report: dict[str, Any], path: str = EVAL_JSON_PATH) -> None:
    slim = {k: v for k, v in report.items() if k not in {"matched_rows", "miss_rows", "extra_rows"}}
    slim["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)


def _hand_zh(hand: str) -> str:
    return {"L": "左手", "R": "右手"}.get(hand, hand)


def write_html(report: dict[str, Any], path: str = HTML_PATH) -> None:
    rows_html = []
    for m in report["matched_rows"]:
        ok = m["zone_ok"]
        cls = "ok" if ok else "wrong"
        label = "正確" if ok else "鼓位錯"
        rows_html.append(
            f"<tr class='{cls}'><td>{m['event_id']}</td><td>{m['csv_center_ms']:.0f}</td>"
            f"<td>{_hand_zh(m['hand'])} {m['drum']}</td>"
            f"<td>{_hand_zh(m['hand'])} {m['pred_drum']}</td>"
            f"<td>{m['dt_ms']:.0f} ms</td><td>{label}</td></tr>"
        )
    for g in report["miss_rows"]:
        rows_html.append(
            f"<tr class='miss'><td>{g['event_id']}</td><td>{g['csv_center_ms']:.0f}</td>"
            f"<td>{_hand_zh(g['hand'])} {g['drum']}</td><td>—</td><td>—</td><td>沒偵測到</td></tr>"
        )
    for p in report["extra_rows"]:
        rows_html.append(
            f"<tr class='extra'><td>—</td><td>{p['peak_ms']:.0f}</td><td>—</td>"
            f"<td>{_hand_zh(p['hand'])} {p['pred_drum']}</td><td>—</td><td>多偵測</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>HoloGrip 產品驗證：有打且打哪</title>
<style>
body{{margin:0;background:#0b1020;color:#e2e8f0;font-family:"Microsoft JhengHei UI",sans-serif}}
.wrap{{padding:24px 28px 48px}}
h1{{margin:0 0 8px;font-size:22px}}
.sub{{color:#94a3b8;margin-bottom:18px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px 16px;min-width:160px}}
.card b{{display:block;font-size:28px;color:#38bdf8}}
.card span{{color:#94a3b8;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 8px;border-bottom:1px solid #1f2937;text-align:left}}
th{{color:#94a3b8;font-weight:600}}
tr.ok td{{color:#86efac}}
tr.wrong td{{color:#fcd34d}}
tr.miss td{{color:#fca5a5}}
tr.extra td{{color:#94a3b8}}
</style></head><body><div class="wrap">
<h1>產品輸出驗證：左手／右手打哪一顆</h1>
<p class="sub">輸入只有手套 IMU。MIDI 只當老師，沒進模型。對齊容差 ±{MATCH_TOL_MS:.0f} ms。</p>
<div class="cards">
<div class="card"><b>{report['product_recall']*100:.1f}%</b><span>完整產品召回<br>手+鼓都對 / 真值</span></div>
<div class="card"><b>{report['hit_recall']*100:.1f}%</b><span>打擊判斷召回<br>有打到 / 真值</span></div>
<div class="card"><b>{report['hit_precision']*100:.1f}%</b><span>打擊判斷精確率<br>對到真值 / 系統輸出</span></div>
<div class="card"><b>{report['zone_wrong']}</b><span>打到了但鼓位錯</span></div>
<div class="card"><b>{report['misses']}</b><span>漏打</span></div>
<div class="card"><b>{report['extras']}</b><span>多偵測</span></div>
<div class="card"><b>{report['joint_ok']}/{report['gt_single']}</b><span>完整正確筆數</span></div>
</div>
<table><thead><tr><th>#</th><th>時間 ms</th><th>真值</th><th>系統輸出</th><th>時間差</th><th>結果</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


@dataclass
class LiveHandState:
    detector: HitDetector
    samples: deque
    pending_peaks: list[dict[str, float]]

    def __init__(self) -> None:
        self.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
        self.samples = deque(maxlen=250)
        self.pending_peaks = []
        self.last_packet: Optional[SensorPacket] = None


class LiveRecognizer:
    """HitDetector + delayed ±80 ms window + zone model. Hand comes from packet."""

    def __init__(self, clf: Pipeline, require_calibration: bool = False):
        self.clf = clf
        self.hands = {"L": LiveHandState(), "R": LiveHandState()}
        self._lock = threading.Lock()
        self.require_calibration = require_calibration
        self.calibrated_hands: set[str] = set()
        self.generation = 0

    def calibrate(self) -> list[str]:
        done = []
        with self._lock:
            self.generation += 1
            for hand, state in self.hands.items():
                pkt = state.last_packet
                if pkt is None:
                    continue
                # A fresh detector makes repeated zeroing absolute, not additive,
                # and discards motion history from the previous coordinate frame.
                state.detector = HitDetector(**PRODUCT_DETECTOR_KWARGS)
                state.detector.calibrate(pkt)
                state.samples.clear()
                state.pending_peaks.clear()
                self.calibrated_hands.add(hand)
                done.append(hand)
        return done

    def push(self, packet: SensorPacket) -> list[dict[str, Any]]:
        with self._lock:
            state = self.hands.get(packet.hand)
            if state is None:
                return []
            if not all(math.isfinite(v) for v in (
                packet.ax, packet.ay, packet.az, packet.yaw, packet.pitch,
                packet.roll, packet.received_monotonic,
            )):
                return []
            state.last_packet = packet
            if self.require_calibration and packet.hand not in self.calibrated_hands:
                return []
            cal_yaw, cal_pitch = state.detector.calibrated(packet)
            mag = math.sqrt(packet.ax ** 2 + packet.ay ** 2 + packet.az ** 2)
            t_ms = packet.received_monotonic * 1000.0
            state.samples.append({
                "song_time_ms": t_ms,
                "ax": packet.ax,
                "ay": packet.ay,
                "az": packet.az,
                "mag": mag,
                "yaw": cal_yaw,
                "pitch": cal_pitch,
                "roll": packet.roll,
            })
            ev = state.detector.add_packet(packet, cal_yaw, cal_pitch)
            if ev is not None:
                state.pending_peaks.append({
                    "peak_ms": t_ms - PEAK_LAG_MS,
                    "peak_accel_g": ev["peak_accel_g"],
                })
            ready = []
            still = []
            now_ms = t_ms
            for item in state.pending_peaks:
                peak_ms = item["peak_ms"]
                if now_ms >= peak_ms + WINDOW_HALF_MS:
                    feat = window_features(slice_window(list(state.samples), peak_ms), peak_ms)
                    if feat is None:
                        if now_ms < peak_ms + WINDOW_HALF_MS + 250.0:
                            still.append(item)
                        continue
                    drum, proba = predict_zone(self.clf, feat)
                    ready.append({
                        "generation": self.generation,
                        "hand": packet.hand,
                        "peak_ms": peak_ms,
                        "pred_drum": drum,
                        "pred_proba": proba,
                        "peak_accel_g": item["peak_accel_g"],
                    })
                else:
                    still.append(item)
            state.pending_peaks = still
            return ready
