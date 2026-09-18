"""IMU-only diagnostic: hit detection + 7-zone classification on Song 2.

MIDI / ground-truth is used only as teacher labels, never as model input.
This matches the product constraint: ESP+JY901S stream + Python, no MIDI at runtime.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from song_collection_server import HitDetector, SensorPacket

CLEANED_DIR = r"C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260812_P01_song02_T1127_cleaned"
GT_CSV_PATH = os.path.join(CLEANED_DIR, "Song2_ground_truth_labels_final_0821.csv")
RAW_CSV_PATH = r"C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260812_P01_song02_raw_100hz_20260812_112101.csv"
OUT_JSON = os.path.join(CLEANED_DIR, "imu_only_product_diagnostic_0821.json")
WINDOW_HALF_MS = 80.0
MATCH_TOL_MS = 90.0
ZONE_MAP = {"小鼓": 0, "高音 Tom": 1, "中音 Tom": 2, "落地 Tom": 3, "Hi-Hat": 4, "Crash": 5, "Ride": 6}


def load_gt():
    with open(GT_CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_raw_rows():
    rows = []
    with open(RAW_CSV_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: (float(r["song_time_ms"]), r["hand"], int(r["packet_id"])))
    return rows


def replay_hit_detector(raw_rows):
    detectors = {"L": HitDetector(), "R": HitDetector()}
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
        ev = detectors[hand].add_packet(
            packet,
            float(row["cal_yaw_deg"]),
            float(row["cal_pitch_deg"]),
        )
        if ev:
            detections.append({
                "hand": hand,
                "song_time_ms": song_ms,
                "peak_accel_g": ev["peak_accel_g"],
                "feature_yaw_deg": ev["feature_yaw_deg"],
                "feature_pitch_deg": ev["feature_pitch_deg"],
                "swing_depth_deg": ev["swing_depth_deg"],
            })
    return detections


def match_detections(gt_rows, detections):
    gt_single = [g for g in gt_rows if g["final_hand"] in ("L", "R")]
    used = set()
    matched = []
    misses = []
    for g in gt_single:
        t = float(g["csv_center_ms"])
        hand = g["final_hand"]
        best_i = None
        best_dt = None
        for i, d in enumerate(detections):
            if i in used or d["hand"] != hand:
                continue
            dt = abs(d["song_time_ms"] - t)
            if dt <= MATCH_TOL_MS and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_i = i
        if best_i is None:
            misses.append(g)
        else:
            used.add(best_i)
            matched.append({"gt": g, "det": detections[best_i], "dt_ms": best_dt})
    extras = [d for i, d in enumerate(detections) if i not in used]
    n_gt = len(gt_single)
    n_det = len(detections)
    recall = len(matched) / n_gt if n_gt else 0.0
    precision = len(matched) / n_det if n_det else 0.0
    return {
        "gt_single": n_gt,
        "detections": n_det,
        "matched": len(matched),
        "misses": len(misses),
        "extras": len(extras),
        "recall": recall,
        "precision": precision,
        "miss_by_drum": dict(Counter(g["drum"] for g in misses)),
        "extra_by_hand": dict(Counter(d["hand"] for d in extras)),
        "mean_match_dt_ms": float(np.mean([m["dt_ms"] for m in matched])) if matched else None,
    }


def slice_hand(samples, start_t, end_t):
    return [s for s in samples if start_t <= s["song_time_ms"] <= end_t]


def window_features(slc, center_t):
    if not slc:
        return None
    mags = np.array([s["mag"] for s in slc])
    axs = np.array([s["ax"] for s in slc])
    ays = np.array([s["ay"] for s in slc])
    azs = np.array([s["az"] for s in slc])
    yaws = np.array([s["yaw"] for s in slc])
    pitches = np.array([s["pitch"] for s in slc])
    rolls = np.array([s["roll"] for s in slc])
    nearest = min(slc, key=lambda s: abs(s["song_time_ms"] - center_t))
    return {
        "max_mag": float(np.max(mags)),
        "mean_mag": float(np.mean(mags)),
        "std_mag": float(np.std(mags)),
        "energy": float(np.sum(np.abs(mags - 1.0))),
        "max_ax": float(np.max(np.abs(axs))),
        "max_ay": float(np.max(np.abs(ays))),
        "max_az": float(np.max(np.abs(azs))),
        "jerk_max": float(np.max(np.abs(np.diff(mags)))) if len(mags) > 1 else 0.0,
        "mean_yaw": float(np.mean(yaws)),
        "mean_pitch": float(np.mean(pitches)),
        "mean_roll": float(np.mean(rolls)),
        "std_yaw": float(np.std(yaws)),
        "std_pitch": float(np.std(pitches)),
        "center_yaw": float(nearest["yaw"]),
        "center_pitch": float(nearest["pitch"]),
        "swing_depth": float(np.max(pitches) - np.min(pitches)),
        "yaw_range": float(np.max(yaws) - np.min(yaws)),
    }


def extract_zone_dataset(gt_rows, samples_by_hand):
    imu_names = [
        "max_mag", "mean_mag", "std_mag", "energy",
        "max_ax", "max_ay", "max_az", "jerk_max",
        "mean_yaw", "mean_pitch", "mean_roll",
        "std_yaw", "std_pitch",
        "center_yaw", "center_pitch", "swing_depth", "yaw_range",
    ]
    old_names = ["center_yaw", "center_pitch", "swing_depth"]
    X_imu, X_old, y, drums, hands = [], [], [], [], []
    for row in gt_rows:
        hand = row["final_hand"]
        drum = row["drum"]
        if hand not in ("L", "R") or drum not in ZONE_MAP:
            continue
        center_t = float(row["csv_center_ms"])
        slc = slice_hand(samples_by_hand[hand], center_t - WINDOW_HALF_MS, center_t + WINDOW_HALF_MS)
        feat = window_features(slc, center_t)
        if feat is None:
            continue
        X_imu.append([feat[k] for k in imu_names])
        X_old.append([feat[k] for k in old_names])
        y.append(ZONE_MAP[drum])
        drums.append(drum)
        hands.append(hand)
    return np.array(X_imu), np.array(X_old), np.array(y), imu_names, old_names, drums, hands


def cv_table(X, y, models, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        out[name] = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "f1_macro_std": float(np.std(scores["test_f1_macro"])),
        }
    return out


def main():
    print("Loading GT + Raw CSV...")
    gt_rows = load_gt()
    raw_rows = load_raw_rows()
    samples_by_hand = {"L": [], "R": []}
    for row in raw_rows:
        h = row["hand"]
        if h in samples_by_hand:
            samples_by_hand[h].append({
                "song_time_ms": float(row["song_time_ms"]),
                "ax": float(row["ax_g"]),
                "ay": float(row["ay_g"]),
                "az": float(row["az_g"]),
                "mag": float(row["accel_magnitude_g"]),
                "yaw": float(row["cal_yaw_deg"]),
                "pitch": float(row["cal_pitch_deg"]),
                "roll": float(row["roll_deg"]),
            })

    print("Replaying HitDetector on Song 2 (no MIDI)...")
    detections = replay_hit_detector(raw_rows)
    hit_report = match_detections(gt_rows, detections)
    print(
        f"  GT single={hit_report['gt_single']}  det={hit_report['detections']}  "
        f"matched={hit_report['matched']}  recall={hit_report['recall']*100:.1f}%  "
        f"precision={hit_report['precision']*100:.1f}%"
    )
    print(f"  misses by drum: {hit_report['miss_by_drum']}")
    print(f"  extras by hand: {hit_report['extra_by_hand']}")

    print("IMU-only 7-zone classification (MIDI used only as labels)...")
    X_imu, X_old, y, imu_names, old_names, drums, hands = extract_zone_dataset(gt_rows, samples_by_hand)
    counts = dict(Counter(drums))
    print(f"  samples={len(y)}  class_counts={counts}")
    min_class = min(Counter(y).values()) if len(y) else 0
    n_splits = max(2, min(5, min_class))
    print(f"  stratified folds={n_splits} (limited by smallest class={min_class})")

    models_imu = {
        "KNN k=5 (IMU 17d)": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]),
        "SVM RBF (IMU 17d)": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0))]),
        "RF (IMU 17d)": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "MLP (IMU 17d)": Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42))]),
    }
    models_old = {
        "KNN k=5 (old 3d yaw/pitch/swing)": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]),
        "RF (old 3d)": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }
    imu_results = cv_table(X_imu, y, models_imu, n_splits)
    old_results = cv_table(X_old, y, models_old, n_splits)
    chance = 1.0 / len(set(y)) if len(y) else 0.0
    print(f"  chance-level 7-class = {chance*100:.1f}%")
    for name, r in {**old_results, **imu_results}.items():
        print(f"  [{name:34s}] Acc {r['accuracy_mean']*100:5.1f}% ±{r['accuracy_std']*100:4.1f}%  F1-macro {r['f1_macro_mean']*100:5.1f}%")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "constraint": "runtime has no MIDI; ESP streams IMU; Python detects hit then classifies zone; hand comes from HAND_ID",
        "window_half_ms": WINDOW_HALF_MS,
        "match_tol_ms": MATCH_TOL_MS,
        "hit_detector_replay": hit_report,
        "zone_class_counts": counts,
        "cv_folds": n_splits,
        "chance_level": chance,
        "old_3d_zone_cv": old_results,
        "imu_17d_zone_cv": imu_results,
        "note": "Current hologrip_song2_best_hand_classifier.joblib predicts L/R using MIDI-centered features; it is not this product path.",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT_JSON}")
    return report


if __name__ == "__main__":
    main()