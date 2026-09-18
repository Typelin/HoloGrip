from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
APP = ROOT / "Apps" / "Song_Collection_COM"
DERIVED = ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
RAW = ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT = DERIVED / "Song2_ground_truth_labels_final_0821.csv"
OUT = LAB / "spatial_reference_0826.json"
VIDEO_REF_DIR = LAB / "video_references"

sys.path.insert(0, str(APP))
from product_hit_and_zone import FEATURE_NAMES, load_raw_rows, rows_to_samples, slice_window, window_features

# Synthetic 3D viewer was previously corrected to show Hi-Hat/Crash on one side
# and Ride/Floor Tom on the other. The old recording was shot facing the drummer,
# so the cockpit view mirrors X back to the drummer's own left/right.
OLD_VIEWER_DRUMS = {
    "小鼓":     {"x": 0.22, "y": 0.74, "z": 0.36},
    "高音 Tom": {"x": 0.16, "y": 0.98, "z": 0.62},
    "中音 Tom": {"x":-0.16, "y": 0.98, "z": 0.62},
    "落地 Tom": {"x":-0.48, "y": 0.72, "z": 0.36},
    "Hi-Hat":   {"x": 0.56, "y": 0.92, "z": 0.42},
    "Crash":    {"x": 0.54, "y": 1.26, "z": 0.68},
    "Ride":     {"x":-0.56, "y": 1.20, "z": 0.58},
}

DRUM_ORDER = ["Hi-Hat", "Crash", "小鼓", "高音 Tom", "中音 Tom", "Ride", "落地 Tom"]
REF_FILES = {
    "Hi-Hat": "hihat.jpg",
    "Crash": "crash.jpg",
    "小鼓": "snare.jpg",
    "高音 Tom": "high_tom.jpg",
    "中音 Tom": "mid_tom.jpg",
    "Ride": "ride.jpg",
    "落地 Tom": "floor_tom.jpg",
}


def q(vals, p):
    return float(np.percentile(np.asarray(vals, dtype=float), p))


def main():
    raw_rows = load_raw_rows(str(RAW))
    samples = rows_to_samples(raw_rows)
    with GT.open("r", encoding="utf-8") as f:
        gt = list(csv.DictReader(f))

    iy = FEATURE_NAMES.index("center_yaw")
    ip = FEATURE_NAMES.index("center_pitch")
    rows = []

    for ev in gt:
        hand = ev.get("final_hand", "")
        drum = ev.get("drum", "")
        if hand not in samples or drum not in OLD_VIEWER_DRUMS:
            continue
        center = float(ev["csv_center_ms"])
        feat = window_features(slice_window(samples[hand], center, 80.0), center)
        if feat is None:
            continue
        rows.append({
            "event_id": int(ev["event_id"]),
            "drum": drum,
            "hand": hand,
            "midi_time_ms": float(ev["midi_time_ms"]),
            "csv_center_ms": center,
            "video_center_ms": float(ev["video_center_ms"]),
            "yaw": float(feat[iy]),
            "pitch": float(feat[ip]),
            "feature": [float(x) for x in feat],
        })

    by = defaultdict(list)
    for r in rows:
        by[r["drum"]].append(r)

    classes = {}
    for drum in DRUM_ORDER:
        rr = by[drum]
        yaws = [r["yaw"] for r in rr]
        pitches = [r["pitch"] for r in rr]
        old = OLD_VIEWER_DRUMS[drum]
        classes[drum] = {
            "n": len(rr),
            "yaw_median": float(np.median(yaws)),
            "pitch_median": float(np.median(pitches)),
            "yaw_p10": q(yaws, 10),
            "yaw_p90": q(yaws, 90),
            "pitch_p10": q(pitches, 10),
            "pitch_p90": q(pitches, 90),
            "yaw_p05": q(yaws, 5),
            "yaw_p95": q(yaws, 95),
            "pitch_p05": q(pitches, 5),
            "pitch_p95": q(pitches, 95),
            "cockpit_x": -float(old["x"]),
            "cockpit_forward": float(old["y"]),
            "cockpit_height": float(old["z"]),
        }

    # Pick one real event nearest each class angle centre and extract the
    # synchronized video frame. The camera faced the drummer, therefore the
    # frame is mirrored back to the drummer's own left/right viewpoint.
    VIDEO_REF_DIR.mkdir(parents=True, exist_ok=True)
    video_path = str(DERIVED / "S20260812_P01_song02_T1127_video_h264.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    for drum in DRUM_ORDER:
        c = classes[drum]
        rr = by[drum]
        sy = max(float(np.std([r["yaw"] for r in rr])), 5.0)
        sp = max(float(np.std([r["pitch"] for r in rr])), 5.0)
        rep = min(
            rr,
            key=lambda r: ((r["yaw"] - c["yaw_median"]) / sy) ** 2
                          + ((r["pitch"] - c["pitch_median"]) / sp) ** 2,
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, float(rep["video_center_ms"]))
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            target_w = 720
            if w > target_w:
                frame = cv2.resize(frame, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
            ref_name = REF_FILES[drum]
            cv2.imwrite(str(VIDEO_REF_DIR / ref_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            c["video_ref_file"] = ref_name
            c["video_ref_event_id"] = rep["event_id"]
            c["video_ref_ms"] = rep["video_center_ms"]
    cap.release()

    # Scale yaw/pitch for the diagnostic 2D nearest-neighbour map.
    ys = np.asarray([r["yaw"] for r in rows], dtype=float)
    ps = np.asarray([r["pitch"] for r in rows], dtype=float)
    scale = {
        "yaw_center": float(np.median(ys)),
        "pitch_center": float(np.median(ps)),
        "yaw_scale": float(np.std(ys) or 1.0),
        "pitch_scale": float(np.std(ps) or 1.0),
    }

    payload = {
        "source": {
            "raw": str(RAW),
            "gt": str(GT),
            "video": str(DERIVED / "S20260812_P01_song02_T1127_video_h264.mp4"),
            "old_viewer": str(DERIVED / "HoloGrip_Song2_3D鼓組空間與打擊重播_0821_ZH_TW.html"),
            "note": "video is front-facing; cockpit_x mirrors the previous viewer X for drummer viewpoint",
        },
        "feature_names": FEATURE_NAMES,
        "event_count": len(rows),
        "drum_order": DRUM_ORDER,
        "classes": classes,
        "scale": scale,
        "events": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}")
    print(f"EVENTS {len(rows)}")
    for drum in DRUM_ORDER:
        c = classes[drum]
        print(f"{drum:8s} n={c['n']:3d} yaw={c['yaw_median']:+6.1f} pitch={c['pitch_median']:+5.1f}  yaw10-90=({c['yaw_p10']:+5.1f},{c['yaw_p90']:+5.1f})")


if __name__ == "__main__":
    main()
