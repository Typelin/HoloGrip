"""Drum-only absolute-world acceleration and relative-rotation trajectories.

World rotation uses ORIGINAL yaw, not zero-subtracted yaw. The world relative
rotation R(t) R(center)^-1 cancels a constant sensor mounting rotation under
the ideal orientation model. Sensor estimates still have real measurement
errors; no practical invariance or future accuracy is asserted by this code.
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from threadpoolctl import threadpool_limits

import pretest_zone_trajectory_runtime_1019 as base

CONTRACT = "hologrip_absolute_world_motion_1019_v1"
KINDS = ["world_motion", "world_pose", "world_hybrid", "world_long"]


def enrich(hands, path):
    counts = {h:0 for h in "LR"}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            h = r["hand"]
            hands[h][counts[h]]["absolute_yaw"] = float(r["yaw_deg"])
            counts[h] += 1
    return hands


def read_raw(path):
    return enrich(base.legacy.read_raw(path), path)


def world_components(rotation, acceleration, center_index):
    linear = rotation.apply(acceleration)-np.array([0., 0., 1.])
    relative = (rotation*rotation[center_index].inv()).as_rotvec()
    velocity = np.vstack([np.zeros(3), (rotation[1:]*rotation[:-1].inv()).as_rotvec()/.01])
    return linear, relative, velocity


def feature(prepared, peak, kind):
    rows = prepared[peak["hand"]]["rows"]
    times = prepared[peak["hand"]]["times"]
    available = min(int(peak["available_index"]), len(rows)-1)
    idx = peak.get("peak_index")
    if idx is None or not 0 <= int(idx) <= available or abs(rows[int(idx)]["t"]-peak["peak_ms"]) > .01:
        delta = np.abs(times[:available+1]-peak["peak_ms"])
        ties = np.flatnonzero(delta == delta.min())
        idx = max(ties, key=lambda i:rows[int(i)]["mag"])
    idx = int(idx)
    before = 400 if kind == "world_long" else 240
    sensor_center = rows[idx]["sensor_t"]
    lo = bisect_left(rows, sensor_center-before-30, 0, available+1, key=lambda r:r["sensor_t"])
    hi = bisect_right(rows, sensor_center+80, 0, available+1, key=lambda r:r["sensor_t"])
    selected = rows[lo:hi]
    if len(selected) < 6:
        return None
    clock = np.array([r["sensor_t"]-sensor_center for r in selected])
    _, unique = np.unique(clock, return_index=True)
    selected = [selected[int(i)] for i in unique]
    clock = clock[unique]
    if len(clock) < 6 or np.any(np.diff(clock) <= 0):
        return None
    grid = np.arange(-before, 81., 10.)
    clipped = np.clip(grid, clock[0], clock[-1])
    raw_accel = np.array([[r["ax"], r["ay"], r["az"]] for r in selected])
    angles = np.array([[r["absolute_yaw"], r["absolute_pitch"], r["roll"]] for r in selected])
    body = np.column_stack([np.interp(grid, clock, raw_accel[:, j]) for j in range(3)])
    rot = Slerp(clock, Rotation.from_euler("ZYX", angles, degrees=True))(clipped)
    center = int(np.argmin(np.abs(grid)))
    linear, relative, angular_velocity = world_components(rot, body, center)
    mag = np.linalg.norm(body, axis=1)
    scale = max(1., float(np.linalg.norm(raw_accel, axis=1).max()))
    snap = np.array([int(np.argmin(abs(grid-t))) for t in [-160, -120, -80, -40, 0, 40, 80]])
    wave = np.column_stack([linear/scale, relative/np.pi, angular_velocity/50., (mag-1)/scale])
    integral = np.cumsum(linear, axis=0)*.01/scale
    scalar = [np.log1p(scale), float(peak["hand"] == "R")]
    for first, last in [(-before, -120), (-120, -40), (-40, 40), (40, 80)]:
        mask = (grid >= first) & (grid <= last)
        a, w, m = linear[mask], angular_velocity[mask], mag[mask]
        scalar.extend(np.r_[a.mean(axis=0)/scale, a.std(axis=0)/scale,
                            w.mean(axis=0)/50., w.std(axis=0)/50.,
                            np.percentile(m, [10, 50, 90])/scale,
                            np.abs(m-1).sum()/scale])
    motion = np.r_[wave.ravel(), integral[snap].ravel(), scalar]
    # Absolute pose is a separate, explicitly non-mount-invariant view.
    pose = np.r_[rot[snap].as_matrix()[:, :, :2].ravel(), scalar]
    if kind in ["world_motion", "world_long"]:
        value = motion
    elif kind == "world_pose":
        value = pose
    else:
        value = np.r_[motion, rot[snap].as_matrix()[:, :, :2].ravel(), body[snap].ravel()/scale]
    assert np.isfinite(value).all()
    return value.astype(np.float32)


def predict(bundle, hands, peaks):
    prepared = base.legacy.prepare(hands)
    selected, vectors = [], []
    for peak in peaks:
        x = feature(prepared, peak, bundle["kind"])
        if x is not None:
            selected.append(peak)
            vectors.append(x)
    return base.classify(bundle, selected, vectors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "model", "output"]:
        ap.add_argument("--"+name, required=True)
    args = ap.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Refuse overwrite")
    bundle = joblib.load(args.model)
    assert bundle["runtime_contract"] == CONTRACT
    hands = read_raw(args.raw)
    peaks, diagnostics = base.detector.detect(hands, bundle["detector_config"])
    with threadpool_limits(limits=1):
        predictions = predict(bundle, hands, peaks)
    target.parent.mkdir(parents=True, exist_ok=True)
    base.legacy.rt.write_csv(target, predictions)
    meta = dict(raw_sha256=base.legacy.rt.sha256(args.raw), model_sha256=base.legacy.rt.sha256(args.model),
                output_sha256=base.legacy.rt.sha256(target), runtime_sha256=base.legacy.rt.sha256(__file__),
                predictions=len(predictions), diagnostics=diagnostics,
                midi_or_labels_read_during_inference=False, expected_drum_input=False,
                additional_calibration=False, answer_derived_reference=False, abstention=False,
                original_yaw_used_for_world=True, future_packets_beyond_emission=False,
                external_detector_changed=False)
    target.with_suffix(".manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
