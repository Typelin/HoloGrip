"""Causal drum-only feature families for fixed external hit candidates.

Only Raw packets and a frozen model are accepted. No MIDI, expected drum,
label-derived pose anchor, protocol calibration or whole-take statistics.
Sensor-clock resampling is bounded by the same available packet as host-clock
resampling. Short acceleration integrals are features, not physical positions.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

import trajectory_runtime_456_1016 as legacy
import drumhead_zero_runtime_1019 as old_rotation
import causal_hit_detector_1016 as detector

CONTRACT = "hologrip_pretest_zone_trajectory_1019_v1"
KINDS = ["legacy348", "cyclic248", "rotation266", "pose_sensor", "multi_sensor", "multi_host", "motion_sensor", "long_sensor"]


def feature(prepared, peak, kind):
    if kind == "legacy348":
        return legacy.feature(prepared, peak, "gravity_world_shape")
    if kind in ["cyclic248", "rotation266"]:
        return old_rotation.feature(prepared, peak, "cyclic_zero" if kind == "cyclic248" else "two_axis_rotation")
    hand = peak["hand"]
    all_rows, ts = prepared[hand]["rows"], prepared[hand]["times"]
    center = peak["peak_ms"]
    available = min(int(peak["available_index"]), len(all_rows)-1)
    center_index = min(available, int(np.argmin(np.abs(ts[:available+1]-center))))
    before = 400 if kind == "long_sensor" else 240
    lo = max(0, int(np.searchsorted(ts, center-before-100, side="left")))
    hi = min(available+1, int(np.searchsorted(ts, center+80, side="right")))
    rows = all_rows[lo:hi]
    if len(rows) < 6:
        return None
    clock = np.array([r["t"]-center for r in rows]) if kind == "multi_host" else np.array(
        [r["sensor_t"]-all_rows[center_index]["sensor_t"] for r in rows]
    )
    grid = np.arange(-before, 81., 10.)
    accel = np.array([[r["ax"], r["ay"], r["az"]] for r in rows])
    angles = np.degrees(np.unwrap(np.radians([[r["yaw"], r["absolute_pitch"], r["roll"]] for r in rows]), axis=0))
    # Duplicate PC timestamps are not rewritten in Raw. As in the legacy model,
    # np.interp is a representation of the already-observed feature window.
    body = np.column_stack([np.interp(grid, clock, accel[:, j]) for j in range(3)])
    euler = np.column_stack([np.interp(grid, clock, angles[:, j]) for j in range(3)])
    zero_pitch = all_rows[center_index]["absolute_pitch"]-all_rows[center_index]["pitch"]
    cal = euler.copy()
    cal[:, 1] -= zero_pitch
    r = Rotation.from_euler("ZYX", euler, degrees=True)
    rz = Rotation.from_euler("Y", -zero_pitch, degrees=True)*r
    ci = int(np.argmin(np.abs(grid)))
    world_relative = (r*r[ci].inv()).as_rotvec()
    body_relative = (r[ci].inv()*r).as_rotvec()
    linear = r.apply(body)-np.array([0., 0., 1.])
    gravity_body = r.inv().apply(np.tile([0., 0., 1.], (len(grid), 1)))
    vertical = np.sum(body*gravity_body, axis=1)
    horizontal = np.sqrt(np.maximum(0., np.sum(body**2, axis=1)-vertical**2))
    mag = np.linalg.norm(body, axis=1)
    scale = max(1., float(np.linalg.norm(accel, axis=1).max()))
    snap_times = np.array([-160., -120., -80., -40., 0., 40., 80.])
    snap = np.array([int(np.argmin(np.abs(grid-t))) for t in snap_times])
    cyclic = lambda x: np.column_stack([np.sin(np.radians(x)), np.cos(np.radians(x))]).ravel()
    pose = np.r_[
        cyclic(euler[snap]), cyclic(cal[snap]),
        r[snap].as_matrix()[:, :, :2].ravel(), rz[snap].as_matrix()[:, :, :2].ravel(),
        float(hand == "R"),
    ]
    summary = [np.log1p(scale), zero_pitch/90., float(hand == "R")]
    for start, stop in [(-before, -120), (-120, -40), (-40, 40), (40, 80)]:
        selected = (grid >= start) & (grid <= stop)
        b, l, m = body[selected], linear[selected], mag[selected]
        summary.extend(np.r_[b.mean(axis=0), b.std(axis=0), l.mean(axis=0), l.std(axis=0),
                             np.percentile(m, [10, 50, 90]), np.mean(np.abs(m-1))])
    summary = np.array(summary)
    wave = np.column_stack([body/scale, linear/scale, world_relative/np.pi,
                            body_relative/np.pi, (vertical-1)/scale, horizontal/scale, (mag-1)/scale])
    # Integrals over <=400ms: motion-shape descriptors, not measured velocity
    # or sensor drift compensation. Starting integration at zero is explicit.
    integral = np.cumsum(linear, axis=0)*.01/scale
    dynamics = np.r_[wave.ravel(), integral[snap].ravel(), summary]
    if kind == "pose_sensor":
        value = np.r_[pose, summary]
    elif kind == "motion_sensor":
        value = dynamics
    else:
        value = np.r_[pose, dynamics]
    assert np.isfinite(value).all()
    return value.astype(np.float32)


def probabilities(bundle, vectors, hands):
    x = np.asarray(vectors, dtype=np.float32)
    classes = bundle["classes"]
    result = np.zeros((len(x), len(classes)))
    if bundle["mode"] == "shared":
        model = bundle["models"]["shared"]
        p = model.predict_proba(x)
        for i, name in enumerate(model.classes_):
            result[:, classes.index(str(name))] = p[:, i]
    else:
        for hand in "LR":
            idx = np.flatnonzero(np.array(hands) == hand)
            if not len(idx):
                continue
            model = bundle["models"][hand]
            p = model.predict_proba(x[idx])
            for i, name in enumerate(model.classes_):
                result[idx, classes.index(str(name))] = p[:, i]
    return result


def classify(bundle, peaks, vectors):
    if not peaks:
        return []
    p = probabilities(bundle, vectors, [r["hand"] for r in peaks])
    return [dict(**{k:v for k,v in row.items() if k != "feature"},
                 drum=bundle["classes"][int(q.argmax())], model_score=float(q.max()),
                 model_margin=float(np.sort(q)[-1]-np.sort(q)[-2])) for row, q in zip(peaks, p)]


def predict(bundle, hands, peaks):
    prepared = legacy.prepare(hands)
    selected, vectors = [], []
    for peak in peaks:
        value = feature(prepared, peak, bundle["kind"])
        if value is not None:
            selected.append(peak)
            vectors.append(value)
    return classify(bundle, selected, vectors)


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
    hands = legacy.read_raw(args.raw)
    peaks, diag = detector.detect(hands, bundle["detector_config"])
    with threadpool_limits(limits=1):
        out = predict(bundle, hands, peaks)
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy.rt.write_csv(target, out)
    meta = dict(raw_sha256=legacy.rt.sha256(args.raw), model_sha256=legacy.rt.sha256(args.model),
                output_sha256=legacy.rt.sha256(target), runtime_sha256=legacy.rt.sha256(__file__),
                raw_rows=sum(map(len, hands.values())), predictions=len(out),
                midi_or_labels_read_during_inference=False, test_answer_reference=False,
                expected_drum_input=False, additional_calibration=False, abstention=False,
                external_detector_changed=False, feature_kind=bundle["kind"], diagnostics=diag)
    target.with_suffix(".manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
