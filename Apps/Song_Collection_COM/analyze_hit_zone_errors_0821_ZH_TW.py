"""Split Song 2 product errors into hit-detect vs zone, then sweep levers."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_hit_and_zone import (
    CLEANED_DIR,
    MATCH_TOL_MS,
    WINDOW_HALF_MS,
    classify_detections,
    evaluate,
    load_gt,
    load_raw_rows,
    load_zone_model,
    replay_hits,
    rows_to_samples,
)

OUT_JSON = os.path.join(CLEANED_DIR, "Song2_hit_zone_error_analysis_0821.json")
HAND_ZH = {"L": "左手", "R": "右手"}


def ms_clock(ms: float) -> str:
    s = max(0.0, ms) / 1000.0
    m = int(s // 60)
    rem = s - 60 * m
    return f"{m:02d}:{rem:06.3f}"


def mag_around(samples: list[dict[str, float]], center_ms: float, half_ms: float = WINDOW_HALF_MS) -> float:
    slc = [s["mag"] for s in samples if abs(s["song_time_ms"] - center_ms) <= half_ms]
    return float(max(slc)) if slc else 0.0


def extra_reason(p: dict[str, Any], gt_single: list[dict[str, Any]], preds: list[dict[str, Any]]) -> str:
    t = p["trigger_ms"]
    mag = p["peak_accel_g"]
    first_gt = min(g["csv_center_ms"] for g in gt_single)
    last_gt = max(g["csv_center_ms"] for g in gt_single)
    if t < first_gt - MATCH_TOL_MS:
        return "曲前熱身"
    if t > last_gt + MATCH_TOL_MS:
        return "曲後殘響"
    same = [abs(t - g["csv_center_ms"]) for g in gt_single if g["hand"] == p["hand"]]
    nearest_same = min(same) if same else 1e9
    other_hits = [
        q for q in preds
        if q["hand"] != p["hand"] and abs(q["trigger_ms"] - t) <= 45.0
    ]
    if other_hits:
        other = max(other_hits, key=lambda q: q["peak_accel_g"])
        if mag < 3.5 and other["peak_accel_g"] >= 2.2 * mag:
            return "對側共振"
    prev_same = [
        q for q in preds
        if q["hand"] == p["hand"] and 0 < t - q["trigger_ms"] <= 80.0
    ]
    if prev_same:
        return "同手連發"
    if mag >= 4.0 and nearest_same > 120.0:
        return "真值沒標到的實打"
    if mag < 2.5:
        return "低g雜訊"
    return "其他多打"


def miss_reason(
    g: dict[str, Any],
    preds: list[dict[str, Any]],
    samples_by_hand: dict[str, list[dict[str, float]]],
) -> str:
    mag = mag_around(samples_by_hand[g["hand"]], g["csv_center_ms"])
    prev = [
        p for p in preds
        if p["hand"] == g["hand"] and p["trigger_ms"] < g["csv_center_ms"]
    ]
    dt_prev = (g["csv_center_ms"] - prev[-1]["trigger_ms"]) if prev else None
    prev_mag = prev[-1]["peak_accel_g"] if prev else None
    if mag <= 1.7:
        return "峰值低於1.7g"
    if dt_prev is not None:
        debounce = 150.0 if (prev_mag or 0) >= 2.5 else 250.0
        if dt_prev <= debounce + 20.0:
            return f"防彈跳吃掉({dt_prev:.0f}ms/{debounce:.0f}ms)"
    return "動作濾波擋掉"


def slim_eval(report: dict[str, Any]) -> dict[str, Any]:
    n_gt = report["gt_single"]
    n_pred = report["predictions"]
    return {
        "gt_single": n_gt,
        "predictions": n_pred,
        "matched": report["matched"],
        "joint_ok": report["joint_ok"],
        "zone_wrong": report["zone_wrong"],
        "misses": report["misses"],
        "extras": report["extras"],
        "hit_recall": round(report["hit_recall"], 4),
        "hit_precision": round(report["hit_precision"], 4),
        "zone_acc_given_hit": round(report["zone_acc_given_hit"], 4),
        "product_recall": round(report["product_recall"], 4),
        "product_ok_pct": round(100.0 * report["joint_ok"] / n_gt, 1) if n_gt else 0.0,
        "hit_detect_ok_pct": round(100.0 * report["matched"] / n_gt, 1) if n_gt else 0.0,
        "hit_precision_pct": round(100.0 * report["hit_precision"], 1),
        "zone_wrong_given_hit_pct": round(100.0 * report["zone_wrong"] / report["matched"], 1) if report["matched"] else 0.0,
        "miss_by_drum": report["miss_by_drum"],
        "wrong_by_drum": report["wrong_by_drum"],
    }


def apply_mag_floor(preds: list[dict[str, Any]], mag_min: float) -> list[dict[str, Any]]:
    return [p for p in preds if p["peak_accel_g"] > mag_min]


def suppress_sympathetic(
    preds: list[dict[str, Any]],
    window_ms: float = 45.0,
    ratio: float = 2.2,
    weak_max: float = 3.2,
) -> list[dict[str, Any]]:
    keep = [True] * len(preds)
    for i, p in enumerate(preds):
        for q in preds:
            if p["hand"] == q["hand"]:
                continue
            if abs(p["trigger_ms"] - q["trigger_ms"]) > window_ms:
                continue
            if p["peak_accel_g"] < weak_max and q["peak_accel_g"] >= ratio * p["peak_accel_g"]:
                keep[i] = False
                break
    return [p for p, ok in zip(preds, keep) if ok]


def crash_second_if_weak(preds: list[dict[str, Any]], mag_max: float) -> list[dict[str, Any]]:
    out = []
    for p in preds:
        q = dict(p)
        if q["pred_drum"] == "Crash" and q["peak_accel_g"] < mag_max and q.get("second_drum"):
            q["pred_drum"] = q["second_drum"]
            q["pred_proba"] = q.get("second_proba", 0.0)
        out.append(q)
    return out


def drop_before_first_gt(preds: list[dict[str, Any]], gt_single: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_gt = min(g["csv_center_ms"] for g in gt_single)
    return [p for p in preds if p["trigger_ms"] >= first_gt - MATCH_TOL_MS]


def confusion(matched_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in matched_rows:
        if not m["zone_ok"]:
            grid[m["drum"]][m["pred_drum"]] += 1
    return {src: dict(dst) for src, dst in grid.items()}


def mag_stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    n = len(xs)
    return {
        "n": n,
        "min": round(xs[0], 2),
        "p25": round(xs[n // 4], 2),
        "median": round(xs[n // 2], 2),
        "p75": round(xs[(3 * n) // 4], 2),
        "max": round(xs[-1], 2),
        "lt_2": sum(v < 2.0 for v in xs),
        "lt_2_5": sum(v < 2.5 for v in xs),
        "lt_3": sum(v < 3.0 for v in xs),
        "ge_4": sum(v >= 4.0 for v in xs),
    }


def current_breakdown(
    gt_rows: list[dict[str, str]],
    preds: list[dict[str, Any]],
    samples_by_hand: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    report = evaluate(gt_rows, preds)
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
    extra_rows = []
    extra_counter = Counter()
    for p in report["extra_rows"]:
        reason = extra_reason(p, gt_single, preds)
        extra_counter[reason] += 1
        extra_rows.append({
            "clock": ms_clock(p["trigger_ms"]),
            "hand": HAND_ZH[p["hand"]],
            "pred_drum": p["pred_drum"],
            "mag": round(p["peak_accel_g"], 2),
            "reason": reason,
        })
    miss_rows = []
    miss_counter = Counter()
    miss_mags = []
    for g in report["miss_rows"]:
        mag = mag_around(samples_by_hand[g["hand"]], g["csv_center_ms"])
        reason = miss_reason(g, preds, samples_by_hand)
        miss_counter[reason] += 1
        miss_mags.append(mag)
        miss_rows.append({
            "clock": ms_clock(g["csv_center_ms"]),
            "hand": HAND_ZH[g["hand"]],
            "drum": g["drum"],
            "window_max_g": round(mag, 2),
            "reason": reason,
        })
    ok_mags = []
    wrong_mags = []
    wrong_pairs = Counter()
    for m in report["matched_rows"]:
        pred = next(p for p in preds if p["pred_id"] == m["pred_id"])
        mag = pred["peak_accel_g"]
        if m["zone_ok"]:
            ok_mags.append(mag)
        else:
            wrong_mags.append(mag)
            wrong_pairs[f"{m['drum']} → {m['pred_drum']}"] += 1
    return {
        "metrics": slim_eval(report),
        "extra_reasons": dict(extra_counter),
        "miss_reasons": dict(miss_counter),
        "wrong_pairs": dict(wrong_pairs.most_common()),
        "confusion": confusion(report["matched_rows"]),
        "extra_mag": mag_stats([p["peak_accel_g"] for p in report["extra_rows"]]),
        "ok_mag": mag_stats(ok_mags),
        "wrong_mag": mag_stats(wrong_mags),
        "miss_window_mag": mag_stats(miss_mags),
        "extra_rows": extra_rows,
        "miss_rows": miss_rows,
    }


def eval_preds(gt_rows: list[dict[str, str]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    return slim_eval(evaluate(gt_rows, preds))


def main() -> None:
    print("load gt/raw/model")
    gt = load_gt()
    raw = load_raw_rows()
    samples = rows_to_samples(raw)
    clf = load_zone_model()
    gt_single = [g for g in gt if g["final_hand"] in ("L", "R")]

    print("replay baseline detector")
    dets = replay_hits(raw, detector_kwargs={
        "mag_min": 1.7,
        "debounce_heavy_s": 0.15,
        "debounce_light_s": 0.25,
        "motion_bypass_mag": 99.0,
    })
    preds = classify_detections(dets, samples, clf)
    current = current_breakdown(gt, preds, samples)
    print("current", json.dumps(current["metrics"], ensure_ascii=False))
    print("extras", json.dumps(current["extra_reasons"], ensure_ascii=False))
    print("misses", json.dumps(current["miss_reasons"], ensure_ascii=False))
    print("wrong", json.dumps(current["wrong_pairs"], ensure_ascii=False))

    postfilters = []
    for mag_min in (1.7, 1.9, 2.1, 2.3, 2.5):
        filtered = apply_mag_floor(preds, mag_min)
        postfilters.append({"name": f"post_mag>{mag_min}", **eval_preds(gt, filtered)})
    postfilters.append({"name": "drop_warmup", **eval_preds(gt, drop_before_first_gt(preds, [
        {"csv_center_ms": float(g["csv_center_ms"]), "hand": g["final_hand"]} for g in gt_single
    ]))})
    postfilters.append({"name": "sympathetic", **eval_preds(gt, suppress_sympathetic(preds))})
    postfilters.append({"name": "crash_second<2.5g", **eval_preds(gt, crash_second_if_weak(preds, 2.5))})
    postfilters.append({"name": "crash_second<3.0g", **eval_preds(gt, crash_second_if_weak(preds, 3.0))})
    combo = suppress_sympathetic(preds)
    combo = crash_second_if_weak(combo, 2.5)
    postfilters.append({"name": "sympathetic+crash_second<2.5g", **eval_preds(gt, combo)})
    print("postfilters done")

    sweep = []
    detector_grid = [
        {"mag_min": 1.7, "debounce_heavy_s": 0.15, "debounce_light_s": 0.25},
        {"mag_min": 1.7, "debounce_heavy_s": 0.10, "debounce_light_s": 0.15},
        {"mag_min": 1.7, "debounce_heavy_s": 0.08, "debounce_light_s": 0.12},
        {"mag_min": 1.7, "debounce_heavy_s": 0.08, "debounce_light_s": 0.10},
        {"mag_min": 1.9, "debounce_heavy_s": 0.10, "debounce_light_s": 0.15},
        {"mag_min": 2.1, "debounce_heavy_s": 0.10, "debounce_light_s": 0.15},
        {"mag_min": 1.7, "debounce_heavy_s": 0.09, "debounce_light_s": 0.12},
    ]
    for kwargs in detector_grid:
        dets_i = replay_hits(raw, detector_kwargs=kwargs)
        preds_i = classify_detections(dets_i, samples, clf)
        metrics = eval_preds(gt, preds_i)
        metrics_sym = eval_preds(gt, suppress_sympathetic(preds_i))
        sweep.append({
            "detector": kwargs,
            "raw": metrics,
            "plus_sympathetic": metrics_sym,
        })
        print("sweep", kwargs, "ok", metrics["joint_ok"], "miss", metrics["misses"], "extra", metrics["extras"])

    payload = {
        "current": current,
        "postfilters_on_current": postfilters,
        "detector_sweep": sweep,
    }
    os.makedirs(CLEANED_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()