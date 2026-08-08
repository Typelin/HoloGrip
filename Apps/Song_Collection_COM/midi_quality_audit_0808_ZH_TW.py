"""Audit MIDI quality and high-confidence MIDI/CSV candidates.

The report deliberately keeps MIDI timing and drum-note identity separate from
left/right hand truth. Hand assignment for a two-note group still needs video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

import midi_label_pipeline as pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CSV = (
    PROJECT_ROOT
    / "Data"
    / "Raw"
    / "Song_Collection_COM"
    / "S20260805_P01_song01_raw_100hz_20260805_161628.csv"
)
DEFAULT_MIDI = (
    PROJECT_ROOT
    / "Data"
    / "External"
    / "FlowAudio_20260805"
    / "Drum Midi_110BPM (0805).mid"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "Data"
    / "Derived"
    / "Song_Collection_COM"
    / "S20260805_P01_song01"
    / "midi_quality_audit_0808"
)


def energy_rows(samples: list[pipeline.RawSample], duration_ms: float) -> list[dict[str, float]]:
    arrays = pipeline._build_energy(samples, duration_ms)
    present_buckets = {
        int(math.floor(max(0.0, sample.song_time_ms) / pipeline.BIN_MS + 0.5)) * pipeline.BIN_MS
        for sample in samples
    }
    return [
        {"t": index * pipeline.BIN_MS, "l": arrays["L"][index], "r": arrays["R"][index]}
        for index in range(len(arrays["L"]))
        if index * pipeline.BIN_MS in present_buckets
    ]


def window_rows(rows: list[dict[str, float]], center_ms: float, radius_ms: float) -> list[dict[str, float]]:
    times = [row["t"] for row in rows]
    start = max(0.0, center_ms - radius_ms)
    end = center_ms + radius_ms
    return rows[bisect_left(times, start) : bisect_right(times, end)]


def annotate_hand(event: pipeline.MidiNoteEvent, rows: list[dict[str, float]], offset_ms: float) -> dict[str, Any]:
    points = window_rows(rows, event.time_ms + offset_ms, 120.0)
    left = max((point["l"] for point in points), default=0.0)
    right = max((point["r"] for point in points), default=0.0)
    total = left + right
    hand = "L" if left > right else "R" if total else "?"
    confidence = max(left, right) / total if total else 0.0
    return {"hand": hand, "confidence": confidence, "left_peak": left, "right_peak": right}


def midi_groups(events: list[dict[str, Any]], tolerance_ms: float) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for event in events:
        group = groups[-1] if groups else None
        if group is None or event["midi_time_ms"] - group["time_ms"] > tolerance_ms:
            group = {"time_ms": event["midi_time_ms"], "events": []}
            groups.append(group)
        group["events"].append(event)
    return groups


def local_peaks(rows: list[dict[str, float]], hand: str, center_ms: float, radius_ms: float) -> list[dict[str, float]]:
    points = window_rows(rows, center_ms, radius_ms)
    candidates: list[dict[str, float]] = []
    for index, point in enumerate(points):
        value = point[hand]
        if value < 1.0:
            continue
        previous = points[index - 1][hand] if index else float("-inf")
        following = points[index + 1][hand] if index + 1 < len(points) else float("-inf")
        if value >= previous and value >= following:
            candidates.append({"time_ms": point["t"], "value": value})
    candidates.sort(key=lambda point: (-point["value"], point["time_ms"]))
    selected: list[dict[str, float]] = []
    for point in candidates:
        if all(abs(item["time_ms"] - point["time_ms"]) >= 40.0 for item in selected):
            selected.append(point)
    return sorted(selected, key=lambda point: point["time_ms"])


def make_event_row(event: pipeline.MidiNoteEvent, hand_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "midi_time_ms": round(event.time_ms, 3),
        "midi_note": event.note,
        "zone_name": event.zone_name,
        "velocity": event.velocity,
        "hand_candidate": hand_info["hand"],
        "hand_confidence": round(hand_info["confidence"], 6),
        "left_peak_g": round(hand_info["left_peak"], 6),
        "right_peak_g": round(hand_info["right_peak"], 6),
    }


def audit_radius(
    events: list[dict[str, Any]],
    rows: list[dict[str, float]],
    radius_ms: float,
    csv_offset_ms: float,
    duplicate_window_ms: float,
    group_tolerance_ms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = midi_groups(events, group_tolerance_ms)
    group_by_event = {
        event["event_id"]: group_index
        for group_index, group in enumerate(groups)
        for event in group["events"]
    }
    duplicate_pairs: list[dict[str, Any]] = []
    same_note_event_ids: set[int] = set()
    same_note_group_indices: set[int] = set()
    for current_index, current in enumerate(events):
        previous_index = current_index - 1
        while previous_index >= 0 and current["midi_time_ms"] - events[previous_index]["midi_time_ms"] <= duplicate_window_ms:
            previous = events[previous_index]
            if current["midi_note"] == previous["midi_note"]:
                duplicate_pairs.append({
                    "first_event_id": previous["event_id"],
                    "second_event_id": current["event_id"],
                    "delta_ms": round(current["midi_time_ms"] - previous["midi_time_ms"], 3),
                    "midi_note": current["midi_note"],
                    "zone_name": current["zone_name"],
                })
                same_note_event_ids.update((previous["event_id"], current["event_id"]))
                same_note_group_indices.update((group_by_event[previous["event_id"]], group_by_event[current["event_id"]]))
            previous_index -= 1

    triple_group_indices: set[int] = set()
    anomaly_event_ids = set(same_note_event_ids)
    for group_index, group in enumerate(groups):
        in_window = [
            event for event in events if abs(event["midi_time_ms"] - group["time_ms"]) <= radius_ms
        ]
        if len(in_window) < 3:
            continue
        anomaly_event_ids.update(event["event_id"] for event in in_window)
        triple_group_indices.update(group_by_event[event["event_id"]] for event in in_window)

    anomaly_group_indices = sorted(same_note_group_indices | triple_group_indices)
    anomaly_area_indices: list[int] = []
    for group_index in anomaly_group_indices:
        previous = anomaly_area_indices[-1] if anomaly_area_indices else None
        if previous is None or groups[group_index]["time_ms"] - groups[previous]["time_ms"] > max(duplicate_window_ms, radius_ms * 2):
            anomaly_area_indices.append(group_index)

    info_by_event: dict[int, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        previous_gap = group["time_ms"] - groups[group_index - 1]["time_ms"] if group_index else math.inf
        next_gap = groups[group_index + 1]["time_ms"] - group["time_ms"] if group_index + 1 < len(groups) else math.inf
        nearest_gap = min(previous_gap, next_gap)
        multi_label = len(group["events"]) > 1
        overlap_review = math.isfinite(nearest_gap) and nearest_gap < radius_ms * 2
        anomaly = group_index in same_note_group_indices or group_index in triple_group_indices
        single_high = False
        if not anomaly and not multi_label and not overlap_review:
            event = group["events"][0]
            peak = max(
                [point["l"] for point in window_rows(rows, group["time_ms"] + csv_offset_ms, radius_ms)]
                + [point["r"] for point in window_rows(rows, group["time_ms"] + csv_offset_ms, radius_ms)]
                + [0.0]
            )
            single_high = (
                event["hand_candidate"] in {"L", "R"}
                and event["hand_confidence"] >= 0.90
                and peak >= 1.0
            )
        for event in group["events"]:
            info_by_event[event["event_id"]] = {
                "group_index": group_index,
                "anomaly": anomaly,
                "anomaly_reasons": [
                    reason
                    for condition, reason in (
                        (group_index in same_note_group_indices, "同一鼓點 20 ms 內重複"),
                        (group_index in triple_group_indices, "窗口內至少 3 個 MIDI"),
                    )
                    if condition
                ],
                "single_high": single_high,
                "multi_label": multi_label,
                "overlap_review": overlap_review,
                "previous_gap_ms": None if not math.isfinite(previous_gap) else round(previous_gap, 3),
                "next_gap_ms": None if not math.isfinite(next_gap) else round(next_gap, 3),
            }

    dual_candidates: list[dict[str, Any]] = []
    dual_clean: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        previous_gap = groups[group_index]["time_ms"] - groups[group_index - 1]["time_ms"] if group_index else math.inf
        next_gap = groups[group_index + 1]["time_ms"] - group["time_ms"] if group_index + 1 < len(groups) else math.inf
        left = local_peaks(rows, "l", group["time_ms"] + csv_offset_ms, radius_ms)
        right = local_peaks(rows, "r", group["time_ms"] + csv_offset_ms, radius_ms)
        two_points = len(group["events"]) == 2
        different_notes = two_points and group["events"][0]["midi_note"] != group["events"][1]["midi_note"]
        candidate = two_points and different_notes and len(left) == 1 and len(right) == 1 and not info_by_event[group["events"][0]["event_id"]]["anomaly"]
        clean = candidate and min(previous_gap, next_gap) >= radius_ms * 2
        item = {
            "group_index": group_index,
            "midi_time_ms": group["time_ms"],
            "event_ids": [event["event_id"] for event in group["events"]],
            "notes": [event["midi_note"] for event in group["events"]],
            "left_peak_g": round(left[0]["value"], 6) if len(left) == 1 else None,
            "right_peak_g": round(right[0]["value"], 6) if len(right) == 1 else None,
            "previous_gap_ms": None if not math.isfinite(previous_gap) else round(previous_gap, 3),
            "next_gap_ms": None if not math.isfinite(next_gap) else round(next_gap, 3),
            "candidate": candidate,
            "clean": clean,
        }
        if candidate:
            dual_candidates.append(item)
        if clean:
            dual_clean.append(item)

    single_events = [
        event for event in events if info_by_event[event["event_id"]]["single_high"]
    ]
    clean_nodes = sum(len(item["event_ids"]) for item in dual_clean)
    effective_denominator = len(events) - len(anomaly_event_ids)
    summary = {
        "window_radius_ms": radius_ms,
        "window_total_ms": radius_ms * 2,
        "total_midi_events": len(events),
        "onset_groups": len(groups),
        "same_note_duplicate_pairs_within_20ms": len(duplicate_pairs),
        "same_note_duplicate_unique_events": len(same_note_event_ids),
        "triple_midi_affected_unique_events": len(anomaly_event_ids - same_note_event_ids),
        "anomaly_unique_events": len(anomaly_event_ids),
        "anomaly_group_count": len(anomaly_group_indices),
        "anomaly_area_count": len(anomaly_area_indices),
        "effective_midi_denominator": effective_denominator,
        "high_confidence_single_events": len(single_events),
        "high_confidence_single_left": sum(event["hand_candidate"] == "L" for event in single_events),
        "high_confidence_single_right": sum(event["hand_candidate"] == "R" for event in single_events),
        "high_confidence_dual_groups": len(dual_clean),
        "high_confidence_dual_midi_events": clean_nodes,
        "high_confidence_total_midi_events": len(single_events) + clean_nodes,
        "high_confidence_share_of_effective_denominator": (len(single_events) + clean_nodes) / effective_denominator if effective_denominator else 0.0,
    }
    event_rows = []
    for event in events:
        info = info_by_event[event["event_id"]]
        dual = next((item for item in dual_clean if event["event_id"] in item["event_ids"]), None)
        status = "異常" if info["anomaly"] else "高信心雙手" if dual else "高信心單手" if info["single_high"] else "待人工判讀"
        event_rows.append({
            **event,
            "window_radius_ms": radius_ms,
            "group_index": info["group_index"] + 1,
            "anomaly": info["anomaly"],
            "anomaly_reasons": "＋".join(info["anomaly_reasons"]),
            "label_status": status,
            "effective_candidate": not info["anomaly"],
            "dual_clean_group": dual["group_index"] + 1 if dual else "",
        })
    return summary, event_rows, duplicate_pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="HoloGrip MIDI quality audit")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-offset-ms", type=float, default=16000.0)
    parser.add_argument("--bpm", type=float, default=110.0)
    parser.add_argument("--window-radius-ms", type=float, default=90.0)
    parser.add_argument("--compare-radius-ms", type=float, default=80.0)
    parser.add_argument("--duplicate-window-ms", type=float, default=20.0)
    parser.add_argument("--group-tolerance-ms", type=float, default=12.0)
    args = parser.parse_args()
    if not args.raw_csv.is_file() or not args.midi.is_file():
        parser.error("raw CSV or MIDI file was not found")
    samples, raw_info = pipeline.read_raw_csv(args.raw_csv)
    rows = energy_rows(samples, raw_info["duration_ms"])
    parsed = pipeline.parse_midi(args.midi, args.bpm)
    events = []
    for event in parsed["events"]:
        hand = annotate_hand(event, rows, args.csv_offset_ms)
        events.append({
            "event_id": event.event_id,
            "midi_time_ms": round(event.time_ms, 3),
            "midi_note": event.note,
            "zone_name": event.zone_name,
            "velocity": event.velocity,
            "hand_candidate": hand["hand"],
            "hand_confidence": round(hand["confidence"], 6),
            "left_peak_g": round(hand["left_peak"], 6),
            "right_peak_g": round(hand["right_peak"], 6),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    all_event_rows = []
    duplicate_rows = []
    for radius in sorted({args.window_radius_ms, args.compare_radius_ms}):
        radius_events = []
        for event in events:
            points = window_rows(rows, event["midi_time_ms"] + args.csv_offset_ms, radius)
            left = max((point["l"] for point in points), default=0.0)
            right = max((point["r"] for point in points), default=0.0)
            total = left + right
            radius_events.append({
                **event,
                "hand_candidate": "L" if left > right else "R" if total else "?",
                "hand_confidence": max(left, right) / total if total else 0.0,
                "left_peak_g": left,
                "right_peak_g": right,
            })
        summary, event_rows, duplicates = audit_radius(radius_events, rows, radius, args.csv_offset_ms, args.duplicate_window_ms, args.group_tolerance_ms)
        summaries[str(int(radius) if radius.is_integer() else radius)] = summary
        all_event_rows.extend(event_rows)
        if radius == args.window_radius_ms:
            duplicate_rows = [{"window_radius_ms": radius, **row} for row in duplicates]
    payload = {
        "source": {
            "midi": str(args.midi.resolve()),
            "raw_csv": str(args.raw_csv.resolve()),
            "midi_event_count": len(events),
            "raw_csv_row_count": len(samples),
            "raw_csv_duration_ms": raw_info["duration_ms"],
            "csv_offset_ms": args.csv_offset_ms,
            "duplicate_window_ms": args.duplicate_window_ms,
        },
        "radius_results": summaries,
    }
    (args.output_dir / "MIDI品質統計摘要_0808_ZH_TW.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "MIDI異常與有效候選事件_0808_ZH_TW.csv", all_event_rows)
    write_csv(args.output_dir / "HiHat_20ms內重複事件_0808_ZH_TW.csv", duplicate_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
