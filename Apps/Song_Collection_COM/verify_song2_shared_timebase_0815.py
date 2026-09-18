"""Verify the Song 2 shared glove timeline and simultaneous-hit peaks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))

import midi_label_pipeline as pipeline  # noqa: E402


CSV_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Raw"
    / "Song_Collection_COM"
    / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
)
OLD_CSV_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Raw"
    / "Song_Collection_COM"
    / "S20260805_P01_song01_raw_100hz_20260805_161628.csv"
)
MIDI_PATH = Path(r"C:\Users\Typelin_Station\Downloads\0812T1127_去除雜訊T1213.mid")
BPM = 110.0
WINDOW_RADIUS_MS = 80
SEARCH_MIN_MS = -5_000
SEARCH_MAX_MS = 15_000


def peak(values: list[float], center_ms: float, radius_ms: int) -> tuple[float, int]:
    first = max(0, int((center_ms - radius_ms) // pipeline.BIN_MS))
    last = min(len(values) - 1, int((center_ms + radius_ms) // pipeline.BIN_MS) + 1)
    index = max(range(first, last + 1), key=values.__getitem__)
    return values[index], index * pipeline.BIN_MS


def simultaneous_groups(events: list[pipeline.MidiNoteEvent]) -> list[list[pipeline.MidiNoteEvent]]:
    groups: list[list[pipeline.MidiNoteEvent]] = []
    for event in events:
        if not groups or event.time_ms - groups[-1][0].time_ms > pipeline.ONSET_GROUP_TOLERANCE_MS:
            groups.append([])
        groups[-1].append(event)
    return [group for group in groups if len(group) >= 2]


def main() -> int:
    samples, raw_info = pipeline.read_raw_csv(CSV_PATH)
    midi = pipeline.parse_midi(MIDI_PATH, BPM)
    energy = pipeline._build_energy(samples, raw_info["duration_ms"])
    combined = [max(right, left) for right, left in zip(energy["R"], energy["L"])]
    offset_ms, candidates = pipeline.estimate_offset(
        midi["events"],
        combined,
        raw_info["duration_ms"],
        midi["track_end_ms"],
        SEARCH_MIN_MS,
        SEARCH_MAX_MS,
    )

    group_rows = []
    for group in simultaneous_groups(midi["events"]):
        center_ms = group[0].time_ms + offset_ms
        right_peak, right_time = peak(energy["R"], center_ms, WINDOW_RADIUS_MS)
        left_peak, left_time = peak(energy["L"], center_ms, WINDOW_RADIUS_MS)
        group_rows.append(
            {
                "event_ids": [event.event_id for event in group],
                "notes": [event.note for event in group],
                "midi_time_ms": round(group[0].time_ms, 3),
                "csv_time_ms": round(center_ms, 3),
                "right_peak": round(right_peak, 6),
                "left_peak": round(left_peak, 6),
                "peak_gap_ms": abs(right_time - left_time),
            }
        )

    dual_thresholds = {}
    for threshold in (0.08, 0.2, 0.5, 1.0):
        for max_gap_ms in (20, 40, 80):
            key = f"both_ge_{threshold:g}_gap_le_{max_gap_ms}ms"
            dual_thresholds[key] = sum(
                row["right_peak"] >= threshold
                and row["left_peak"] >= threshold
                and row["peak_gap_ms"] <= max_gap_ms
                for row in group_rows
            )

    old_samples, old_info = pipeline.read_raw_csv(OLD_CSV_PATH)
    result = {
        "time_base": raw_info["time_base"],
        "time_base_description": raw_info["time_base_description"],
        "song2_sensor_origins_by_hand_ms": raw_info["sensor_origins_by_hand_ms"],
        "song2_sensor_origin_gap_ms": raw_info["sensor_origin_gap_ms"],
        "old_0805_sensor_origin_gap_ms": old_info["sensor_origin_gap_ms"],
        "best_offset_ms": offset_ms,
        "previous_wrong_offset_ms": 15_657,
        "offset_difference_ms": 15_657 - offset_ms,
        "top_candidates": candidates[:5],
        "midi_event_count": len(midi["events"]),
        "simultaneous_group_count": len(group_rows),
        "simultaneous_midi_event_count": sum(len(group) for group in simultaneous_groups(midi["events"])),
        "dual_peak_counts": dual_thresholds,
        "strongest_dual_examples": sorted(
            group_rows,
            key=lambda row: min(row["right_peak"], row["left_peak"]),
            reverse=True,
        )[:10],
        "weakest_dual_examples": sorted(
            group_rows,
            key=lambda row: min(row["right_peak"], row["left_peak"]),
        )[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
