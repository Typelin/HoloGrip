"""Audit the complete MIDI, video soundtrack, external WAV, and glove CSV timeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Tools" / "Audio"))
sys.path.insert(0, str(REPO / "Apps" / "Song_Collection_COM"))

from analyze_drum_audio_0808 import nearest_distance, probe_audio, score_offset  # noqa: E402
from midi_label_pipeline import (  # noqa: E402
    BIN_MS,
    _build_energy,
    _local_max,
    _score_at,
    parse_midi,
    read_raw_csv,
)


TOLERANCE_MS = 35.0
SEGMENT_MS = 60_000
LOCAL_SEARCH_MS = 500
CSV_PEAK_RADIUS_MS = 120


def read_detected_onsets(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("is_detected_onset_35ms") or "").lower() != "true":
                continue
            try:
                values.append(float(row["audio_time_ms"]))
            except (KeyError, ValueError):
                continue
    return np.asarray(values, dtype=np.float64)


def midi_rows(midi_info: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "id": event.event_id,
            "time_ms": float(event.time_ms),
            "note": event.note,
            "zone": event.zone_name,
        }
        for event in midi_info["events"]
    ]


def audio_best_offset(
    events: list[dict[str, object]],
    onsets: np.ndarray,
    start_ms: int = -20_000,
    end_ms: int = 30_000,
) -> dict[str, object]:
    scores = [score_offset(events, onsets, offset, TOLERANCE_MS) for offset in range(start_ms, end_ms + 1, 10)]
    return sorted(
        scores,
        key=lambda item: (
            -int(item["matched_events"]),
            float(item["median_abs_error_ms"] or 999999),
            abs(int(item["offset_ms"])),
        ),
    )[0]


def csv_score(events: list[dict[str, object]], combined: list[float], offset_ms: int) -> dict[str, float]:
    scores = [_score_at(combined, float(event["time_ms"]) + offset_ms) for event in events]
    return {
        "offset_ms": offset_ms,
        "mean_motion_score": float(np.mean(scores)) if scores else 0.0,
        "median_motion_score": float(np.median(scores)) if scores else 0.0,
        "active_ratio": float(np.mean(np.asarray(scores) > 0.08)) if scores else 0.0,
    }


def csv_best_offset(
    events: list[dict[str, object]],
    combined: list[float],
    start_ms: int = -5_000,
    end_ms: int = 30_000,
) -> dict[str, float]:
    scores = [csv_score(events, combined, offset) for offset in range(start_ms, end_ms + 1, BIN_MS)]
    return sorted(scores, key=lambda item: (-item["mean_motion_score"], -item["active_ratio"], abs(item["offset_ms"])))[0]


def local_audio_offset(events: list[dict[str, object]], onsets: np.ndarray, global_offset: int) -> dict[str, object] | None:
    if not events:
        return None
    return audio_best_offset(events, onsets, global_offset - LOCAL_SEARCH_MS, global_offset + LOCAL_SEARCH_MS)


def local_csv_offset(events: list[dict[str, object]], combined: list[float], global_offset: int) -> dict[str, float] | None:
    if not events:
        return None
    return csv_best_offset(events, combined, global_offset - LOCAL_SEARCH_MS, global_offset + LOCAL_SEARCH_MS)


def nearest_record(target_ms: float, onsets: np.ndarray) -> dict[str, float | None]:
    if onsets.size == 0:
        return {"detected_ms": None, "signed_error_ms": None, "abs_error_ms": None}
    detected = float(onsets[np.argmin(np.abs(onsets - target_ms))])
    error = detected - target_ms
    return {"detected_ms": round(detected, 3), "signed_error_ms": round(error, 3), "abs_error_ms": round(abs(error), 3)}


def audio_event_summary(events: list[dict[str, object]], onsets: np.ndarray, offset_ms: int) -> dict[str, object]:
    records = [nearest_record(float(event["time_ms"]) + offset_ms, onsets) for event in events]
    errors = np.asarray([float(record["signed_error_ms"]) for record in records if record["signed_error_ms"] is not None])
    matched = errors[np.abs(errors) <= TOLERANCE_MS]
    slope = float(np.polyfit([float(event["time_ms"]) for event, record in zip(events, records) if record["signed_error_ms"] is not None], errors, 1)[0]) if errors.size >= 2 else 0.0
    return {
        "event_count": len(events),
        "matched_within_35ms": int(matched.size),
        "match_rate": float(matched.size / len(events)) if events else 0.0,
        "median_abs_error_ms": float(np.median(np.abs(matched))) if matched.size else None,
        "mean_abs_error_ms": float(np.mean(np.abs(matched))) if matched.size else None,
        "median_signed_error_ms": float(np.median(matched)) if matched.size else None,
        "drift_ms_per_minute": slope * 60_000.0,
    }


def csv_peak_summary(events: list[dict[str, object]], combined: list[float], offset_ms: int) -> dict[str, object]:
    errors: list[float] = []
    scores: list[float] = []
    for event in events:
        target = float(event["time_ms"]) + offset_ms
        left = max(0, int(round((target - CSV_PEAK_RADIUS_MS) / BIN_MS)))
        right = min(len(combined), int(round((target + CSV_PEAK_RADIUS_MS) / BIN_MS)) + 1)
        values = combined[left:right]
        if not values:
            continue
        peak_index = left + int(np.argmax(values))
        errors.append(peak_index * BIN_MS - target)
        scores.append(float(values[peak_index - left]))
    abs_errors = np.abs(np.asarray(errors))
    return {
        "event_count": len(events),
        "events_with_peak_in_±120ms": len(errors),
        "active_ratio_over_0.08": float(np.mean(np.asarray(scores) > 0.08)) if scores else 0.0,
        "median_peak_abs_error_ms": float(np.median(abs_errors)) if errors else None,
        "mean_peak_abs_error_ms": float(np.mean(abs_errors)) if errors else None,
        "median_peak_signed_error_ms": float(np.median(errors)) if errors else None,
    }


def segment_report(
    events: list[dict[str, object]],
    video_onsets: np.ndarray,
    wav_onsets: np.ndarray,
    combined: list[float],
    video_offset: int,
    wav_offset: int,
    csv_offset: int,
) -> list[dict[str, object]]:
    last_time = max((float(event["time_ms"]) for event in events), default=0.0)
    output: list[dict[str, object]] = []
    for start in range(0, int(last_time) + SEGMENT_MS, SEGMENT_MS):
        end = min(last_time + 1, start + SEGMENT_MS)
        rows = [event for event in events if start <= float(event["time_ms"]) < end]
        if not rows:
            continue
        video = local_audio_offset(rows, video_onsets, video_offset)
        wav = local_audio_offset(rows, wav_onsets, wav_offset)
        csv = local_csv_offset(rows, combined, csv_offset)
        output.append({
            "midi_range_ms": [start, end],
            "event_count": len(rows),
            "midi_to_video_offset_ms": video["offset_ms"] if video else None,
            "midi_to_wav_offset_ms": wav["offset_ms"] if wav else None,
            "midi_to_csv_offset_ms": csv["offset_ms"] if csv else None,
            "video_match_rate": video["match_rate"] if video else None,
            "wav_match_rate": wav["match_rate"] if wav else None,
            "csv_mean_motion_score": csv["mean_motion_score"] if csv else None,
        })
    return output


def event_check(event: dict[str, object], video_onsets: np.ndarray, wav_onsets: np.ndarray, combined: list[float], video_offset: int, wav_offset: int, csv_offset: int) -> dict[str, object]:
    midi_ms = float(event["time_ms"])
    video = nearest_record(midi_ms + video_offset, video_onsets)
    wav = nearest_record(midi_ms + wav_offset, wav_onsets)
    target = midi_ms + csv_offset
    left = max(0, int(round((target - CSV_PEAK_RADIUS_MS) / BIN_MS)))
    right = min(len(combined), int(round((target + CSV_PEAK_RADIUS_MS) / BIN_MS)) + 1)
    values = combined[left:right]
    csv_peak = left + int(np.argmax(values)) if values else None
    return {
        "event_id": int(event["id"]),
        "zone": event["zone"],
        "midi_time_ms": round(midi_ms, 3),
        "video_onset_ms": video["detected_ms"],
        "video_error_ms": video["signed_error_ms"],
        "wav_onset_ms": wav["detected_ms"],
        "wav_error_ms": wav["signed_error_ms"],
        "csv_target_ms": round(target, 3),
        "csv_peak_ms": csv_peak * BIN_MS if csv_peak is not None else None,
        "csv_peak_error_ms": round(csv_peak * BIN_MS - target, 3) if csv_peak is not None else None,
        "csv_peak_activity": round(float(combined[csv_peak]), 6) if csv_peak is not None else None,
    }


def main() -> None:
    external = REPO / "Data" / "External" / "FlowAudio_20260805"
    derived = REPO / "Data" / "Derived" / "Song_Collection_COM" / "S20260805_P01_song01"
    parser = argparse.ArgumentParser(description="四方檢查 MIDI、WAV、影片音訊與手套 CSV 對齊")
    parser.add_argument("--video", type=Path, default=external / "IMG_6060.MOV")
    parser.add_argument("--wav", type=Path, default=external / "Drum Audio_110BPM (0805).wav")
    parser.add_argument("--midi", type=Path, default=external / "Drum Midi_110BPM (0805).mid")
    parser.add_argument("--csv", type=Path, default=REPO / "Data" / "Raw" / "Song_Collection_COM" / "S20260805_P01_song01_raw_100hz_20260805_161628.csv")
    parser.add_argument("--video-onsets", type=Path, default=derived / "video_alignment_0808" / "audio_onsets_0808.csv")
    parser.add_argument("--wav-onsets", type=Path, default=derived / "audio_alignment_0808" / "audio_onsets_0808.csv")
    parser.add_argument("--output", type=Path, default=derived / "video_alignment_0808" / "full_alignment_report_0808_ZH_TW.json")
    args = parser.parse_args()

    raw_samples, raw_info = read_raw_csv(args.csv)
    midi_info = parse_midi(args.midi, 110.0)
    events = midi_rows(midi_info)
    energy = _build_energy(raw_samples, raw_info["duration_ms"])
    smoothed = {hand: _local_max(values, 4) for hand, values in energy.items()}
    combined = [max(smoothed["L"][index], smoothed["R"][index]) for index in range(len(smoothed["L"]))]
    video_onsets = read_detected_onsets(args.video_onsets)
    wav_onsets = read_detected_onsets(args.wav_onsets)
    video_score = audio_best_offset(events, video_onsets)
    wav_score = audio_best_offset(events, wav_onsets)
    csv_score_result = csv_best_offset(events, combined)
    video_offset = int(video_score["offset_ms"])
    wav_offset = int(wav_score["offset_ms"])
    csv_offset = int(csv_score_result["offset_ms"])
    checks = [event_check(event, video_onsets, wav_onsets, combined, video_offset, wav_offset, csv_offset) for event in events]
    first_crash = next((event for event in events if int(event["note"]) == 49), None)
    checkpoint_indices = sorted({0, len(events) // 2, len(events) - 1}) if events else []
    checkpoint_events = [event_check(events[index], video_onsets, wav_onsets, combined, video_offset, wav_offset, csv_offset) for index in checkpoint_indices]
    audio_drift_video = audio_event_summary(events, video_onsets, video_offset)
    audio_drift_wav = audio_event_summary(events, wav_onsets, wav_offset)
    report = {
        "status": "full_alignment_audit",
        "inputs": {
            "video": str(args.video.resolve()),
            "wav": str(args.wav.resolve()),
            "midi": str(args.midi.resolve()),
            "csv": str(args.csv.resolve()),
            "video_onsets": str(args.video_onsets.resolve()),
            "wav_onsets": str(args.wav_onsets.resolve()),
        },
        "duration_ms": {
            "video": round(float(probe_audio(args.video)["duration_s"]) * 1000.0, 3),
            "wav": round(float(probe_audio(args.wav)["duration_s"]) * 1000.0, 3),
            "midi_last_event": round(float(events[-1]["time_ms"]), 3) if events else None,
            "midi_track_end": round(float(midi_info["track_end_ms"]), 3),
            "csv": round(float(raw_info["duration_ms"]), 3),
        },
        "counts": {
            "midi_events": len(events),
            "csv_rows": raw_info["row_count"],
            "csv_rows_left": raw_info["hand_counts"].get("L", 0),
            "csv_rows_right": raw_info["hand_counts"].get("R", 0),
            "video_detected_onsets": int(video_onsets.size),
            "wav_detected_onsets": int(wav_onsets.size),
        },
        "global_offsets": {
            "midi_to_video_ms": video_score,
            "midi_to_wav_ms": wav_score,
            "midi_to_csv_ms": csv_score_result,
            "video_minus_wav_ms": video_offset - wav_offset,
            "csv_minus_wav_ms": csv_offset - wav_offset,
            "csv_minus_video_ms": csv_offset - video_offset,
        },
        "all_event_error_summary": {
            "video": audio_drift_video,
            "wav": audio_drift_wav,
            "csv_motion_peak": csv_peak_summary(events, combined, csv_offset),
        },
        "checkpoints_first_middle_last": checkpoint_events,
        "first_crash": event_check(first_crash, video_onsets, wav_onsets, combined, video_offset, wav_offset, csv_offset) if first_crash else None,
        "local_60s_offsets": segment_report(events, video_onsets, wav_onsets, combined, video_offset, wav_offset, csv_offset),
        "interpretation": [
            "影片、WAV、MIDI 先以音訊 onset 對齊；CSV 以 accel_magnitude_g 偏離 1g 的左右手活動度，10ms bin 後取約 ±40ms local-max，再與 MIDI 搜尋整體 offset。",
            "影片偏移 +14520ms、WAV 偏移 +0ms、CSV 偏移 +16000ms 是三個不同資料源的獨立偏移，不能合併成同一個數字。",
            "全曲逐事件誤差與 60 秒局部 offset 只能確認時間規律，不能單靠音訊確認左手、右手或鼓面；手別仍需影片畫面人工標記。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"midi_to_video_ms={video_offset}")
    print(f"midi_to_wav_ms={wav_offset}")
    print(f"midi_to_csv_ms={csv_offset}")
    print(f"first_crash={report['first_crash']}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
