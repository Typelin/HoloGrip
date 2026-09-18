"""Build reviewable seven-zone labels from a Raw 100 Hz CSV and drum MIDI.

The source CSV and MIDI files are never modified. This first-pass pipeline
uses the confirmed 0805 mapping and an explicit BPM override, estimates a
candidate recording offset from glove motion, and marks all generated labels
as pending review until the alignment is accepted by a human.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BIN_MS = 10
DEFAULT_BPM = 110.0
DEFAULT_PRE_MS = 90
DEFAULT_POST_MS = 90
ONSET_GROUP_TOLERANCE_MS = 12.0
HAND_PEAK_SEARCH_MS = 40
TIME_BASE = "song_time_ms"

# Stable mapping for this SONG dataset. 37 and 38 are intentionally merged.
ZONE_BY_NOTE: dict[int, tuple[int, str]] = {
    37: (0, "小鼓"),
    38: (0, "小鼓"),
    48: (1, "高音 Tom"),
    45: (2, "中音 Tom"),
    43: (3, "落地 Tom"),
    46: (4, "Hi-Hat"),
    49: (5, "Crash"),
    51: (6, "Ride"),
}
EXCLUDED_NOTES: dict[int, str] = {
    36: "大鼓（腳部事件）",
    44: "Hi-Hat 腳踏（腳部事件）",
}
RAW_REQUIRED = {
    "session_id",
    "sample_index",
    "hand",
    "song_time_ms",
    "sensor_time_ms",
    "packet_id",
    "ax_g",
    "ay_g",
    "az_g",
    "accel_magnitude_g",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "cal_yaw_deg",
    "cal_pitch_deg",
}


@dataclass(frozen=True)
class MidiNoteEvent:
    event_id: int
    tick: int
    time_ms: float
    note: int
    velocity: int
    channel: int
    track: int
    zone_id: int
    zone_name: str


@dataclass(frozen=True)
class RawSample:
    session_id: str
    sample_index: int
    hand: str
    song_time_ms: float
    sensor_time_ms: int
    packet_id: int
    ax_g: float
    ay_g: float
    az_g: float
    accel_magnitude_g: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    cal_yaw_deg: float
    cal_pitch_deg: float


class MidiParseError(ValueError):
    """Raised when the input is not a supported Standard MIDI file."""


def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], "big")


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], "big")


def _vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if offset >= len(data):
            raise MidiParseError("MIDI variable-length value is truncated")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset
    raise MidiParseError("MIDI variable-length value is invalid")


def parse_midi(path: Path, bpm: float) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"MThd" or len(data) < 14:
        raise MidiParseError(f"Not a Standard MIDI file: {path}")

    header_length = _u32(data, 4)
    midi_format = _u16(data, 8)
    track_count = _u16(data, 10)
    division = _u16(data, 12)
    if division & 0x8000:
        raise MidiParseError("SMPTE MIDI division is not supported; expected ticks per beat")
    if division == 0:
        raise MidiParseError("MIDI ticks per beat cannot be zero")
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero")

    tick_to_ms = 60_000.0 / (bpm * division)
    offset = 8 + header_length
    notes: list[tuple[int, int, int, int, int]] = []
    tempo_events: list[dict[str, int]] = []
    track_end_ticks: list[int] = []
    track_names: list[str] = []

    for track_index in range(track_count):
        if data[offset:offset + 4] != b"MTrk":
            raise MidiParseError(f"Missing MTrk header for track {track_index}")
        length = _u32(data, offset + 4)
        track_end = offset + 8 + length
        cursor = offset + 8
        tick = 0
        running_status: int | None = None

        while cursor < track_end:
            delta, cursor = _vlq(data, cursor)
            tick += delta
            if cursor >= track_end:
                raise MidiParseError("MIDI event is truncated")
            status = data[cursor]
            if status < 0x80:
                if running_status is None:
                    raise MidiParseError("MIDI running status appears before a status byte")
                status = running_status
            else:
                cursor += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                if cursor >= track_end:
                    raise MidiParseError("MIDI meta event is truncated")
                meta_type = data[cursor]
                cursor += 1
                size, cursor = _vlq(data, cursor)
                payload = data[cursor:cursor + size]
                cursor += size
                if len(payload) != size:
                    raise MidiParseError("MIDI meta payload is truncated")
                if meta_type == 0x03:
                    track_names.append(payload.decode("utf-8", "replace"))
                elif meta_type == 0x51 and len(payload) == 3:
                    tempo_events.append({"tick": tick, "microseconds_per_beat": int.from_bytes(payload, "big")})
                if meta_type == 0x2F:
                    break
                continue

            if status in (0xF0, 0xF7):
                size, cursor = _vlq(data, cursor)
                cursor += size
                if cursor > track_end:
                    raise MidiParseError("MIDI sysex payload is truncated")
                continue

            message_kind = status >> 4
            channel = status & 0x0F
            data_size = 1 if message_kind in (0xC, 0xD) else 2
            values = data[cursor:cursor + data_size]
            cursor += data_size
            if len(values) != data_size:
                raise MidiParseError("MIDI channel event is truncated")
            if message_kind == 0x9:
                note, velocity = values
                if velocity > 0:
                    notes.append((tick, note, velocity, channel, track_index))

        track_end_ticks.append(tick)
        offset = track_end

    notes.sort(key=lambda item: (item[0], item[4], item[3], item[1]))
    mapped: list[MidiNoteEvent] = []
    unknown_counts: Counter[int] = Counter()
    excluded_counts: Counter[int] = Counter()
    for event_id, (tick, note, velocity, channel, track) in enumerate(notes, start=1):
        mapping = ZONE_BY_NOTE.get(note)
        if mapping is None:
            if note in EXCLUDED_NOTES:
                excluded_counts[note] += 1
            else:
                unknown_counts[note] += 1
            continue
        zone_id, zone_name = mapping
        mapped.append(
            MidiNoteEvent(
                event_id=event_id,
                tick=tick,
                time_ms=tick * tick_to_ms,
                note=note,
                velocity=velocity,
                channel=channel,
                track=track,
                zone_id=zone_id,
                zone_name=zone_name,
            )
        )

    return {
        "format": midi_format,
        "track_count": track_count,
        "ticks_per_beat": division,
        "bpm_override": bpm,
        "tick_to_ms": tick_to_ms,
        "tempo_events_in_source": tempo_events,
        "track_end_ticks": track_end_ticks,
        "track_end_ms": max(track_end_ticks, default=0) * tick_to_ms,
        "track_names": track_names,
        "source_note_on_count": len(notes),
        "events": mapped,
        "unknown_note_counts": dict(unknown_counts),
        "excluded_note_counts": dict(excluded_counts),
    }


def read_raw_csv(path: Path) -> tuple[list[RawSample], dict[str, Any]]:
    samples: list[RawSample] = []
    sessions: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = RAW_REQUIRED - fields
        if missing:
            raise ValueError(f"Raw CSV is missing fields: {sorted(missing)}")
        for row in reader:
            hand = row["hand"]
            if hand not in {"R", "L"}:
                continue
            values = {key: float(row[key]) for key in (
                "ax_g", "ay_g", "az_g", "accel_magnitude_g", "yaw_deg",
                "pitch_deg", "roll_deg", "cal_yaw_deg", "cal_pitch_deg",
            )}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite sensor value at sample {row['sample_index']}")
            sample = RawSample(
                session_id=row["session_id"],
                sample_index=int(row["sample_index"]),
                hand=hand,
                song_time_ms=float(row["song_time_ms"]),
                sensor_time_ms=int(row["sensor_time_ms"]),
                packet_id=int(row["packet_id"]),
                **values,
            )
            samples.append(sample)
            sessions[sample.session_id] += 1
    # song_time_ms is stamped by the recorder's shared monotonic clock. Each
    # glove has an independent sensor_time_ms origin, so that field cannot be
    # used to place left and right samples on one timeline.
    samples.sort(key=lambda sample: (sample.song_time_ms, sample.hand, sample.sample_index))
    sensor_origin_ms = min((sample.sensor_time_ms for sample in samples), default=0)
    sensor_origins_by_hand = {
        hand: min(
            (sample.sensor_time_ms for sample in samples if sample.hand == hand),
            default=0,
        )
        for hand in ("L", "R")
    }
    song_duration_ms = max((sample.song_time_ms for sample in samples), default=0.0)
    sensor_duration_ms = max(
        (sample.sensor_time_ms - sensor_origin_ms for sample in samples),
        default=0,
    )
    return samples, {
        "row_count": len(samples),
        "session_counts": dict(sessions),
        "duration_ms": float(song_duration_ms),
        "song_duration_ms": song_duration_ms,
        "sensor_duration_ms": float(sensor_duration_ms),
        "sensor_origin_ms": sensor_origin_ms,
        "sensor_origins_by_hand_ms": sensor_origins_by_hand,
        "sensor_origin_gap_ms": abs(
            sensor_origins_by_hand["R"] - sensor_origins_by_hand["L"]
        ),
        "time_base": TIME_BASE,
        "time_base_description": "CSV song_time_ms (shared recorder monotonic clock)",
        "hand_counts": dict(Counter(sample.hand for sample in samples)),
    }


def _motion_energy(sample: RawSample) -> float:
    # Static gravity is close to 1 g; deviation is a simple, explainable hit signal.
    return abs(sample.accel_magnitude_g - 1.0)


def sample_timeline_ms(sample: RawSample) -> float:
    """Return the recorder clock shared by both gloves."""
    return float(sample.song_time_ms)


def _build_energy(
    samples: Iterable[RawSample],
    duration_ms: float,
) -> dict[str, list[float]]:
    """Bin both gloves on song_time_ms; each 10 ms bin keeps its max signal."""
    sample_list = list(samples)
    size = max(1, int(math.ceil(duration_ms / BIN_MS)) + 1)
    result = {"R": [0.0] * size, "L": [0.0] * size}
    for sample in sample_list:
        index = max(
            0,
            min(size - 1, int(round(sample_timeline_ms(sample) / BIN_MS))),
        )
        result[sample.hand][index] = max(result[sample.hand][index], _motion_energy(sample))
    return result


def _local_max(values: list[float], radius_bins: int) -> list[float]:
    result = [0.0] * len(values)
    window: deque[tuple[int, float]] = deque()
    for index, value in enumerate(values):
        while window and window[0][0] < index - radius_bins:
            window.popleft()
        while window and window[-1][1] <= value:
            window.pop()
        window.append((index, value))
        if index >= radius_bins:
            result[index - radius_bins] = window[0][1]
    for index in range(max(0, len(values) - radius_bins), len(values)):
        while window and window[0][0] < index - radius_bins:
            window.popleft()
        result[index] = window[0][1] if window else 0.0
    return result


def _score_at(values: list[float], time_ms: float) -> float:
    index = int(round(time_ms / BIN_MS))
    if index < 0 or index >= len(values):
        return 0.0
    return values[index]


def _peak_score(values: list[float], time_ms: float, radius_ms: int) -> float:
    """Return the local activity peak near an event center for hand scoring."""
    first = max(0, int(math.floor((time_ms - radius_ms) / BIN_MS)))
    last = min(len(values) - 1, int(math.ceil((time_ms + radius_ms) / BIN_MS)))
    if first > last:
        return 0.0
    return max(values[first:last + 1], default=0.0)


def estimate_offset(
    midi_events: list[MidiNoteEvent],
    combined_energy: list[float],
    raw_duration_ms: float,
    midi_duration_ms: float,
    min_offset_ms: int,
    max_offset_ms: int,
) -> tuple[int, list[dict[str, float]]]:
    if not midi_events:
        raise ValueError("No mapped MIDI events available for alignment")
    lower = max(min_offset_ms, int(math.ceil(-midi_events[0].time_ms)))
    upper = min(max_offset_ms, int(math.floor(raw_duration_ms - midi_events[-1].time_ms)))
    if upper < lower:
        raise ValueError("MIDI timeline cannot fit inside the Raw CSV search range")

    candidates: list[dict[str, float]] = []
    for offset_ms in range(lower, upper + 1, BIN_MS):
        scores = [_score_at(combined_energy, event.time_ms + offset_ms) for event in midi_events]
        mean_score = sum(scores) / len(scores)
        active_ratio = sum(score > 0.08 for score in scores) / len(scores)
        candidates.append({"offset_ms": offset_ms, "mean_motion_score": mean_score, "active_ratio": active_ratio})
    candidates.sort(key=lambda item: (item["mean_motion_score"], item["active_ratio"]), reverse=True)
    return int(candidates[0]["offset_ms"]), candidates[:20]


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _format_optional_ms(value: float | None) -> str:
    return "" if value is None else _format_ms(value)


def _density_annotations(
    events: list[MidiNoteEvent],
    pre_ms: int,
    post_ms: int,
) -> dict[int, dict[str, Any]]:
    """Classify events that cannot safely become one fixed single-label window."""
    groups: list[list[MidiNoteEvent]] = []
    for event in events:
        if not groups or event.time_ms - groups[-1][0].time_ms > ONSET_GROUP_TOLERANCE_MS:
            groups.append([])
        groups[-1].append(event)

    overlap_span_ms = float(pre_ms + post_ms)
    annotations: dict[int, dict[str, Any]] = {}
    for group_index, group in enumerate(groups, start=1):
        previous_gap = (
            group[0].time_ms - groups[group_index - 2][-1].time_ms
            if group_index > 1
            else None
        )
        next_gap = (
            groups[group_index][0].time_ms - group[-1].time_ms
            if group_index < len(groups)
            else None
        )
        finite_gaps = [gap for gap in (previous_gap, next_gap) if gap is not None]
        nearest_gap = min(finite_gaps) if finite_gaps else None
        multi_label = len(group) > 1
        overlap_review = any(gap < overlap_span_ms for gap in finite_gaps)
        if multi_label and overlap_review:
            density_status = "multi_label_overlap"
        elif multi_label:
            density_status = "multi_label"
        elif overlap_review:
            density_status = "overlap_review"
        else:
            density_status = "single_label"
        group_notes = "、".join(dict.fromkeys(f"{event.note} {event.zone_name}" for event in group))
        safe_half_ms = (
            min(overlap_span_ms / 2.0, nearest_gap / 2.0)
            if nearest_gap is not None
            else overlap_span_ms / 2.0
        )
        for event in group:
            annotations[event.event_id] = {
                "onset_group_id": group_index,
                "group_event_count": len(group),
                "group_notes": group_notes,
                "previous_gap_ms": previous_gap,
                "next_gap_ms": next_gap,
                "nearest_gap_ms": nearest_gap,
                "multi_label": multi_label,
                "overlap_review": overlap_review,
                "single_label_eligible": density_status == "single_label",
                "density_status": density_status,
                "safe_half_window_ms": safe_half_ms,
            }
    return annotations


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_outputs(
    raw_csv: Path,
    midi_path: Path,
    output_dir: Path,
    bpm: float,
    offset_ms: int | None,
    min_offset_ms: int,
    max_offset_ms: int | None,
    pre_ms: int,
    post_ms: int,
) -> dict[str, Any]:
    samples, raw_info = read_raw_csv(raw_csv)
    midi_info = parse_midi(midi_path, bpm)
    events: list[MidiNoteEvent] = midi_info["events"]
    if not events:
        raise ValueError("MIDI contains no mapped 7-zone events")

    energy = _build_energy(samples, raw_info["duration_ms"])
    combined = [max(energy["R"][i], energy["L"][i]) for i in range(len(energy["R"]))]
    midi_duration_ms = float(midi_info["track_end_ms"])
    if offset_ms is None:
        search_max = max_offset_ms
        if search_max is None:
            search_max = int(max(5_000, raw_info["duration_ms"] - midi_duration_ms + 5_000))
        selected_offset_ms, candidates = estimate_offset(
            events, combined, raw_info["duration_ms"], midi_duration_ms,
            min_offset_ms, search_max,
        )
        alignment_mode = "auto_candidate"
    else:
        selected_offset_ms = int(offset_ms)
        candidates = []
        alignment_mode = "manual_offset_pending_review"

    status = "pending_manual_alignment_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    density = _density_annotations(events, pre_ms, post_ms)
    event_rows: list[dict[str, Any]] = []
    for event in events:
        density_row = density[event.event_id]
        aligned_ms = event.time_ms + selected_offset_ms
        peak_radius_ms = min(HAND_PEAK_SEARCH_MS, pre_ms, post_ms)
        r_score = _peak_score(energy["R"], aligned_ms, peak_radius_ms)
        l_score = _peak_score(energy["L"], aligned_ms, peak_radius_ms)
        total = r_score + l_score
        hand_candidate = "R" if r_score >= l_score else "L"
        hand_confidence = abs(r_score - l_score) / total if total > 0 else 0.0
        event_rows.append({
            "event_id": event.event_id,
            "midi_tick": event.tick,
            "midi_time_ms": _format_ms(event.time_ms),
            "aligned_time_ms": _format_ms(aligned_ms),
            "midi_note": event.note,
            "velocity": event.velocity,
            "channel": event.channel,
            "track": event.track,
            "zone_id": event.zone_id,
            "zone_name": event.zone_name,
            "hand_candidate": hand_candidate,
            "hand_score_r": f"{r_score:.6f}",
            "hand_score_l": f"{l_score:.6f}",
            "hand_confidence": f"{hand_confidence:.6f}",
            "alignment_signal": f"raw_10ms_song_time_local_peak_max_{peak_radius_ms}ms",
            "onset_group_id": density_row["onset_group_id"],
            "group_event_count": density_row["group_event_count"],
            "group_notes": density_row["group_notes"],
            "previous_onset_gap_ms": _format_optional_ms(density_row["previous_gap_ms"]),
            "next_onset_gap_ms": _format_optional_ms(density_row["next_gap_ms"]),
            "nearest_onset_gap_ms": _format_optional_ms(density_row["nearest_gap_ms"]),
            "multi_label": str(density_row["multi_label"]).lower(),
            "overlap_review": str(density_row["overlap_review"]).lower(),
            "single_label_eligible": str(density_row["single_label_eligible"]).lower(),
            "density_status": density_row["density_status"],
            "safe_half_window_ms": _format_ms(density_row["safe_half_window_ms"]),
            "label_status": status,
            "label_source": "midi_timed_event",
        })

    _write_csv(
        output_dir / "midi_events.csv",
        list(event_rows[0].keys()),
        event_rows,
    )

    window_rows = []
    for row in event_rows:
        aligned = float(row["aligned_time_ms"])
        window_rows.append({
            "event_id": row["event_id"],
            "zone_id": row["zone_id"],
            "zone_name": row["zone_name"],
            "midi_note": row["midi_note"],
            "midi_time_ms": row["midi_time_ms"],
            "aligned_time_ms": row["aligned_time_ms"],
            "window_start_ms": _format_ms(aligned - pre_ms),
            "window_end_ms": _format_ms(aligned + post_ms),
            "hand_candidate": row["hand_candidate"],
            "hand_confidence": row["hand_confidence"],
            "onset_group_id": row["onset_group_id"],
            "group_event_count": row["group_event_count"],
            "group_notes": row["group_notes"],
            "previous_onset_gap_ms": row["previous_onset_gap_ms"],
            "next_onset_gap_ms": row["next_onset_gap_ms"],
            "nearest_onset_gap_ms": row["nearest_onset_gap_ms"],
            "multi_label": row["multi_label"],
            "overlap_review": row["overlap_review"],
            "single_label_eligible": row["single_label_eligible"],
            "density_status": row["density_status"],
            "safe_half_window_ms": row["safe_half_window_ms"],
            "label_status": row["label_status"],
        })
    _write_csv(output_dir / "candidate_windows.csv", list(window_rows[0].keys()), window_rows)

    samples_by_hand = {hand: [] for hand in ("R", "L")}
    for sample in samples:
        samples_by_hand[sample.hand].append(sample)
    times_by_hand = {
        hand: [sample_timeline_ms(sample) for sample in hand_samples]
        for hand, hand_samples in samples_by_hand.items()
    }
    sample_rows = []
    for row in event_rows:
        aligned = float(row["aligned_time_ms"])
        start = bisect_left(times_by_hand["R"], aligned - pre_ms)
        end = bisect_right(times_by_hand["R"], aligned + post_ms)
        for hand in ("R", "L"):
            start = bisect_left(times_by_hand[hand], aligned - pre_ms)
            end = bisect_right(times_by_hand[hand], aligned + post_ms)
            for sample in samples_by_hand[hand][start:end]:
                sample_rows.append({
                    "event_id": row["event_id"],
                    "zone_id": row["zone_id"],
                    "zone_name": row["zone_name"],
                    "midi_note": row["midi_note"],
                    "aligned_event_time_ms": row["aligned_time_ms"],
                    "sample_relative_ms": _format_ms(
                        sample_timeline_ms(sample) - aligned
                    ),
                    "event_hand_candidate": row["hand_candidate"],
                    "event_hand_confidence": row["hand_confidence"],
                    "onset_group_id": row["onset_group_id"],
                    "multi_label": row["multi_label"],
                    "overlap_review": row["overlap_review"],
                    "single_label_eligible": row["single_label_eligible"],
                    "density_status": row["density_status"],
                    "hand": sample.hand,
                    "sample_index": sample.sample_index,
                    "song_time_ms": _format_ms(sample.song_time_ms),
                    "sensor_timeline_ms": _format_ms(
                        sample_timeline_ms(sample)
                    ),
                    "sensor_time_ms": sample.sensor_time_ms,
                    "packet_id": sample.packet_id,
                    "ax_g": sample.ax_g,
                    "ay_g": sample.ay_g,
                    "az_g": sample.az_g,
                    "accel_magnitude_g": sample.accel_magnitude_g,
                    "yaw_deg": sample.yaw_deg,
                    "pitch_deg": sample.pitch_deg,
                    "roll_deg": sample.roll_deg,
                    "cal_yaw_deg": sample.cal_yaw_deg,
                    "cal_pitch_deg": sample.cal_pitch_deg,
                    "label_status": row["label_status"],
                })
    if sample_rows:
        _write_csv(output_dir / "labeled_window_samples.csv", list(sample_rows[0].keys()), sample_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "alignment": {
            "mode": alignment_mode,
            "bpm_override": bpm,
            "offset_ms": selected_offset_ms,
            "search_min_ms": min_offset_ms,
            "search_max_ms": max_offset_ms,
            "pre_window_ms": pre_ms,
            "post_window_ms": post_ms,
            "time_base": TIME_BASE,
            "time_base_description": "CSV song_time_ms (shared recorder monotonic clock)",
            "score_signal": "max(abs(accel_magnitude_g - 1.0)) at the exact 10 ms song-time bin",
            "hand_score_signal": f"local max within ±{min(HAND_PEAK_SEARCH_MS, pre_ms, post_ms)} ms of the MIDI event center",
            "hand_peak_search_ms": min(HAND_PEAK_SEARCH_MS, pre_ms, post_ms),
            "uses_local_max_smoothing": False,
            "onset_group_tolerance_ms": ONSET_GROUP_TOLERANCE_MS,
            "window_overlap_span_ms": pre_ms + post_ms,
            "density_event_counts": dict(Counter(row["density_status"] for row in event_rows)),
            "single_label_eligible_count": sum(row["single_label_eligible"] == "true" for row in event_rows),
            "top_candidates": candidates,
        },
        "raw_csv": {"path": str(raw_csv.resolve()), "sha256": _sha256(raw_csv), **raw_info},
        "midi": {
            "path": str(midi_path.resolve()),
            "sha256": _sha256(midi_path),
            "format": midi_info["format"],
            "track_count": midi_info["track_count"],
            "ticks_per_beat": midi_info["ticks_per_beat"],
            "source_tempo_events": midi_info["tempo_events_in_source"],
            "track_end_ms_at_override_bpm": midi_info["track_end_ms"],
            "track_names": midi_info["track_names"],
            "source_note_on_count": midi_info["source_note_on_count"],
            "mapped_event_count": len(events),
            "unknown_note_counts": midi_info["unknown_note_counts"],
            "excluded_note_counts": midi_info["excluded_note_counts"],
            "mapped_note_counts": dict(Counter(event.note for event in events)),
        },
        "mapping": {
            str(note): {"zone_id": zone_id, "zone_name": zone_name}
            for note, (zone_id, zone_name) in sorted(ZONE_BY_NOTE.items())
        },
        "outputs": {
            "midi_events": str((output_dir / "midi_events.csv").resolve()),
            "candidate_windows": str((output_dir / "candidate_windows.csv").resolve()),
            "labeled_window_samples": str((output_dir / "labeled_window_samples.csv").resolve()),
        },
        "review_gate": [
            "Confirm the selected offset against WAV or a known sync hit.",
            "Review simultaneous notes and hand_candidate before per-hand training.",
            "Only promote verified events to final training labels.",
        ],
    }
    (output_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HoloGrip Raw CSV + MIDI 7-zone label pipeline")
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--midi", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM, help="Tempo override; confirmed 0805 value is 110")
    parser.add_argument("--offset-ms", type=int, help="Use a known CSV-to-MIDI offset instead of auto search")
    parser.add_argument("--search-min-ms", type=int, default=-5_000)
    parser.add_argument("--search-max-ms", type=int)
    parser.add_argument("--pre-ms", type=int, default=DEFAULT_PRE_MS)
    parser.add_argument("--post-ms", type=int, default=DEFAULT_POST_MS)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _self_test() -> None:
    assert ZONE_BY_NOTE[37][0] == ZONE_BY_NOTE[38][0] == 0
    assert len({zone_id for zone_id, _ in ZONE_BY_NOTE.values()}) == 7
    assert 36 in EXCLUDED_NOTES and 44 in EXCLUDED_NOTES
    assert abs(60_000.0 / (110.0 * 96) - 5.681818) < 0.00001
    print("midi_label_pipeline self-test: OK")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.raw_csv is None or args.midi is None or args.out_dir is None:
        parser.error("--raw-csv, --midi, and --out-dir are required unless --self-test is used")
    try:
        report = build_outputs(
            raw_csv=args.raw_csv,
            midi_path=args.midi,
            output_dir=args.out_dir,
            bpm=args.bpm,
            offset_ms=args.offset_ms,
            min_offset_ms=args.search_min_ms,
            max_offset_ms=args.search_max_ms,
            pre_ms=args.pre_ms,
            post_ms=args.post_ms,
        )
    except (OSError, ValueError, MidiParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"status={report['status']}")
    print(f"output_dir={args.out_dir.resolve()}")
    print(f"offset_ms={report['alignment']['offset_ms']}")
    print(f"midi_events={report['midi']['mapped_event_count']}")
    print(f"raw_rows={report['raw_csv']['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
