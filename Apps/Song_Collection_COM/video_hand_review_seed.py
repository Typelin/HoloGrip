"""Create a compact video-review queue for verified left/right hand labels.

This script never changes the source MIDI, raw glove CSV, or existing label
outputs.  It selects only single-label-eligible MIDI events and writes a JSON
queue that the companion HTML reviewer can load from the local filesystem.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_EVENT_CSV = (
    DATA_ROOT
    / "Derived"
    / "Song_Collection_COM"
    / "S20260805_P01_song01"
    / "midi_events.csv"
)
DEFAULT_OUTPUT = DEFAULT_EVENT_CSV.with_name("video_hand_review_seed_0807.json")
ONSET_GROUP_TOLERANCE_MS = 12.0
DEFAULT_CLEAN_WINDOW_MS = 180.0


def _as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {field!r} in MIDI event CSV") from exc


def _as_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {field!r} in MIDI event CSV") from exc


def build_seed(
    event_csv: Path,
    video_offset_ms: float,
    clip_pre_ms: int,
    clip_post_ms: int,
    clean_window_ms: float,
) -> dict[str, Any]:
    with event_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {
            "event_id",
            "midi_time_ms",
            "midi_note",
            "velocity",
            "zone_id",
            "zone_name",
            "hand_candidate",
            "hand_confidence",
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"CSV lacks required fields: {', '.join(missing)}")

        source_rows: list[dict[str, Any]] = []
        for row in reader:
            midi_time_ms = _as_float(row, "midi_time_ms")
            source_rows.append({
                "event_id": _as_int(row, "event_id"),
                "midi_time_ms": round(midi_time_ms, 3),
                "midi_note": _as_int(row, "midi_note"),
                "velocity": _as_int(row, "velocity"),
                "zone_id": _as_int(row, "zone_id"),
                "zone_name": row["zone_name"].strip(),
                "hand_candidate": row["hand_candidate"].strip().upper(),
                "hand_confidence": round(_as_float(row, "hand_confidence"), 6),
            })

    source_rows.sort(key=lambda item: (item["midi_time_ms"], item["event_id"]))
    groups: list[list[dict[str, Any]]] = []
    for row in source_rows:
        if not groups or row["midi_time_ms"] - groups[-1][0]["midi_time_ms"] > ONSET_GROUP_TOLERANCE_MS:
            groups.append([])
        groups[-1].append(row)

    events: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        previous_gap = group[0]["midi_time_ms"] - groups[group_index - 1][-1]["midi_time_ms"] if group_index else None
        next_gap = groups[group_index + 1][0]["midi_time_ms"] - group[-1]["midi_time_ms"] if group_index + 1 < len(groups) else None
        finite_gaps = [value for value in (previous_gap, next_gap) if value is not None]
        is_clean_single = len(group) == 1 and all(value >= clean_window_ms for value in finite_gaps)
        if not is_clean_single:
            continue
        row = group[0]
        events.append({
            **row,
            "video_time_ms": round(row["midi_time_ms"] + video_offset_ms, 3),
            "previous_onset_gap_ms": "" if previous_gap is None else round(previous_gap, 3),
            "next_onset_gap_ms": "" if next_gap is None else round(next_gap, 3),
            "nearest_onset_gap_ms": "" if not finite_gaps else round(min(finite_gaps), 3),
            "safe_half_window_ms": round(min(clean_window_ms / 2, min(finite_gaps) / 2) if finite_gaps else clean_window_ms / 2, 3),
        })

    if not events:
        raise ValueError("No single_label_eligible events were found")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_event_csv": str(event_csv.resolve()),
        "selection_rule": (
            f"single MIDI onset only; onset groups use {ONSET_GROUP_TOLERANCE_MS:g} ms; "
            f"both neighboring onset groups must be at least {clean_window_ms:g} ms away"
        ),
        "clean_window_total_ms": clean_window_ms,
        "video_offset_ms": video_offset_ms,
        "clip_pre_ms": clip_pre_ms,
        "clip_post_ms": clip_post_ms,
        "event_count": len(events),
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local video hand-review queue from clean MIDI events."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--video-offset-ms",
        type=float,
        default=14500.0,
        help="Video time = MIDI time + this offset. 0805 initial estimate: 14500 ms.",
    )
    parser.add_argument(
        "--clip-pre-ms",
        type=int,
        default=80,
        help="Model input starts this many milliseconds before the MIDI event.",
    )
    parser.add_argument(
        "--clip-post-ms",
        type=int,
        default=80,
        help="Model input ends this many milliseconds after the MIDI event.",
    )
    parser.add_argument("--clean-window-ms", type=float, default=DEFAULT_CLEAN_WINDOW_MS)
    args = parser.parse_args()

    if args.clip_pre_ms < 0 or args.clip_post_ms < 0:
        parser.error("clip durations must be zero or greater")
    if args.clean_window_ms <= 0:
        parser.error("clean window duration must be greater than zero")
    if not args.events.is_file():
        parser.error(f"event CSV was not found: {args.events}")

    payload = build_seed(
        args.events,
        args.video_offset_ms,
        args.clip_pre_ms,
        args.clip_post_ms,
        args.clean_window_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"events={payload['event_count']}")
    print(f"video_offset_ms={payload['video_offset_ms']:.3f}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
