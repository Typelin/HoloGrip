"""Split MIDI events into safe auto-label candidates and manual chord groups.

The input MIDI supplies time and drum-zone labels.  The glove CSV supplies a
left/right activity candidate only.  Simultaneous MIDI notes are intentionally
kept as one review group: they must never inherit the same hand automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import midi_label_pipeline as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = (
    PROJECT_ROOT
    / "CSV_Data"
    / "Song_Collection_COM"
    / "S20260805_P01_song01_raw_100hz_20260805_161628.csv"
)
DEFAULT_MIDI = PROJECT_ROOT / "流音給" / "Drum Midi_110BPM (0805).mid"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "Derived_Data"
    / "Song_Collection_COM"
    / "S20260805_P01_song01"
    / "hand_label_triage_0807"
)


def format_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def midi_groups(events: list[pipeline.MidiNoteEvent], tolerance_ms: float) -> list[list[pipeline.MidiNoteEvent]]:
    groups: list[list[pipeline.MidiNoteEvent]] = []
    for event in events:
        if not groups or event.time_ms - groups[-1][0].time_ms > tolerance_ms:
            groups.append([])
        groups[-1].append(event)
    return groups


def peak_in_window(values: list[float], center_ms: float, radius_ms: int) -> float:
    start = max(0, int(math.ceil((center_ms - radius_ms) / pipeline.BIN_MS)))
    end = min(len(values), int(math.floor((center_ms + radius_ms) / pipeline.BIN_MS)) + 1)
    return max(values[start:end], default=0.0)


def hand_metrics(
    energy: dict[str, list[float]],
    midi_time_ms: float,
    csv_offset_ms: float,
    window_radius_ms: int,
) -> dict[str, Any]:
    center_ms = midi_time_ms + csv_offset_ms
    left_peak = peak_in_window(energy["L"], center_ms, window_radius_ms)
    right_peak = peak_in_window(energy["R"], center_ms, window_radius_ms)
    dominant_peak = max(left_peak, right_peak)
    total_peak = left_peak + right_peak
    candidate = "L" if left_peak > right_peak else "R"
    share = dominant_peak / total_peak if total_peak > 0 else 0.0
    return {
        "aligned_csv_time_ms": round(center_ms, 3),
        "left_peak_g": round(left_peak, 6),
        "right_peak_g": round(right_peak, 6),
        "dominant_hand": candidate,
        "dominant_hand_share": round(share, 6),
        "dominant_peak_g": round(dominant_peak, 6),
    }


def event_row(
    event: pipeline.MidiNoteEvent,
    group_id: int,
    previous_gap_ms: float | None,
    next_gap_ms: float | None,
    metrics: dict[str, Any],
    video_offset_ms: float,
    status: str,
) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "onset_group_id": group_id,
        "midi_time_ms": format_ms(event.time_ms),
        "aligned_csv_time_ms": format_ms(metrics["aligned_csv_time_ms"]),
        "video_time_ms": format_ms(event.time_ms + video_offset_ms),
        "midi_note": event.note,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "velocity": event.velocity,
        "previous_onset_gap_ms": format_ms(previous_gap_ms),
        "next_onset_gap_ms": format_ms(next_gap_ms),
        "left_peak_g": f"{metrics['left_peak_g']:.6f}",
        "right_peak_g": f"{metrics['right_peak_g']:.6f}",
        "dominant_hand": metrics["dominant_hand"],
        "dominant_hand_share": f"{metrics['dominant_hand_share']:.6f}",
        "dominant_peak_g": f"{metrics['dominant_peak_g']:.6f}",
        "label_status": status,
    }


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    samples, raw_info = pipeline.read_raw_csv(args.raw_csv)
    midi_info = pipeline.parse_midi(args.midi, args.bpm)
    events = midi_info["events"]
    energy = pipeline._build_energy(samples, raw_info["duration_ms"])
    groups = midi_groups(events, args.group_tolerance_ms)

    auto_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    manual_groups: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, start=1):
        previous_gap = group[0].time_ms - groups[group_index - 2][-1].time_ms if group_index > 1 else None
        next_gap = groups[group_index][0].time_ms - group[-1].time_ms if group_index < len(groups) else None
        finite_gaps = [gap for gap in (previous_gap, next_gap) if gap is not None]
        clean_spacing = all(gap >= args.clean_gap_ms for gap in finite_gaps)
        metrics = hand_metrics(
            energy,
            group[0].time_ms,
            args.csv_offset_ms,
            args.window_radius_ms,
        )

        if len(group) > 1:
            manual_groups.append({
                "group_id": group_index,
                "midi_time_ms": round(group[0].time_ms, 3),
                "aligned_csv_time_ms": metrics["aligned_csv_time_ms"],
                "video_time_ms": round(group[0].time_ms + args.video_offset_ms, 3),
                "event_count": len(group),
                "previous_onset_gap_ms": None if previous_gap is None else round(previous_gap, 3),
                "next_onset_gap_ms": None if next_gap is None else round(next_gap, 3),
                "left_peak_g": metrics["left_peak_g"],
                "right_peak_g": metrics["right_peak_g"],
                "dominant_hand": metrics["dominant_hand"],
                "dominant_hand_share": metrics["dominant_hand_share"],
                "dominant_peak_g": metrics["dominant_peak_g"],
                "events": [
                    event_row(
                        event,
                        group_index,
                        previous_gap,
                        next_gap,
                        metrics,
                        args.video_offset_ms,
                        "manual_simultaneous_group",
                    )
                    for event in group
                ],
            })
            continue

        event = group[0]
        if not clean_spacing:
            excluded_rows.append(event_row(event, group_index, previous_gap, next_gap, metrics, args.video_offset_ms, "excluded_nearby_midi"))
            continue

        auto_ok = (
            metrics["dominant_hand_share"] >= args.dominant_hand_share
            and metrics["dominant_peak_g"] >= args.min_dominant_peak_g
        )
        status = "auto_accepted_high_confidence" if auto_ok else "excluded_single_hand_ambiguous"
        row = event_row(event, group_index, previous_gap, next_gap, metrics, args.video_offset_ms, status)
        if auto_ok:
            row["assigned_hand"] = metrics["dominant_hand"]
            auto_rows.append(row)
        else:
            excluded_rows.append(row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_csv_rows: list[dict[str, Any]] = []
    for group in manual_groups:
        manual_csv_rows.append({
            "group_id": group["group_id"],
            "midi_time_ms": format_ms(group["midi_time_ms"]),
            "aligned_csv_time_ms": format_ms(group["aligned_csv_time_ms"]),
            "video_time_ms": format_ms(group["video_time_ms"]),
            "event_count": group["event_count"],
            "event_ids": "|".join(str(event["event_id"]) for event in group["events"]),
            "notes": "|".join(str(event["midi_note"]) for event in group["events"]),
            "zones": "|".join(event["zone_name"] for event in group["events"]),
            "left_peak_g": f"{group['left_peak_g']:.6f}",
            "right_peak_g": f"{group['right_peak_g']:.6f}",
            "dominant_hand": group["dominant_hand"],
            "dominant_hand_share": f"{group['dominant_hand_share']:.6f}",
            "manual_status": "pending_left_right_assignment",
        })

    write_csv(output_dir / "auto_accepted_single_events.csv", auto_rows)
    write_csv(output_dir / "excluded_events.csv", excluded_rows)
    write_csv(output_dir / "manual_simultaneous_groups.csv", manual_csv_rows)
    manual_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_offset_ms": args.video_offset_ms,
        "clip_pre_ms": args.window_radius_ms,
        "clip_post_ms": args.window_radius_ms,
        "manual_group_count": len(manual_groups),
        "groups": manual_groups,
    }
    (output_dir / "manual_simultaneous_groups.json").write_text(
        json.dumps(manual_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    excluded_counts = Counter(row["label_status"] for row in excluded_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "onset_group_tolerance_ms": args.group_tolerance_ms,
            "clean_neighbor_gap_ms": args.clean_gap_ms,
            "csv_window_radius_ms": args.window_radius_ms,
            "dominant_hand_share_min": args.dominant_hand_share,
            "dominant_peak_min_g": args.min_dominant_peak_g,
            "csv_offset_ms": args.csv_offset_ms,
            "video_offset_ms": args.video_offset_ms,
        },
        "counts": {
            "all_midi_events": len(events),
            "auto_accepted_single_events": len(auto_rows),
            "manual_simultaneous_groups": len(manual_groups),
            "manual_simultaneous_events": sum(len(group["events"]) for group in manual_groups),
            "excluded": dict(excluded_counts),
        },
        "auto_by_zone": dict(sorted(Counter(row["zone_name"] for row in auto_rows).items())),
        "outputs": {
            "auto_accepted_single_events": str((output_dir / "auto_accepted_single_events.csv").resolve()),
            "manual_simultaneous_groups_csv": str((output_dir / "manual_simultaneous_groups.csv").resolve()),
            "manual_simultaneous_groups_json": str((output_dir / "manual_simultaneous_groups.json").resolve()),
            "excluded_events": str((output_dir / "excluded_events.csv").resolve()),
        },
    }
    (output_dir / "hand_label_triage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 0807 手別標記分流規則\n\n## 目的\n\n將 MIDI 的「時間與鼓點」結合手套 CSV 的左右活動量。單一、乾淨且單手明顯的事件先自動標記；同時 MIDI 事件必須整組交由影片確認。\n\n## 自動標記條件\n\n1. 同一 onset 群組只有 1 個 MIDI 事件。群組判定容許 {args.group_tolerance_ms:g} ms。\n2. 前後相鄰 MIDI 群組都距離至少 {args.clean_gap_ms:g} ms。\n3. 在 MIDI 對齊 CSV 時間前後 {args.window_radius_ms:g} ms 中，主手活動占比 `max(L, R) / (L + R)` 至少 {args.dominant_hand_share:.0%}。\n4. 主手活動峰值至少 {args.min_dominant_peak_g:.1f} g，避免兩手都幾乎未動時只有比例高。\n\n## 本次結果\n\n- 全部 7 鼓點 MIDI：{len(events):,} 筆\n- 自動標記成功：{len(auto_rows):,} 筆\n- 同時 MIDI 人工群組：{len(manual_groups):,} 組，合計 {sum(len(group['events']) for group in manual_groups):,} 筆 MIDI\n- 乾淨但單手不明確：{excluded_counts['excluded_single_hand_ambiguous']:,} 筆，暫不訓練\n- 靠近其他 MIDI 的單點：{excluded_counts['excluded_nearby_midi']:,} 筆，暫不訓練\n\n## 同時 MIDI 的處理\n\n同時 MIDI 群組不能把同一個右手候選複製給每一筆。例如 MIDI #9 小鼓（5.392 s）與 #10 Hi-Hat（5.398 s）只差 5.7 ms：它們是一組，須由影片確認各自左手或右手。\n\n## 輸出\n\n- `auto_accepted_single_events.csv`：可先使用的單手高信心候選。\n- `manual_simultaneous_groups.csv`／`.json`：每一組同時 MIDI 的完整事件列表，供影片逐組分配左右手。\n- `excluded_events.csv`：本輪不納入訓練的事件與原因。\n\n自動標記仍應在模型驗證時抽樣稽核；它是高信心自動候選，不是 MIDI 原生提供的左右手真值。\n"""
    (output_dir / "HAND_LABEL_TRIAGE_RULES_0807_ZH_TW.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="HoloGrip MIDI hand-label triage")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--midi", type=Path, default=DEFAULT_MIDI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bpm", type=float, default=110.0)
    parser.add_argument("--csv-offset-ms", type=float, default=16000.0)
    parser.add_argument("--video-offset-ms", type=float, default=14500.0)
    parser.add_argument("--window-radius-ms", type=int, default=80)
    parser.add_argument("--clean-gap-ms", type=float, default=180.0)
    parser.add_argument("--group-tolerance-ms", type=float, default=12.0)
    parser.add_argument("--dominant-hand-share", type=float, default=0.90)
    parser.add_argument("--min-dominant-peak-g", type=float, default=1.0)
    args = parser.parse_args()
    if not args.raw_csv.is_file() or not args.midi.is_file():
        parser.error("raw CSV or MIDI file was not found")
    if not 0.5 <= args.dominant_hand_share <= 1.0:
        parser.error("dominant-hand-share must be between 0.5 and 1.0")
    if args.window_radius_ms <= 0 or args.clean_gap_ms <= 0 or args.min_dominant_peak_g < 0:
        parser.error("window, gap, and peak settings must be positive")
    summary = build_outputs(args)
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
