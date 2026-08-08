"""Compare the video soundtrack, external WAV, and MIDI drum timeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_drum_audio_0808 import nearest_distance, probe_audio, read_midi_events, score_offset


def read_detected_onsets(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("is_detected_onset_35ms") or "").lower() == "true":
                try:
                    values.append(float(row["audio_time_ms"]))
                except (KeyError, ValueError):
                    continue
    return np.asarray(values, dtype=np.float64)


def best_offset(events: list[dict[str, object]], onsets: np.ndarray) -> dict[str, object]:
    scores = [score_offset(events, onsets, offset, 35.0) for offset in range(-20000, 30001, 10)]
    return sorted(
        scores,
        key=lambda item: (
            -int(item["matched_events"]),
            float(item["median_abs_error_ms"] or 999999),
            abs(int(item["offset_ms"])),
        ),
    )[0]


def nearest_record(target_ms: float, onsets: np.ndarray) -> dict[str, float | None]:
    if onsets.size == 0:
        return {"detected_ms": None, "error_ms": None}
    detected = float(onsets[np.argmin(np.abs(onsets - target_ms))])
    return {"detected_ms": round(detected, 3), "error_ms": round(abs(detected - target_ms), 3)}


def first_event(events: list[dict[str, object]], note: int) -> dict[str, object] | None:
    for event in events:
        if int(event["note"]) == note:
            return event
    return None


def event_check(event: dict[str, object], video_onsets: np.ndarray, wav_onsets: np.ndarray, video_offset: float, wav_offset: float) -> dict[str, object]:
    midi_ms = float(event["time_ms"])
    video_expected = midi_ms + video_offset
    wav_expected = midi_ms + wav_offset
    video_match = nearest_record(video_expected, video_onsets)
    wav_match = nearest_record(wav_expected, wav_onsets)
    return {
        "event_id": int(event["id"]),
        "zone": event["zone"],
        "midi_time_ms": round(midi_ms, 3),
        "video_expected_ms": round(video_expected, 3),
        "video_onset_ms": video_match["detected_ms"],
        "video_error_ms": video_match["error_ms"],
        "wav_expected_ms": round(wav_expected, 3),
        "wav_onset_ms": wav_match["detected_ms"],
        "wav_error_ms": wav_match["error_ms"],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    external = repo / "Data" / "External" / "FlowAudio_20260805"
    derived = repo / "Data" / "Derived" / "Song_Collection_COM" / "S20260805_P01_song01"
    parser = argparse.ArgumentParser(description="比較影片音訊、外部 WAV 與 MIDI 的時間關係")
    parser.add_argument("--video", type=Path, default=external / "IMG_6060.MOV")
    parser.add_argument("--wav", type=Path, default=external / "Drum Audio_110BPM (0805).wav")
    parser.add_argument("--midi", type=Path, default=derived / "midi_events.csv")
    parser.add_argument("--video-onsets", type=Path, default=derived / "video_alignment_0808" / "audio_onsets_0808.csv")
    parser.add_argument("--wav-onsets", type=Path, default=derived / "audio_alignment_0808" / "audio_onsets_0808.csv")
    parser.add_argument("--output", type=Path, default=derived / "video_alignment_0808" / "video_wav_midi_comparison_0808_ZH_TW.json")
    args = parser.parse_args()

    events = read_midi_events(args.midi)
    video_onsets = read_detected_onsets(args.video_onsets)
    wav_onsets = read_detected_onsets(args.wav_onsets)
    video_score = best_offset(events, video_onsets)
    wav_score = best_offset(events, wav_onsets)
    video_offset = float(video_score["offset_ms"])
    wav_offset = float(wav_score["offset_ms"])
    crash = first_event(events, 49)
    first_events = [event_check(event, video_onsets, wav_onsets, video_offset, wav_offset) for event in events[:12]]
    report = {
        "status": "video_wav_midi_comparison",
        "generated_from": {
            "video": str(args.video.resolve()),
            "wav": str(args.wav.resolve()),
            "midi_events": str(args.midi.resolve()),
            "video_onsets": str(args.video_onsets.resolve()),
            "wav_onsets": str(args.wav_onsets.resolve()),
        },
        "media": {"video": probe_audio(args.video), "wav": probe_audio(args.wav)},
        "first_detected_onset_ms": {
            "video": float(video_onsets[0]) if video_onsets.size else None,
            "wav": float(wav_onsets[0]) if wav_onsets.size else None,
        },
        "midi": {
            "event_count": len(events),
            "first_event_ms": float(events[0]["time_ms"]) if events else None,
            "first_crash": event_check(crash, video_onsets, wav_onsets, video_offset, wav_offset) if crash else None,
        },
        "alignment": {
            "midi_to_video_offset_ms": int(video_score["offset_ms"]),
            "midi_to_wav_offset_ms": int(wav_score["offset_ms"]),
            "video_minus_wav_offset_ms": int(video_score["offset_ms"] - wav_score["offset_ms"]),
            "video_score": video_score,
            "wav_score": wav_score,
        },
        "first_12_midi_event_checks": first_events,
        "interpretation": [
            "影片第一個偵測 onset 不一定是鼓擊；本次影片 2440 ms 的第一個 onset 沒有對應第一筆 MIDI，因此不拿它當第一個鼓點。",
            "第一個 Crash 的 MIDI、影片音訊與外部 WAV 分別在 4318.182 ms、約 18840 ms、約 4320 ms，支持 MIDI → 影片 +14520 ms 與 MIDI → WAV +0 ms。",
            "影片與 WAV 的共同 onset 規律支持 +14520 ms，但仍應以影片畫面確認手別與鼓面；音訊本身不能分辨左右手。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"midi_to_video_offset_ms={video_score['offset_ms']}")
    print(f"midi_to_wav_offset_ms={wav_score['offset_ms']}")
    print(f"video_minus_wav_offset_ms={video_score['offset_ms'] - wav_score['offset_ms']}")
    print(f"first_crash={report['midi']['first_crash']}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
