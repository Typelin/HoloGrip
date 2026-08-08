"""Analyze a rendered drum WAV against the MIDI event timeline.

The result is a review aid for timing and suspicious duplicate events. It does
not infer left/right hand labels from a mixed audio file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, medfilt, stft


ZONE_NAMES = {
    37: "小鼓",
    38: "小鼓",
    43: "落地 Tom",
    45: "中音 Tom",
    46: "Hi-Hat",
    48: "高音 Tom",
    49: "Crash",
    51: "Ride",
}


def default_paths(repo: Path) -> tuple[Path, Path, Path]:
    external = repo / "Data" / "External" / "FlowAudio_20260805"
    derived = repo / "Data" / "Derived" / "Song_Collection_COM" / "S20260805_P01_song01"
    return (
        external / "Drum Audio_110BPM (0805).wav",
        derived / "midi_events.csv",
        derived / "audio_alignment_0808",
    )


def probe_audio(path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,channel_layout,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
        return {
            "codec": stream.get("codec_name"),
            "sample_rate_hz": int(stream.get("sample_rate") or 0),
            "channels": int(stream.get("channels") or 0),
            "channel_layout": stream.get("channel_layout"),
            "duration_s": duration,
        }
    with wave.open(str(path), "rb") as handle:
        return {
            "codec": f"pcm_{handle.getsampwidth() * 8}bit",
            "sample_rate_hz": handle.getframerate(),
            "channels": handle.getnchannels(),
            "channel_layout": None,
            "duration_s": handle.getnframes() / handle.getframerate(),
        }


def decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
        if audio.size:
            return audio
    with wave.open(str(path), "rb") as handle:
        source_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("找不到 ffmpeg，且 fallback 只支援 16/32-bit WAV。")
    mono = values.reshape(-1, channels).mean(axis=1)
    if source_rate != sample_rate:
        target_size = int(round(mono.size * sample_rate / source_rate))
        old_x = np.linspace(0.0, 1.0, mono.size, endpoint=False)
        new_x = np.linspace(0.0, 1.0, target_size, endpoint=False)
        mono = np.interp(new_x, old_x, mono).astype(np.float32)
    return mono


def onset_envelope(audio: np.ndarray, sample_rate: int, hop_ms: int = 10) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, round(sample_rate * hop_ms / 1000))
    frame = max(hop * 4, round(sample_rate * 0.04))
    _, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=frame,
        noverlap=frame - hop,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum).astype(np.float32)
    if magnitude.shape[1] < 2:
        return times * 1000.0, np.zeros_like(times, dtype=np.float32)
    flux = np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0.0).sum(axis=0)
    flux = np.pad(flux, (1, 0))
    baseline = medfilt(flux, kernel_size=9 if flux.size >= 9 else 3)
    envelope = np.maximum(flux - baseline, 0.0)
    scale = np.percentile(envelope, 99.5) or 1.0
    return times * 1000.0, (envelope / scale).astype(np.float32)


def detect_onsets(times_ms: np.ndarray, envelope: np.ndarray, min_gap_ms: float = 35.0) -> tuple[np.ndarray, dict[str, float]]:
    if not envelope.size:
        return np.array([], dtype=np.float64), {"threshold": 0.0, "prominence": 0.0}
    robust = np.median(envelope) + 1.4826 * np.median(np.abs(envelope - np.median(envelope)))
    threshold = max(float(np.percentile(envelope, 72)), float(robust * 1.35), 0.035)
    prominence = max(float(np.percentile(envelope, 80) * 0.18), 0.025)
    hop_ms = float(np.median(np.diff(times_ms))) if times_ms.size > 1 else 10.0
    distance = max(1, round(min_gap_ms / hop_ms))
    peaks, _ = find_peaks(envelope, height=threshold, prominence=prominence, distance=distance)
    return times_ms[peaks], {"threshold": threshold, "prominence": prominence, "min_gap_ms": min_gap_ms}


def read_midi_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                time_ms = float(row.get("midi_time_ms", ""))
                note = int(float(row.get("midi_note", "")))
            except ValueError:
                continue
            events.append({
                "id": int(float(row.get("event_id", len(events) + 1))),
                "time_ms": time_ms,
                "note": note,
                "zone": row.get("zone_name") or ZONE_NAMES.get(note, "未知"),
            })
    return events


def nearest_distance(target: float, values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    index = int(np.searchsorted(values, target))
    candidates = values[max(0, index - 1) : min(values.size, index + 1)]
    return float(np.min(np.abs(candidates - target)))


def score_offset(events: list[dict[str, object]], onsets: np.ndarray, offset_ms: int, tolerance_ms: float) -> dict[str, object]:
    distances = np.array([nearest_distance(float(event["time_ms"]) + offset_ms, onsets) for event in events])
    matched = distances <= tolerance_ms
    errors = distances[matched]
    return {
        "offset_ms": offset_ms,
        "matched_events": int(matched.sum()),
        "match_rate": float(matched.mean()) if len(events) else 0.0,
        "median_abs_error_ms": float(np.median(errors)) if errors.size else None,
        "mean_abs_error_ms": float(np.mean(errors)) if errors.size else None,
    }


def per_zone(events: list[dict[str, object]], onsets: np.ndarray, offset_ms: int, tolerance_ms: float) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        grouped[int(event["note"])].append(event)
    result = []
    for note, rows in sorted(grouped.items()):
        score = score_offset(rows, onsets, offset_ms, tolerance_ms)
        result.append({"midi_note": note, "zone_name": ZONE_NAMES.get(note, rows[0]["zone"]), **score})
    return result


def audit_same_note_pairs(events: list[dict[str, object]], onsets: np.ndarray, offset_ms: int, tolerance_ms: float) -> list[dict[str, object]]:
    by_note: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        by_note[int(event["note"])].append(event)
    audits: list[dict[str, object]] = []
    for note, rows in sorted(by_note.items()):
        rows.sort(key=lambda item: float(item["time_ms"]))
        for first, second in zip(rows, rows[1:]):
            gap = float(second["time_ms"]) - float(first["time_ms"])
            if gap <= 20.0:
                first_target = float(first["time_ms"]) + offset_ms
                second_target = float(second["time_ms"]) + offset_ms
                nearby = onsets[(onsets >= first_target - tolerance_ms) & (onsets <= second_target + tolerance_ms)]
                first_error = nearest_distance(first_target, onsets)
                second_error = nearest_distance(second_target, onsets)
                status = "音訊候選有兩個 onset" if nearby.size >= 2 else "音訊目前無法分辨兩次"
                audits.append({
                    "midi_note": note,
                    "zone_name": ZONE_NAMES.get(note, first["zone"]),
                    "first_event_id": first["id"],
                    "second_event_id": second["id"],
                    "midi_gap_ms": round(gap, 3),
                    "first_audio_error_ms": round(first_error, 3) if np.isfinite(first_error) else None,
                    "second_audio_error_ms": round(second_error, 3) if np.isfinite(second_error) else None,
                    "fast_onsets_in_pair_window": int(nearby.size),
                    "status": status,
                })
    return audits


def write_onsets(path: Path, times_ms: np.ndarray, envelope: np.ndarray, onset_times: np.ndarray, fast_onset_times: np.ndarray) -> None:
    onset_set = {round(float(value), 1) for value in onset_times}
    fast_onset_set = {round(float(value), 1) for value in fast_onset_times}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_time_ms", "onset_envelope_normalized", "is_detected_onset_35ms", "is_fast_onset_12ms"])
        for time_ms, value in zip(times_ms, envelope):
            rounded = round(float(time_ms), 1)
            writer.writerow([f"{time_ms:.1f}", f"{float(value):.6f}", "true" if rounded in onset_set else "false", "true" if rounded in fast_onset_set else "false"])


def analyze(audio_path: Path, midi_path: Path, output_dir: Path) -> dict[str, object]:
    audio_meta = probe_audio(audio_path)
    analysis_rate = 24000
    audio = decode_mono(audio_path, analysis_rate)
    times_ms, envelope = onset_envelope(audio, analysis_rate, hop_ms=5)
    onset_times, detector = detect_onsets(times_ms, envelope, min_gap_ms=35.0)
    fast_onset_times, fast_detector = detect_onsets(times_ms, envelope, min_gap_ms=12.0)
    events = read_midi_events(midi_path)
    candidate_offsets = range(-20000, 30001, 10)
    scores = [score_offset(events, onset_times, offset, 35.0) for offset in candidate_offsets]
    ranked = sorted(scores, key=lambda item: (-int(item["matched_events"]), float(item["median_abs_error_ms"] or 999999), abs(int(item["offset_ms"]))))
    best = ranked[0] if ranked else score_offset(events, onset_times, 0, 35.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicate_audit = audit_same_note_pairs(events, fast_onset_times, int(best["offset_ms"]), 35.0)
    write_onsets(output_dir / "audio_onsets_0808.csv", times_ms, envelope, onset_times, fast_onset_times)
    report = {
        "status": "audio_alignment_candidate",
        "generated_from": {
            "audio": str(audio_path.resolve()),
            "midi_events": str(midi_path.resolve()),
        },
        "audio": {**audio_meta, "analysis_sample_rate_hz": analysis_rate, "detected_onset_count": int(onset_times.size), "fast_detected_onset_count": int(fast_onset_times.size), "first_onset_ms": float(onset_times[0]) if onset_times.size else None, "last_onset_ms": float(onset_times[-1]) if onset_times.size else None, "detector": detector, "fast_detector": fast_detector},
        "midi": {"event_count": len(events), "first_event_ms": float(events[0]["time_ms"]) if events else None, "last_event_ms": float(events[-1]["time_ms"]) if events else None},
        "alignment": {"tolerance_ms": 35.0, "search_range_ms": [-20000, 30000], "best_offset_ms": best["offset_ms"], "best_score": best, "top_candidates": ranked[:20], "per_zone": per_zone(events, onset_times, int(best["offset_ms"]), 35.0)},
        "same_note_under_20ms": {"pair_count": len(duplicate_audit), "audio_supported_pair_count": sum(row["status"] == "音訊候選有兩個 onset" for row in duplicate_audit), "audio_unresolved_pair_count": sum(row["status"] == "音訊目前無法分辨兩次" for row in duplicate_audit), "pairs": duplicate_audit},
        "interpretation": [
            "WAV onset 只能證明音訊中有時間上的聲音瞬間，不能單靠混音可靠分辨左手與右手。",
            "若同一 MIDI note 在 20 ms 內有兩筆，但 WAV 只有一個可辨識 onset，列為需要影片與流音確認的疑似重複，不自動刪除。",
            "若 WAV 與 MIDI 的最佳偏移穩定且命中率高，可增加對齊信心；仍不能取代 CSV 活動峰與影片人工手別標記。",
        ],
    }
    (output_dir / "audio_alignment_report_0808.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_audio, default_midi, default_output = default_paths(repo)
    parser = argparse.ArgumentParser(description="分析 WAV 與 MIDI 事件的時間對齊候選")
    parser.add_argument("--audio", type=Path, default=default_audio)
    parser.add_argument("--midi", type=Path, default=default_midi)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    report = analyze(args.audio, args.midi, args.output_dir)
    best = report["alignment"]["best_score"]
    print(f"audio_duration_s={report['audio']['duration_s']:.6f}")
    print(f"detected_onsets_35ms={report['audio']['detected_onset_count']}")
    print(f"detected_onsets_12ms={report['audio']['fast_detected_onset_count']}")
    print(f"best_offset_ms={best['offset_ms']}")
    print(f"matched_events={best['matched_events']}/{report['midi']['event_count']}")
    print(f"match_rate={best['match_rate']:.4f}")
    print(f"same_note_pairs_under_20ms={report['same_note_under_20ms']['pair_count']}")
    print(f"audio_supported_fast_pairs={report['same_note_under_20ms']['audio_supported_pair_count']}")
    print(f"report={args.output_dir.resolve() / 'audio_alignment_report_0808.json'}")
    print(f"onsets={args.output_dir.resolve() / 'audio_onsets_0808.csv'}")


if __name__ == "__main__":
    main()
