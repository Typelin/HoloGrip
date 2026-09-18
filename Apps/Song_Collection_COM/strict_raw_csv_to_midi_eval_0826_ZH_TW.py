"""Strict Song2 evaluation: Raw CSV -> peak -> drum class -> original MIDI.

Training uses previously approved human labels, which is necessary to teach a
seven-class classifier.  At evaluation/inference time this script never reads
MIDI or Ground Truth to select a peak or an IMU window:

    Raw CSV -> HitDetector -> detected peak +/- 80 ms -> MLP drum prediction

Only after the full prediction list is complete, it compares the list one-to-
one against the original cleaned MIDI events on the shared song timeline.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))

import midi_label_pipeline as midi_pipeline  # noqa: E402
from product_hit_and_zone import (  # noqa: E402
    CLEANED_DIR,
    GT_CSV_PATH,
    PRODUCT_DETECTOR_KWARGS,
    RAW_CSV_PATH,
    WINDOW_HALF_MS,
    ZONE_MAP,
    ZONE_NAMES,
    load_gt,
    load_raw_rows,
    replay_hits,
    rows_to_samples,
    slice_window,
    window_features,
)

OUT_DIR = Path(CLEANED_DIR)
MIDI_PATH = OUT_DIR / "0812T1127_去除雜訊T1213.mid"
REPORT_JSON = OUT_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_0826_ZH_TW.json"
REPORT_MD = OUT_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_0826_ZH_TW.md"
PREDICTIONS_CSV = OUT_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_預測對照_0826.csv"

BPM = 110.0
MIDI_TO_CSV_OFFSET_MS = 6027.0
MATCH_TOLERANCE_MS = 80.0
BLOCK_MS = 4000.0
SEED = 42


def make_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200, random_state=SEED)),
    ])


def build_training_dataset(gt_rows: list[dict[str, str]], samples_by_hand: dict[str, list[dict[str, float]]]):
    """Build teacher-labelled training rows only; not used to choose test peaks."""
    rows, X, y, groups = [], [], [], []
    for row in gt_rows:
        hand, drum = row.get("final_hand", ""), row.get("drum", "")
        if hand not in {"L", "R"} or drum not in ZONE_MAP:
            continue
        center_ms = float(row["csv_center_ms"])
        feature = window_features(slice_window(samples_by_hand[hand], center_ms, WINDOW_HALF_MS), center_ms)
        if feature is None:
            continue
        rows.append({"event_id": int(row["event_id"]), "group": int(center_ms // BLOCK_MS)})
        X.append(feature)
        y.append(ZONE_MAP[drum])
        groups.append(int(center_ms // BLOCK_MS))
    return rows, np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(groups, dtype=int)


def train_fold_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Return a model for each held-out 4 s block, plus a final display model."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    group_models: dict[int, Pipeline] = {}
    fold_rows = []
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = make_mlp()
        model.fit(X[train_idx], y[train_idx])
        test_groups = sorted(set(int(value) for value in groups[test_idx]))
        for group in test_groups:
            group_models[group] = model
        fold_rows.append({"fold": fold_index, "held_out_blocks": test_groups, "train_events": len(train_idx), "test_events": len(test_idx)})
    final_model = make_mlp()
    final_model.fit(X, y)
    return group_models, final_model, fold_rows


def greedy_time_match(preds: list[dict[str, Any]], midi_events: list[dict[str, Any]], require_same_drum: bool):
    """Maximum-cardinality chronological 1:1 matching on fixed +/-80 ms intervals."""
    if require_same_drum:
        drums = sorted(set(item["drum"] for item in midi_events) | set(item["pred_drum"] for item in preds))
        partitions = [
            (
                [item for item in preds if item["pred_drum"] == drum],
                [item for item in midi_events if item["drum"] == drum],
            )
            for drum in drums
        ]
    else:
        partitions = [(preds, midi_events)]

    matched, unmatched_pred_ids, unmatched_midi_ids = [], set(), set()
    for pred_part, midi_part in partitions:
        ordered_pred = sorted(pred_part, key=lambda item: (item["peak_ms"], item["pred_id"]))
        ordered_midi = sorted(midi_part, key=lambda item: (item["csv_time_ms"], item["midi_event_id"]))
        p_index = m_index = 0
        while p_index < len(ordered_pred) and m_index < len(ordered_midi):
            pred = ordered_pred[p_index]
            midi = ordered_midi[m_index]
            delta = pred["peak_ms"] - midi["csv_time_ms"]
            if delta < -MATCH_TOLERANCE_MS:
                unmatched_pred_ids.add(pred["pred_id"])
                p_index += 1
            elif delta > MATCH_TOLERANCE_MS:
                unmatched_midi_ids.add(midi["midi_event_id"])
                m_index += 1
            else:
                matched.append({
                    "pred_id": pred["pred_id"],
                    "midi_event_id": midi["midi_event_id"],
                    "time_error_ms": round(abs(delta), 3),
                })
                p_index += 1
                m_index += 1
        unmatched_pred_ids.update(item["pred_id"] for item in ordered_pred[p_index:])
        unmatched_midi_ids.update(item["midi_event_id"] for item in ordered_midi[m_index:])
    return matched, unmatched_pred_ids, unmatched_midi_ids


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    if not MIDI_PATH.exists():
        raise FileNotFoundError(f"找不到原始清理版 MIDI：{MIDI_PATH}")

    print("1/5 讀取已核准訓練標記、Raw CSV、原始 MIDI")
    gt_rows = load_gt()
    raw_rows = load_raw_rows()
    samples_by_hand = rows_to_samples(raw_rows)
    parsed_midi = midi_pipeline.parse_midi(MIDI_PATH, BPM)
    midi_events = [
        {
            "midi_event_id": event.event_id,
            "midi_time_ms": round(float(event.time_ms), 3),
            "csv_time_ms": round(float(event.time_ms) + MIDI_TO_CSV_OFFSET_MS, 3),
            "note": event.note,
            "drum": event.zone_name,
        }
        for event in parsed_midi["events"]
        if event.zone_name in ZONE_MAP
    ]

    print("2/5 訓練 5-fold 外推 MLP（僅教師標記階段使用答案）")
    _training_rows, X, y, groups = build_training_dataset(gt_rows, samples_by_hand)
    group_models, final_model, fold_rows = train_fold_models(X, y, groups)

    print("3/5 Raw CSV 自行找 Peak 並分類；此階段不讀 MIDI 或 Ground Truth")
    detections = replay_hits(raw_rows, PRODUCT_DETECTOR_KWARGS)
    predictions = []
    for pred_id, detection in enumerate(detections, start=1):
        peak_ms = float(detection["peak_ms"])
        block = int(peak_ms // BLOCK_MS)
        model = group_models.get(block, final_model)
        feature = window_features(
            slice_window(samples_by_hand[detection["hand"]], peak_ms, WINDOW_HALF_MS),
            peak_ms,
        )
        probabilities = model.predict_proba([feature])[0]
        zone_id = int(np.argmax(probabilities))
        predictions.append({
            "pred_id": pred_id,
            "hand": detection["hand"],
            "peak_ms": round(peak_ms, 3),
            "trigger_ms": round(float(detection["trigger_ms"]), 3),
            "peak_accel_g": round(float(detection["peak_accel_g"]), 4),
            "prediction_block": block,
            "pred_drum": ZONE_NAMES[zone_id],
            "pred_confidence": round(float(probabilities[zone_id]), 6),
        })

    print("4/5 全部預測完成後，才與原始 MIDI 做 +/-80 ms 一對一比對")
    temporal_matches, _, _ = greedy_time_match(predictions, midi_events, require_same_drum=False)
    exact_matches, exact_unmatched_pred, exact_unmatched_midi = greedy_time_match(
        predictions, midi_events, require_same_drum=True
    )
    exact_by_pred = {item["pred_id"]: item for item in exact_matches}
    midi_by_id = {item["midi_event_id"]: item for item in midi_events}

    true_positive = len(exact_matches)
    false_positive = len(exact_unmatched_pred)
    false_negative = len(exact_unmatched_midi)
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)

    per_drum = {}
    for drum in ZONE_NAMES:
        midi_count = sum(item["drum"] == drum for item in midi_events)
        pred_count = sum(item["pred_drum"] == drum for item in predictions)
        tp_count = sum(midi_by_id[item["midi_event_id"]]["drum"] == drum for item in exact_matches)
        drum_precision = safe_ratio(tp_count, pred_count)
        drum_recall = safe_ratio(tp_count, midi_count)
        per_drum[drum] = {
            "midi_events": midi_count,
            "raw_csv_predictions": pred_count,
            "exact_correct": tp_count,
            "precision": drum_precision,
            "recall": drum_recall,
            "f1": safe_ratio(2 * drum_precision * drum_recall, drum_precision + drum_recall),
        }

    csv_rows = []
    for prediction in predictions:
        matched = exact_by_pred.get(prediction["pred_id"])
        target = midi_by_id.get(matched["midi_event_id"]) if matched else None
        csv_rows.append({
            **prediction,
            "strict_status": "正確" if matched else "多偵測或鼓位錯",
            "matched_midi_event_id": target["midi_event_id"] if target else "",
            "matched_midi_time_ms": target["midi_time_ms"] if target else "",
            "matched_midi_drum": target["drum"] if target else "",
            "time_error_ms": matched["time_error_ms"] if matched else "",
        })
    for event_id in sorted(exact_unmatched_midi):
        target = midi_by_id[event_id]
        csv_rows.append({
            "pred_id": "",
            "hand": "",
            "peak_ms": "",
            "trigger_ms": "",
            "peak_accel_g": "",
            "prediction_block": "",
            "pred_drum": "",
            "pred_confidence": "",
            "strict_status": "漏打或鼓位錯",
            "matched_midi_event_id": target["midi_event_id"],
            "matched_midi_time_ms": target["midi_time_ms"],
            "matched_midi_drum": target["drum"],
            "time_error_ms": "",
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "Raw CSV 自行找 Peak -> 七鼓點 MLP -> 311 筆清理後正確 MIDI 事後一對一評分",
        "inference_contract": "推論階段未讀 MIDI、Ground Truth、csv_center_ms 或 final_hand；僅用 Raw CSV 與 HAND_ID。",
        "training_contract": "MLP 訓練使用已核准人工 Ground Truth；這是監督式學習必要的教師答案。",
        "source_raw_csv": RAW_CSV_PATH,
        "source_reference_midi": str(MIDI_PATH),
        "source_training_ground_truth": GT_CSV_PATH,
        "bpm": BPM,
        "midi_to_csv_global_offset_ms": MIDI_TO_CSV_OFFSET_MS,
        "window_half_ms_after_detected_peak": WINDOW_HALF_MS,
        "midi_match_tolerance_ms": MATCH_TOLERANCE_MS,
        "midi_event_count": len(midi_events),
        "raw_csv_detection_count": len(predictions),
        "training_event_count": len(y),
        "cross_validation": {
            "scheme": "StratifiedGroupKFold, 5 folds, group=floor(time_ms/4000)",
            "folds": fold_rows,
            "unseen_block_policy": "A prediction in a block without any labelled event uses the final model; it cannot create a true positive without a MIDI event and remains fully counted if extra.",
        },
        "temporal_hit_detection_regardless_of_drum": {
            "matched": len(temporal_matches),
            "midi_total": len(midi_events),
            "recall": safe_ratio(len(temporal_matches), len(midi_events)),
        },
        "strict_exact_event_metrics": {
            "definition": "同一預測需在 MIDI 對應 CSV 時間 +/-80 ms 且七鼓點類別相同，才算正確；MIDI 與預測均一對一使用。",
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_drum": per_drum,
        "notes": [
            "這是同一位演奏者、同一首 Song2 的開發期時間區塊外推驗證，非跨人跨歌曲泛化測試。",
            "MIDI 對 CSV 僅使用一個預先確認的全域偏移 +6027 ms；MIDI 不用於找個別 Peak 或擷取推論窗口。",
            "原始 MIDI 沒有左右手資訊，因此嚴格 MIDI 直接評分衡量時間與鼓點，不把手別列入正確條件；手別來自手套 HAND_ID。",
        ],
    }

    with open(PREDICTIONS_CSV, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Song2 Raw CSV 到 311 筆正確 MIDI 嚴格驗證報告",
        "",
        "## 這份報告驗證什麼",
        "",
        "推論時只輸入 Raw CSV 原始 G 力與航向角。系統自行找擊打 Peak，從該 Peak 前後 ±80 ms 取 IMU 特徵，再預測七種鼓點。直到所有預測都完成，才以 311 筆清理後正確 MIDI + 全域偏移 +6,027 ms 做 ±80 ms 一對一評分。",
        "",
        "## 完整端到端結果",
        "",
        f"- 清理後正確 MIDI 事件：{len(midi_events)} 筆",
        f"- Raw CSV 系統輸出：{len(predictions)} 筆",
        f"- 時間上找到 MIDI（不看鼓位）：{len(temporal_matches)} / {len(midi_events)} = {safe_ratio(len(temporal_matches), len(midi_events))*100:.1f}%",
        f"- 時間與鼓位都正確（True Positive）：{true_positive} 筆",
        f"- 完整事件 Precision：{precision*100:.1f}%",
        f"- 完整事件 Recall：{recall*100:.1f}%",
        f"- 完整事件 F1：{f1*100:.1f}%",
        "",
        "## 嚴格計分定義",
        "",
        "每一筆 Raw CSV 預測只能配對一筆 MIDI；每一筆 MIDI 也只能被配對一次。預測 Peak 與 MIDI 對應 CSV 時間差不超過 ±80 ms，且七鼓點類別相同，才算完全正確。多偵測、漏打、或時間對到但鼓位錯，都會扣分。",
        "",
        "## 各鼓點",
        "",
        "| 鼓點 | MIDI 真值 | Raw CSV 輸出 | 完全正確 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for drum in ZONE_NAMES:
        row = per_drum[drum]
        lines.append(f"| {drum} | {row['midi_events']} | {row['raw_csv_predictions']} | {row['exact_correct']} | {row['precision']*100:.1f}% | {row['recall']*100:.1f}% | {row['f1']*100:.1f}% |")
    lines += [
        "",
        "## 防止誤解",
        "",
        "- 這份分數不是 MIDI 中心窗口分類率；MIDI 不會參與推論時的 Peak 選擇或窗口位置。",
        "- 訓練仍使用已標記資料，否則監督式七鼓點模型沒有學習答案。",
        "- 原始 MIDI 無法提供左右手，因此本表的正確條件是『時間 + 鼓點』；左右手由手套 HAND_ID 直接提供。",
        "- 目前仍是同人、同曲的開發驗證，不可說成跨歌曲或跨演奏者的產品泛化率。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("5/5 寫出嚴格報告")
    print(f"完整事件 Precision {precision*100:.1f}% | Recall {recall*100:.1f}% | F1 {f1*100:.1f}%")
    print(f"JSON：{REPORT_JSON}")
    print(f"MD：{REPORT_MD}")
    print(f"CSV：{PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
