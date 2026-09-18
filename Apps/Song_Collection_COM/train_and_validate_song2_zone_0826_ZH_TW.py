"""Train and honestly validate the Song2 seven-drum IMU classifier.

This script keeps MIDI/ground truth out of model features. MIDI and the final
hand/zone labels are used only to define training targets and to score output.
It reports both event-level cross-validation and an end-to-end Raw CSV replay
score using fold models that did not train on the scored event.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_hit_and_zone import (  # noqa: E402
    CLEANED_DIR,
    GT_CSV_PATH,
    RAW_CSV_PATH,
    PRODUCT_DETECTOR_KWARGS,
    MATCH_TOL_MS,
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
REPORT_PATH = OUT_DIR / "Song2_七鼓點模型真正驗證報告_0826_ZH_TW.json"
REPORT_MD_PATH = OUT_DIR / "Song2_七鼓點模型真正驗證報告_0826_ZH_TW.md"
MODEL_PATH = OUT_DIR / "hologrip_song2_七鼓點模型_真正驗證版_0826.joblib"
PRED_PATH = OUT_DIR / "Song2_七鼓點模型_fold外推預測_0826.csv"
BLOCK_MS = 4000.0
SEED = 42
EVAL_MATCH_TOL_MS = WINDOW_HALF_MS

FEATURE_NAMES = [
    "max_mag", "mean_mag", "std_mag", "energy",
    "max_ax", "max_ay", "max_az", "jerk_max",
    "mean_yaw", "mean_pitch", "mean_roll",
    "std_yaw", "std_pitch", "center_yaw", "center_pitch",
    "swing_depth", "yaw_range",
]


def build_dataset(gt_rows, samples_by_hand):
    rows, X, y, groups = [], [], [], []
    for row in gt_rows:
        hand = row["final_hand"]
        if hand not in {"L", "R"} or row["drum"] not in ZONE_MAP:
            continue
        center = float(row["csv_center_ms"])
        slc = slice_window(samples_by_hand[hand], center, WINDOW_HALF_MS)
        feat = window_features(slc, center)
        if feat is None:
            continue
        rows.append({
            "event_id": int(row["event_id"]),
            "csv_center_ms": center,
            "hand": hand,
            "drum": row["drum"],
            "n_samples": len(slc),
        })
        X.append(feat)
        y.append(ZONE_MAP[row["drum"]])
        groups.append(int(center // BLOCK_MS))
    return rows, np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(groups, dtype=int)


def models():
    return {
        "多層感知器神經網路（MLP）": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200, random_state=SEED)),
        ]),
        "線性支援向量機（Linear SVM）": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)),
        ]),
        "徑向基核支援向量機（RBF SVM）": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ]),
        "邏輯斯蒂迴歸（Logistic Regression）": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "梯度提升分類器（Gradient Boosting）": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED
        ),
        "隨機森林（Random Forest）": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=SEED, class_weight="balanced_subsample"
        ),
    }


def cv_splits(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))
    if len(splits) != 5:
        raise RuntimeError("交叉驗證沒有產生 5 個時間區塊")
    return splits


def evaluate_models(X, y, groups):
    splits = cv_splits(X, y, groups)
    results = {}
    predictions = {}
    for name, clf in models().items():
        fold_metrics = []
        pred = np.full(len(y), -1, dtype=int)
        for train_idx, test_idx in splits:
            clf.fit(X[train_idx], y[train_idx])
            p = clf.predict(X[test_idx]).astype(int)
            pred[test_idx] = p
            fold_metrics.append({
                "accuracy": float(accuracy_score(y[test_idx], p)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], p)),
                "macro_f1": float(f1_score(y[test_idx], p, average="macro", zero_division=0)),
            })
        results[name] = {
            "folds": fold_metrics,
            "accuracy_mean": float(accuracy_score(y, pred)),
            "balanced_accuracy_mean": float(balanced_accuracy_score(y, pred)),
            "macro_f1_mean": float(f1_score(y, pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        }
        predictions[name] = pred
    best_name = max(results, key=lambda n: (results[n]["macro_f1_mean"], results[n]["accuracy_mean"]))
    return results, predictions, best_name, splits


def match_predictions(gt_rows, detections):
    gt = [
        {
            "event_id": int(row["event_id"]),
            "csv_center_ms": float(row["csv_center_ms"]),
            "hand": row["final_hand"],
            "drum": row["drum"],
        }
        for row in gt_rows if row["final_hand"] in {"L", "R"}
    ]
    used = set()
    matched, misses = [], []
    for g in gt:
        candidates = [
            (i, abs(float(p["trigger_ms"]) - g["csv_center_ms"]))
            for i, p in enumerate(detections)
            if i not in used and p["hand"] == g["hand"]
        ]
        candidates = [item for item in candidates if item[1] <= EVAL_MATCH_TOL_MS]
        if not candidates:
            misses.append(g)
            continue
        pred_i, dt = min(candidates, key=lambda item: item[1])
        used.add(pred_i)
        matched.append({"gt": g, "pred_index": pred_i, "dt_ms": dt})
    extras = [i for i in range(len(detections)) if i not in used]
    return matched, misses, extras


def fold_end_to_end(gt_rows, dataset_rows, X, y, groups, samples_by_hand, detections, best_name):
    splits = cv_splits(X, y, groups)
    event_to_index = {row["event_id"]: i for i, row in enumerate(dataset_rows)}
    matched, misses, extras = match_predictions(gt_rows, detections)
    scored = []
    for train_idx, test_idx in splits:
        test_set = set(test_idx.tolist())
        model = models()[best_name]
        model.fit(X[train_idx], y[train_idx])
        for item in matched:
            event_idx = event_to_index.get(item["gt"]["event_id"])
            if event_idx not in test_set:
                continue
            pred = detections[item["pred_index"]]
            center = float(pred["peak_ms"])
            feat = window_features(
                slice_window(samples_by_hand[pred["hand"]], center, WINDOW_HALF_MS), center
            )
            pred_zone = ZONE_NAMES[int(model.predict([feat])[0])]
            scored.append({
                "event_id": item["gt"]["event_id"],
                "true_drum": item["gt"]["drum"],
                "pred_drum": pred_zone,
                "hand": item["gt"]["hand"],
                "time_error_ms": item["dt_ms"],
                "zone_ok": pred_zone == item["gt"]["drum"],
            })
    zone_ok = sum(1 for row in scored if row["zone_ok"])
    matched_count = len(matched)
    gt_count = len(matched) + len(misses)
    return {
        "ground_truth_single_events": gt_count,
        "detector_matched_events": matched_count,
        "detector_misses": len(misses),
        "detector_extras": len(extras),
        "detector_recall": matched_count / gt_count if gt_count else 0.0,
        "fold_external_zone_correct": zone_ok,
        "fold_external_zone_total_matched": len(scored),
        "fold_external_zone_accuracy_on_matched": zone_ok / len(scored) if scored else 0.0,
        "fold_external_product_correct": zone_ok,
        "fold_external_product_accuracy": zone_ok / gt_count if gt_count else 0.0,
        "scored_events": scored,
        "note": "每個事件以其所在時間區塊的外部 fold 模型預測；HitDetector 本身為規則，不是用 MIDI 輸入。",
    }


def write_predictions(dataset_rows, y, pred, out_path):
    fields = ["event_id", "csv_center_ms", "hand", "true_drum", "fold_pred_drum", "correct"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, true, guess in zip(dataset_rows, y, pred):
            writer.writerow({
                "event_id": row["event_id"],
                "csv_center_ms": row["csv_center_ms"],
                "hand": row["hand"],
                "true_drum": ZONE_NAMES[int(true)],
                "fold_pred_drum": ZONE_NAMES[int(guess)],
                "correct": int(true == guess),
            })


def write_report_md(report):
    lines = [
        "# Song2 七鼓點模型真正驗證報告",
        "",
        f"產生時間：{report['generated_at']} ",
        "",
        "## 先說結論",
        "",
        f"本報告使用 Raw CSV 的 IMU 特徵訓練七類鼓點模型；MIDI 與人工 Ground Truth 沒有作為模型輸入，只用於建立事件答案與最後評分。最佳模型為 **{report['best_model']}**。",
        "",
        f"時間區塊交叉驗證的事件分類正確率：**{report['model_comparison'][report['best_model']]['accuracy_mean'] * 100:.1f}%**。這是七類鼓點分類的模型分數，不是左右手分數。",
        "",
        f"Raw CSV 端到端 fold 外推結果：**{report['end_to_end_fold_external']['fold_external_product_correct']} / {report['end_to_end_fold_external']['ground_truth_single_events']} = {report['end_to_end_fold_external']['fold_external_product_accuracy'] * 100:.1f}%**。",
        "",
        "## 模型比較（ZH-TW）",
        "",
        "| 模型 | 7 類事件分類正確率 | Macro F1 | Macro Precision | Macro Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["model_comparison"].items():
        lines.append(
            f"| {name} | {result['accuracy_mean'] * 100:.1f}% | {result['macro_f1_mean'] * 100:.1f}% | {result['precision_macro'] * 100:.1f}% | {result['recall_macro'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## 每個鼓點的外部 Fold 分類結果",
        "",
        "| 鼓點 | Precision | Recall | F1 | 支援筆數 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["per_drum_best_model"].items():
        lines.append(f"| {name} | {row['precision'] * 100:.1f}% | {row['recall'] * 100:.1f}% | {row['f1'] * 100:.1f}% | {row['support']} |")
    e = report["end_to_end_fold_external"]
    lines += [
        "",
        "## Raw CSV 端到端評估",
        "",
        f"- Ground Truth 單手事件：{e['ground_truth_single_events']} 筆",
        f"- Raw CSV HitDetector 對上的事件：{e['detector_matched_events']} 筆，召回 {e['detector_recall'] * 100:.1f}%",
        f"- 漏打：{e['detector_misses']} 筆；多偵測：{e['detector_extras']} 筆",
        f"- 外部 Fold 模型鼓位正確：{e['fold_external_zone_correct']} / {e['fold_external_zone_total_matched']} = {e['fold_external_zone_accuracy_on_matched'] * 100:.1f}%",
        f"- 外部 Fold 完整產品正確：{e['fold_external_product_correct']} / {e['ground_truth_single_events']} = {e['fold_external_product_accuracy'] * 100:.1f}%",
        "",
        "## 防止誤解",
        "",
        "- 這是七類鼓點分類模型，不是左右手分類模型。左右手由手套 HAND_ID 提供。",
        "- 模型特徵只來自 Raw CSV 的加速度與姿態欄位；沒有使用 MIDI note、MIDI velocity 或鼓位答案。",
        "- 每一筆事件的評估模型沒有看過該事件的訓練窗口；使用 4 秒時間區塊的 Stratified Group 5-Fold。",
        "- 這仍然是同一位演奏者、同一首 Song2 的離線資料驗證，不能宣稱跨歌曲或現場泛化率。",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("讀取最終 Ground Truth 與 Raw CSV")
    gt_rows = load_gt()
    raw_rows = load_raw_rows()
    samples = rows_to_samples(raw_rows)
    dataset_rows, X, y, groups = build_dataset(gt_rows, samples)
    print(f"7 類訓練事件：{len(y)}，Raw CSV 列：{len(raw_rows)}")
    print("以 4 秒時間區塊做 5-Fold 外部驗證")
    comparison, predictions, best_name, _splits = evaluate_models(X, y, groups)
    for name, result in comparison.items():
        print(f"{name}: {result['accuracy_mean'] * 100:.1f}% / Macro F1 {result['macro_f1_mean'] * 100:.1f}%")
    best_pred = predictions[best_name]
    best_report = classification_report(
        y, best_pred, labels=list(range(len(ZONE_NAMES))), target_names=ZONE_NAMES,
        output_dict=True, zero_division=0,
    )
    per_drum = {
        name: {
            "precision": float(best_report[name]["precision"]),
            "recall": float(best_report[name]["recall"]),
            "f1": float(best_report[name]["f1-score"]),
            "support": int(best_report[name]["support"]),
        }
        for name in ZONE_NAMES
    }
    final_model = models()[best_name]
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    detections = replay_hits(raw_rows, PRODUCT_DETECTOR_KWARGS)
    e2e = fold_end_to_end(gt_rows, dataset_rows, X, y, groups, samples, detections, best_name)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_gt": GT_CSV_PATH,
        "source_raw_csv": RAW_CSV_PATH,
        "window_half_ms": WINDOW_HALF_MS,
        "match_tolerance_ms": EVAL_MATCH_TOL_MS,
        "raw_csv_rows": len(raw_rows),
        "ground_truth_total_events": len(gt_rows),
        "ground_truth_single_events_used": len(y),
        "ground_truth_dual_events_excluded_from_single_class_model": len(gt_rows) - len(y),
        "feature_count": X.shape[1],
        "features_used": FEATURE_NAMES,
        "cv_scheme": "StratifiedGroupKFold 5 folds; groups=floor(csv_center_ms/4000)",
        "best_model": best_name,
        "model_comparison": comparison,
        "per_drum_best_model": per_drum,
        "best_model_confusion_matrix_labels": ZONE_NAMES,
        "best_model_confusion_matrix": confusion_matrix(y, best_pred, labels=list(range(len(ZONE_NAMES)))).tolist(),
        "end_to_end_fold_external": e2e,
        "model_path": str(MODEL_PATH),
        "prediction_path": str(PRED_PATH),
        "limitations": [
            "同一首 Song2、同一位演奏者的離線資料；不是跨歌曲泛化測試。",
            "HitDetector 是規則式擊打偵測，七鼓位部分才是 MLP。",
            "4 筆雙手事件未納入單手七類模型的訓練與端到端分母。",
        ],
    }
    write_predictions(dataset_rows, y, best_pred, PRED_PATH)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_md(report)
    print(f"最佳模型：{best_name}")
    print(f"外部 Fold 端到端：{e2e['fold_external_product_correct']}/{e2e['ground_truth_single_events']} ({e2e['fold_external_product_accuracy'] * 100:.1f}%)")
    print(f"JSON：{REPORT_PATH}")
    print(f"MD：{REPORT_MD_PATH}")
    print(f"模型：{MODEL_PATH}")


if __name__ == "__main__":
    main()
