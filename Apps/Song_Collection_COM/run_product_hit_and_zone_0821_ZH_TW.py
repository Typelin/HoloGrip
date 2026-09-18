"""Train IMU-only zone model on Song 2, then replay hit+zone without MIDI input."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_hit_and_zone import (
    CLEANED_DIR,
    EVAL_JSON_PATH,
    HTML_PATH,
    MODEL_PATH,
    PRED_CSV_PATH,
    classify_detections,
    evaluate,
    load_gt,
    load_raw_rows,
    replay_hits,
    rows_to_samples,
    build_zone_dataset,
    save_zone_model,
    train_zone_model,
    write_eval_json,
    write_html,
    write_pred_csv,
)


def main() -> None:
    print("1/4 讀取 Song 2 真值 + Raw 100Hz")
    gt = load_gt()
    raw = load_raw_rows()
    samples = rows_to_samples(raw)

    print("2/4 用 MIDI/影片真值當老師，訓練 IMU-only 7 類鼓位")
    X, y = build_zone_dataset(gt, samples)
    clf = train_zone_model(X, y)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    save_zone_model(clf)
    print(f"   樣本 {len(y)}  模型 {MODEL_PATH}")

    print("3/4 重放 HitDetector（不餵 MIDI）並分類鼓位")
    dets = replay_hits(raw)
    preds = classify_detections(dets, samples, clf)
    write_pred_csv(preds)

    print("4/4 對答案並寫驗證頁")
    report = evaluate(gt, preds)
    write_eval_json(report)
    write_html(report)
    print(
        f"   有打召回 {report['hit_recall']*100:.1f}%  "
        f"偵測後鼓位 {report['zone_acc_given_hit']*100:.1f}%  "
        f"手+鼓完整 {report['product_recall']*100:.1f}%  "
        f"({report['joint_ok']}/{report['gt_single']})"
    )
    print(f"   預測 CSV  {PRED_CSV_PATH}")
    print(f"   評估 JSON {EVAL_JSON_PATH}")
    print(f"   驗證頁    {HTML_PATH}")


if __name__ == "__main__":
    main()