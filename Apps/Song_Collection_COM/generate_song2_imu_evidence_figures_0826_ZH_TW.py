"""Create PPT-ready evidence figures for the Song2 IMU-only product path.

The figures deliberately separate runtime inference from later MIDI scoring:
Raw CSV detects a peak and supplies a +/-80 ms IMU window to the MLP.  MIDI
appears only after all predictions have been generated, for one-to-one scoring.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from product_hit_and_zone import FEATURE_NAMES, WINDOW_HALF_MS, window_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
RAW_CSV = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT_CSV = CLEANED_DIR / "Song2_ground_truth_labels_final_0821.csv"
PRED_CSV = CLEANED_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_預測對照_0826.csv"
REPORT_JSON = CLEANED_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_0826_ZH_TW.json"
MODEL_INPUT_FIGURE = CLEANED_DIR / "Song2_一次打擊_模型判斷輸入_0826_ZH_TW.png"
DRUM_ACCURACY_FIGURE = CLEANED_DIR / "Song2_七鼓點端到端正確率_0826_ZH_TW.png"

DRUM_ORDER = ["小鼓", "高音 Tom", "中音 Tom", "落地 Tom", "Hi-Hat", "Crash", "Ride"]
DRUM_COLORS = ["#c65d00", "#b44c92", "#ba7b00", "#176d9c", "#007c69", "#b09700", "#3a8fc0"]


def choose_cjk_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft JhengHei", "Microsoft JhengHei UI", "Noto Sans CJK TC", "SimHei"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def label_box(ax, x: float, y: float, title: str, body: str, color: str = "#173042") -> None:
    ax.text(
        x, y, f"{title}\n{body}", transform=ax.transAxes, va="top", ha="left",
        fontsize=11, color="#16222a", linespacing=1.52,
        bbox={"boxstyle": "round,pad=0.52", "facecolor": "#f5f8fa", "edgecolor": color, "linewidth": 1.15},
    )


def get_event_data() -> tuple[dict[str, str], dict[str, str], dict[str, str], list[dict[str, float]]]:
    ground_truth = read_csv(GT_CSV, "utf-8")
    gt = next(row for row in ground_truth if row["event_id"] == "1")
    predictions = read_csv(PRED_CSV)
    prediction = next(row for row in predictions if row["matched_midi_event_id"] == "1")
    raw_rows = read_csv(RAW_CSV)
    peak_ms = float(prediction["peak_ms"])
    samples: list[dict[str, float]] = []
    for row in raw_rows:
        if row["hand"] != prediction["hand"]:
            continue
        song_time = float(row["song_time_ms"])
        if abs(song_time - peak_ms) <= 250.0:
            samples.append({
                "song_time_ms": song_time,
                "ax": float(row["ax_g"]),
                "ay": float(row["ay_g"]),
                "az": float(row["az_g"]),
                "mag": float(row["accel_magnitude_g"]),
                "yaw": float(row["cal_yaw_deg"]),
                "pitch": float(row["cal_pitch_deg"]),
                "roll": float(row["roll_deg"]),
            })
    if not samples:
        raise RuntimeError("Representative event samples were not found.")
    return gt, prediction, {"peak_ms": str(peak_ms)}, samples


def draw_model_input_figure(output: Path) -> None:
    gt, prediction, meta, samples = get_event_data()
    peak_ms = float(meta["peak_ms"])
    times = np.array([sample["song_time_ms"] - peak_ms for sample in samples])
    window = [sample for sample in samples if abs(sample["song_time_ms"] - peak_ms) <= WINDOW_HALF_MS]
    feature_values = window_features(window, peak_ms)
    if feature_values is None:
        raise RuntimeError("Feature window is empty.")
    feature = dict(zip(FEATURE_NAMES, feature_values))

    fig = plt.figure(figsize=(17.2, 9.2), dpi=180, facecolor="#ffffff")
    grid = fig.add_gridspec(2, 2, width_ratios=(2.55, 1.0), height_ratios=(1, 1), left=0.06, right=0.965, top=0.83, bottom=0.09, hspace=0.28, wspace=0.08)
    accel_ax = fig.add_subplot(grid[0, 0])
    angle_ax = fig.add_subplot(grid[1, 0], sharex=accel_ax)
    side_ax = fig.add_subplot(grid[:, 1])
    side_ax.axis("off")

    fig.suptitle("一次打擊的模型判斷：不只看航向角", x=0.06, y=0.965, ha="left", fontsize=25, fontweight="bold", color="#14212b")
    fig.text(0.06, 0.905, "真實範例：右手 Hi-Hat。系統在 Peak 前後 ±80 ms 同時讀取加速度與姿態變化，而非只用單一角度。", fontsize=12, color="#50616c")

    accel_ax.axvspan(-WINDOW_HALF_MS, WINDOW_HALF_MS, color="#86cfa1", alpha=0.25, zorder=0, label="MLP 特徵視窗 ±80 ms")
    accel_ax.axvline(0, color="#1d6f42", linewidth=1.35, zorder=2)
    accel_ax.axhline(2.3, color="#bd5a45", linewidth=1.1, linestyle="--")
    accel_ax.plot(times, [s["mag"] for s in samples], color="#101820", linewidth=3.1)
    accel_ax.scatter([0], [float(prediction["peak_accel_g"])], s=80, marker="o", color="#f4c542", edgecolors="#171717", zorder=5)
    accel_ax.annotate("Peak 8.19 g\n判定有擊打", xy=(0, float(prediction["peak_accel_g"])), xytext=(37, 7), textcoords="offset points", fontsize=11.5, fontweight="bold", color="#172027")
    accel_ax.text(0.985, 0.10, "綠色區域：模型取用的 ±80 ms 資料", transform=accel_ax.transAxes, ha="right", fontsize=10.5, color="#23754c")
    accel_ax.set_title("1. 加速度：判斷『有沒有打』與打擊時機", loc="left", fontsize=16, fontweight="bold", pad=11)
    accel_ax.set_ylabel("加速度（g）", fontsize=12)
    accel_ax.grid(color="#d9e0e4", linewidth=0.7)

    angle_ax.axvspan(-WINDOW_HALF_MS, WINDOW_HALF_MS, color="#86cfa1", alpha=0.25, zorder=0)
    angle_ax.axvline(0, color="#1d6f42", linewidth=1.35, zorder=2)
    angle_ax.plot(times, [s["yaw"] for s in samples], color="#4d78ad", linewidth=2.0, label="Yaw（航向）")
    angle_ax.plot(times, [s["pitch"] for s in samples], color="#cf7a2d", linewidth=2.0, label="Pitch（俯仰）")
    angle_ax.plot(times, [s["roll"] for s in samples], color="#8c5e9f", linewidth=2.0, label="Roll（翻滾）")
    angle_ax.set_title("2. 姿態與揮動：判斷『打到哪裡』", loc="left", fontsize=16, fontweight="bold", pad=11)
    angle_ax.set_xlabel("相對 Raw CSV 偵測 Peak 的時間（ms）", fontsize=12)
    angle_ax.set_ylabel("歸零後姿態角（度）", fontsize=12)
    angle_ax.grid(color="#d9e0e4", linewidth=0.7)
    angle_ax.legend(loc="upper left", ncol=3, fontsize=10, frameon=False)

    label_box(side_ax, 0.0, 0.98, "判斷『有沒有打』", "加速度量級 |a|\n三軸加速度 ax / ay / az\n加速度變化速度\n\n本次 Peak：8.19 g", "#24717f")
    label_box(side_ax, 0.0, 0.63, "判斷『打到哪裡』", f"Yaw、Pitch、Roll\n揮動深度、航向角變化範圍\n\n本次中心 Yaw / Pitch：\n{feature['center_yaw']:.1f}° / {feature['center_pitch']:.1f}°", "#2b855f")
    label_box(side_ax, 0.0, 0.30, "手別與模型輸出", f"HAND_ID：{prediction['hand']}（韌體直接提供）\n\n17 個特徵一起輸入 MLP\n預測結果：{prediction['pred_drum']}\n模型信心：{float(prediction['pred_confidence'])*100:.1f}%", "#b76827")

    for axis in (accel_ax, angle_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=10.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def draw_per_drum_accuracy_figure(output: Path) -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    per_drum = report["per_drum"]
    exact = report["strict_exact_event_metrics"]
    fig, bars_ax = plt.subplots(figsize=(17.2, 9.2), dpi=180, facecolor="#ffffff")
    fig.subplots_adjust(left=0.20, right=0.95, top=0.75, bottom=0.16)
    fig.suptitle("七鼓點端到端正確率", x=0.06, y=0.96, ha="left", fontsize=27, fontweight="bold", color="#14212b")
    fig.text(0.06, 0.90, "Raw CSV 自主找打擊、判斷鼓點後，再與 MIDI 一對一比對；時間差在 ±80 ms 內且鼓點相同，才算正確。", fontsize=12, color="#50616c")
    fig.text(0.06, 0.80, f"整體：{exact['true_positive']} / {report['midi_event_count']} 筆時間與鼓點都正確 = {exact['recall']*100:.1f}%", fontsize=20, color="#178b55", fontweight="bold")

    values = [per_drum[drum]["recall"] * 100 for drum in DRUM_ORDER]
    correct_counts = [per_drum[drum]["exact_correct"] for drum in DRUM_ORDER]
    totals = [per_drum[drum]["midi_events"] for drum in DRUM_ORDER]
    positions = np.arange(len(DRUM_ORDER))
    colors = ["#3d8fba"] * len(DRUM_ORDER)
    colors[DRUM_ORDER.index("Crash")] = "#c56a35"
    bars_ax.barh(positions, values, color=colors, height=0.60)
    bars_ax.set_yticks(positions, DRUM_ORDER, fontsize=15)
    bars_ax.invert_yaxis()
    bars_ax.set_xlim(0, 105)
    bars_ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=11)
    bars_ax.grid(axis="x", color="#d9e0e4", linewidth=0.7)
    bars_ax.set_axisbelow(True)
    bars_ax.set_title("每一根：該鼓點『時間與鼓點都正確』的比例", loc="left", fontsize=15, fontweight="bold", pad=16)
    for idx, value in enumerate(values):
        bars_ax.text(value + 1.2, idx, f"{correct_counts[idx]} / {totals[idx]} = {value:.1f}%", va="center", fontsize=13, fontweight="bold", color="#1d333e")
    bars_ax.spines["top"].set_visible(False)
    bars_ax.spines["right"].set_visible(False)
    bars_ax.spines["left"].set_visible(False)
    bars_ax.tick_params(axis="y", length=0)
    fig.text(0.06, 0.055, "橘色 Crash 為目前優先改善項；其餘皆使用相同的驗證條件。", fontsize=11, color="#667780")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    choose_cjk_font()
    draw_model_input_figure(MODEL_INPUT_FIGURE)
    draw_per_drum_accuracy_figure(DRUM_ACCURACY_FIGURE)
    print(f"Created: {MODEL_INPUT_FIGURE}")
    print(f"Created: {DRUM_ACCURACY_FIGURE}")


if __name__ == "__main__":
    main()
