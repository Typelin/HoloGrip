"""Render a truthful local Yaw/Pitch decision slice from the verified Song2 MLP.

This is not a hand-drawn drum layout or a distribution ellipse.  Every colored
pixel is a real MLP prediction.  The other 15 IMU features are held to a
verified, representative right-hand Hi-Hat event, so this figure is an
interpretable local slice through the 17-dimensional model.
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from product_hit_and_zone import FEATURE_NAMES, slice_window, rows_to_samples, window_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
RAW_CSV = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
PRED_CSV = CLEANED_DIR / "Song2_RawCSV到311正確MIDI嚴格驗證_預測對照_0826.csv"
MODEL_PATH = CLEANED_DIR / "hologrip_song2_七鼓點模型_真正驗證版_0826.joblib"
OUTPUT_PATH = CLEANED_DIR / "Song2_MLP真實判斷範圍切面_0826_ZH_TW.png"

ZONE_NAMES = ["小鼓", "高音 Tom", "中音 Tom", "落地 Tom", "Hi-Hat", "Crash", "Ride"]
# Okabe-Ito-inspired categorical palette: high hue separation and PPT-friendly.
ZONE_COLORS = ["#E69F00", "#CC79A7", "#F0E442", "#56B4E9", "#009E73", "#D55E00", "#0072B2"]
REP_PRED_ID = "5"
GRID_STEP_DEG = 0.75


def choose_cjk_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft JhengHei", "Microsoft JhengHei UI", "Noto Sans CJK TC", "SimHei"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_rows(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def representative_features() -> tuple[np.ndarray, dict[str, str]]:
    prediction = next(row for row in read_rows(PRED_CSV) if row["pred_id"] == REP_PRED_ID)
    if not prediction["matched_midi_event_id"]:
        raise RuntimeError("Representative prediction must be MIDI-matched.")
    raw_rows = read_rows(RAW_CSV)
    samples = rows_to_samples(raw_rows)
    peak_ms = float(prediction["peak_ms"])
    features = window_features(slice_window(samples[prediction["hand"]], peak_ms), peak_ms)
    if features is None:
        raise RuntimeError("Could not create the representative +/-80 ms feature window.")
    return np.array(features, dtype=float), prediction


def observed_center_domains() -> tuple[tuple[float, float], tuple[float, float]]:
    """Use actual raw hit windows only to define a readable plotting domain."""
    raw_rows = read_rows(RAW_CSV)
    samples = rows_to_samples(raw_rows)
    prediction_rows = read_rows(PRED_CSV)
    vectors = []
    for row in prediction_rows:
        if not row["matched_midi_event_id"] or row["hand"] not in samples:
            continue
        peak_ms = float(row["peak_ms"])
        features = window_features(slice_window(samples[row["hand"]], peak_ms), peak_ms)
        if features is not None:
            vectors.append(features)
    if not vectors:
        raise RuntimeError("No verified peak windows were found for axis ranges.")
    values = np.asarray(vectors, dtype=float)
    yaw = values[:, FEATURE_NAMES.index("center_yaw")]
    pitch = values[:, FEATURE_NAMES.index("center_pitch")]

    def padded_range(series: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(series, [1, 99])
        padding = max(8.0, (hi - lo) * 0.12)
        return float(np.floor((lo - padding) / 5.0) * 5.0), float(np.ceil((hi + padding) / 5.0) * 5.0)

    return padded_range(yaw), padded_range(pitch)


def render() -> Path:
    choose_cjk_font()
    model = joblib.load(MODEL_PATH)
    classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    if not np.array_equal(classes, np.arange(len(ZONE_NAMES))):
        raise RuntimeError(f"Unexpected class order: {classes.tolist()}")

    base, prediction = representative_features()
    yaw_domain, pitch_domain = observed_center_domains()
    yaws = np.arange(yaw_domain[0], yaw_domain[1] + GRID_STEP_DEG, GRID_STEP_DEG)
    pitches = np.arange(pitch_domain[0], pitch_domain[1] + GRID_STEP_DEG, GRID_STEP_DEG)
    yy, pp = np.meshgrid(yaws, pitches)
    grid = np.tile(base, (yy.size, 1))
    grid[:, FEATURE_NAMES.index("center_yaw")] = yy.ravel()
    grid[:, FEATURE_NAMES.index("center_pitch")] = pp.ravel()

    probabilities = model.predict_proba(grid)
    predicted = np.argmax(probabilities, axis=1).reshape(pp.shape)

    fig, axis = plt.subplots(figsize=(17.2, 9.2), dpi=180, facecolor="#ffffff")
    fig.subplots_adjust(left=0.085, right=0.965, top=0.76, bottom=0.18)
    fig.suptitle("MLP 真實判斷範圍切面：固定同一打擊型態，掃描 Yaw / Pitch", x=0.06, y=0.965, ha="left", fontsize=24, fontweight="bold", color="#14212b")
    fig.text(0.06, 0.905, "其他 15 個 IMU 特徵固定為一筆真實右手 Hi-Hat；每一個色塊都由已訓練 MLP 實際輸出，不是人工畫的鼓位範圍。", fontsize=12, color="#50616c")
    fig.text(0.06, 0.867, "用途：看『同樣的加速度與揮動型態下，姿態角改變時模型會判成哪一顆鼓』；每個顏色交界就是模型判斷改變處。", fontsize=11, color="#50616c")

    mesh = axis.pcolormesh(yy, pp, predicted, cmap=ListedColormap(ZONE_COLORS), shading="nearest", alpha=0.82)

    rep_yaw = base[FEATURE_NAMES.index("center_yaw")]
    rep_pitch = base[FEATURE_NAMES.index("center_pitch")]
    axis.scatter([rep_yaw], [rep_pitch], s=205, marker="*", color="#f7c948", edgecolor="#15212a", linewidth=1.25, zorder=5)
    axis.annotate(
        "真實右手 Hi-Hat 範例\n中心 Yaw / Pitch = " + f"{rep_yaw:.1f}° / {rep_pitch:.1f}°\n模型信心 {float(prediction['pred_confidence']) * 100:.1f}%",
        xy=(rep_yaw, rep_pitch), xytext=(18, 16), textcoords="offset points", fontsize=11, fontweight="bold", color="#14212b",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#ffffff", "edgecolor": "#31434f", "linewidth": 1.0},
        arrowprops={"arrowstyle": "-", "color": "#31434f", "linewidth": 1.0},
    )

    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in zip(ZONE_NAMES, ZONE_COLORS)]
    fig.text(0.085, 0.064, "模型輸出顏色：", fontsize=11.5, fontweight="bold", color="#24313a")
    axis.legend(handles=legend_handles, ncol=7, loc="upper center", bbox_to_anchor=(0.56, -0.15), fontsize=10.5, frameon=False, columnspacing=1.25, handlelength=1.5)
    axis.set_xlabel("中心 Yaw（航向角，度）", fontsize=13)
    axis.set_ylabel("中心 Pitch（俯仰角，度）", fontsize=13)
    axis.set_xlim(yaw_domain)
    axis.set_ylim(pitch_domain)
    axis.grid(color="#ffffff", linewidth=0.55, alpha=0.38)
    axis.tick_params(labelsize=10.5)
    for spine in axis.spines.values():
        spine.set_color("#52636e")
        spine.set_linewidth(0.9)
    fig.savefig(OUTPUT_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output = render()
    print(f"Created: {output}")
