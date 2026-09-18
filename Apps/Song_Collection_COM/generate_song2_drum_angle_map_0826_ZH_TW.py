"""Render Song2 seven-drum IMU direction distribution for the presentation.

Each point is one approved single-hand Ground Truth event.  It reads the
aligned Raw CSV and Ground Truth labels, so its positions are not hand-entered
illustrative values.  The chart shows why a seven-drum classifier cannot use
Yaw alone: the groups overlap and need the full IMU feature window.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
RAW_CSV = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GROUND_TRUTH_CSV = CLEANED_DIR / "Song2_ground_truth_labels_final_0821.csv"
DEFAULT_OUTPUT = CLEANED_DIR / "Song2_七鼓點_IMU方向分布_0826_ZH_TW.png"

DRUM_ORDER = ["小鼓", "高音 Tom", "中音 Tom", "落地 Tom", "Hi-Hat", "Crash", "Ride"]
COLORS = {
    "小鼓": "#D55E00",
    "高音 Tom": "#CC79A7",
    "中音 Tom": "#E69F00",
    "落地 Tom": "#0072B2",
    "Hi-Hat": "#009E73",
    "Crash": "#F0E442",
    "Ride": "#56B4E9",
}
MARKERS = {"L": "o", "R": "^"}


def choose_cjk_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_rows() -> list[dict[str, object]]:
    with RAW_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.DictReader(fh))
    by_hand: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        if row["hand"] in {"L", "R"}:
            by_hand[row["hand"]].append(row)

    with GROUND_TRUTH_CSV.open("r", encoding="utf-8", newline="") as fh:
        ground_truth = list(csv.DictReader(fh))

    records: list[dict[str, object]] = []
    for event in ground_truth:
        hand = event.get("final_hand", "")
        drum = event.get("drum", "")
        if hand not in {"L", "R"} or drum not in DRUM_ORDER:
            continue
        center_ms = float(event["csv_center_ms"])
        samples = [
            row for row in by_hand[hand]
            if abs(float(row["song_time_ms"]) - center_ms) <= 80.0
        ]
        if not samples:
            continue
        nearest = min(samples, key=lambda row: abs(float(row["song_time_ms"]) - center_ms))
        records.append({
            "drum": drum,
            "hand": hand,
            "yaw": float(nearest["cal_yaw_deg"]),
            "pitch": float(nearest["cal_pitch_deg"]),
        })
    return records


def add_std_ellipse(ax, values: np.ndarray, color: str) -> tuple[float, float]:
    center = values.mean(axis=0)
    cov = np.cov(values.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(eigvals, 0.0))
    ax.add_patch(Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor=color, edgecolor=color, alpha=0.12, linewidth=1.4, zorder=1,
    ))
    ax.scatter(*center, s=46, c=color, marker="X", edgecolors="white", linewidths=0.7, zorder=5)
    return float(center[0]), float(center[1])


def render(output: Path) -> None:
    choose_cjk_font()
    records = load_rows()
    if len(records) != 307:
        raise RuntimeError(f"Expected 307 single-hand events, received {len(records)}")

    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="#FFFFFF")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.55, 1.0), left=0.055, right=0.965, top=0.84, bottom=0.1, wspace=0.11)
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])
    side.axis("off")

    ax.set_title("每一點 = 一次已確認的單手擊打（共 307 筆）", fontsize=17, fontweight="bold", loc="left", pad=15)
    ax.set_xlabel("展示用鏡像航向角 -Yaw（歸零後，度）", fontsize=13)
    ax.set_ylabel("俯仰角 Pitch（歸零後，度）", fontsize=13)
    ax.axhline(0, color="#BAC1C8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#BAC1C8", linewidth=0.8, zorder=0)
    ax.grid(color="#DDE2E6", linewidth=0.7, alpha=0.85)

    positions: dict[str, tuple[float, float]] = {}
    for drum in DRUM_ORDER:
        group = [row for row in records if row["drum"] == drum]
        # Mirror only the horizontal presentation so the plot follows the
        # physical drum layout: Hi-Hat/Crash on the left, Ride on the right.
        values = np.array([[-row["yaw"], row["pitch"]] for row in group], dtype=float)
        positions[drum] = add_std_ellipse(ax, values, COLORS[drum])
        for hand in ("L", "R"):
            hand_values = np.array([[-row["yaw"], row["pitch"]] for row in group if row["hand"] == hand], dtype=float)
            if len(hand_values):
                ax.scatter(
                    hand_values[:, 0], hand_values[:, 1], s=33, marker=MARKERS[hand],
                    c=COLORS[drum], alpha=0.74, edgecolors="#263238", linewidths=0.3,
                    label=drum if hand == "L" else None, zorder=3,
                )

    label_offsets = {
        "小鼓": (-5, -5), "高音 Tom": (-5, 6), "中音 Tom": (17, -9),
        "落地 Tom": (5, -13), "Hi-Hat": (-4, 4), "Crash": (-6, 5), "Ride": (6, 7),
    }
    for drum, (x, y) in positions.items():
        dx, dy = label_offsets[drum]
        ax.annotate(drum, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=11, fontweight="bold", color="#172027", zorder=6)

    ax.text(
        0.015, 0.02,
        "圓點 = 左手    三角 = 右手    半透明橢圓 = 各鼓點約 1 個標準差    水平軸為展示鏡像",
        transform=ax.transAxes, fontsize=10.5, color="#27323A", va="bottom",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.9, "pad": 3.5}, zorder=7,
    )

    side.text(0, 0.98, "目前 MLP 的輸入", fontsize=18, fontweight="bold", va="top", color="#172027")
    side.text(0, 0.89, "先找擊打 Peak，再取前後 ±80 ms\n的 IMU 視窗，輸入 17 維特徵。", fontsize=12.5, va="top", linespacing=1.55, color="#27323A")
    side.text(0, 0.72, "加速度動態（8 維）", fontsize=13.5, fontweight="bold", va="top", color="#0072B2")
    side.text(0, 0.665, "量級最大／平均／波動\n活動能量、X/Y/Z 最大值、瞬間變化量", fontsize=11.2, va="top", linespacing=1.5, color="#27323A")
    side.text(0, 0.51, "姿態與揮動（9 維）", fontsize=13.5, fontweight="bold", va="top", color="#009E73")
    side.text(0, 0.455, "Yaw、Pitch、Roll 的平均／波動\nPeak 中心 Yaw、Pitch\nPitch 揮動深度、Yaw 變化範圍", fontsize=11.2, va="top", linespacing=1.5, color="#27323A")
    side.text(0, 0.25, "圖上重點", fontsize=13.5, fontweight="bold", va="top", color="#D55E00")
    side.text(0, 0.195, "小鼓、高音 Tom、中音 Tom 明顯重疊；\n落地 Tom 與 Ride 的航向中心也接近。\n因此不能只用 Yaw 設固定門檻，\n必須綜合完整 IMU 動作特徵。", fontsize=11.2, va="top", linespacing=1.5, color="#27323A")

    fig.suptitle("HoloGrip Song2：七鼓點的 IMU 方向分布與模型輸入", x=0.055, y=0.955, ha="left", fontsize=24, fontweight="bold", color="#111827")
    fig.text(0.055, 0.895, "資料來源：Song2 Ground Truth 311 筆；圖中排除 4 筆雙手同時事件。半透明橢圓為各鼓點約 1 標準差的分布範圍。", fontsize=10.5, color="#58636C")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Song2 IMU direction scatter plot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
