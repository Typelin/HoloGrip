from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

LAB = Path(__file__).resolve().parent
REF = json.loads((LAB / "spatial_reference_0826.json").read_text(encoding="utf-8"))
OUT = LAB / "angle_map_background.png"

ORDER = REF["drum_order"]
COLORS = {
    "Hi-Hat": "#eab308",
    "Crash": "#f97316",
    "小鼓": "#06b6d4",
    "高音 Tom": "#22c55e",
    "中音 Tom": "#8b5cf6",
    "Ride": "#10b981",
    "落地 Tom": "#0ea5e9",
}

YAW_LEFT = 60.0
YAW_RIGHT = -120.0
PITCH_BOTTOM = -25.0
PITCH_TOP = 65.0


def classify(yaw: float, pitch: float) -> int:
    ev = REF["events"]
    sc = REF["scale"]
    dy = np.asarray([(r["yaw"] - yaw) / sc["yaw_scale"] for r in ev], dtype=float)
    dp = np.asarray([(r["pitch"] - pitch) / sc["pitch_scale"] for r in ev], dtype=float)
    d = np.hypot(dy, dp)
    k = min(11, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    votes = {name: 0.0 for name in ORDER}
    for i in idx:
        votes[ev[int(i)]["drum"]] += 1.0 / (float(d[int(i)]) + 0.08) ** 2
    best = max(ORDER, key=lambda x: votes[x])
    return ORDER.index(best)


def main():
    xs = np.linspace(YAW_RIGHT, YAW_LEFT, 181)
    ys = np.linspace(PITCH_BOTTOM, PITCH_TOP, 91)
    z = np.zeros((len(ys), len(xs)), dtype=int)
    for iy, p in enumerate(ys):
        for ix, y in enumerate(xs):
            z[iy, ix] = classify(float(y), float(p))

    fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    cmap = ListedColormap([COLORS[x] for x in ORDER])
    ax.pcolormesh(xs, ys, z, cmap=cmap, shading="auto", alpha=0.20)

    # Draw class-change contours as the empirical posture boundaries.
    levels = np.arange(-0.5, len(ORDER) + 0.5, 1)
    ax.contour(xs, ys, z, levels=levels, colors="#94a3b8", linewidths=0.65, alpha=0.65)

    # Real 307 event points.
    for drum in ORDER:
        rr = [r for r in REF["events"] if r["drum"] == drum]
        ax.scatter(
            [r["yaw"] for r in rr],
            [r["pitch"] for r in rr],
            s=10,
            c=COLORS[drum],
            alpha=0.30,
            edgecolors="none",
        )
        c = REF["classes"][drum]
        w = max(4.0, c["yaw_p90"] - c["yaw_p10"])
        h = max(4.0, c["pitch_p90"] - c["pitch_p10"])
        ell = Ellipse(
            (c["yaw_median"], c["pitch_median"]),
            width=w,
            height=h,
            fill=False,
            edgecolor=COLORS[drum],
            linewidth=2.2,
            alpha=0.95,
        )
        ax.add_patch(ell)
        ax.scatter([c["yaw_median"]], [c["pitch_median"]], s=105, c=COLORS[drum], edgecolors="white", linewidths=1.2, zorder=5)
        ax.text(
            c["yaw_median"], c["pitch_median"] + 2.5, drum,
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#111827", ec=COLORS[drum], alpha=0.85),
        )

    ax.axvline(0, color="white", lw=1.1, alpha=0.75)
    ax.axhline(0, color="white", lw=1.1, alpha=0.45)
    ax.set_xlim(YAW_LEFT, YAW_RIGHT)
    ax.set_ylim(PITCH_BOTTOM, PITCH_TOP)
    ax.set_xticks(np.arange(-120, 61, 10))
    ax.set_yticks(np.arange(-20, 61, 10))
    ax.grid(True, color="#64748b", alpha=0.18, linewidth=0.7)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")

    ax.set_xlabel("鼓手左  ←  +Yaw（度）     0°     -Yaw（度）  →  鼓手右", color="#f8fafc", fontsize=12, labelpad=10)
    ax.set_ylabel("Pitch（度）  ↑ 抬高", color="#f8fafc", fontsize=12)
    ax.set_title("HoloGrip 鼓手視角角度地圖｜姿態參考分界（非完整 17 維 MLP 邊界）", color="white", fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
