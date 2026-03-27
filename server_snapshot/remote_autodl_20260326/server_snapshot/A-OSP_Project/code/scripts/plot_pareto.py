#!/usr/bin/env python3
"""
A-OSP Figure 1 — Pareto Bubble Plot
=====================================
X-axis: Throughput (tok/s)
Y-axis: POPE F1 score
Bubble size: Average Generation Length (MMHal-Bench)

Demonstrates the "Punching Above Weight" narrative: A-OSP achieves
near-identical quality with negligible throughput cost.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT_DIR = Path("/root/autodl-tmp/A-OSP_Project/logs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════
# Data — verified from eval_results
# ═════════════════════════════════════════════════════════════════

data_points = [
    {
        "label": "Qwen2-VL-7B\n(Base)",
        "throughput": 48.92,
        "pope_f1": 0.8727,
        "mmhal_agl": 27.5,
        "color": "#4A90D9",
        "marker": "o",
    },
    {
        "label": "Qwen2-VL-7B\n+ A-OSP",
        "throughput": 48.42,
        "pope_f1": 0.8727,
        "mmhal_agl": 28.0,
        "color": "#E85D4A",
        "marker": "D",
    },
    # Reference points from literature (approximate)
    {
        "label": "InternVL2-26B",
        "throughput": 18.0,
        "pope_f1": 0.89,
        "mmhal_agl": 35.0,
        "color": "#888888",
        "marker": "s",
    },
    {
        "label": "LLaVA-1.5-13B",
        "throughput": 32.0,
        "pope_f1": 0.86,
        "mmhal_agl": 30.0,
        "color": "#AAAAAA",
        "marker": "^",
    },
    {
        "label": "Qwen-VL-Chat",
        "throughput": 38.0,
        "pope_f1": 0.84,
        "mmhal_agl": 25.0,
        "color": "#CCCCCC",
        "marker": "v",
    },
]


def render_pareto():
    fig, ax = plt.subplots(figsize=(8, 6))

    for pt in data_points:
        s = pt["mmhal_agl"] * 15
        is_ours = "A-OSP" in pt["label"] or "Base" in pt["label"]
        alpha = 1.0 if is_ours else 0.45
        zorder = 10 if is_ours else 5
        edgecolor = "black" if is_ours else "#666666"
        linewidth = 1.8 if is_ours else 0.8

        ax.scatter(
            pt["throughput"], pt["pope_f1"],
            s=s, c=pt["color"], marker=pt["marker"],
            alpha=alpha, zorder=zorder,
            edgecolors=edgecolor, linewidths=linewidth,
        )

        offset_x = 1.5 if not is_ours else (-0.5 if "Base" in pt["label"] else 1.5)
        offset_y = 0.006 if not is_ours else (-0.012 if "Base" in pt["label"] else 0.008)
        fontsize = 10 if is_ours else 8
        fontweight = "bold" if is_ours else "normal"

        txt = ax.annotate(
            pt["label"],
            (pt["throughput"], pt["pope_f1"]),
            xytext=(pt["throughput"] + offset_x, pt["pope_f1"] + offset_y),
            fontsize=fontsize, fontweight=fontweight,
            color=pt["color"] if is_ours else "#555555",
            ha="left", va="center",
        )
        if is_ours:
            txt.set_path_effects([
                pe.withStroke(linewidth=2, foreground="white")
            ])

    # Arrow showing A-OSP overhead
    ax.annotate(
        "", xy=(48.42, 0.8727), xytext=(48.92, 0.8727),
        arrowprops=dict(arrowstyle="<->", color="#E85D4A", lw=1.5),
    )
    ax.text(
        48.67, 0.8695, "Δ=1.0%\noverhead",
        ha="center", va="top", fontsize=7.5, color="#E85D4A",
        fontstyle="italic",
    )

    ax.set_xlabel("Throughput (tokens/s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("POPE F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Accuracy–Throughput Pareto Front", fontsize=13, fontweight="bold")

    # AGL legend
    for agl_val in [25, 30, 35]:
        ax.scatter([], [], s=agl_val * 15, c="gray", alpha=0.4,
                   label=f"AGL = {agl_val}", edgecolors="gray", linewidths=0.5)
    ax.legend(title="Bubble = AGL (MMHal)", loc="lower left",
              fontsize=8, title_fontsize=9, framealpha=0.85)

    ax.set_xlim(12, 55)
    ax.set_ylim(0.82, 0.90)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"pareto_optimal_fig1.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[Fig1] Saved → {OUT_DIR}/pareto_optimal_fig1.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    render_pareto()
