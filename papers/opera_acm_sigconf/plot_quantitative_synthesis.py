from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

MODEL_LABEL = {
    "llava-v1.5-7b": "LLaVA-v1.5-7B",
    "instructblip-7b": "InstructBLIP-7B",
}

METHOD_STYLE = {
    "base": {"label": "Greedy", "color": "#7a7f87", "marker": "o", "family": False},
    "opera": {"label": "OPERA", "color": "#4c78a8", "marker": "s", "family": False},
    "vcd": {"label": "VCD", "color": "#d95f5f", "marker": "P", "family": False},
    "dola": {"label": "DoLa", "color": "#f58518", "marker": "^", "family": False},
    "chord_past": {"label": "CHORD-P", "color": "#7fbf7b", "marker": "o", "family": True},
    "chord_past_current": {"label": "CHORD-P+C", "color": "#2f8f5b", "marker": "D", "family": True},
    "chord_past_future": {"label": "CHORD-P+F", "color": "#166a6a", "marker": "X", "family": True},
    "chord": {"label": "Full CHORD", "color": "#163d2a", "marker": "*", "family": True},
}

MODEL_DATA = {
    "llava-v1.5-7b": {
        "methods": {
            "base": {"adv_f1": 0.8036, "chair_s": 0.2291, "mmbench": 68.63, "itl": 19.73},
            "opera": {"adv_f1": 0.8042, "chair_s": 0.2284, "mmbench": 68.65, "itl": 21.69},
            "vcd": {"adv_f1": 0.8031, "chair_s": 0.2183, "mmbench": 68.29, "itl": 33.07},
            "dola": {"adv_f1": 0.8103, "chair_s": 0.3276, "mmbench": 68.63, "itl": 23.11},
            "chord_past": {"adv_f1": 0.8196, "chair_s": 0.2042, "mmbench": 68.84, "itl": 24.38},
            "chord_past_current": {"adv_f1": 0.8318, "chair_s": 0.1745, "mmbench": 69.15, "itl": 27.24},
            "chord_past_future": {"adv_f1": 0.8251, "chair_s": 0.1794, "mmbench": 69.46, "itl": 34.62},
            "chord": {"adv_f1": 0.8453, "chair_s": 0.1548, "mmbench": 69.78, "itl": 37.31},
        }
    },
    "instructblip-7b": {
        "methods": {
            "base": {"adv_f1": 0.8327, "chair_s": 0.2140, "mmbench": 69.34, "itl": 16.47},
            "opera": {"adv_f1": 0.8327, "chair_s": 0.2180, "mmbench": 69.27, "itl": 16.90},
            "vcd": {"adv_f1": 0.8173, "chair_s": 0.2060, "mmbench": 68.70, "itl": 24.80},
            "dola": {"adv_f1": 0.8408, "chair_s": 0.3090, "mmbench": 69.41, "itl": 19.03},
            "chord_past": {"adv_f1": 0.8413, "chair_s": 0.1842, "mmbench": 69.64, "itl": 21.43},
            "chord_past_current": {"adv_f1": 0.8517, "chair_s": 0.1543, "mmbench": 70.08, "itl": 24.51},
            "chord_past_future": {"adv_f1": 0.8485, "chair_s": 0.1651, "mmbench": 70.43, "itl": 32.55},
            "chord": {"adv_f1": 0.8651, "chair_s": 0.1347, "mmbench": 70.82, "itl": 35.86},
        }
    },
}

FRONTIER_METHODS = ["base", "opera", "vcd", "dola", "chord_past", "chord_past_current", "chord_past_future", "chord"]
QUALITY_METHODS = ["base", "opera", "vcd", "dola", "chord_past_current", "chord"]

FRONTIER_LABEL_OFFSETS = {
    "llava-v1.5-7b": {
        "base": (0, 0),
        "opera": (0, 0),
        "vcd": (0, 0),
        "dola": (0, 0),
        "chord_past": (-26, 9),
        "chord_past_current": (7, 12),
        "chord_past_future": (-48, -10),
        "chord": (8, -12),
    },
    "instructblip-7b": {
        "base": (0, 0),
        "opera": (0, 0),
        "vcd": (0, 0),
        "dola": (0, 0),
        "chord_past": (-14, 14),
        "chord_past_current": (8, 11),
        "chord_past_future": (-48, 4),
        "chord": (7, -13),
    },
}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle=(0, (3, 2)), linewidth=0.45, alpha=0.2, color="#94a3b8")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=6.4, width=0.8)


def annotate(ax: plt.Axes, x: float, y: float, text: str, dx: int, dy: int, color: str) -> None:
    note = ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=5.1,
        color=color,
        zorder=5,
        fontweight="bold" if "CHORD" in text else "normal",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    note.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])


def add_frontier_panel(ax: plt.Axes, model: str) -> None:
    style_axis(ax)
    model_points = MODEL_DATA[model]["methods"]
    family_methods = [name for name in FRONTIER_METHODS if METHOD_STYLE[name]["family"]]
    panel = "(a)" if model == "llava-v1.5-7b" else "(b)"

    ax.plot(
        [model_points[name]["itl"] for name in family_methods],
        [model_points[name]["adv_f1"] for name in family_methods],
        linestyle="--",
        linewidth=1.0,
        color="#2f8f5b",
        alpha=0.55,
        zorder=1,
    )

    base_chair = model_points["base"]["chair_s"]
    for name in FRONTIER_METHODS:
        point = model_points[name]
        style = METHOD_STYLE[name]
        chair_gain = max(base_chair - point["chair_s"], 0.0)
        size = 28 + 1180 * chair_gain
        if name == "chord":
            size += 20
        ax.scatter(
            point["itl"],
            point["adv_f1"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="#1b1b1b" if name == "chord" else "#ffffff",
            linewidth=0.75,
            zorder=4 if style["family"] else 3,
            alpha=1.0 if style["family"] else 0.88,
        )
        if style["family"]:
            dx, dy = FRONTIER_LABEL_OFFSETS[model][name]
            annotate(ax, point["itl"], point["adv_f1"], style["label"], dx, dy, style["color"])

    x_vals = [model_points[name]["itl"] for name in FRONTIER_METHODS]
    y_vals = [model_points[name]["adv_f1"] for name in FRONTIER_METHODS]
    ax.set_xlim(min(x_vals) - 2.0, max(x_vals) + 4.0)
    ax.set_ylim(min(y_vals) - 0.01, max(y_vals) + 0.012)
    ax.set_xlabel("ITL per token (ms)", fontsize=7.2, labelpad=2.0)
    ax.set_ylabel("Adversarial F1", fontsize=7.3)
    ax.set_title(f"{panel} {MODEL_LABEL[model]} frontier", loc="left", fontsize=7.0, fontweight="bold", pad=5)


def add_quality_panel(ax: plt.Axes, model: str) -> None:
    style_axis(ax)
    model_points = MODEL_DATA[model]["methods"]
    panel = "(c)" if model == "llava-v1.5-7b" else "(d)"

    for name in QUALITY_METHODS:
        point = model_points[name]
        style = METHOD_STYLE[name]
        size = 26 + 1.6 * point["itl"]
        if name == "chord":
            size += 18
        ax.scatter(
            point["mmbench"],
            point["chair_s"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="#1b1b1b" if name == "chord" else "#ffffff",
            linewidth=0.75,
            zorder=4 if style["family"] else 3,
            alpha=1.0 if style["family"] else 0.88,
        )
        if name in {"chord_past_current", "chord"}:
            dx = 8
            dy = -10 if name == "chord" else 10
            annotate(ax, point["mmbench"], point["chair_s"], style["label"], dx, dy, style["color"])

    ax.plot(
        [model_points[name]["mmbench"] for name in ["chord_past_current", "chord"]],
        [model_points[name]["chair_s"] for name in ["chord_past_current", "chord"]],
        linestyle="--",
        linewidth=1.0,
        color="#2f8f5b",
        alpha=0.55,
        zorder=1,
    )
    ax.set_xlabel("MMBench accuracy", fontsize=7.2, labelpad=2.0)
    ax.set_ylabel("CHAIR$_S$", fontsize=7.3)
    ax.invert_yaxis()
    x_vals = [model_points[name]["mmbench"] for name in QUALITY_METHODS]
    y_vals = [model_points[name]["chair_s"] for name in QUALITY_METHODS]
    ax.set_xlim(min(x_vals) - 0.35, max(x_vals) + 0.48)
    ax.set_ylim(max(y_vals) + 0.018, min(y_vals) - 0.012)
    ax.set_title(f"{panel} {MODEL_LABEL[model]} trade-off map", loc="left", fontsize=7.0, fontweight="bold", pad=6)
    ax.text(
        0.03,
        0.95,
        "larger markers indicate higher ITL",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.9,
        color="#5a5f66",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.6,
            "axes.labelsize": 7.3,
            "axes.titlesize": 7.8,
            "legend.fontsize": 5.6,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.35, 4.18), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.992, top=0.842, bottom=0.092, wspace=0.20, hspace=0.50)

    add_frontier_panel(axes[0, 0], "llava-v1.5-7b")
    add_frontier_panel(axes[0, 1], "instructblip-7b")
    add_quality_panel(axes[1, 0], "llava-v1.5-7b")
    add_quality_panel(axes[1, 1], "instructblip-7b")

    legend_order = ["base", "opera", "vcd", "dola", "chord_past_current", "chord"]
    handles = []
    labels = []
    for name in legend_order:
        style = METHOD_STYLE[name]
        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            lw=0,
            markersize=4.4 if name != "chord" else 6.0,
        )
        handles.append(handle)
        labels.append(style["label"])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.982),
        columnspacing=0.78,
        handletextpad=0.22,
    )

    png_path = OUT_DIR / "figure3.png"
    pdf_path = OUT_DIR / "figure3.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
