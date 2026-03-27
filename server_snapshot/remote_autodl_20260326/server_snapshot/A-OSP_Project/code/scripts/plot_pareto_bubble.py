"""
A-OSP 帕累托气泡图 (Pareto Bubble Chart)
=========================================
X 轴 = Throughput (Tokens/s) — 线性刻度，凸显 +1.5% vs +98% 的鸿沟
Y 轴 = POPE F1 Score
气泡大小 = AGL (Average Generation Length) — 证明未陷入长度-偏见陷阱
气泡颜色 = 方法类别

学术排版标准：Nature / CVPR 级别，高对比度，无视觉噪声。
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ────────────────────────────────────────────────────────────────
# 路径
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")
FIGURE_DIR = PROJECT_ROOT / "logs" / "figures"
EVAL_DIR = PROJECT_ROOT / "logs" / "eval_results"

# ────────────────────────────────────────────────────────────────
# Mock 数据 — 等 Agent 2 跑完后替换为真实数据
# 格式：(method, throughput, pope_f1, agl, color_key)
# ────────────────────────────────────────────────────────────────
MOCK_DATA = [
    # method           throughput  pope_f1  agl   category
    ("Base (7B)",          38.2,   85.1,   72.0, "base"),
    ("VCD",                19.5,   87.3,   51.0, "baseline"),
    ("OPERA",              22.8,   86.8,   54.0, "baseline"),
    ("DoLa",               25.1,   86.2,   49.0, "baseline"),
    ("HALC",               23.0,   87.0,   53.0, "baseline"),
    ("ITI (Static)",       36.5,   86.5,   68.0, "baseline"),
    ("A-OSP (Ours, 7B)",   37.6,   89.8,   71.5, "ours"),
    ("A-OSP (Ours, 13B)",  24.8,   91.2,   73.0, "ours_large"),
    ("Qwen2-VL-72B†",      5.2,   90.5,   75.0, "ceiling"),
]

# ────────────────────────────────────────────────────────────────
# 配色方案 — 高对比度学术风格
# ────────────────────────────────────────────────────────────────
PALETTE = {
    "base":       "#8C8C8C",   # 中灰 — 原始模型
    "baseline":   "#4E79A7",   # 深蓝 — 现有方法
    "ours":       "#E15759",   # 鲜红 — A-OSP（核心亮点）
    "ours_large": "#B03A2E",   # 深红 — A-OSP 13B
    "ceiling":    "#59A14F",   # 翠绿 — 72B 天花板
}

MARKER_MAP = {
    "base": "D",
    "baseline": "o",
    "ours": "★",
    "ours_large": "★",
    "ceiling": "^",
}


def load_real_data(eval_dir: Path):
    """尝试从 eval_results 加载真实数据，否则返回 None 使用 Mock。"""
    pareto_json = eval_dir / "pareto_data.json"
    if pareto_json.exists():
        with open(pareto_json) as f:
            data = json.load(f)
        print(f"[INFO] 从 {pareto_json} 加载真实数据。")
        return [(d["method"], d["throughput"], d["pope_f1"],
                 d["agl"], d["category"]) for d in data]
    return None


def plot_pareto(data, output_path: Path):
    """绘制帕累托气泡图。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for method, tp, f1, agl, cat in data:
        color = PALETTE.get(cat, "#333333")
        bubble_size = (agl / 10) ** 2  # 非线性缩放，让 AGL 差异更明显

        edge_color = "white" if cat in ("ours", "ours_large") else "#555555"
        edge_width = 2.0 if cat in ("ours", "ours_large") else 0.8
        zorder = 10 if cat in ("ours", "ours_large") else 5

        if cat in ("ours", "ours_large"):
            ax.scatter(tp, f1, s=bubble_size, c=color, alpha=0.92,
                       edgecolors=edge_color, linewidths=edge_width,
                       zorder=zorder, marker="o")
            ax.annotate(
                method, (tp, f1),
                textcoords="offset points", xytext=(8, 10),
                fontsize=9.5, fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=zorder + 1,
            )
        elif cat == "ceiling":
            ax.scatter(tp, f1, s=bubble_size, c=color, alpha=0.75,
                       edgecolors="#333", linewidths=1.0,
                       zorder=4, marker="^")
            ax.annotate(
                method, (tp, f1),
                textcoords="offset points", xytext=(-10, -16),
                fontsize=8.5, fontstyle="italic", color="#3a7d32",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=6,
            )
        else:
            ax.scatter(tp, f1, s=bubble_size, c=color, alpha=0.7,
                       edgecolors=edge_color, linewidths=edge_width,
                       zorder=zorder)
            ax.annotate(
                method, (tp, f1),
                textcoords="offset points", xytext=(6, -12),
                fontsize=8, color="#444444",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    # ── 帕累托前沿虚线（连接 A-OSP 点） ───────────────────────
    ours_pts = [(tp, f1) for _, tp, f1, _, cat in data
                if cat in ("ours", "ours_large")]
    if len(ours_pts) >= 2:
        ours_pts.sort()
        xs, ys = zip(*ours_pts)
        ax.plot(xs, ys, "--", color="#E15759", alpha=0.4, linewidth=1.5,
                zorder=3)

    # ── 轴标签与标题 ─────────────────────────────────────────────
    ax.set_xlabel("Throughput (Tokens/s)  →  Linear Scale", fontsize=12,
                  fontweight="bold", labelpad=8)
    ax.set_ylabel("POPE F1 Score (%)", fontsize=12, fontweight="bold",
                  labelpad=8)
    ax.set_title("Pareto Frontier: Throughput vs. Faithfulness vs. Verbosity",
                 fontsize=13, fontweight="bold", pad=14)

    # AGL 图例（手动气泡大小说明）
    agl_legend_vals = [50, 65, 75]
    legend_bubbles = []
    for v in agl_legend_vals:
        s = (v / 10) ** 2
        lb = ax.scatter([], [], s=s, c="gray", alpha=0.35, edgecolors="gray",
                        linewidths=0.5)
        legend_bubbles.append(lb)
    leg = ax.legend(legend_bubbles, [f"AGL={v}" for v in agl_legend_vals],
                    scatterpoints=1, title="Bubble Size = AGL",
                    title_fontsize=9, fontsize=8, loc="lower right",
                    framealpha=0.85, edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.8)

    # ── 网格与美化 ───────────────────────────────────────────────
    ax.grid(True, alpha=0.25, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    x_margin = 2
    y_margin = 1.5
    all_tp = [d[1] for d in data]
    all_f1 = [d[2] for d in data]
    ax.set_xlim(min(all_tp) - x_margin, max(all_tp) + x_margin)
    ax.set_ylim(min(all_f1) - y_margin, max(all_f1) + y_margin)

    # ── 标注"长度-偏见陷阱"区域 ─────────────────────────────────
    trap_x = [d[1] for d in data if d[4] == "baseline"]
    trap_y = [d[2] for d in data if d[4] == "baseline"]
    if trap_x:
        rect_x = min(trap_x) - 1.5
        rect_y = min(trap_y) - 0.8
        rect_w = max(trap_x) - min(trap_x) + 3
        rect_h = max(trap_y) - min(trap_y) + 1.6
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (rect_x, rect_y), rect_w, rect_h,
            boxstyle="round,pad=0.3", facecolor="#4E79A7", alpha=0.06,
            edgecolor="#4E79A7", linewidth=1.2, linestyle="--", zorder=1,
        )
        ax.add_patch(box)
        ax.text(
            rect_x + rect_w / 2, rect_y - 0.4,
            "Length-Bias Trap Zone",
            ha="center", fontsize=8, fontstyle="italic", color="#4E79A7",
            alpha=0.7
        )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] 帕累托气泡图已保存 → {output_path}")
    print(f"       PDF 版本 → {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="A-OSP 帕累托气泡图")
    parser.add_argument("--output", type=str,
                        default=str(FIGURE_DIR / "pareto_bubble.png"))
    parser.add_argument("--data", type=str, default=None,
                        help="pareto_data.json 路径（默认自动检测）")
    args = parser.parse_args()

    data = None
    if args.data:
        with open(args.data) as f:
            raw = json.load(f)
        data = [(d["method"], d["throughput"], d["pope_f1"],
                 d["agl"], d["category"]) for d in raw]
    else:
        data = load_real_data(EVAL_DIR)

    if data is None:
        print("[INFO] 未检测到真实数据，使用 Mock 数据调试排版。")
        data = MOCK_DATA

    plot_pareto(data, Path(args.output))


if __name__ == "__main__":
    main()
