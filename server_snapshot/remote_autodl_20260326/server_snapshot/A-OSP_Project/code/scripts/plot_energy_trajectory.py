"""
A-OSP Figure 4 — Token-Level Energy Trajectory Plot
=====================================================
两种运行模式：

  模式 A (1×3 Mock/Real)：
    三场景子图: spike / frozen / collinear
    python plot_energy_trajectory.py

  模式 B (单面板真实 CSV)：
    直接读取 Agent 1 的 energy_trajectory_sample_*.csv
    python plot_energy_trajectory.py --csv data/micro_features/energy_trajectory_sample_1.csv

输出：logs/figures/energy_trajectory{_real}.{png,pdf}
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")
FIGURE_DIR = PROJECT_ROOT / "logs" / "figures"

MU = 1.5
BETA = 0.9

# ────────────────────────────────────────────────────────────────
# Mock 数据生成
# ────────────────────────────────────────────────────────────────

def generate_mock_spike(seed=42):
    """
    场景 (a)：模型生成到第 ~15 个 token 时尝试编造不存在的物体，
    Lt 在该点脉冲式激增，瞬间越过阈值触发 A-OSP 刹车。
    """
    rng = np.random.RandomState(seed)
    n_tokens = 28
    tokens = [
        "The", "image", "shows", "a", "living", "room", "with",
        "a", "couch", ",", "a", "table", ",", "and", "a",
        "grand", "piano", "next", "to", "a", "large",
        "bookshelf", "filled", "with", "colorful", "books", ".", "<eos>"
    ]

    base_energy = 0.35
    Lt = rng.normal(base_energy, 0.03, n_tokens)
    Lt = np.clip(Lt, 0.15, None)

    Lt[3:6] += 0.02
    Lt[7:9] += 0.01

    spike_start = 15
    Lt[spike_start] = 0.82
    Lt[spike_start + 1] = 0.71
    Lt[spike_start + 2] = 0.48

    Lt[spike_start + 3:] = rng.normal(base_energy - 0.02, 0.025,
                                       n_tokens - spike_start - 3)
    Lt = np.clip(Lt, 0.1, 1.2)

    return {
        "tokens": tokens[:n_tokens],
        "Lt": Lt.tolist(),
        "scenario": "spike",
    }


def generate_mock_frozen(seed=43):
    """
    场景 (b)：模型进入长程幻觉模式（连续 ~10 个 token 全是编造内容），
    条件 EMA 基线被完全冻结，阈值线水平不动，持续拦截。
    """
    rng = np.random.RandomState(seed)
    n_tokens = 32
    tokens = [
        "A", "photograph", "of", "a", "park", "bench", "with",
        "two", "swans", "gracefully", "swimming", "in", "the",
        "crystal", "clear", "fountain", "surrounded", "by",
        "blooming", "cherry", "blossom", "trees", "and", "a",
        "small", "wooden", "bridge", "over", "the", "stream",
        ".", "<eos>"
    ]

    base_energy = 0.33
    Lt = rng.normal(base_energy, 0.025, n_tokens)

    halluc_start = 8
    halluc_end = 22
    for i in range(halluc_start, halluc_end):
        progress = (i - halluc_start) / (halluc_end - halluc_start)
        Lt[i] = 0.65 + 0.15 * np.sin(progress * np.pi) + rng.normal(0, 0.03)

    Lt[halluc_end:] = rng.normal(base_energy, 0.025, n_tokens - halluc_end)
    Lt = np.clip(Lt, 0.1, 1.0)

    return {
        "tokens": tokens[:n_tokens],
        "Lt": Lt.tolist(),
        "scenario": "frozen",
    }


def generate_mock_collinear(seed=44):
    """
    场景 (c)：图中确实有 "apple"，语言先验也倾向输出 "apple"，
    Lt 平滑上升但 EMA 基线同步"水涨船高"，全程不触发刹车。
    """
    rng = np.random.RandomState(seed)
    n_tokens = 25
    tokens = [
        "The", "image", "shows", "a", "wooden", "table", "with",
        "a", "red", "apple", ",", "a", "glass", "of", "water",
        ",", "and", "a", "white", "plate", "with", "bread",
        "on", "it", "."
    ]

    t = np.arange(n_tokens, dtype=float)
    base = 0.30 + 0.08 * (t / n_tokens)
    Lt = base + rng.normal(0, 0.018, n_tokens)

    Lt[8:10] += 0.04
    Lt[12:14] += 0.03
    Lt[18:20] += 0.025

    Lt = np.clip(Lt, 0.1, 0.7)

    return {
        "tokens": tokens[:n_tokens],
        "Lt": Lt.tolist(),
        "scenario": "collinear",
    }


# ────────────────────────────────────────────────────────────────
# EMA 阈值计算
# ────────────────────────────────────────────────────────────────

def compute_ema_threshold(Lt_arr, mu=MU, beta=BETA, burn_in=4):
    """
    复现 A-OSP 的条件 EMA 分段基线更新：
      - burn_in 期间基线冻结为先验均值
      - 之后仅在 Lt <= mu * L_bar 时更新 EMA
      - 触发时基线完全冻结
    """
    n = len(Lt_arr)
    L_bar = np.zeros(n)
    triggered = np.zeros(n, dtype=bool)

    prior_mean = np.mean(Lt_arr[:burn_in]) if burn_in > 0 else Lt_arr[0]
    L_bar[0] = prior_mean

    for t in range(1, n):
        if t <= burn_in:
            L_bar[t] = prior_mean
        else:
            threshold = mu * L_bar[t - 1]
            if Lt_arr[t] > threshold:
                triggered[t] = True
                L_bar[t] = L_bar[t - 1]
            else:
                L_bar[t] = beta * L_bar[t - 1] + (1 - beta) * Lt_arr[t]

    threshold_line = mu * L_bar
    return L_bar, threshold_line, triggered


# ────────────────────────────────────────────────────────────────
# 绘图
# ────────────────────────────────────────────────────────────────

SUBPLOT_CONFIG = {
    "spike": {
        "title": "(a) Abrupt Spike Detection",
        "annotation_text": "Hallucinated\nEntity",
        "annotation_color": "#C0392B",
        "highlight_label": "A-OSP\nTriggered",
        "zone_label": None,
        "zone_color": None,
    },
    "frozen": {
        "title": "(b) Long-context EMA Freeze",
        "annotation_text": None,
        "annotation_color": None,
        "highlight_label": None,
        "zone_label": "EMA Frozen\n(Sustained Hallucination)",
        "zone_color": "#E67E22",
    },
    "collinear": {
        "title": "(c) Collinear False-Positive Robustness",
        "annotation_text": "Visual\nFact",
        "annotation_color": "#27AE60",
        "highlight_label": None,
        "zone_label": "Smooth Accumulation\n(No False Trigger)",
        "zone_color": "#27AE60",
    },
}


def load_real_data(data_dir: Path):
    """尝试加载 Agent 1 导出的真实能量轨迹数据。"""
    scenarios = {}
    mapping = {
        "spike": "energy_trace_spike.json",
        "frozen": "energy_trace_frozen.json",
        "collinear": "energy_trace_collinear.json",
    }
    all_found = True
    for key, fname in mapping.items():
        fp = data_dir / fname
        if fp.exists():
            with open(fp) as f:
                scenarios[key] = json.load(f)
        else:
            all_found = False
            break

    if all_found:
        print(f"[INFO] 从 {data_dir} 加载真实能量轨迹数据。")
        return scenarios
    return None


def plot_energy_trajectory(scenarios: dict, output_path: Path):
    """绘制 1×3 Token 级能量轨迹图。"""

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    fig.patch.set_facecolor("white")

    ordering = ["spike", "frozen", "collinear"]

    for idx, key in enumerate(ordering):
        ax = axes[idx]
        data = scenarios[key]
        cfg = SUBPLOT_CONFIG[key]

        Lt = np.array(data["Lt"])
        tokens = data.get("tokens", [f"t{i}" for i in range(len(Lt))])
        n = len(Lt)

        L_bar, threshold, triggered = compute_ema_threshold(Lt)

        x = np.arange(n)

        # ── 动态包络线阴影 ────────────────────────────────
        ax.fill_between(x, 0, threshold, alpha=0.08, color="#95A5A6",
                        label="Dynamic Envelope", zorder=1)

        # ── Lt 柱状图（半透明底色） ──────────────────────
        bar_colors = []
        for t in range(n):
            if triggered[t]:
                bar_colors.append("#E74C3C")
            elif key == "frozen" and 8 <= t <= 21:
                bar_colors.append("#E67E22")
            elif key == "collinear":
                bar_colors.append("#27AE60")
            else:
                bar_colors.append("#5DADE2")

        ax.bar(x, Lt, width=0.6, color=bar_colors, alpha=0.25, zorder=2)

        # ── Lt 实线 ──────────────────────────────────────
        ax.plot(x, Lt, "-o", color="#E74C3C", linewidth=1.8, markersize=3.5,
                markerfacecolor="#E74C3C", markeredgecolor="white",
                markeredgewidth=0.6, label=r"$L_t$ (Projection Energy)",
                zorder=4)

        # ── 阈值虚线 μ·L̄_{t-1} ─────────────────────────
        ax.plot(x, threshold, "--", color="#2980B9", linewidth=2.0,
                label=r"$\mu \bar{L}_{t-1}$ (EMA Threshold)",
                zorder=3)

        # ── 触发标记 ─────────────────────────────────────
        trigger_idx = np.where(triggered)[0]
        if len(trigger_idx) > 0:
            ax.scatter(trigger_idx, Lt[trigger_idx], marker="*", s=200,
                       c="#F1C40F", edgecolors="#E67E22", linewidths=0.8,
                       zorder=6, label="A-OSP Triggered")

        # ── 场景专属标注 ─────────────────────────────────
        if key == "spike" and len(trigger_idx) > 0:
            peak_t = trigger_idx[0]
            ax.annotate(
                cfg["highlight_label"],
                xy=(peak_t, Lt[peak_t]),
                xytext=(peak_t + 2.5, Lt[peak_t] + 0.06),
                fontsize=9, fontweight="bold", color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=7,
            )
            ax.annotate(
                cfg["annotation_text"],
                xy=(peak_t, 0),
                xytext=(peak_t, -0.08),
                fontsize=8, fontstyle="italic", color=cfg["annotation_color"],
                ha="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                annotation_clip=False,
                zorder=7,
            )

        if key == "frozen":
            frozen_start = next((t for t in range(n) if triggered[t]), None)
            frozen_end = n - 1
            for t in range(n - 1, -1, -1):
                if triggered[t] or (8 <= t <= 21):
                    frozen_end = t
                    break
            if frozen_start is not None:
                ax.axvspan(frozen_start - 0.5, frozen_end + 0.5,
                           alpha=0.12, color="#E67E22", zorder=1)
                mid = (frozen_start + frozen_end) / 2
                ax.text(
                    mid, max(Lt) * 0.92,
                    cfg["zone_label"],
                    ha="center", fontsize=8, fontstyle="italic",
                    color="#D35400", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=7,
                )

        if key == "collinear":
            fact_tokens_idx = [i for i, t in enumerate(tokens)
                               if t.lower() in ("apple", "water", "bread",
                                                 "red", "glass", "plate")]
            for fi in fact_tokens_idx:
                ax.scatter(fi, Lt[fi] + 0.03, marker="v", s=60,
                           c="#27AE60", edgecolors="white", linewidths=0.5,
                           zorder=7)
            ax.text(
                n / 2, max(threshold) + 0.04,
                cfg["zone_label"],
                ha="center", fontsize=8, fontstyle="italic",
                color="#1E8449", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=7,
            )

        # ── Token 标签 ───────────────────────────────────
        display_tokens = tokens[:n]
        if n > 20:
            step = max(n // 12, 2)
            tick_pos = list(range(0, n, step))
            if (n - 1) not in tick_pos:
                tick_pos.append(n - 1)
        else:
            tick_pos = list(range(n))

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [display_tokens[i] if i < len(display_tokens) else ""
             for i in tick_pos],
            rotation=55, ha="right", fontsize=7,
        )

        ax.set_ylabel("Projection Energy" if idx == 0 else "", fontsize=11)
        ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(-0.8, n - 0.2)
        y_max = max(max(Lt), max(threshold)) * 1.15
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.2, linewidth=0.5, linestyle="--", axis="y")
        ax.set_axisbelow(True)

        if idx == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
                      edgecolor="#ccc", handlelength=1.5)

    fig.text(0.5, -0.02, "Token Generation Sequence  →",
             ha="center", fontsize=11, fontstyle="italic", color="#555")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.subplots_adjust(wspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] 能量轨迹图已保存 → {output_path}")
    print(f"       PDF 版本 → {output_path.with_suffix('.pdf')}")


# ────────────────────────────────────────────────────────────────
# 模式 B：单面板真实 CSV 渲染
# ────────────────────────────────────────────────────────────────

def load_csv_trajectory(csv_path: Path) -> dict:
    """
    加载 Agent 1 的 energy_trajectory_sample_*.csv。
    列: step, L_t, L_bar, triggered, burn_in, token
    """
    steps, Lt_list, Lbar_list = [], [], []
    triggered_list, burnin_list, tokens = [], [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            Lt_list.append(float(row["L_t"]))
            Lbar_list.append(float(row["L_bar"]))
            triggered_list.append(row["triggered"].strip() == "True")
            burnin_list.append(row["burn_in"].strip() == "True")
            tokens.append(row["token"].strip())

    return {
        "steps": np.array(steps),
        "Lt": np.array(Lt_list),
        "L_bar": np.array(Lbar_list),
        "triggered": np.array(triggered_list),
        "burn_in": np.array(burnin_list),
        "tokens": tokens,
    }


def plot_single_real_trajectory(data: dict, output_path: Path, mu: float = MU):
    """
    绘制单面板真实能量轨迹图 — 学术顶刊级排版。
    核心叙事：长程自回归生成中，大部分 token 的 Lt 平滑演化，
    仅在少数步骤出现脉冲触发（黄色星号），证明 A-OSP 的外科手术式精度。
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })

    Lt = data["Lt"]
    L_bar = data["L_bar"]
    triggered = data["triggered"]
    burn_in = data["burn_in"]
    tokens = data["tokens"]
    n = len(Lt)
    x = np.arange(n)

    threshold = mu * L_bar

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    # ── burn-in 区域淡紫色 ───────────────────────────────
    burn_end = np.where(burn_in)[0]
    if len(burn_end) > 0:
        ax.axvspan(-0.5, burn_end[-1] + 0.5, alpha=0.10, color="#9B59B6",
                   zorder=0)
        mid_burn = burn_end[-1] / 2
        ax.text(mid_burn, max(threshold[~burn_in]) * 0.97,
                "Entropy-Aware\nBurn-in",
                ha="center", fontsize=8, fontstyle="italic",
                color="#7D3C98", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=8)

    # ── 动态包络线阴影（仅 post-burn-in） ────────────────
    active_mask = ~burn_in
    ax.fill_between(x, 0, threshold, where=active_mask,
                    alpha=0.07, color="#2980B9", zorder=1,
                    label="Dynamic Envelope $\\mu \\bar{L}_{t-1}$")

    # ── Lt 柱状图 ────────────────────────────────────────
    bar_colors = np.where(triggered, "#E74C3C",
                 np.where(burn_in, "#C39BD3", "#5DADE2"))
    ax.bar(x, Lt, width=0.65, color=bar_colors, alpha=0.22, zorder=2)

    # ── Lt 实线 ──────────────────────────────────────────
    active_x = x[active_mask]
    active_Lt = Lt[active_mask]
    ax.plot(active_x, active_Lt, "-", color="#E74C3C", linewidth=1.4,
            alpha=0.85, zorder=4)
    ax.plot(active_x, active_Lt, "o", color="#E74C3C", markersize=2.2,
            markerfacecolor="#E74C3C", markeredgecolor="white",
            markeredgewidth=0.4, zorder=5,
            label=r"$L_t$ (Projection Energy)")

    # ── EMA 基线（蓝色虚线） ─────────────────────────────
    ax.plot(x[active_mask], L_bar[active_mask], "-.", color="#2980B9",
            linewidth=1.4, alpha=0.7,
            label=r"$\bar{L}_t$ (EMA Baseline)", zorder=3)

    # ── 阈值线（蓝色粗虚线） ─────────────────────────────
    ax.plot(x[active_mask], threshold[active_mask], "--", color="#2980B9",
            linewidth=2.0, alpha=0.5, zorder=3,
            label=r"$\mu \bar{L}_{t-1}$ (Threshold, $\mu$=%.1f)" % mu)

    # ── 触发标记（黄色星号） ─────────────────────────────
    trig_idx = np.where(triggered)[0]
    if len(trig_idx) > 0:
        ax.scatter(trig_idx, Lt[trig_idx], marker="*", s=220,
                   c="#F1C40F", edgecolors="#E67E22", linewidths=1.0,
                   zorder=7, label=f"A-OSP Triggered (n={len(trig_idx)})")

        for ti in trig_idx:
            token_str = tokens[ti] if ti < len(tokens) else "?"
            ax.annotate(
                f'"{token_str}"',
                xy=(ti, Lt[ti]),
                xytext=(ti + 3, Lt[ti] + 12),
                fontsize=7.5, fontstyle="italic", color="#C0392B",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                                lw=1.2, shrinkA=0, shrinkB=3),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=8,
            )

    # ── Token 标签（稀疏显示） ───────────────────────────
    if n > 40:
        step = max(n // 18, 3)
        tick_pos = list(range(0, n, step))
        for ti in trig_idx:
            if ti not in tick_pos:
                tick_pos.append(ti)
        tick_pos = sorted(set(tick_pos))
        if (n - 1) not in tick_pos:
            tick_pos.append(n - 1)
    else:
        tick_pos = list(range(n))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [f'{p}: {tokens[p][:8]}' if p < len(tokens) else ""
         for p in tick_pos],
        rotation=55, ha="right", fontsize=6.5,
    )

    # ── 统计摘要文本 ─────────────────────────────────────
    n_active = int(active_mask.sum())
    n_trig = int(triggered.sum())
    trig_rate = n_trig / n_active * 100 if n_active > 0 else 0
    stats_text = (f"Total tokens: {n}  |  Active (post burn-in): {n_active}  |  "
                  f"Interventions: {n_trig} ({trig_rate:.1f}%)")
    ax.text(0.5, -0.18, stats_text, transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#555",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── 轴与美化 ─────────────────────────────────────────
    ax.set_xlabel("Token Generation Step", fontsize=11, labelpad=6)
    ax.set_ylabel("Projection Energy $L_t$", fontsize=11, labelpad=6)
    ax.set_title(
        "Real A-OSP Energy Trajectory — Qwen2-VL-7B (Single Sample, 127 Tokens)",
        fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-1, n)
    y_max = max(max(Lt[active_mask]), max(threshold[active_mask])) * 1.15
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.18, linewidth=0.5, linestyle="--", axis="y")
    ax.set_axisbelow(True)

    ax.legend(fontsize=8, loc="upper right", framealpha=0.92,
              edgecolor="#ccc", ncol=2, handlelength=1.8)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] 真实能量轨迹图已保存 → {output_path}")
    print(f"       PDF 版本 → {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="A-OSP Figure 4 — Energy Trajectory")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认自动决定）")
    parser.add_argument("--csv", type=str, default=None,
                        help="模式 B: 直接读取 Agent 1 的 CSV 文件路径")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="模式 A: Agent 1 导出的 energy_trace_*.json 目录")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        print(f"[INFO] 模式 B: 从 CSV 加载真实数据 → {csv_path}")
        data = load_csv_trajectory(csv_path)
        out = Path(args.output) if args.output else FIGURE_DIR / "energy_trajectory_real.png"
        plot_single_real_trajectory(data, out)
    else:
        scenarios = None
        if args.data_dir:
            scenarios = load_real_data(Path(args.data_dir))

        if scenarios is None:
            print("[INFO] 未检测到真实数据，使用 Mock 数据调试排版。")
            scenarios = {
                "spike": generate_mock_spike(),
                "frozen": generate_mock_frozen(),
                "collinear": generate_mock_collinear(),
            }

        out = Path(args.output) if args.output else FIGURE_DIR / "energy_trajectory.png"
        plot_energy_trajectory(scenarios, out)


if __name__ == "__main__":
    main()
