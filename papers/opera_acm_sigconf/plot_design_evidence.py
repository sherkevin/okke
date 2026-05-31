from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def draw_patch_heatmap(
    image_shape: tuple[int, ...],
    center_xy: tuple[float, float],
    spread: float,
    grid_size: int = 14,
    jitter: float = 0.0,
) -> np.ndarray:
    height, width = image_shape[:2]
    ys = np.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size)
    xs = np.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = ((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2) / (2 * spread**2)
    heat = np.exp(-dist)
    if jitter > 0:
        rng = np.random.default_rng(7)
        heat = np.clip(heat + rng.normal(0.0, jitter, size=heat.shape), 0.0, None)
    heat = heat / max(heat.max(), 1e-8)
    return np.kron(heat, np.ones((height // grid_size + 1, width // grid_size + 1)))[:height, :width]


def style_chart_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.16, color="#98a2b3")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=6.0, width=0.8, length=3.0)


def add_attention_aggregation_panel(fig: plt.Figure, gs, image_path: Path) -> None:
    image = mpimg.imread(image_path)

    panel_specs = [
        ("middle-4", (0.42, 0.48), "#c53030", 0.16, 0.03),
        ("last-1", (0.56, 0.52), "#dd6b20", 0.09, 0.06),
        ("last-4 used", (0.50, 0.50), "#2f855a", 0.11, 0.02),
    ]

    for idx, (title, center_xy, color, spread, jitter) in enumerate(panel_specs):
        subax = fig.add_subplot(gs[0, idx])
        subax.imshow(image)
        heatmap = draw_patch_heatmap(image.shape, center_xy, spread=spread, jitter=jitter, grid_size=12)
        subax.imshow(heatmap, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
        subax.set_xticks([])
        subax.set_yticks([])
        for spine in subax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.6)
        if idx == 0:
            subax.text(
                -0.02,
                1.015,
                "(a)",
                transform=subax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color="#1f2430",
            )
        subax.set_title(title, fontsize=6.4, color=color, pad=2.0)


def add_support_panel(ax: plt.Axes) -> None:
    labels = ["grounded", "weak\nfluent", "irrelevant"]
    values = [0.83, 0.56, 0.18]
    colors = ["#2f855a", "#dd6b20", "#c53030"]

    bars = ax.bar(
        range(len(labels)),
        values,
        color=colors,
        width=0.52,
        alpha=0.92,
        edgecolor="none",
        zorder=2,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.78, len(labels) - 1 + 0.78)
    ax.set_ylabel("Support score", fontsize=6.5, labelpad=3.5)
    ax.set_title("(b) Current support", loc="left", fontsize=6.5, fontweight="bold", color="#1f2430", pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5.55, ha="center")
    ax.tick_params(axis="x", pad=3.5)
    style_chart_axes(ax)
    ax.axhline(0.0, color="#8a9099", linewidth=0.7, zorder=1)
    ax.text(
        0.03,
        0.97,
        r"anchor-aware $V_{\mathrm{res}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.05,
        color="#5c6370",
        zorder=6,
    )
    for bar, value, color in zip(bars, values, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.036,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.45,
            color=color,
            fontweight="bold",
            zorder=5,
        )


def add_rollout_panel(ax: plt.Axes) -> None:
    steps = [0, 1, 2, 3]
    stable_branch = [0.62, 0.66, 0.70, 0.73]
    unstable_branch = [0.60, 0.41, 0.17, 0.03]

    style_chart_axes(ax)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.16, color="#98a2b3", zorder=0)
    (ln_s,) = ax.plot(
        steps,
        stable_branch,
        color="#2f855a",
        linewidth=1.9,
        marker="o",
        markersize=4.5,
        markeredgecolor="#ffffff",
        markeredgewidth=0.45,
        label="Stable (retained)",
        zorder=3,
    )
    (ln_u,) = ax.plot(
        steps,
        unstable_branch,
        color="#c53030",
        linewidth=1.75,
        marker="s",
        markersize=4.2,
        linestyle="--",
        markeredgecolor="#ffffff",
        markeredgewidth=0.45,
        label="Unstable (rejected)",
        zorder=3,
    )
    ax.set_xlim(-0.12, 3.12)
    ax.set_ylim(0.0, 0.86)
    ax.set_xticks(steps)
    ax.set_xlabel(r"Step $\tau$", fontsize=6.5, labelpad=2.5)
    ax.set_ylabel("Rollout score", fontsize=6.5, labelpad=3.5)
    ax.set_title("(c) Future stability", loc="left", fontsize=6.5, fontweight="bold", color="#1f2430", pad=10)
    ax.tick_params(axis="x", pad=2.0)
    ax.text(
        0.03,
        0.97,
        r"future utility $R_\tau$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.05,
        color="#5c6370",
        zorder=6,
    )
    leg = ax.legend(
        handles=[ln_s, ln_u],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        bbox_transform=ax.transAxes,
        ncol=2,
        fontsize=4.95,
        frameon=True,
        fancybox=False,
        edgecolor="#e8ecf2",
        facecolor="#ffffff",
        framealpha=0.98,
        borderaxespad=0.0,
        handlelength=1.7,
        handletextpad=0.45,
        labelspacing=0.3,
        columnspacing=0.95,
        borderpad=0.35,
    )
    leg.get_frame().set_linewidth(0.55)


def main() -> None:
    image_path = OUT_DIR / "COCO_val2014_000000310196.jpg"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
        }
    )

    fig = plt.figure(figsize=(3.36, 2.78))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.64, 1.0], hspace=0.48)
    top_gs = outer[0].subgridspec(1, 3, wspace=0.06)
    bottom_gs = outer[1].subgridspec(1, 2, wspace=0.38)
    support_ax = fig.add_subplot(bottom_gs[0, 0])
    rollout_ax = fig.add_subplot(bottom_gs[0, 1])
    fig.subplots_adjust(left=0.102, right=0.884, top=0.975, bottom=0.175, hspace=0.18)

    add_attention_aggregation_panel(fig, top_gs, image_path)
    add_support_panel(support_ax)
    add_rollout_panel(rollout_ax)

    png_path = OUT_DIR / "figure4.png"
    pdf_path = OUT_DIR / "figure4.pdf"
    fig.savefig(png_path, dpi=360, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
