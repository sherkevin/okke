from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent

CASES = [
    {
        "title": "(a) Snowboard",
        "image_path": OUT_DIR / "COCO_val2014_000000310196.jpg",
    },
    {
        "title": "(b) Car",
        "image_path": OUT_DIR / "COCO_val2014_000000310196.jpg",
    },
    {
        "title": "(c) Person",
        "image_path": OUT_DIR / "COCO_val2014_000000429109.jpg",
    },
    {
        "title": "(d) Sheep",
        "image_path": OUT_DIR / "COCO_val2014_000000429109.jpg",
    },
]


def style_panel(ax: plt.Axes, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.02, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6, fontweight="bold")


def draw_placeholder_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor="#cfd6e0",
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    if label:
        ax.text(x + 0.02, y + height / 2, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.9, color="#a1a8b3")


def add_case_card(ax: plt.Axes, image_path: Path, title: str) -> None:
    style_panel(ax, title)
    image = mpimg.imread(image_path)

    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        0.95,
        transform=ax.transAxes,
        boxstyle="round,pad=0.007,rounding_size=0.014",
        linewidth=0.6,
        edgecolor="#d9dfe8",
        facecolor="#fbfcfd",
    )
    ax.add_patch(card)

    image_ax = ax.inset_axes([0.04, 0.20, 0.41, 0.63])
    image_ax.imshow(image)
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_edgecolor("#c6ccd6")
        spine.set_linewidth(0.6)

    ax.text(0.50, 0.80, "Question", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color="#5a6270")
    draw_placeholder_box(ax, 0.50, 0.67, 0.44, 0.09, "fill")

    ax.text(0.50, 0.56, "Pred.", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color="#5a6270")
    draw_placeholder_box(ax, 0.50, 0.46, 0.18, 0.06, "fill")

    ax.text(0.74, 0.56, "GT", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color="#5a6270")
    draw_placeholder_box(ax, 0.74, 0.46, 0.20, 0.06, "fill")

    ax.text(0.50, 0.34, "Openers", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color="#5a6270")
    draw_placeholder_box(ax, 0.50, 0.24, 0.13, 0.05)
    draw_placeholder_box(ax, 0.65, 0.24, 0.13, 0.05)
    draw_placeholder_box(ax, 0.80, 0.24, 0.13, 0.05)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
        }
    )

    fig = plt.figure(figsize=(3.28, 2.85))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    fig.subplots_adjust(left=0.06, right=0.985, top=0.975, bottom=0.05, hspace=0.32, wspace=0.22)

    for ax, case in zip(axes, CASES):
        add_case_card(ax, case["image_path"], case["title"])

    png_path = OUT_DIR / "figure5.png"
    pdf_path = OUT_DIR / "figure5.pdf"
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
