"""
A-OSP Figure 3 — UMAP Topological Unphasing Visualization
===========================================================
1×3 布局：浅层(L5) → 中层(L15) → 深层(L24)

核心学术叙事：
  - 浅层：视觉特征与语言先验呈螺旋状非线性纠缠
  - 中层：部分解耦但边界模糊
  - 深层：清晰拓扑解相 — SVM/LogReg 拟合的超平面完美分离

数据源（优先级从高到低）：
  1. --pt_file  : Agent 1 导出的 .pt 张量（torch.load）
  2. --feature_dir : CSV 文件目录
  3. 内置合成数据

.pt 文件期望结构（单样本 mini 或多样本 batch）：
  Mini:  { "coco_XXXX": { "base": {5: [T,D], 15: [T,D], 24: [T,D]},
                           "aosp": {5: [T,D], 15: [T,D], 24: [T,D]} } }
  Batch: { "grounded": {5: [N,D], 15: [N,D], 24: [N,D]},
            "prior":    {5: [N,D], 15: [N,D], 24: [N,D]} }

用法：
  python plot_umap_topology.py
  python plot_umap_topology.py --pt_file data/micro_features/umap_features_mini.pt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

try:
    import umap
except ImportError:
    import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")
FIGURE_DIR = PROJECT_ROOT / "logs" / "figures"

LAYER_KEYS = [5, 15, 24]
LAYER_LABELS = {
    5:  {"name": "Layer 5 (Shallow)",       "depth": "shallow"},
    15: {"name": "Layer 15 (Mid)",          "depth": "mid"},
    24: {"name": "Layer 24 (Deep, L-4)",    "depth": "deep"},
}

UMAP_PARAMS = {
    "n_neighbors": 25,
    "min_dist": 0.25,
    "metric": "cosine",
    "random_state": 42,
}

COLOR_GROUNDED = "#2171B5"
COLOR_PRIOR    = "#E15759"

# ────────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────────

def load_pt_features(pt_path: Path) -> dict:
    """
    加载 Agent 1 导出的 .pt 张量。
    自动检测 mini（单/多样本嵌套字典）或 batch 格式。
    返回: {layer_key: (features [N, D], labels [N])}
    """
    import torch
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # ── 检测格式 ─────────────────────────────────────────
    if "grounded" in raw and "prior" in raw:
        return _parse_batch_format(raw)
    else:
        return _parse_mini_format(raw)


def _parse_batch_format(raw: dict) -> dict:
    """Batch 格式: grounded/prior 各含 {layer: [N,D]}。"""
    import torch
    features_by_layer = {}
    for lk in LAYER_KEYS:
        g = raw["grounded"][lk]
        p = raw["prior"][lk]
        if isinstance(g, torch.Tensor):
            g, p = g.numpy(), p.numpy()
        features = np.vstack([g, p])
        labels = np.array([0] * len(g) + [1] * len(p))
        features_by_layer[lk] = (features, labels)
    return features_by_layer


def _parse_mini_format(raw: dict) -> dict:
    """
    Mini 格式: 一个或多个样本的 base/aosp 字典。
    将所有样本的 token-level 特征堆叠：base→class 0, aosp→class 1。
    """
    import torch
    all_base = {lk: [] for lk in LAYER_KEYS}
    all_aosp = {lk: [] for lk in LAYER_KEYS}

    for sample_key, sample_data in raw.items():
        base_dict = sample_data.get("base", {})
        aosp_dict = sample_data.get("aosp", {})
        for lk in LAYER_KEYS:
            if lk in base_dict:
                t = base_dict[lk]
                all_base[lk].append(t.numpy() if isinstance(t, torch.Tensor) else t)
            if lk in aosp_dict:
                t = aosp_dict[lk]
                all_aosp[lk].append(t.numpy() if isinstance(t, torch.Tensor) else t)

    features_by_layer = {}
    for lk in LAYER_KEYS:
        if all_base[lk] and all_aosp[lk]:
            base_cat = np.concatenate(all_base[lk], axis=0)
            aosp_cat = np.concatenate(all_aosp[lk], axis=0)
            features = np.vstack([base_cat, aosp_cat])
            labels = np.array([0] * len(base_cat) + [1] * len(aosp_cat))
            features_by_layer[lk] = (features, labels)

    return features_by_layer


def load_csv_features(feature_dir: Path) -> dict:
    """CSV fallback: layer_5_features.csv etc, last column = label."""
    features_by_layer = {}
    tag_map = {5: "layer_5", 15: "layer_15", 24: "layer_24"}
    for lk, tag in tag_map.items():
        csv_path = feature_dir / f"{tag}_features.csv"
        if not csv_path.exists():
            return None
        data = np.loadtxt(str(csv_path), delimiter=",")
        features_by_layer[lk] = (data[:, :-1], data[:, -1].astype(int))
    return features_by_layer


def generate_synthetic_features(n_per_class=300, dim=3584, seed=42):
    """合成数据：模拟三层级拓扑演化。"""
    rng = np.random.RandomState(seed)
    features_by_layer = {}

    for lk in LAYER_KEYS:
        depth = LAYER_LABELS[lk]["depth"]
        if depth == "shallow":
            t = np.linspace(0, 4 * np.pi, n_per_class)
            vis_base = np.column_stack([
                np.cos(t) * (1 + 0.5 * np.sin(3 * t)),
                np.sin(t) * (1 + 0.5 * np.cos(3 * t)),
                t / (4 * np.pi),
            ])
            prior_base = np.column_stack([
                np.cos(t + np.pi / 3) * (1 + 0.5 * np.cos(3 * t)),
                np.sin(t + np.pi / 3) * (1 + 0.5 * np.sin(3 * t)),
                t / (4 * np.pi) + 0.1,
            ])
            noise = 0.15
        elif depth == "mid":
            vis_base = rng.randn(n_per_class, 3) * 0.8 + [1.5, 0, 0]
            prior_base = rng.randn(n_per_class, 3) * 0.8 + [-1.0, 0.5, 0]
            noise = 0.10
        else:
            vis_base = rng.randn(n_per_class, 3) * 0.4 + [3.0, 0, 0]
            prior_base = rng.randn(n_per_class, 3) * 0.4 + [-3.0, 0, 0]
            noise = 0.05

        pad = dim - 3
        vis = np.hstack([vis_base, rng.randn(n_per_class, pad) * noise])
        pri = np.hstack([prior_base, rng.randn(n_per_class, pad) * noise])
        features_by_layer[lk] = (
            np.vstack([vis, pri]),
            np.array([0] * n_per_class + [1] * n_per_class),
        )

    return features_by_layer


# ────────────────────────────────────────────────────────────────
# SVM / LogReg 分离超平面
# ────────────────────────────────────────────────────────────────

def fit_separating_hyperplane(embedding_2d, labels):
    """
    在 2D UMAP 空间中用 LogisticRegression 拟合决策边界，
    返回绘制虚线所需的 (x_line, y_line)。
    """
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(embedding_2d, labels)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    margin = (x_max - x_min) * 0.15
    x_line = np.linspace(x_min - margin, x_max + margin, 200)

    if abs(w[1]) < 1e-10:
        x_val = -b / w[0] if abs(w[0]) > 1e-10 else 0
        y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
        return np.array([x_val, x_val]), np.array([y_min, y_max]), clf.score(embedding_2d, labels)

    y_line = -(w[0] * x_line + b) / w[1]
    acc = clf.score(embedding_2d, labels)
    return x_line, y_line, acc


# ────────────────────────────────────────────────────────────────
# 绘图
# ────────────────────────────────────────────────────────────────

SUBTITLE_STYLE = {
    "shallow": ("Spiral Entanglement\n(Multiplicative Coupling)", "#B85450"),
    "mid":     ("Partial Decoupling\n(Fuzzy Boundaries)",         "#D4A843"),
    "deep":    ("Topological Unphasing\n(Orthogonal Separability)", "#2E8B57"),
}


def plot_umap_triptych(features_by_layer: dict, output_path: Path,
                       class0_label: str = "Visually Grounded",
                       class1_label: str = "Language Prior"):
    """绘制 1×3 UMAP 拓扑解相图。"""

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.patch.set_facecolor("white")

    for idx, lk in enumerate(LAYER_KEYS):
        ax = axes[idx]
        cfg = LAYER_LABELS[lk]
        depth = cfg["depth"]

        features, labels = features_by_layer[lk]
        n_total = len(labels)
        n_class0 = int((labels == 0).sum())
        n_class1 = int((labels == 1).sum())

        print(f"  [Layer {lk}] {n_total} points ({n_class0} grounded + {n_class1} prior), dim={features.shape[1]}")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.astype(np.float64))

        reducer = umap.UMAP(**UMAP_PARAMS)
        embedding = reducer.fit_transform(features_scaled)

        vis_mask = labels == 0
        prior_mask = labels == 1

        ax.scatter(
            embedding[vis_mask, 0], embedding[vis_mask, 1],
            c=COLOR_GROUNDED, s=10, alpha=0.50, edgecolors="none",
            label=class0_label, rasterized=True,
        )
        ax.scatter(
            embedding[prior_mask, 0], embedding[prior_mask, 1],
            c=COLOR_PRIOR, s=10, alpha=0.50, edgecolors="none",
            label=class1_label, rasterized=True,
        )

        # ── 深层子图：SVM/LogReg 分离超平面 ──────────────
        if depth == "deep":
            x_line, y_line, acc = fit_separating_hyperplane(embedding, labels)

            y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
            y_margin = (y_max - y_min) * 0.3
            line_mask = (y_line > y_min - y_margin) & (y_line < y_max + y_margin)

            ax.plot(x_line[line_mask], y_line[line_mask],
                    "--", color="#333333", linewidth=2.0, alpha=0.7,
                    zorder=6)

            mid_idx = len(x_line[line_mask]) // 2
            if mid_idx > 0 and len(x_line[line_mask]) > 0:
                tx = x_line[line_mask][mid_idx]
                ty = y_line[line_mask][mid_idx]
                ax.annotate(
                    f"Separating Hyperplane\n(LogReg Acc={acc:.1%})",
                    xy=(tx, ty),
                    xytext=(tx + (embedding[:, 0].max() - embedding[:, 0].min()) * 0.15,
                            ty + (embedding[:, 1].max() - embedding[:, 1].min()) * 0.2),
                    fontsize=8, fontstyle="italic", color="#333",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2),
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=7,
                )

        ax.set_title(cfg["name"], fontsize=12, fontweight="bold", pad=10)

        sub_text, sub_color = SUBTITLE_STYLE[depth]
        ax.text(
            0.5, -0.08, sub_text,
            transform=ax.transAxes, ha="center", fontsize=9,
            fontstyle="italic", color=sub_color,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── 全局图例 ─────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLOR_GROUNDED, markersize=8,
                   label=f"{class0_label} Features"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLOR_PRIOR, markersize=8,
                   label=f"{class1_label} Features"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2, fontsize=10,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Feature Manifold Topology Across Network Depths",
        fontsize=14, fontweight="bold", y=1.08,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] UMAP 拓扑解相图已保存 → {output_path}")
    print(f"       PDF 版本 → {output_path.with_suffix('.pdf')}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A-OSP Figure 3 — UMAP Topology")
    parser.add_argument("--output", type=str,
                        default=str(FIGURE_DIR / "umap_topology.png"))
    parser.add_argument("--pt_file", type=str, default=None,
                        help="Agent 1 导出的 .pt 张量路径")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="CSV 特征目录（fallback）")
    parser.add_argument("--n_samples", type=int, default=300,
                        help="合成数据每类样本数")
    parser.add_argument("--dim", type=int, default=3584,
                        help="合成数据维度")
    parser.add_argument("--class0", type=str, default="Base Model",
                        help="Class 0 标签名")
    parser.add_argument("--class1", type=str, default="A-OSP Intervened",
                        help="Class 1 标签名")
    args = parser.parse_args()

    features_by_layer = None

    if args.pt_file:
        pt_path = Path(args.pt_file)
        if pt_path.exists():
            print(f"[INFO] 从 .pt 加载特征: {pt_path}")
            features_by_layer = load_pt_features(pt_path)
        else:
            print(f"[WARN] .pt 文件不存在: {pt_path}")

    if features_by_layer is None and args.feature_dir:
        features_by_layer = load_csv_features(Path(args.feature_dir))

    if features_by_layer is None:
        print("[INFO] 未检测到真实特征数据，使用合成数据调试排版。")
        features_by_layer = generate_synthetic_features(
            n_per_class=args.n_samples, dim=args.dim
        )
        args.class0 = "Visually Grounded"
        args.class1 = "Language Prior"

    plot_umap_triptych(features_by_layer, Path(args.output),
                       class0_label=args.class0, class1_label=args.class1)


if __name__ == "__main__":
    main()
