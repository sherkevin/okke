"""
A-OSP §4.4 — Principal Angle Subspace Similarity Calculator
=============================================================
Pivoted from full-rank Frobenius Grassmannian distance (which accumulates
noise from K=20 tail eigenvalues) to the **Principal Angles** metric.

Key insight: The leading singular vectors of V₁ᵀV₂ capture the most
meaningful mechanistic alignment. Reporting mean cos(θ) for top-K
principal directions at K=1, 3, 5 provides a noise-robust proof of
"Mechanistic Homology" for Section 4.4.

Math:
    cos(θᵢ) = σᵢ(V₁ᵀ V₂)     (principal angles, sorted desc.)
    mean_cos(K) = (1/K) Σᵢ₌₁ᴷ cos(θᵢ)

Interpretation:
    mean_cos(K=1) ≥ 0.6 → dominant hallucination direction is shared
    mean_cos(K=3) ≥ 0.5 → leading 3-D subspace is well-aligned
    mean_cos(K=5) ≥ 0.4 → robust structural homology

Usage:
    python calc_grassmannian.py --all_pairs
    python calc_grassmannian.py --v1 V_blur.pt --v2 V_solid.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


MODELS_DIR = Path("/root/autodl-tmp/A-OSP_Project/models")
REPORT_DIR = Path("/root/autodl-tmp/A-OSP_Project/logs/grassmannian")


def load_subspace(pt_path: str) -> tuple[np.ndarray, dict]:
    """Load V_bias from .pt → [D, K] column-orthonormal matrix + metadata."""
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    V = data["V_bias"].float().numpy()            # [K, D]
    V = V.T                                        # → [D, K]

    meta = {
        "evr": float(data.get("evr", 0)),
        "K": int(data.get("K", V.shape[1])),
        "sv_top5": data["singular_values"][:5].tolist() if "singular_values" in data else [],
    }
    return V, meta


def ensure_orthonormal(V: np.ndarray, name: str) -> np.ndarray:
    K = V.shape[1]
    gram = V.T @ V
    err = np.linalg.norm(gram - np.eye(K), "fro")
    if err > 1e-4:
        print(f"  [QR] {name}: ‖VᵀV − I‖_F = {err:.6f} → re-orthonormalized")
        V, _ = np.linalg.qr(V)
    return V


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Return cos(θ) for all K principal angles, sorted descending."""
    M = V1.T @ V2                          # [K, K]
    cos_theta = np.linalg.svd(M, compute_uv=False)
    return np.clip(cos_theta, 0.0, 1.0)   # numerical safety


def analyze_pair(V1: np.ndarray, V2: np.ndarray, label1: str, label2: str) -> dict:
    """Full analysis of one subspace pair."""
    cos_theta = principal_angles(V1, V2)
    K = len(cos_theta)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # Top-K mean cosine similarity
    topk_results = {}
    for k in [1, 3, 5, 10, K]:
        if k > K:
            continue
        mean_cos = float(np.mean(cos_theta[:k]))
        topk_results[f"mean_cos_top{k}"] = round(mean_cos, 4)

    # Legacy full-rank distance (for reference)
    d_G_full = float(np.sqrt(np.sum(theta ** 2)))
    d_G_proj = float(np.linalg.norm(V1 @ V1.T - V2 @ V2.T, "fro") / np.sqrt(2))

    result = {
        "pair": f"{label1} ↔ {label2}",
        **topk_results,
        "cos_theta_all": [round(float(c), 4) for c in cos_theta],
        "theta_deg_top5": [round(float(np.degrees(t)), 1) for t in theta[:5]],
        "d_G_full_rank": round(d_G_full, 4),
        "d_G_projection": round(d_G_proj, 4),
        "overlap_full": round(float(np.mean(cos_theta ** 2)), 4),
    }
    return result


def print_pair_report(r: dict):
    """Pretty-print one pair's results."""
    print(f"\n{'━' * 60}")
    print(f"  {r['pair']}")
    print(f"{'━' * 60}")

    print(f"\n  ┌─ Principal Angle Cosines (sorted desc.) ─┐")
    cos_all = r["cos_theta_all"]
    for i, c in enumerate(cos_all):
        bar = "█" * int(c * 30)
        marker = " ← DOMINANT" if i == 0 else ""
        print(f"  │ θ_{i+1:2d}: cos = {c:.4f}  {bar}{marker}")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  ┌─ Top-K Mean Cosine Similarity ──────────┐")
    for k in [1, 3, 5, 10]:
        key = f"mean_cos_top{k}"
        if key in r:
            val = r[key]
            verdict = "✓ STRONG" if val >= 0.6 else ("~ moderate" if val >= 0.4 else "✗ weak")
            print(f"  │ K={k:2d}: mean cos(θ) = {val:.4f}  {verdict}")
    k_full = f"mean_cos_top{len(cos_all)}"
    if k_full in r:
        print(f"  │ K={len(cos_all):2d}: mean cos(θ) = {r[k_full]:.4f}  (full rank)")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  Top-5 principal angles: {r['theta_deg_top5']}°")
    print(f"  Legacy d_G (full-rank): {r['d_G_full_rank']} "
          f"(of max √K = {np.sqrt(len(cos_all)):.2f})")


def run_all_pairs():
    """Compute principal angles for all three subspace pairs."""
    files = {
        "S_blur": MODELS_DIR / "V_matrix.pt",
        "S_solid": MODELS_DIR / "V_solid.pt",
        "S_text_only": MODELS_DIR / "V_text_only.pt",
    }

    subspaces = {}
    print("=" * 60)
    print("  A-OSP §4.4 — Principal Angle Subspace Analysis")
    print("=" * 60)

    for label, path in files.items():
        if not path.exists():
            print(f"  [SKIP] {label}: {path} not found")
            continue
        V, meta = load_subspace(str(path))
        V = ensure_orthonormal(V, label)
        subspaces[label] = V
        print(f"  {label}: shape={V.shape}, EVR={meta['evr']:.4f}, "
              f"sv[0]={meta['sv_top5'][0]:.1f}" if meta['sv_top5'] else "")

    pairs = [
        ("S_blur", "S_solid"),
        ("S_blur", "S_text_only"),
        ("S_solid", "S_text_only"),
    ]

    all_results = []
    for l1, l2 in pairs:
        if l1 not in subspaces or l2 not in subspaces:
            continue
        r = analyze_pair(subspaces[l1], subspaces[l2], l1, l2)
        print_pair_report(r)
        all_results.append(r)

    # Summary table
    print(f"\n{'=' * 60}")
    print("  §4.4 SUMMARY TABLE (for paper)")
    print(f"{'=' * 60}")
    print(f"  {'Pair':<28s} {'cos₁':>6s} {'top-3':>6s} {'top-5':>6s}")
    print(f"  {'─' * 50}")
    for r in all_results:
        cos1 = r.get("mean_cos_top1", 0)
        cos3 = r.get("mean_cos_top3", 0)
        cos5 = r.get("mean_cos_top5", 0)
        print(f"  {r['pair']:<28s} {cos1:>6.3f} {cos3:>6.3f} {cos5:>6.3f}")

    # Save JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "principal_angles_full_report.json"
    with open(out, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\n  Report → {out}")


def run_single_pair(args):
    """Compute for a single pair specified via CLI."""
    V1, m1 = load_subspace(args.v1)
    V2, m2 = load_subspace(args.v2)
    V1 = ensure_orthonormal(V1, args.label1)
    V2 = ensure_orthonormal(V2, args.label2)

    r = analyze_pair(V1, V2, args.label1, args.label2)
    print_pair_report(r)

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\n  Report → {args.report}")


def main():
    p = argparse.ArgumentParser(description="A-OSP §4.4 Principal Angle Calculator")
    p.add_argument("--all_pairs", action="store_true", help="Run all 3 pairs")
    p.add_argument("--v1", type=str, default=None)
    p.add_argument("--v2", type=str, default=None)
    p.add_argument("--label1", default="S_1")
    p.add_argument("--label2", default="S_2")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    if args.all_pairs:
        run_all_pairs()
    elif args.v1 and args.v2:
        run_single_pair(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
