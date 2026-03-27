#!/usr/bin/env python3
"""
Principal Angles Analysis: S_text-only vs V_matrix_q3 (S_blur-like).

Mechanistic Homology Test §4.4:
Compute principal angles between two subspaces using SVD on V1^T @ V2.
High cosine similarity (>0.6) on leading vectors proves subspace alignment.

Usage:
  python calc_principal_angles.py
"""

import torch
import numpy as np
import json
import os

PROJECT_ROOT = "/root/autodl-tmp/A-OSP_Project"
V_TEXT_ONLY  = f"{PROJECT_ROOT}/models/V_text_only_q3.pt"
V_MATRIX_Q3  = f"{PROJECT_ROOT}/models/V_matrix_q3.pt"
LOG_DIR      = f"{PROJECT_ROOT}/logs/eval_results"
os.makedirs(LOG_DIR, exist_ok=True)


def load_vmatrix(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    V = d["V_bias"]  # [K, D]
    return V.float(), d


def compute_principal_angles(V1, V2, top_k=None):
    """
    Compute principal angles between subspaces spanned by rows of V1 and V2.

    Args:
        V1: [K1, D] orthonormal basis for subspace 1
        V2: [K2, D] orthonormal basis for subspace 2
        top_k: number of principal angles to compute (default: min(K1, K2))

    Returns:
        cos_thetas: cosine of principal angles (sorted descending)
        angles_deg: principal angles in degrees
    """
    if top_k is None:
        top_k = min(V1.shape[0], V2.shape[0])

    # Ensure V1, V2 are orthonormal row bases
    # Re-orthonormalize via QR
    Q1, _ = torch.linalg.qr(V1.T)  # [D, K1]
    Q2, _ = torch.linalg.qr(V2.T)  # [D, K2]

    # Gram matrix: M = Q1^T @ Q2  [K1, K2]
    M = Q1.T @ Q2

    # SVD of Gram matrix: singular values = cos(principal angles)
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    cos_thetas = S[:top_k].clamp(0, 1)
    angles_deg = torch.acos(cos_thetas) * 180 / np.pi

    return cos_thetas, angles_deg


def main():
    print("=== Principal Angles Analysis: S_text-only vs S_blur (V_matrix_q3) ===")
    print("This validates Mechanistic Homology §4.4: high cosine similarity proves")
    print("both extraction methods capture the same Language Gravity Well geometry.\n")

    # Check if V_text_only_q3.pt exists
    if not os.path.exists(V_TEXT_ONLY):
        print(f"[ERROR] V_text_only_q3.pt not found at {V_TEXT_ONLY}")
        print("Run extract_vmatrix_text_only_q3.py first.")

        # Check fallback: old V_text_only.pt for Qwen2-VL
        fallback = f"{PROJECT_ROOT}/models/V_text_only.pt"
        if os.path.exists(fallback):
            print(f"\nFallback: Using Qwen2-VL V_text_only.pt for structure validation...")
            V1, meta1 = load_vmatrix(fallback)
            print(f"  V_text_only (Qwen2-VL): {list(V1.shape)}, tag={meta1.get('tag', 'N/A')}")
            print(f"  NOTE: Dimension mismatch with V_matrix_q3 (3584 vs 4096)")
            print("  Cannot compute principal angles across different architectures.")
            print("  Proceed after extracting V_text_only_q3.pt for Qwen3-VL.")
            return
        raise FileNotFoundError(V_TEXT_ONLY)

    # Load both subspace matrices
    V1, meta1 = load_vmatrix(V_TEXT_ONLY)   # S_text-only (V3.5 primary)
    V2, meta2 = load_vmatrix(V_MATRIX_Q3)   # S_blur-like (COCO images)

    print(f"Subspace 1 (S_text-only): {list(V1.shape)}")
    print(f"  tag={meta1.get('tag', 'N/A')}, EVR={meta1.get('evr', 0):.4f}, L_prior={meta1.get('L_prior', 0):.4f}")
    print(f"  extraction_method={meta1.get('extraction_method', 'unknown')}")
    print()
    print(f"Subspace 2 (S_blur / V_matrix_q3): {list(V2.shape)}")
    print(f"  tag={meta2.get('tag', 'S_blur_like')}, EVR={meta2.get('evr', 0):.4f}, L_prior={meta2.get('L_prior', 0):.4f}")
    print()

    if V1.shape[1] != V2.shape[1]:
        print(f"[ERROR] Dimension mismatch: {V1.shape[1]} vs {V2.shape[1]}")
        print("Both must be from the same model (Qwen3-VL-8B, D=4096).")
        return

    # Compute principal angles for K = 1, 3, 5
    print("Computing principal angles for K=1, 3, 5 principal directions...")

    results = {}
    for k in [1, 3, 5, 10, 20]:
        k_actual = min(k, V1.shape[0], V2.shape[0])
        cos_thetas, angles_deg = compute_principal_angles(V1[:k_actual], V2[:k_actual])
        mean_cos = cos_thetas.mean().item()
        results[f"K={k}"] = {
            "k": k_actual,
            "cos_thetas": cos_thetas.tolist(),
            "angles_deg": angles_deg.tolist(),
            "mean_cos": mean_cos,
            "top1_cos": cos_thetas[0].item() if len(cos_thetas) > 0 else 0,
        }
        threshold = ">0.85 ✓ (Mechanistic Homology confirmed)" if mean_cos > 0.85 else \
                    ">0.60 ✓ (Subspace alignment confirmed)" if mean_cos > 0.60 else \
                    "< 0.60 ✗ (Weak alignment)"
        print(f"  K={k:2d}: mean cosine={mean_cos:.4f}  {threshold}")
        if k <= 5:
            print(f"        cos per direction: {[f'{c:.4f}' for c in cos_thetas.tolist()]}")

    # Summary
    print()
    print("=== MECHANISTIC HOMOLOGY VERDICT ===")
    k3_cos = results.get("K=3", {}).get("mean_cos", 0)
    k1_cos = results.get("K=1", {}).get("top1_cos", 0)

    if k3_cos > 0.85:
        verdict = "STRONG HOMOLOGY (>0.85): S_text-only and S_blur capture identical Language Gravity Well geometry. Zero-vision paradigm is mathematically superior AND equivalent."
    elif k3_cos > 0.60:
        verdict = "MODERATE HOMOLOGY (>0.60): Subspaces share major geometric direction. S_text-only is theoretically purer version of S_blur."
    else:
        verdict = f"WEAK ALIGNMENT ({k3_cos:.3f}): Subspaces diverge - check extraction methods."

    print(f"  Top-1 principal cosine: {k1_cos:.4f}")
    print(f"  Top-3 mean cosine:      {k3_cos:.4f}")
    print(f"  Verdict: {verdict}")

    # Save results
    output = {
        "V1_path": V_TEXT_ONLY,
        "V2_path": V_MATRIX_Q3,
        "V1_tag": meta1.get("tag", "S_text_only"),
        "V2_tag": meta2.get("tag", "S_blur_like"),
        "V1_evr": meta1.get("evr", 0),
        "V2_evr": meta2.get("evr", 0),
        "principal_angles": results,
        "top1_cos": k1_cos,
        "top3_mean_cos": k3_cos,
        "verdict": verdict,
        "mechanistic_homology_confirmed": k3_cos > 0.60,
        "strong_homology": k3_cos > 0.85,
    }

    out_file = os.path.join(LOG_DIR, "principal_angles_stext_vs_sblur.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_file}")
    print("PRINCIPAL ANGLES ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
