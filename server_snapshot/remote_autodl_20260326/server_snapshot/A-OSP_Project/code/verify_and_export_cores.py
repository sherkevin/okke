"""
Subspace Verification & Low-Rank Core Export
=============================================
1. Mini-batch verification: Load model, run 10 diverse pure-text prompts,
   extract hidden states, run SVD, verify non-zero singular values.
2. Export K=3 and K=5 low-rank cores from all three existing subspaces
   (V_blur, V_solid, V_text_only).
3. Compute pairwise principal angles between cores to prove dominant
   momentum directions align across regimes.
4. Update DATA_REGISTRY.md.

Usage:
  python verify_and_export_cores.py              # full pipeline
  python verify_and_export_cores.py --skip_mini  # skip mini-batch, export only
"""

import sys, os, gc, math, argparse, time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_PATH = PROJECT / "models" / "Qwen2-VL-7B-Instruct"
MODELS_DIR = PROJECT / "models"
REGISTRY_PATH = PROJECT / "DATA_REGISTRY.md"
TARGET_LAYER_OFFSET = -4

MINI_PROMPTS = [
    "A photo of a dog running in a park",
    "Describe the kitchen in detail",
    "A tall building in a modern city",
    "The cat sat on the windowsill watching birds",
    "A red sports car parked on a rainy street",
    "Two children playing with a ball in the garden",
    "An old wooden boat on a calm lake at sunset",
    "A plate of spaghetti with fresh basil and tomatoes",
    "The astronaut floated weightlessly inside the space station",
    "A colorful market with fruits and vegetables",
]

DIVERSE_PREFIXES = [
    "The image shows a", "The image shows an", "The image depicts a",
    "In the image there is a", "The photograph captures a",
    "The picture features a", "Visible in the image is a",
    "The main subject is a", "I can see a", "The scene contains a",
]


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_principal_angles(V1, V2):
    """
    Compute principal angles between two orthonormal subspaces.
    V1: [k1, D], V2: [k2, D] — rows are orthonormal basis vectors.
    Returns angles in radians, sorted ascending.
    """
    M = V1 @ V2.T
    svals = torch.linalg.svdvals(M)
    svals = torch.clamp(svals, 0.0, 1.0)
    angles = torch.acos(svals)
    return angles


def grassmann_distance(V1, V2):
    """Chordal Grassmann distance = sqrt(sum(sin^2(theta_i)))."""
    angles = compute_principal_angles(V1, V2)
    return torch.sqrt((torch.sin(angles) ** 2).sum()).item()


def export_low_rank_cores(ckpt_path, name, ks=(3, 5)):
    """Export K=3 and K=5 sub-basis from a full K=20 checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    V_full = ckpt["V_bias"]
    S_full = ckpt.get("singular_values", None)
    evr_full = ckpt.get("evr", None)
    results = {}

    for k in ks:
        V_k = V_full[:k, :]
        out_path = MODELS_DIR / f"{name}_k{k}.pt"

        payload = {
            "V_bias": V_k,
            "K": k,
            "parent": str(ckpt_path),
            "parent_K": V_full.shape[0],
        }

        if S_full is not None and len(S_full) >= k:
            S_k = S_full[:k]
            total_var_full = (S_full ** 2).sum()
            evr_k = ((S_k ** 2).sum() / total_var_full).item()
            payload["singular_values"] = S_k
            payload["evr"] = evr_k
        else:
            evr_k = None

        torch.save(payload, str(out_path))
        results[k] = {"path": out_path, "evr": evr_k, "V": V_k}
        print(f"  {out_path.name}: [{k}, {V_k.shape[1]}], "
              f"EVR={evr_k:.4f}" if evr_k else f"  {out_path.name}: [{k}, {V_k.shape[1]}]")

    return results


class LayerCapture:
    def __init__(self):
        self.step_states = []
        self._prefill_done = False

    def reset(self):
        self.step_states.clear()
        self._prefill_done = False

    def __call__(self, module, inp, out):
        h = out[0]
        if not self._prefill_done:
            self._prefill_done = True
            return
        self.step_states.append(h[:, -1, :].detach().float().cpu())


@torch.no_grad()
def run_mini_batch_verification():
    """Run 10 diverse pure-text prompts through Qwen2-VL (no images)
    and verify SVD produces non-zero singular values."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig

    print("=" * 60)
    print("MINI-BATCH VERIFICATION: 10 diverse text prompts")
    print("=" * 60)

    print("Loading model ...")
    try:
        import flash_attn; attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model.eval()
    tokenizer = processor.tokenizer

    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    target_idx = num_layers + TARGET_LAYER_OFFSET
    print(f"Layers: {num_layers}, target: {target_idx}")

    capture = LayerCapture()
    handle = decoder_layers[target_idx].register_forward_hook(capture)

    gen_config = GenerationConfig(
        max_new_tokens=20, do_sample=True, temperature=1.2, top_p=0.9, top_k=50)

    all_h = []
    print(f"\nGenerating with {len(MINI_PROMPTS)} diverse prompts (pure text, no images):\n")

    for idx, prompt in enumerate(MINI_PROMPTS):
        prefix = DIVERSE_PREFIXES[idx % len(DIVERSE_PREFIXES)]
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text += prefix
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        capture.reset()
        output_ids = model.generate(**inputs, generation_config=gen_config)
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if capture.step_states:
            mid = len(capture.step_states) // 2
            all_h.append(capture.step_states[mid])

        print(f"  [{idx+1:>2}/{len(MINI_PROMPTS)}] steps={len(capture.step_states):>2} | "
              f"{prefix}{gen_text[:50]}...")

        del inputs, output_ids, gen_ids
        flush_vram()

    handle.remove()
    del model
    flush_vram()

    print(f"\nCollected {len(all_h)} hidden state vectors")
    assert len(all_h) >= 5, f"Too few hidden states collected: {len(all_h)}"

    H = torch.cat(all_h, dim=0)
    print(f"H shape: {H.shape}")
    H_mean = H.mean(dim=0, keepdim=True)
    R = H - H_mean
    U, S, Vt = torch.linalg.svd(R, full_matrices=False)

    print(f"\n{'='*40}")
    print("SVD VERIFICATION RESULTS")
    print(f"{'='*40}")
    print(f"  Singular values (all {len(S)}):")
    for i, s in enumerate(S):
        marker = " ✓" if s.item() > 1e-6 else " ✗ ZERO!"
        print(f"    σ_{i+1} = {s.item():.4f}{marker}")

    n_nonzero = (S > 1e-6).sum().item()
    total_var = (S ** 2).sum()
    top3_evr = ((S[:3] ** 2).sum() / total_var).item() if total_var > 0 else 0
    top5_evr = ((S[:5] ** 2).sum() / total_var).item() if total_var > 0 else 0
    print(f"\n  Non-zero singular values: {n_nonzero}/{len(S)}")
    print(f"  Top-3 EVR: {top3_evr:.4f}")
    print(f"  Top-5 EVR: {top5_evr:.4f}")
    print(f"  Total variance: {total_var.item():.2f}")

    all_nonzero = n_nonzero == len(S)
    has_nan = torch.isnan(S).any().item()
    print(f"\n  ✓ All singular values non-zero: {all_nonzero}")
    print(f"  ✓ No NaN in singular values: {not has_nan}")
    print(f"  ✓ Sufficient variance for SVD: {total_var.item() > 1.0}")

    if all_nonzero and not has_nan and total_var.item() > 1.0:
        print("\n  ★ MINI-BATCH VERIFICATION PASSED ★")
    else:
        print("\n  ✗ MINI-BATCH VERIFICATION FAILED")
        sys.exit(1)

    return {
        "n_prompts": len(MINI_PROMPTS),
        "n_collected": len(all_h),
        "n_nonzero_sv": n_nonzero,
        "top3_evr": top3_evr,
        "top5_evr": top5_evr,
        "total_variance": total_var.item(),
    }


def run_core_export_and_analysis():
    """Export K=3 and K=5 cores, compute pairwise principal angles."""
    print(f"\n{'='*60}")
    print("LOW-RANK CORE EXPORT & PRINCIPAL ANGLE ANALYSIS")
    print(f"{'='*60}")

    sources = {
        "V_blur":      MODELS_DIR / "V_matrix.pt",
        "V_solid":     MODELS_DIR / "V_solid.pt",
        "V_text_only": MODELS_DIR / "V_text_only.pt",
    }

    all_cores = {}
    for name, path in sources.items():
        print(f"\n--- {name} ({path.name}) ---")
        cores = export_low_rank_cores(path, name, ks=(3, 5))
        all_cores[name] = cores

    ckpts = {}
    for name, path in sources.items():
        c = torch.load(str(path), map_location="cpu", weights_only=True)
        ckpts[name] = c["V_bias"]

    print(f"\n{'='*60}")
    print("PRINCIPAL ANGLE ANALYSIS")
    print(f"{'='*60}")

    pairs = [
        ("V_blur", "V_solid"),
        ("V_blur", "V_text_only"),
        ("V_solid", "V_text_only"),
    ]

    results_table = []

    for k_val in [3, 5, 20]:
        print(f"\n--- K = {k_val} ---")
        for n1, n2 in pairs:
            if k_val == 20:
                V1, V2 = ckpts[n1][:k_val], ckpts[n2][:k_val]
            else:
                V1 = all_cores[n1][k_val]["V"]
                V2 = all_cores[n2][k_val]["V"]

            angles = compute_principal_angles(V1, V2)
            angles_deg = angles * 180.0 / math.pi
            d_grass = grassmann_distance(V1, V2)
            mean_cos = torch.cos(angles).mean().item()

            row = {
                "K": k_val, "pair": f"{n1} <-> {n2}",
                "d_G": d_grass, "mean_cos": mean_cos,
                "angles_deg": [round(a.item(), 2) for a in angles_deg],
            }
            results_table.append(row)

            print(f"  {n1} <-> {n2}:")
            print(f"    d_G = {d_grass:.4f}, mean_cos = {mean_cos:.4f}")
            print(f"    angles (deg): {', '.join(f'{a:.1f}' for a in angles_deg[:min(k_val,5)])}"
                  + ("..." if k_val > 5 else ""))

    torch.save(results_table, str(MODELS_DIR / "principal_angles_analysis.pt"))
    print(f"\nSaved analysis → models/principal_angles_analysis.pt")

    return all_cores, results_table


def update_registry(mini_results, all_cores, angle_results):
    """Append structured entry to DATA_REGISTRY.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    block = f"""
## §8b. Subspace Verification & Low-Rank Cores — §4.4 Figure 4 (Agent 1 — {timestamp})

### Mini-Batch Verification (10 diverse pure-text prompts, Layer 24)

| Metric | Value |
|--------|-------|
| Prompts | {mini_results['n_prompts']} |
| Hidden states collected | {mini_results['n_collected']} |
| Non-zero singular values | {mini_results['n_nonzero_sv']}/{mini_results['n_collected']} |
| Top-3 EVR | {mini_results['top3_evr']:.4f} |
| Top-5 EVR | {mini_results['top5_evr']:.4f} |
| Total variance | {mini_results['total_variance']:.2f} |
| **Status** | **PASSED** |

### Low-Rank Core Files

| File | Shape | EVR | Parent | Purpose |
|------|-------|-----|--------|---------|"""

    for name in ["V_blur", "V_solid", "V_text_only"]:
        for k in [3, 5]:
            core = all_cores[name][k]
            evr_str = f"{core['evr']:.4f}" if core['evr'] else "N/A"
            block += f"\n| `models/{name}_k{k}.pt` | [{k}, 3584] | {evr_str} | `{name}.pt` | §4.4 dominant direction alignment |"

    block += f"\n| `models/principal_angles_analysis.pt` | analysis dict | — | All 3 subspaces | Figure 4: Principal Angles |"

    block += """

### Principal Angle Alignment (Dominant Directions)

| K | Pair | d_G | mean_cos | θ_1 (deg) | θ_2 (deg) | θ_3 (deg) |
|---|------|-----|----------|-----------|-----------|-----------|"""

    for r in angle_results:
        if r["K"] in (3, 5):
            angles = r["angles_deg"]
            a_strs = [f"{a:.1f}" for a in angles[:3]]
            while len(a_strs) < 3:
                a_strs.append("—")
            block += f"\n| {r['K']} | {r['pair']} | {r['d_G']:.4f} | {r['mean_cos']:.4f} | {a_strs[0]} | {a_strs[1]} | {a_strs[2]} |"

    block += """

> **§4.4 Key Evidence**:
> - Mini-batch verification confirms pure-text hidden states have full-rank SVD (no NaN, no zero σ).
> - K=3 cores capture the dominant inertia directions; if cos(θ_1) ≈ 1 across regimes,
>   the PRIMARY momentum direction is regime-invariant (= model-intrinsic language prior).
> - Grassmann distance at K=3 vs K=20 reveals whether alignment is concentrated in top
>   principal components (low-rank alignment) or distributed (full-subspace alignment).
"""

    with open(REGISTRY_PATH, "a") as f:
        f.write("\n" + block + "\n")
    print(f"\nUpdated {REGISTRY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mini", action="store_true",
                        help="Skip mini-batch verification, export cores only")
    args = parser.parse_args()

    if args.skip_mini:
        mini_results = {
            "n_prompts": "skipped", "n_collected": "skipped",
            "n_nonzero_sv": "skipped", "top3_evr": 0,
            "top5_evr": 0, "total_variance": 0,
        }
    else:
        mini_results = run_mini_batch_verification()

    all_cores, angle_results = run_core_export_and_analysis()

    if not args.skip_mini:
        update_registry(mini_results, all_cores, angle_results)

    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
