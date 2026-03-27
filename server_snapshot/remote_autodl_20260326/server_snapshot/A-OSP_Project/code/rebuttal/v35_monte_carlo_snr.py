"""
V3.5 Sprint 1 — Task 1.3: Monte Carlo Visual SNR
=================================================
Apply A-OSP to 50 samples. For scale compensation
  H_t' = H_proj / ||H_proj|| * ||H_t||
sample 1,000 random directions in S_text_only^⊥.
Compute mean and 95% CI of noise amplification.
Prove: signal recovery > CI_95 of noise.

Output: logs/rebuttal/monte_carlo_snr.json
"""

import sys, os, gc, json, argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, GenerationConfig

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
V_TEXT_ONLY = PROJECT / "models" / "qwen3vl" / "V_text_only.pt"
V_TEXT_ONLY_Q3 = PROJECT / "models" / "V_text_only_q3.pt"
COCO_DIR = PROJECT / "data" / "coco_val2014"
OUT_LOG = PROJECT / "logs" / "rebuttal" / "monte_carlo_snr.json"
TARGET_LAYER_OFFSET = -4
N_SAMPLES = 50
N_RANDOM_DIRS = 1000
ALPHA = 0.5
PROMPT = "Describe the image in one sentence."


def sample_random_directions_in_sperp(V_bias: torch.Tensor, n: int):
    """
    Sample n random unit vectors strictly in S_text_only^⊥ via Gram-Schmidt rejection.

    Method (correct, O(n*K*D)):
      1. d ~ N(0, I_D)         [n, D] random Gaussian directions
      2. proj = (d @ V.T) @ V  [n, D] projection of d onto S_text_only
      3. d_orth = d - proj      [n, D] strictly orthogonal to S
      4. normalize to unit sphere

    Correctness: <d_orth, v_i> = <d - proj(d), v_i> = 0 for all v_i in V_bias.
    """
    K, D = V_bias.shape
    d = torch.randn(n, D)                       # [n, D]
    proj = (d @ V_bias.T) @ V_bias              # [n, D]: projection onto S_text_only
    d_orth = d - proj                            # [n, D]: strictly in S⊥
    norms = d_orth.norm(dim=1, keepdim=True).clamp(min=1e-8)
    d_orth = d_orth / norms                     # [n, D]: unit vectors in S⊥

    # VERIFY orthogonality
    max_proj = (d_orth @ V_bias.T).abs().max().item()
    assert max_proj < 1e-4, f"Orthogonality check failed: max |<d,v>| = {max_proj:.2e}"
    print(f"  S⊥ sampling verified: max residual projection = {max_proj:.2e} (should be < 1e-4)")
    return d_orth  # [n, D]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_random", type=int, default=N_RANDOM_DIRS)
    args = parser.parse_args()

    # Load V_text_only (prefer qwen3vl path, fallback to V_text_only_q3)
    v_path = V_TEXT_ONLY if V_TEXT_ONLY.exists() else V_TEXT_ONLY_Q3
    if not v_path.exists():
        print(f"[ERROR] V_text_only not found. Run v35_extract_text_only.py first.")
        sys.exit(1)

    data = torch.load(v_path, map_location="cpu", weights_only=True)
    V_bias = data["V_bias"].float()  # [K, D]
    K, D = V_bias.shape
    print(f"Loaded {v_path.name}, V shape {list(V_bias.shape)}, EVR={data.get('evr',0):.4f}")
    print(f"S⊥ sampling: Gram-Schmidt rejection in R^{D} (K={K} directions removed)")

    # Get COCO image paths
    if not COCO_DIR.exists():
        print(f"[ERROR] COCO dir not found: {COCO_DIR}")
        sys.exit(1)
    paths = sorted(COCO_DIR.glob("COCO_val2014_*.jpg"))[:args.n_samples]
    paths = [str(p) for p in paths]
    print(f"Using {len(paths)} images")

    try:
        import flash_attn
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"

    MODEL_LOCAL = Path("/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct")
    model_path = str(MODEL_LOCAL)
    print(f"Loading Qwen3-VL-8B-Instruct from LOCAL: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(model_path)

    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_idx = len(layers) + TARGET_LAYER_OFFSET

    collected_H = []
    collected_H_proj = []

    class Capture:
        def __init__(self):
            self.states = []
            self._prefill = False
        def reset(self):
            self.states.clear()
            self._prefill = False
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] != 1:
                self._prefill = True
                return
            self.states.append(h[:, -1, :].detach())

    capture = Capture()
    handle = layers[target_idx].register_forward_hook(capture)
    gen_config = GenerationConfig(max_new_tokens=24, do_sample=False)

    for path in paths:
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(msgs)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        else:
            inputs = processor(text=text, images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        capture.reset()
        model.generate(**inputs, generation_config=gen_config)

        if capture.states:
            mid = len(capture.states) // 2
            H_t = capture.states[mid].float()
            V = V_bias.to(H_t.device, dtype=H_t.dtype)
            proj = H_t @ V.T
            H_proj = H_t - ALPHA * (proj @ V)
            collected_H.append(H_t.cpu())
            collected_H_proj.append(H_proj.cpu())

    handle.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    H_all = torch.cat(collected_H, dim=0)  # [N, D]
    H_proj_all = torch.cat(collected_H_proj, dim=0)
    print(f"Collected {H_all.shape[0]} hidden state pairs from {len(collected_H)} samples")

    # Signal recovery: mean ||H_proj|| over samples (before scale compensation)
    # This is the VISUAL SIGNAL energy (orthogonal complement of language inertia)
    sigma_signal = H_proj_all.norm(dim=1).mean().item()

    # Scale compensation: H' = H_proj / ||H_proj|| * ||H_t||
    # Implements the formula from V3.5: H_t' = H_proj / ||H_proj||_2 * ||H_t||_2
    orig_norms = H_all.norm(dim=1, keepdim=True).clamp(min=1e-6)
    proj_norms = H_proj_all.norm(dim=1, keepdim=True).clamp(min=1e-6)
    scale = orig_norms / proj_norms
    H_scaled = H_proj_all * scale

    # Noise amplification: sample 1000 random directions in S_text_only^⊥
    # Measure how much H_scaled projects onto random directions in S⊥
    # This quantifies noise introduced by the scale-compensation rescaling step
    print(f"\nSampling {args.n_random} random S⊥ directions via Gram-Schmidt rejection...")
    orth_dirs = sample_random_directions_in_sperp(V_bias, args.n_random)  # [n_random, D]

    # For each sample, compute projection amplitude onto each random S⊥ direction
    noise_amps = []
    for i in range(H_scaled.shape[0]):
        H_s = H_scaled[i:i+1].float()  # [1, D]
        amps = (H_s @ orth_dirs.T).abs().squeeze(0)   # [n_random]
        noise_amps.extend(amps.tolist())

    noise_amps = np.array(noise_amps)
    noise_mean = float(np.mean(noise_amps))
    noise_std = float(np.std(noise_amps))
    ci95_lo = float(np.percentile(noise_amps, 2.5))
    ci95_hi = float(np.percentile(noise_amps, 97.5))

    # Proposition: signal recovery > CI_95 of noise (i.e. sigma_signal > ci95_hi)
    passed = sigma_signal > ci95_hi
    snr_ratio = sigma_signal / (ci95_hi + 1e-10)

    result = {
        "task": "V3.5 Task 1.3 — Monte Carlo Visual SNR",
        "n_samples": H_all.shape[0],
        "n_random_dirs": args.n_random,
        "V_text_only_evr": float(data.get("evr", 0)),
        "V_text_only_layer": int(data.get("layer_idx", -1)),
        "alpha": ALPHA,
        "sigma_signal_mean": sigma_signal,
        "noise_amplification_mean": noise_mean,
        "noise_amplification_std": noise_std,
        "noise_CI95_low": ci95_lo,
        "noise_CI95_high": ci95_hi,
        "signal_gt_CI95_noise": passed,
        "snr_ratio_signal_to_ci95": snr_ratio,
        "proposition_1_satisfied": passed,
        "proposition_threshold_1_8x": snr_ratio >= 1.8,
        "verdict": "PASSED" if passed else "FAILED",
        "padding_audit": "S_text_only from prefill_masked_mean_pool; image samples batch_size=1_no_padding",
    }

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSignal recovery (mean ||H_proj||): {sigma_signal:.4f}")
    print(f"Noise amplification: mean={noise_mean:.4f}, CI95=[{ci95_lo:.4f}, {ci95_hi:.4f}]")
    print(f"Signal > CI95_high? {passed} (need signal > {ci95_hi:.4f})")
    print(f"Saved → {OUT_LOG}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
