#!/usr/bin/env python3
"""
Bias Subspace Extractor for Qwen3-VL
=====================================
Extracts V_matrix_q3.pt from blurred COCO val2014 images using Qwen3-VL-8B.

Key differences from Qwen2-VL extraction:
  - hidden_size = 4096 (vs 3584)
  - num_layers  = 36   (vs 28)
  - target layer: 32   (倒数第4层 = 36 - 4 = 32)
  - Model class: Qwen3VLForConditionalGeneration

Usage:
    python extract_vmatrix_q3.py --n_images 200 --layer 32 --K 20 --output models/V_matrix_q3.pt
    python extract_vmatrix_q3.py --n_images 10  --layer 32 --K 20 --output models/V_matrix_q3_mini.pt
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from eval.eval_utils import load_qwen2vl, log_gpu_memory

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
COCO_DIR = PROJECT / "data" / "coco_val2014"
OUT_DIR = PROJECT / "models"

BLUR_PROMPT = "Describe everything you can see in this image."


def get_coco_images(n: int) -> list:
    imgs = sorted(COCO_DIR.glob("*.jpg"))[:n]
    if len(imgs) < n:
        imgs += sorted(COCO_DIR.glob("*.png"))
    imgs = imgs[:n]
    print(f"[Data] Found {len(imgs)} COCO images in {COCO_DIR}")
    return imgs


def blur_image(path: str, radius: int = 20) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def extract_hidden_states(model, processor, img_paths: list, layer_idx: int,
                           device: str) -> torch.Tensor:
    """Returns [N, hidden_size] float32 tensor of mean pooled layer activations."""

    hidden_states_list = []
    hook_output = [None]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Mean-pool over sequence length → [hidden_size]
        hook_output[0] = h.detach().float().mean(dim=1).squeeze(0).cpu()

    # Attach hook
    if hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[layer_idx]
    else:
        target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    for i, img_path in enumerate(img_paths):
        try:
            blurred = blur_image(str(img_path))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": blurred},
                    {"type": "text", "text": BLUR_PROMPT},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                _ = model(
                    **inputs,
                    output_hidden_states=False,
                )

            if hook_output[0] is not None:
                hidden_states_list.append(hook_output[0])
                hook_output[0] = None

        except Exception as e:
            print(f"  [SKIP] {img_path.name}: {e}")
        finally:
            try:
                del inputs
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1}/{len(img_paths)}] extracted {len(hidden_states_list)} states")
            log_gpu_memory(f"layer{layer_idx}_step{i+1}")

    handle.remove()

    if not hidden_states_list:
        raise RuntimeError("No hidden states collected!")

    return torch.stack(hidden_states_list, dim=0)  # [N, D]


def extract_svd(hidden_states: torch.Tensor, K: int) -> dict:
    """Compute SVD on centered hidden states, return top-K right singular vectors."""
    H = hidden_states  # [N, D]
    N, D = H.shape
    print(f"[SVD] Input: {N} samples × {D} dims")

    # Center
    mean = H.mean(dim=0)
    H_centered = H - mean

    # Covariance: [D, D] = H^T H / N
    # For large D, use full SVD on H directly
    print("[SVD] Computing SVD (may take 30-60s for D=4096, N=200)...")
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    print(f"[SVD] Done in {time.time()-t0:.1f}s | top-3 σ: {S[:3].tolist()}")

    total_var = (S ** 2).sum().item()
    evr_k = (S[:K] ** 2).sum().item() / total_var
    evr_full = (S ** 2).sum().item() / total_var

    V_bias = Vh[:K]  # [K, D]  top-K right singular vectors

    # Compute L_prior: mean L2 norm of projection onto bias subspace
    # Must match compute_projection_energy() which returns sqrt(||proj||²)
    proj = (H_centered @ V_bias.T)  # [N, K]
    L_prior = torch.sqrt((proj ** 2).sum(dim=1)).mean().item()

    print(f"[SVD] K={K} | EVR={evr_k:.4f} | L_prior={L_prior:.2f} | σ range: [{S[0]:.1f}, {S[-1]:.1f}]")

    return {
        "V_bias": V_bias,           # [K, D]
        "singular_values": S[:K],
        "mean_hidden": mean,
        "K": K,
        "N": N,
        "D": D,
        "evr": evr_k,
        "L_prior": L_prior,
        "top3_sigma": S[:3].tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(PROJECT / "models" / "Qwen3-VL-8B-Instruct"))
    p.add_argument("--n_images", type=int, default=50,
                   help="Number of COCO images to use (start with 50 for mini-test)")
    p.add_argument("--layer", type=int, default=32,
                   help="Target intervention layer (倒数第4层 = 36-4=32 for 36-layer model)")
    p.add_argument("--K", type=int, default=20, help="Number of bias directions to keep")
    p.add_argument("--output", default=str(OUT_DIR / "V_matrix_q3.pt"),
                   help="Output path for V_matrix checkpoint")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Qwen3-VL Bias Subspace Extraction")
    print(f"  model : {args.model_path}")
    print(f"  layer : {args.layer}")
    print(f"  K     : {args.K}")
    print(f"  images: {args.n_images}")
    print(f"  output: {args.output}")
    print(f"{'='*60}\n")

    img_paths = get_coco_images(args.n_images)
    if len(img_paths) == 0:
        print(f"[ERROR] No images found in {COCO_DIR}")
        sys.exit(1)

    model, processor = load_qwen2vl(args.model_path)
    device = str(next(model.parameters()).device)

    # Verify layer exists
    if hasattr(model.model, 'language_model'):
        n_layers = len(model.model.language_model.layers)
    else:
        n_layers = len(model.model.layers)
    print(f"[Model] n_layers={n_layers} | hidden_size={model.config.text_config.hidden_size}")
    if args.layer >= n_layers:
        print(f"[WARN] layer={args.layer} >= n_layers={n_layers}, clamping to {n_layers-4}")
        args.layer = n_layers - 4

    t0 = time.time()
    hidden_states = extract_hidden_states(model, processor, img_paths, args.layer, device)
    print(f"[Extraction] Done in {time.time()-t0:.1f}s | shape={list(hidden_states.shape)}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after_model_del")

    result = extract_svd(hidden_states, args.K)
    result["layer_idx"] = args.layer
    result["model"] = "Qwen3-VL-8B-Instruct"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"\n[SAVED] → {args.output}")
    print(f"  V_bias shape : {list(result['V_bias'].shape)}")
    print(f"  EVR          : {result['evr']:.4f}")
    print(f"  L_prior      : {result['L_prior']:.2f}")
    print(f"  layer_idx    : {result['layer_idx']}")


if __name__ == "__main__":
    main()
