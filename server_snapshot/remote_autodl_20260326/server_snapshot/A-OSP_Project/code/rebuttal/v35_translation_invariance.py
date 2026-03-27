"""
V3.5 Sprint 1 — Task 1.2: Spatial Translation Invariance
=========================================================
Prove Attention-weighted Pooling doesn't destroy spatial logic.
10 clear MSCOCO images. Pad main object to Top-Left → S_A, Bottom-Right → S_B.
Principal angles (cosine similarity) of top 3 directions must be ≥ 0.85.

PADDING MASK AUDIT (Reviewer 1 Response):
  During the forward pass, processor() returns an attention_mask tensor where
  padding positions have value 0 and real token positions have value 1.
  The hook captures the full hidden state h [1, seq_len, D] and computes a
  MASKED mean pool:
      mask = attention_mask.float().T  → [seq_len, 1]  (captured from input)
      h_pooled = (h * mask).sum(dim=1) / mask.sum()
  This guarantees STRICT 0 WEIGHT for all padding tokens in the pooled vector.
  The captured attention_mask is validated: all positions must be 0 or 1, and
  at least 1 position must be 1 (no all-padding sequences allowed).

  For Qwen3-VL dynamic resolution: visual tokens appear as contiguous real tokens
  (value=1) in attention_mask — no special visual-token padding exists.
  Any gray-canvas padding we ADD to the IMAGE changes visual content, not the
  attention_mask; attention_mask only marks tokenizer-level sequence padding.

Output: logs/rebuttal/translation_invariance.json
"""

import sys, os, gc, json, argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_LOCAL = PROJECT / "models" / "Qwen3-VL-8B-Instruct"
COCO_DIR = PROJECT / "data" / "coco_val2014"
OUT_LOG = PROJECT / "logs" / "rebuttal" / "translation_invariance.json"
TARGET_LAYER_OFFSET = -4
K = 20
MAX_NEW_TOKENS = 32
N_IMAGES = 10
PROMPT = "Describe the image concisely."


def get_coco_image_paths(n=10):
    """Get n clear MSCOCO image paths from coco_val2014."""
    if not COCO_DIR.exists():
        raise FileNotFoundError(f"COCO dir not found: {COCO_DIR}")
    files = sorted(COCO_DIR.glob("COCO_val2014_*.jpg"))[:n]
    return [str(f) for f in files]


def pad_image_top_left(img: Image.Image, pad_ratio=0.5):
    """Pad so original content stays in top-left. Add padding to right and bottom."""
    w, h = img.size
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    new_w, new_h = w + pad_w, h + pad_h
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(img, (0, 0))
    return canvas


def pad_image_bottom_right(img: Image.Image, pad_ratio=0.5):
    """Pad so original content moves to bottom-right. Add padding to left and top."""
    w, h = img.size
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    new_w, new_h = w + pad_w, h + pad_h
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(img, (pad_w, pad_h))
    return canvas


def compute_principal_angles(V1: torch.Tensor, V2: torch.Tensor, top_k=3):
    """Principal angles via SVD of Gram matrix. Returns cos(angles)."""
    Q1, _ = torch.linalg.qr(V1.T)
    Q2, _ = torch.linalg.qr(V2.T)
    M = Q1.T @ Q2
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    return S[:top_k].clamp(0, 1)


@torch.no_grad()
def extract_subspace_from_images(model, processor, image_paths, pad_fn, decoder_layers, target_idx, device):
    """
    Forward-pass on padded images, collect MASKED-MEAN-POOLED hidden states → SVD.

    Padding Mask Protocol:
    - attention_mask from processor: 1=real token, 0=padding token.
    - We pass attention_mask into the hook closure and compute:
        h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    - Strict 0 weight for all padding positions. Validated per sample.
    """
    hook_out = [None]
    captured_mask = [None]   # will hold attention_mask for current sample

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out  # [1, seq_len, D]
        mask = captured_mask[0]                         # [1, seq_len]

        if mask is not None:
            # PADDING AUDIT: validate mask is binary and has real tokens
            assert mask.min().item() >= 0 and mask.max().item() <= 1, \
                f"attention_mask not binary: min={mask.min()}, max={mask.max()}"
            n_real = mask.sum().item()
            assert n_real > 0, "attention_mask is all-zero (no real tokens)"

            # Masked mean pool: strictly zero weight for padding positions
            m = mask.float().unsqueeze(-1).to(h.device)   # [1, seq_len, 1]
            h_pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)  # [1, D]
        else:
            # Fallback (should not happen): unmasked mean
            h_pooled = h.mean(dim=1)

        hook_out[0] = h_pooled.squeeze(0).detach().float().cpu()

    handle = decoder_layers[target_idx].register_forward_hook(hook_fn)

    all_h = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_padded = pad_fn(img)

        msgs = [{"role": "user", "content": [{"type": "image", "image": img_padded}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(msgs)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        else:
            inputs = processor(text=text, images=[img_padded], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Capture attention_mask for this sample before forward pass
        captured_mask[0] = inputs.get("attention_mask", None)
        if captured_mask[0] is not None:
            n_pad = (captured_mask[0] == 0).sum().item()
            n_real = (captured_mask[0] == 1).sum().item()
            print(f"    seq_len={captured_mask[0].shape[1]}, real={n_real}, padding={n_pad}")

        hook_out[0] = None
        _ = model(**inputs, output_hidden_states=False)

        if hook_out[0] is not None:
            all_h.append(hook_out[0])

        gc.collect()
        torch.cuda.empty_cache()

    handle.remove()
    if len(all_h) < 5:
        raise RuntimeError(f"Too few hidden states: {len(all_h)}")

    H = torch.stack(all_h, dim=0)
    R = H - H.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(R, full_matrices=False)
    return Vt[:K, :].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=N_IMAGES)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    paths = get_coco_image_paths(args.n_images)
    print(f"Using {len(paths)} COCO images")

    try:
        import flash_attn
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"

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
    num_layers = len(layers)
    target_idx = args.layer if args.layer is not None else num_layers + TARGET_LAYER_OFFSET
    print(f"Target layer: {target_idx} (of {num_layers})")

    print("Extracting S_A (top-left padding)...")
    V_A = extract_subspace_from_images(model, processor, paths, pad_image_top_left, layers, target_idx, model.device)
    gc.collect()
    torch.cuda.empty_cache()

    print("Extracting S_B (bottom-right padding)...")
    V_B = extract_subspace_from_images(model, processor, paths, pad_image_bottom_right, layers, target_idx, model.device)

    cos_top3 = compute_principal_angles(V_A, V_B, top_k=3)
    cos_list = cos_top3.tolist()
    mean_cos = float(cos_top3.mean())
    passed = mean_cos >= 0.85

    result = {
        "task": "V3.5 Task 1.2 — Spatial Translation Invariance",
        "n_images": len(paths),
        "layer_idx": target_idx,
        "top3_cos_thetas": cos_list,
        "mean_cos_top3": mean_cos,
        "threshold": 0.85,
        "passed": passed,
        "verdict": "PASSED" if passed else "FAILED",
    }

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTop-3 principal cosines: {cos_list}")
    print(f"Mean: {mean_cos:.4f} (need ≥ 0.85)")
    print(f"Saved → {OUT_LOG}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
