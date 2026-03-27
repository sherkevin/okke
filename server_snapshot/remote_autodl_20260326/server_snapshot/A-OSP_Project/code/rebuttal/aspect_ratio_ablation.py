"""
Aspect Ratio Ablation — Reviewer Q1 (Dynamic Resolution)
=========================================================
Proves that attention-weighted pooling in Qwen3-VL doesn't absorb spatial priors.
Extracts S_blur for 3 extreme aspect ratios (1:1, 16:9, 9:16) and computes
pairwise Grassmann distances. Goal: d_G < 0.15 across spatial grids.

Model: Qwen3-VL (2B or 8B). Mini-batch: 50 images.

Output:
  models/qwen3vl/V_blur_1x1.pt
  models/qwen3vl/V_blur_16x9.pt
  models/qwen3vl/V_blur_9x16.pt
  logs/rebuttal/aspect_ratio_grassmann.json

Usage:
  python aspect_ratio_ablation.py --model 2b
  python aspect_ratio_ablation.py --model 8b --limit 50
"""

import sys, os, gc, json, argparse, time
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image, ImageFilter

sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
BLUR_DIR = PROJECT / "data" / "blurred_calibration" / "blur"
OUT_MODELS = PROJECT / "models" / "qwen3vl"
OUT_LOGS = PROJECT / "logs" / "rebuttal"
REGISTRY_PATH = PROJECT / "DATA_REGISTRY.md"

K = 20
TARGET_LAYER_OFFSET = -4
MAX_NEW_TOKENS = 20
BLUR_RADIUS = 20

# Target sizes for aspect ratios (H, W)
SIZE_1x1 = (224, 224)
SIZE_16x9 = (216, 384)
SIZE_9x16 = (384, 216)


def resize_to_aspect(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Resize image to target size via center crop + resize (no padding)."""
    w, h = image.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        cropped = image.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        cropped = image.crop((0, top, w, top + new_h))
    return cropped.resize((target_w, target_h), Image.BILINEAR)


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def grassmann_distance(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """Chordal Grassmann distance between orthonormal subspaces."""
    M = V1 @ V2.T
    svals = torch.linalg.svdvals(M)
    svals = torch.clamp(svals, 0.0, 1.0)
    return torch.sqrt((torch.sin(torch.acos(svals)) ** 2).sum()).item()


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


def locate_entity_position(gen_ids, tokenizer):
    DETERMINERS = frozenset({"a", "an", "the"})
    TEMPLATE = frozenset({"image", "picture", "photo", "shows", "depicts"})
    decoded = [tokenizer.decode([tid], skip_special_tokens=True).strip().lower() for tid in gen_ids]
    for i, tok in enumerate(decoded):
        if tok in DETERMINERS and (i + 1) < len(decoded):
            c = decoded[i + 1]
            if len(c) > 2 and c.isalpha() and c not in TEMPLATE:
                return i + 1
    for i, tok in enumerate(decoded):
        if len(tok) > 3 and tok.isalpha() and tok not in TEMPLATE and tok not in DETERMINERS:
            return i
    return max(1, len(decoded) // 3)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["2b", "8b"], default="2b")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-VL-2B-Instruct" if args.model == "2b" else "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading {model_id} ...")

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("ERROR: Qwen3VLForConditionalGeneration not found. Install: pip install transformers>=4.57.0")
        sys.exit(1)

    try:
        import flash_attn
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
        print("[WARN] SDPA fallback")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    if hasattr(model.model, "language_model"):
        decoder_layers = model.model.language_model.layers
    elif hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        decoder_layers = model.model.model.layers
    else:
        decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    target_idx = num_layers + TARGET_LAYER_OFFSET
    print(f"Layers: {num_layers}, target: {target_idx}")

    import glob
    cal_paths = sorted(glob.glob(str(BLUR_DIR / "*.jpg")))[:args.limit]
    if not cal_paths:
        cal_paths = sorted(glob.glob(str(BLUR_DIR / "*.png")))[:args.limit]
    assert cal_paths, f"No images in {BLUR_DIR}"
    print(f"Calibration images: {len(cal_paths)}")

    aspect_configs = [
        ("1x1", SIZE_1x1, "V_blur_1x1.pt"),
        ("16x9", SIZE_16x9, "V_blur_16x9.pt"),
        ("9x16", SIZE_9x16, "V_blur_9x16.pt"),
    ]

    capture = LayerCapture()
    handle = decoder_layers[target_idx].register_forward_hook(capture)

    try:
        from qwen_vl_utils import process_vision_info
        HAS_QVL = True
    except ImportError:
        HAS_QVL = False

    def build_inputs(image):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image concisely:"}]}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        return inputs

    results = {}
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    for aspect_name, (target_h, target_w), out_name in aspect_configs:
        print(f"\n{'='*50}")
        print(f"Aspect: {aspect_name} ({target_h}x{target_w})")
        print(f"{'='*50}")

        all_h = []
        t0 = time.time()
        for idx, img_path in enumerate(cal_paths):
            image = Image.open(img_path).convert("RGB")
            image = resize_to_aspect(image, target_h, target_w)
            image = image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

            inputs = build_inputs(image)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            else:
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            capture.reset()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gen_ids = out[0, input_len:]
            ent_pos = locate_entity_position(gen_ids, processor.tokenizer)
            cap_step = max(0, ent_pos - 1)

            if cap_step < len(capture.step_states):
                all_h.append(capture.step_states[cap_step])
            elif capture.step_states:
                all_h.append(capture.step_states[len(capture.step_states) // 2])

            del inputs, out, gen_ids, image
            flush_vram()
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(cal_paths)}] [{time.time()-t0:.0f}s]")

        H = torch.cat(all_h, dim=0)
        H_mean = H.mean(dim=0, keepdim=True)
        R = H - H_mean
        U, S, Vt = torch.linalg.svd(R, full_matrices=False)
        V_bias = Vt[:K, :]
        total_var = (S ** 2).sum()
        evr = ((S[:K] ** 2).sum() / total_var).item()
        L_prior = torch.sqrt((S[:K] ** 2).sum()).item() / H.shape[0] ** 0.5

        out_path = OUT_MODELS / out_name
        torch.save({
            "V_bias": V_bias,
            "singular_values": S[:K],
            "evr": evr,
            "L_prior": L_prior,
            "K": K,
            "num_samples": H.shape[0],
            "aspect": aspect_name,
            "model_id": model_id,
        }, str(out_path))
        results[aspect_name] = {"path": str(out_path), "evr": evr, "L_prior": L_prior, "V": V_bias}
        print(f"  Saved → {out_path}, EVR={evr:.4f}, N={H.shape[0]}")

    handle.remove()
    del model
    flush_vram()

    # Pairwise Grassmann distances
    pairs = [("1x1", "16x9"), ("1x1", "9x16"), ("16x9", "9x16")]
    dG_results = {}
    for a, b in pairs:
        dG = grassmann_distance(results[a]["V"], results[b]["V"])
        dG_results[f"{a}_vs_{b}"] = round(dG, 4)
        print(f"  d_G({a}, {b}) = {dG:.4f}")

    report = {
        "model_id": model_id,
        "n_images": len(cal_paths),
        "timestamp": datetime.now().isoformat(),
        "evr": {k: v["evr"] for k, v in results.items()},
        "L_prior": {k: v["L_prior"] for k, v in results.items()},
        "grassmann_distances": dG_results,
        "goal_dG": 0.15,
        "passed": all(v < 0.15 for v in dG_results.values()),
    }
    report_path = OUT_LOGS / "aspect_ratio_grassmann.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport → {report_path}")
    print(f"All d_G < 0.15: {report['passed']}")

    # Update registry
    block = f"""
### §13.1 Result — Aspect Ratio Ablation (Qwen3-VL-{args.model.upper()}, {datetime.now().strftime('%Y-%m-%d %H:%M')})

| File | Aspect | EVR | N |
|------|--------|-----|---|"""
    for k, v in results.items():
        block += f"\n| `models/qwen3vl/V_blur_{k}.pt` | {k} | {v['evr']:.4f} | {len(cal_paths)} |"
    block += f"\n\n| Pair | d_G | Goal |\n|------|-----|------|"
    for k, v in dG_results.items():
        block += f"\n| {k} | {v} | < 0.15 |"
    block += f"\n\n**Passed**: {report['passed']}\n"
    with open(REGISTRY_PATH, "a") as f:
        f.write(block)
    print(f"Updated {REGISTRY_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
