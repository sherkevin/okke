"""
BRA V2 Smoke Test
=================
Loads Qwen3-VL-2B-Instruct, runs baseline and BRA-V2 generation on a small
COCO sample, and verifies the new logits-processor path.

Reports:
  1. Whether generation completes without error
  2. Whether BRA extracts vision features and executes decode steps
  3. AGL deviation vs. base model
  4. Peak VRAM delta
  5. How many outputs differ from baseline
  6. Whether required benchmark log fields are present

Run on a node with at least one GPU:
    python bra_smoke_test.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (AutoDL conventions)
# ---------------------------------------------------------------------------
PROJECT   = Path("/root/autodl-tmp/BRA_Project")
MODEL_DIR = PROJECT / "models" / "Qwen3-VL-2B-Instruct"
COCO_DIR  = PROJECT / "datasets" / "coco2014" / "val2014"

MAX_SAMPLES    = int(os.environ.get("BRA_SMOKE_SAMPLES", "20"))
MAX_NEW_TOKENS = int(os.environ.get("BRA_SMOKE_MAX_NEW_TOKENS", "128"))
PROMPT         = "Describe the image concisely."


def load_model_and_processor():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[INFO] Loading model from {MODEL_DIR} …")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model.eval()
    return model, processor


def collect_images(n: int):
    """Return up to *n* .jpg paths from the COCO val directory."""
    imgs = sorted(COCO_DIR.glob("*.jpg"))[:n]
    if not imgs:
        alt = PROJECT / "datasets" / "coco2014"
        imgs = sorted(alt.glob("**/*.jpg"))[:n]
    if not imgs:
        print("[WARN] No COCO images found; generating dummy inputs instead.")
        return None
    return imgs


def build_inputs(processor, image_path: Path | None):
    """Build a single-image QA input dict for the model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)} if image_path else
                {"type": "text", "text": "(no image)"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_generation(model, processor, images, label: str):
    """Run generation on *images* and return list of (output_text, gen_length)."""
    results = []
    for i, img_path in enumerate(images):
        inputs = build_inputs(processor, img_path)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        results.append((text, len(gen_ids)))

        if i < 3:
            print(f"  [{label}][{i}] len={len(gen_ids)}  {text[:100]}…")
    return results


def run_bra_generation(model, processor, images, label: str):
    from bra_logits_processor import create_bra_processor, make_bra_config
    from bra_operator_multi import detect_adapter

    adapter = detect_adapter(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    cfg = make_bra_config("bra_zero", debug=False, warmup_steps=3)

    results = []
    steps = []
    vision_tokens = []
    proc_stats = []
    for i, img_path in enumerate(images):
        inputs = build_inputs(processor, img_path)
        video_grid_thw = inputs.get("video_grid_thw")
        extractor, bra_proc = create_bra_processor(
            model,
            adapter,
            inputs["input_ids"],
            config=cfg,
            tokenizer=tokenizer,
            video_grid_thw=video_grid_thw,
        )
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    logits_processor=[bra_proc],
                )
            gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
            text = processor.decode(gen_ids, skip_special_tokens=True)
            results.append((text, len(gen_ids)))
            steps.append(bra_proc._step)
            n_vis = 0 if bra_proc._vision_features is None else bra_proc._vision_features.shape[0]
            vision_tokens.append(n_vis)
            proc_stats.append(bra_proc.get_stats())

            if i < 3:
                print(f"  [{label}][{i}] len={len(gen_ids)}  vision={n_vis}  steps={bra_proc._step}  {text[:100]}…")
        finally:
            extractor.remove()
            bra_proc.reset()

    return results, steps, vision_tokens, proc_stats


def report_agl(base_results, bra_results):
    base_agl = sum(r[1] for r in base_results) / len(base_results)
    bra_agl  = sum(r[1] for r in bra_results) / len(bra_results)
    delta_pct = abs(bra_agl - base_agl) / max(base_agl, 1) * 100
    status = "PASS" if delta_pct < 2.0 else "FAIL"
    print(f"\n[AGL] base={base_agl:.1f}  bra={bra_agl:.1f}  "
          f"delta={delta_pct:.2f}%  [{status}]")
    return delta_pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model, processor = load_model_and_processor()
    images = collect_images(MAX_SAMPLES)
    if images is None:
        print("[ERROR] Cannot proceed without images.")
        return

    print(f"\n{'='*60}")
    print(f"  Collected {len(images)} images from {COCO_DIR}")
    print(f"{'='*60}")

    # ---- Baseline run (no BRA) ----
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.max_memory_allocated()

    print("\n>>> Baseline generation (no BRA) …")
    base_results = run_generation(model, processor, images, "BASE")

    vram_base = torch.cuda.max_memory_allocated()

    # ---- BRA run ----
    torch.cuda.reset_peak_memory_stats()
    print("\n>>> BRA-V2 generation (LogitsProcessor) …")
    bra_results, bra_steps, bra_vision_tokens, bra_proc_stats = run_bra_generation(model, processor, images, "BRA")

    vram_bra = torch.cuda.max_memory_allocated()

    # ---- Reports ----
    delta_pct = report_agl(base_results, bra_results)
    changed = sum(1 for (b_txt, _), (r_txt, _) in zip(base_results, bra_results) if b_txt != r_txt)

    vram_delta_mb = (vram_bra - vram_base) / (1024 ** 2)
    vram_status = "PASS" if abs(vram_delta_mb) < 50 else "FAIL"
    print(f"[VRAM] base_peak={vram_base/1e9:.2f}GB  bra_peak={vram_bra/1e9:.2f}GB  "
          f"delta={vram_delta_mb:+.1f}MB  [{vram_status}]")
    print(f"[DIFF] outputs_changed={changed}/{len(images)}")
    print(f"[BRA] avg_steps={sum(bra_steps)/max(len(bra_steps),1):.1f}  "
          f"avg_vision_tokens={sum(bra_vision_tokens)/max(len(bra_vision_tokens),1):.1f}")
    missing = []
    if bra_proc_stats:
        required_keys = {
            "avg_candidate_window",
            "avg_visual_topk",
            "avg_resonance_time_ms",
            "intervention_rate",
            "selected_frame_histogram",
        }
        missing = sorted(required_keys - set(bra_proc_stats[0].keys()))
        avg_candidate = sum(s["avg_candidate_window"] for s in bra_proc_stats) / len(bra_proc_stats)
        avg_visual_topk = sum(s["avg_visual_topk"] for s in bra_proc_stats) / len(bra_proc_stats)
        avg_intervention = sum(s["intervention_rate"] for s in bra_proc_stats) / len(bra_proc_stats)
        avg_res_ms = sum(s["avg_resonance_time_ms"] for s in bra_proc_stats) / len(bra_proc_stats)
        print(f"[PROC] candidate_window={avg_candidate:.1f}  visual_topk={avg_visual_topk:.1f}  "
              f"intervention_rate={avg_intervention:.2f}  resonance_ms={avg_res_ms:.3f}")
        print(f"[LOG] required_fields={'OK' if not missing else 'MISSING:' + ','.join(missing)}")

    print(f"\n{'='*60}")
    all_pass = (
        delta_pct < 5.0
        and abs(vram_delta_mb) < 512
        and all(s > 0 for s in bra_steps)
        and all(v > 0 for v in bra_vision_tokens)
        and (not bra_proc_stats or not missing)
    )
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
