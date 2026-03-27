#!/usr/bin/env python3
"""
A-OSP Evaluation Pipeline — Base Model POPE Evaluation
=======================================================
Evaluates *native* Qwen2-VL-7B-Instruct on POPE benchmark (no intervention).
This produces the absolute reference baseline for all subsequent A-OSP comparisons.

Key engineering guarantees:
  1. JSONL Append mode — every sample is flushed immediately after inference,
     enabling crash-safe checkpoint / resume.
  2. Strict VRAM reclamation — del + gc.collect + torch.cuda.empty_cache
     at the end of every sample's for-loop iteration.
  3. AGL (Average Generation Length) is recorded per-sample to break the
     "Length-Bias Trap" narrative.

Usage:
    python run_base_eval.py \
        --model_path /root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct \
        --pope_file  /root/autodl-tmp/A-OSP_Project/data/pope/pope_coco_popular.jsonl \
        --image_dir  /path/to/coco/val2014 \
        --output_dir /root/autodl-tmp/A-OSP_Project/logs/eval_results \
        --tag        base_qwen2vl7b_pope_popular
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_utils import (
    append_jsonl,
    load_completed_ids,
    load_jsonl,
    compute_pope_metrics,
    save_csv_summary,
    aggressive_vram_cleanup,
    log_gpu_memory,
    load_qwen2vl,
    Timer,
)


# ──────────────────────────────────────────────────────────────────────────
# Inference for a single POPE sample
# ──────────────────────────────────────────────────────────────────────────

def infer_single_pope(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 64,
) -> tuple[str, int, float]:
    """
    Run one POPE question through Qwen2-VL / Qwen3-VL.
    Returns (generated_text, generation_length_in_tokens, latency_seconds).
    """
    # Use apply_chat_template for cross-model compatibility (Qwen2-VL, Qwen3-VL)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question} Answer with yes or no."},
        ],
    }]
    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        # Fallback: legacy direct path
        prompt = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question} Answer with yes or no.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True,
        ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    gen_ids = output_ids[0, input_len:]
    gen_len = gen_ids.shape[0]
    gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    del inputs, output_ids, gen_ids
    return gen_text, int(gen_len), latency


# ──────────────────────────────────────────────────────────────────────────
# Fallback: generate a placeholder image when COCO images are unavailable
# ──────────────────────────────────────────────────────────────────────────

def get_image_or_placeholder(image_dir: str, filename: str) -> Image.Image:
    """Load image from disk; tries .jpg/.png suffixes if bare name given."""
    path = os.path.join(image_dir, filename)
    for candidate in [path, path + ".jpg", path + ".png", path + ".jpeg"]:
        if os.path.isfile(candidate):
            return Image.open(candidate).convert("RGB")
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ──────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────

def run_pope_eval(args):
    # ── paths ──
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{args.tag}_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── checkpoint: load already-completed IDs ──
    completed_ids = load_completed_ids(result_file)
    if completed_ids:
        print(f"[Resume] Found {len(completed_ids)} completed samples — skipping them.")

    # ── dataset ──
    dataset = load_jsonl(args.pope_file)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(dataset)} | Pending={len(pending)}")

    if not pending:
        print("[Done] All samples already evaluated. Computing final metrics ...")
        all_results = load_jsonl(result_file)
        metrics = compute_pope_metrics(all_results)
        save_csv_summary(summary_file, metrics)
        print(json.dumps(metrics, indent=2))
        return

    # ── model ──
    model, processor = load_qwen2vl(args.model_path)

    # ── inference loop ──
    total_latency = 0.0
    total_gen_tokens = 0
    errors = 0

    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        question = sample["question"]
        gt = sample["ground_truth"]
        img_name = sample.get("image", "")

        image = get_image_or_placeholder(args.image_dir, img_name)

        try:
            gen_text, gen_len, latency = infer_single_pope(
                model, processor, image, question,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            gen_text, gen_len, latency = "error", 0, 0.0
            errors += 1

        total_latency += latency
        total_gen_tokens += gen_len

        record = {
            "question_id": qid,
            "image": img_name,
            "question": question,
            "prediction": gen_text,
            "ground_truth": gt,
            "generation_length": gen_len,
            "latency_s": round(latency, 4),
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pending):
            done = len(completed_ids) + idx + 1
            avg_tok_s = total_gen_tokens / max(total_latency, 1e-9)
            print(
                f"[Progress] {done}/{len(dataset)} | "
                f"pred='{gen_text[:40]}' | gt={gt} | "
                f"gen_len={gen_len} | latency={latency:.2f}s | "
                f"avg_throughput={avg_tok_s:.1f} tok/s"
            )

        # ── STRICT VRAM reclamation ──
        del image, gen_text
        gc.collect()
        torch.cuda.empty_cache()

    # ── final metrics ──
    print("\n" + "=" * 60)
    print("[Eval] Computing final POPE metrics ...")
    all_results = load_jsonl(result_file)
    metrics = compute_pope_metrics(all_results)
    metrics["method"] = "base"
    metrics["model"] = "Qwen2-VL-7B-Instruct"
    metrics["tag"] = args.tag
    metrics["errors"] = errors
    metrics["total_latency_s"] = round(total_latency, 2)

    save_csv_summary(summary_file, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"[Done] Results  → {result_file}")
    print(f"[Done] Summary  → {summary_file}")

    log_gpu_memory("final")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="A-OSP Base POPE Evaluation")
    p.add_argument(
        "--model_path", type=str,
        default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct",
    )
    p.add_argument(
        "--pope_file", type=str,
        default="/root/autodl-tmp/A-OSP_Project/data/pope/pope_coco_popular_mini.jsonl",
    )
    p.add_argument(
        "--image_dir", type=str,
        default="/root/autodl-tmp/A-OSP_Project/data/coco_val2014",
        help="Directory containing COCO val2014 images. Missing images → grey placeholder.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="/root/autodl-tmp/A-OSP_Project/logs/eval_results",
    )
    p.add_argument("--tag", type=str, default="base_qwen2vl7b_pope_popular")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="Max samples to eval (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  A-OSP Evaluation Pipeline — POPE Base Baseline")
    print(f"  Model : {args.model_path}")
    print(f"  Dataset: {args.pope_file}")
    print(f"  Output : {args.output_dir}/{args.tag}_*")
    print("=" * 60)
    run_pope_eval(args)
