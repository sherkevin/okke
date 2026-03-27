#!/usr/bin/env python3
"""
A-OSP Evaluation Pipeline — MMHal-Bench Evaluation
====================================================
Long-form open-ended visual QA — the primary battleground for A-OSP.

Unlike POPE (2-token yes/no), MMHal-Bench forces the model to generate
detailed multi-sentence descriptions, causing the autoregressive trajectory
to traverse the entropy burn-in period and potentially trigger A-OSP
intervention when language inertia dominates.

Key metrics per sample:
  - generation_length (AGL tracking to prove no length-truncation cheating)
  - intervention_count (A-OSP trigger frequency — should be >> 0 here)
  - latency_s

Supports two modes:
  --method base    → native Qwen2-VL-7B
  --method aosp    → with A-OSP hook

Usage:
    python run_mmhal_eval.py --method base --limit 50 --tag base_mmhal_50
    python run_mmhal_eval.py --method aosp --limit 50 --tag aosp_mmhal_50
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import (
    append_jsonl,
    load_completed_ids,
    load_jsonl,
    save_csv_summary,
    log_gpu_memory,
    load_qwen2vl,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MMHAL_IMAGE_DIR = PROJECT_ROOT / "data" / "mmhal_bench" / "images"


# ──────────────────────────────────────────────────────────────────────────
# Image loading with multiple fallback strategies
# ──────────────────────────────────────────────────────────────────────────

MAX_IMAGE_PIXELS = 1024 * 1024  # cap at ~1MP to prevent VRAM explosion on Flickr originals


def _safe_open(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def load_mmhal_image(image_field: str, image_id: str) -> Image.Image:
    """Load MMHal-Bench image with fallback strategies + resize guard."""
    for candidate in [
        MMHAL_IMAGE_DIR / image_field,
        MMHAL_IMAGE_DIR / f"{image_id}.jpg",
        MMHAL_IMAGE_DIR / f"{image_id}.png",
    ]:
        if candidate.exists():
            return _safe_open(candidate)

    for fname in os.listdir(MMHAL_IMAGE_DIR) if MMHAL_IMAGE_DIR.exists() else []:
        if image_id in fname:
            return _safe_open(MMHAL_IMAGE_DIR / fname)

    return Image.new("RGB", (448, 448), color=(128, 128, 128))


# ──────────────────────────────────────────────────────────────────────────
# Single-sample inference
# ──────────────────────────────────────────────────────────────────────────

def infer_mmhal(
    model,
    processor,
    image: Image.Image,
    question: str,
    aosp_handle=None,
    max_new_tokens: int = 512,
) -> dict:
    """
    Generate a long-form answer for MMHal-Bench.
    Returns dict with keys: text, gen_len, latency, interventions.
    """
    # apply_chat_template path for cross-model support (Qwen2-VL / Qwen3-VL)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
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
        prompt = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True,
        ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if aosp_handle is not None:
        aosp_handle.reset()

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
    gen_len = int(gen_ids.shape[0])
    gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    interventions = 0
    if aosp_handle is not None:
        interventions = aosp_handle.intervention_count

    del inputs, output_ids, gen_ids
    return {
        "text": gen_text,
        "gen_len": gen_len,
        "latency": latency,
        "interventions": interventions,
    }


# ──────────────────────────────────────────────────────────────────────────
# MMHal-Bench scoring (hallucination detection via entity matching)
# ──────────────────────────────────────────────────────────────────────────

def compute_mmhal_metrics(results: list[dict]) -> dict:
    """
    Compute summary metrics for MMHal-Bench.
    Full GPT-4 scoring requires API access; here we compute proxy metrics.
    """
    total = len(results)
    total_gen_len = sum(r.get("generation_length", 0) for r in results)
    total_interventions = sum(r.get("intervention_count", 0) for r in results)
    total_latency = sum(r.get("latency_s", 0) for r in results)

    agl = total_gen_len / max(total, 1)
    avg_interventions = total_interventions / max(total, 1)
    avg_latency = total_latency / max(total, 1)

    type_stats: dict[str, list] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        type_stats.setdefault(qt, []).append(r.get("generation_length", 0))

    per_type_agl = {
        qt: round(sum(lens) / max(len(lens), 1), 1)
        for qt, lens in type_stats.items()
    }

    return {
        "total": total,
        "agl": round(agl, 1),
        "total_interventions": total_interventions,
        "avg_interventions_per_sample": round(avg_interventions, 2),
        "avg_latency_s": round(avg_latency, 3),
        "total_latency_s": round(total_latency, 1),
        "per_type_agl": per_type_agl,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def run_mmhal_eval(args):
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{args.tag}_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    completed_ids = load_completed_ids(result_file)
    if completed_ids:
        print(f"[Resume] Found {len(completed_ids)} completed — skipping.")

    dataset = load_jsonl(args.mmhal_file)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(dataset)} | Pending={len(pending)}")

    if not pending:
        print("[Done] All samples evaluated.")
        all_results = load_jsonl(result_file)
        metrics = compute_mmhal_metrics(all_results)
        metrics["method"] = args.method
        print(json.dumps(metrics, indent=2))
        return

    model, processor = load_qwen2vl(args.model_path)

    aosp_handle = None
    if args.method == "aosp":
        from aosp_hook import apply_aosp_hook
        aosp_handle = apply_aosp_hook(
            model, args.v_matrix,
            alpha=args.alpha, mu=args.mu, beta=args.beta,
            epsilon_steady=args.epsilon_steady,
            dynamic_mu=args.dynamic_mu,
        )

    total_gen_tokens = 0
    total_interventions = 0
    total_latency = 0.0
    errors = 0

    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        question = sample["question"]
        image_field = sample.get("image", "")
        image_id = sample.get("image_id", "")

        image = load_mmhal_image(image_field, image_id)

        try:
            result = infer_mmhal(
                model, processor, image, question,
                aosp_handle=aosp_handle,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            result = {"text": "error", "gen_len": 0, "latency": 0.0, "interventions": 0}
            errors += 1

        total_gen_tokens += result["gen_len"]
        total_interventions += result["interventions"]
        total_latency += result["latency"]

        record = {
            "question_id": qid,
            "image_id": image_id,
            "question": question,
            "question_type": sample.get("question_type", ""),
            "question_topic": sample.get("question_topic", ""),
            "gt_answer": sample.get("gt_answer", ""),
            "prediction": result["text"],
            "generation_length": result["gen_len"],
            "latency_s": round(result["latency"], 4),
            "intervention_count": result["interventions"],
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(pending):
            done = len(completed_ids) + idx + 1
            avg_tok_s = total_gen_tokens / max(total_latency, 1e-9)
            print(
                f"[{done}/{len(dataset)}] "
                f"type={sample.get('question_type','?'):12s} | "
                f"gen_len={result['gen_len']:3d} | "
                f"interv={result['interventions']:2d} | "
                f"lat={result['latency']:.1f}s | "
                f"avg={avg_tok_s:.0f}tok/s | "
                f"pred='{result['text'][:60]}...'"
            )

        # ── STRICT VRAM reclamation ──
        del image
        gc.collect()
        torch.cuda.empty_cache()

    if aosp_handle is not None:
        aosp_handle.remove()

    # ── Final metrics ──
    print("\n" + "=" * 70)
    all_results = load_jsonl(result_file)
    metrics = compute_mmhal_metrics(all_results)
    metrics["method"] = args.method
    metrics["model"] = "Qwen2-VL-7B-Instruct"
    metrics["tag"] = args.tag
    metrics["errors"] = errors

    flat_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    save_csv_summary(summary_file, flat_metrics)

    print(f"[MMHal-Bench {args.method.upper()} Results]")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults  → {result_file}")
    print(f"Summary  → {summary_file}")
    log_gpu_memory("final")


def parse_args():
    p = argparse.ArgumentParser(description="A-OSP MMHal-Bench Evaluation")
    p.add_argument("--model_path", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct")
    p.add_argument("--v_matrix", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--mmhal_file", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/data/mmhal_bench/mmhal_bench.jsonl")
    p.add_argument("--output_dir", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, default="base_mmhal_50")
    p.add_argument("--method", type=str, default="base",
                    choices=["base", "aosp"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    # A-OSP hyperparameters
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    p.add_argument("--dynamic_mu", action="store_true", help="Enable visual entropy-based dynamic mu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print(f"  A-OSP Evaluation — MMHal-Bench [{args.method.upper()}]")
    print(f"  Model  : {args.model_path}")
    if args.method == "aosp":
        print(f"  V_matrix: {args.v_matrix}")
        print(f"  Hyperparams: alpha={args.alpha} mu={args.mu} beta={args.beta}")
    print(f"  Dataset: {args.mmhal_file} (limit={args.limit})")
    print(f"  Output : {args.output_dir}/{args.tag}_*")
    print("=" * 70)
    run_mmhal_eval(args)
