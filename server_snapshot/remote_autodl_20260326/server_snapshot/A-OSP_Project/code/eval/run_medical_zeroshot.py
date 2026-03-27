#!/usr/bin/env python3
"""
A-OSP Evaluation — Medical Cross-Domain Zero-Shot Test (Section 4.6)
=====================================================================
Core Hypothesis:
  The bias subspace S_mscoco extracted from natural images encodes
  *language-side* hallucination momentum that is **domain-agnostic**.
  Applying A-OSP with this natural-image V_matrix to medical radiology
  images (VQA-RAD) should still suppress hallucination without degrading
  factual grounding — proving cross-domain transferability.

Dataset: VQA-RAD (flaviagiammarino/vqa-rad) — 451 radiology VQA samples
  - 56% yes/no (expect zero A-OSP intervention — same as POPE)
  - 44% open-ended (expect moderate intervention on longer answers)

Supports two modes:
  --method base    → native Qwen2-VL-7B
  --method aosp    → with A-OSP hook (natural-image V_matrix)

Usage:
    python run_medical_zeroshot.py --method base --limit 50 --tag med_base_50
    python run_medical_zeroshot.py --method aosp --limit 50 --tag med_aosp_50
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


# ──────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ──────────────────────────────────────────────────────────────────────────

def load_vqa_rad(split: str = "test", limit: int = 0, cache_dir: str = None) -> list[dict]:
    """Load VQA-RAD from HuggingFace and convert to our JSONL-compatible format."""
    from datasets import load_dataset

    ds = load_dataset("flaviagiammarino/vqa-rad", split=split, cache_dir=cache_dir)
    records = []
    for i, row in enumerate(ds):
        if limit > 0 and i >= limit:
            break
        answer = row["answer"].strip().lower()
        is_yn = answer in ("yes", "no")
        records.append({
            "question_id": i,
            "question": row["question"],
            "gt_answer": row["answer"],
            "answer_type": "yes/no" if is_yn else "open",
            "image": row["image"],  # PIL.Image object
        })
    return records


def save_vqa_rad_jsonl(records: list[dict], outpath: str):
    """Save VQA-RAD records to JSONL (without image bytes)."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        for r in records:
            row = {k: v for k, v in r.items() if k != "image"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────

MAX_IMAGE_PIXELS = 1024 * 1024


def _safe_resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.convert("RGB")


def infer_medical(
    model,
    processor,
    image: Image.Image,
    question: str,
    aosp_handle=None,
    max_new_tokens: int = 256,
) -> dict:
    """Generate answer for a medical VQA sample."""
    image = _safe_resize(image)

    prompt = (
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"You are a radiologist. Answer the following question about this medical image.\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
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
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def compute_medical_metrics(results: list[dict]) -> dict:
    """Compute accuracy for yes/no, AGL for open-ended, overall stats."""
    yn_results = [r for r in results if r.get("answer_type") == "yes/no"]
    open_results = [r for r in results if r.get("answer_type") == "open"]

    # Yes/No exact match
    yn_correct = sum(
        1 for r in yn_results
        if r["prediction"].strip().lower().startswith(r["gt_answer"].strip().lower())
    )
    yn_acc = yn_correct / max(len(yn_results), 1)

    # Open-ended: relaxed containment match
    open_correct = sum(
        1 for r in open_results
        if r["gt_answer"].strip().lower() in r["prediction"].strip().lower()
    )
    open_acc = open_correct / max(len(open_results), 1)

    total_gen_len = sum(r.get("generation_length", 0) for r in results)
    total_interventions = sum(r.get("intervention_count", 0) for r in results)
    total_latency = sum(r.get("latency_s", 0) for r in results)

    yn_interventions = sum(r.get("intervention_count", 0) for r in yn_results)
    open_interventions = sum(r.get("intervention_count", 0) for r in open_results)

    yn_agl = sum(r.get("generation_length", 0) for r in yn_results) / max(len(yn_results), 1)
    open_agl = sum(r.get("generation_length", 0) for r in open_results) / max(len(open_results), 1)

    return {
        "total": len(results),
        "yn_count": len(yn_results),
        "yn_accuracy": round(yn_acc, 4),
        "yn_agl": round(yn_agl, 1),
        "yn_interventions": yn_interventions,
        "open_count": len(open_results),
        "open_accuracy": round(open_acc, 4),
        "open_agl": round(open_agl, 1),
        "open_interventions": open_interventions,
        "overall_agl": round(total_gen_len / max(len(results), 1), 1),
        "total_interventions": total_interventions,
        "avg_interventions_per_sample": round(total_interventions / max(len(results), 1), 2),
        "avg_latency_s": round(total_latency / max(len(results), 1), 3),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def run_medical_eval(args):
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{args.tag}_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    completed_ids = load_completed_ids(result_file)
    if completed_ids:
        print(f"[Resume] Found {len(completed_ids)} completed — skipping.")

    print("[Dataset] Loading VQA-RAD ...")
    dataset = load_vqa_rad(split="test", limit=args.limit)
    print(f"[Dataset] Loaded {len(dataset)} samples "
          f"({sum(1 for r in dataset if r['answer_type']=='yes/no')} yes/no, "
          f"{sum(1 for r in dataset if r['answer_type']=='open')} open)")

    # Save dataset JSONL for reference
    ref_jsonl = os.path.join(
        str(PROJECT_ROOT / "data" / "medical"),
        f"vqa_rad_test_{args.limit}.jsonl",
    )
    save_vqa_rad_jsonl(dataset, ref_jsonl)

    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Pipeline] Total={len(dataset)} | Pending={len(pending)}")

    if not pending:
        print("[Done] All samples evaluated.")
        all_results = load_jsonl(result_file)
        metrics = compute_medical_metrics(all_results)
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
        )
        print(f"[Cross-Domain] Using NATURAL-IMAGE V_matrix for MEDICAL data!")
        print(f"[Cross-Domain] This tests domain-agnostic hallucination suppression.")

    errors = 0
    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        question = sample["question"]
        image = sample["image"]

        try:
            result = infer_medical(
                model, processor, image, question,
                aosp_handle=aosp_handle,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            result = {"text": "error", "gen_len": 0, "latency": 0.0, "interventions": 0}
            errors += 1

        record = {
            "question_id": qid,
            "question": question,
            "gt_answer": sample["gt_answer"],
            "answer_type": sample["answer_type"],
            "prediction": result["text"],
            "generation_length": result["gen_len"],
            "latency_s": round(result["latency"], 4),
            "intervention_count": result["interventions"],
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(pending):
            done = len(completed_ids) + idx + 1
            print(
                f"[{done}/{len(dataset)}] "
                f"type={sample['answer_type']:5s} | "
                f"gen_len={result['gen_len']:3d} | "
                f"interv={result['interventions']:2d} | "
                f"lat={result['latency']:.1f}s | "
                f"GT='{sample['gt_answer'][:30]}' | "
                f"pred='{result['text'][:50]}'"
            )

        del image
        gc.collect()
        torch.cuda.empty_cache()

    if aosp_handle is not None:
        aosp_handle.remove()

    # ── Final metrics ──
    print("\n" + "=" * 70)
    all_results = load_jsonl(result_file)
    metrics = compute_medical_metrics(all_results)
    metrics["method"] = args.method
    metrics["model"] = "Qwen2-VL-7B-Instruct"
    metrics["tag"] = args.tag
    metrics["v_matrix_domain"] = "MSCOCO (natural images)" if args.method == "aosp" else "N/A"
    metrics["errors"] = errors

    flat_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    save_csv_summary(summary_file, flat_metrics)

    print(f"[Medical Cross-Domain {args.method.upper()} Results]")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults  → {result_file}")
    print(f"Summary  → {summary_file}")
    print(f"Dataset  → {ref_jsonl}")
    log_gpu_memory("final")


def parse_args():
    p = argparse.ArgumentParser(description="A-OSP Medical Cross-Domain Zero-Shot Test")
    p.add_argument("--model_path", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct")
    p.add_argument("--v_matrix", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--output_dir", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, default="med_base_50")
    p.add_argument("--method", type=str, default="base",
                    choices=["base", "aosp"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    # A-OSP hyperparameters
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print(f"  A-OSP Medical Cross-Domain Zero-Shot — [{args.method.upper()}]")
    print(f"  Model     : {args.model_path}")
    if args.method == "aosp":
        print(f"  V_matrix  : {args.v_matrix} (NATURAL-IMAGE domain)")
        print(f"  Hyperparams: alpha={args.alpha} mu={args.mu} beta={args.beta}")
    print(f"  Dataset   : VQA-RAD test (limit={args.limit})")
    print(f"  Output    : {args.output_dir}/{args.tag}_*")
    print("=" * 70)
    run_medical_eval(args)
