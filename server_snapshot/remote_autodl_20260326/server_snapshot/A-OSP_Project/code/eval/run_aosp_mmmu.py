#!/usr/bin/env python3
"""
A-OSP Evaluation Pipeline — A-OSP Intervention MMMU Evaluation
================================================================
Evaluates Qwen2-VL-7B-Instruct WITH A-OSP on MMMU to prove the orthogonal
projection does NOT damage commonsense reasoning (< 1% accuracy drop).

Usage:
    python run_aosp_mmmu.py \
        --model_path  /root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct \
        --v_matrix    /root/autodl-tmp/A-OSP_Project/models/V_matrix.pt \
        --output_dir  /root/autodl-tmp/A-OSP_Project/logs/eval_results \
        --tag         aosp_qwen2vl7b_mmmu_val \
        --max_samples 200
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

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
from aosp_hook import apply_aosp_hook


OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_mmmu_prompt(question: str, options: list[str]) -> str:
    opts_str = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options) if opt
    )
    return (
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}\n{opts_str}\n"
        f"Answer with the letter of the correct option only.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_answer_letter(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r"[A-H]", text)
    return m.group(0) if m else text[:1]


def load_mmmu_dataset(split: str = "validation", max_samples: int = 0) -> list[dict]:
    from datasets import load_dataset, concatenate_datasets
    print(f"[MMMU] Loading split='{split}' from HuggingFace ...")
    ds = load_dataset("MMMU/MMMU", split=split)
    if isinstance(ds, dict):
        ds = concatenate_datasets(list(ds.values()))

    records = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        options = [row.get(f"option_{j}", "") for j in range(8)]
        options = [o for o in options if o]
        img = row.get("image", None)
        if img is None:
            for col in row:
                val = row[col]
                if isinstance(val, Image.Image):
                    img = val
                    break
        records.append({
            "question_id": row.get("id", f"mmmu_{i}"),
            "question": row["question"],
            "options": options,
            "answer": row["answer"],
            "image": img,
        })
    print(f"[MMMU] Loaded {len(records)} samples.")
    return records


def run_aosp_mmmu_eval(args):
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{args.tag}_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    completed_ids = load_completed_ids(result_file)
    if completed_ids:
        print(f"[Resume] Found {len(completed_ids)} completed — skipping.")

    dataset = load_mmmu_dataset(split="validation", max_samples=args.max_samples)
    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(dataset)} | Pending={len(pending)}")

    if not pending:
        print("[Done] All samples evaluated.")
        all_results = load_jsonl(result_file)
        _print_mmmu_metrics(all_results, summary_file, args.tag)
        return

    model, processor = load_qwen2vl(args.model_path)
    handle = apply_aosp_hook(
        model, args.v_matrix,
        alpha=args.alpha, mu=args.mu,
        beta=args.beta, epsilon_steady=args.epsilon_steady,
    )

    correct = 0
    total = 0
    total_interventions = 0
    errors = 0

    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        question = sample["question"]
        options = sample["options"]
        gt_answer = sample["answer"].strip().upper()
        image = sample.get("image")

        if not isinstance(image, Image.Image):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        prompt = format_mmmu_prompt(question, options)

        try:
            inputs = processor(
                text=[prompt], images=[image],
                return_tensors="pt", padding=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[1]

            handle.reset()

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                )

            gen_ids = output_ids[0, input_len:]
            gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
            pred_letter = extract_answer_letter(gen_text)
            gen_len = int(gen_ids.shape[0])
            n_interv = handle.intervention_count

            del inputs, output_ids, gen_ids

        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            gen_text, pred_letter, gen_len, n_interv = "error", "", 0, 0
            errors += 1

        is_correct = pred_letter == gt_answer
        correct += int(is_correct)
        total += 1
        total_interventions += n_interv

        record = {
            "question_id": qid,
            "question": question[:80],
            "prediction_raw": gen_text,
            "prediction": pred_letter,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "generation_length": gen_len,
            "intervention_count": n_interv,
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(pending):
            running_acc = correct / max(total, 1)
            print(
                f"[Progress] {len(completed_ids) + idx + 1}/{len(dataset)} | "
                f"pred={pred_letter} gt={gt_answer} | "
                f"acc={running_acc:.4f} | interventions={n_interv}"
            )

        # ── STRICT VRAM reclamation ──
        del image, gen_text
        gc.collect()
        torch.cuda.empty_cache()

    handle.remove()

    print("\n" + "=" * 60)
    all_results = load_jsonl(result_file)
    _print_mmmu_metrics(all_results, summary_file, args.tag, total_interventions)
    log_gpu_memory("final")


def _print_mmmu_metrics(
    results: list[dict], summary_file: str, tag: str, total_interv: int = 0
):
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    accuracy = correct / max(total, 1)
    agl = sum(r.get("generation_length", 0) for r in results) / max(total, 1)

    metrics = {
        "method": "aosp",
        "model": "Qwen2-VL-7B-Instruct",
        "benchmark": "MMMU",
        "tag": tag,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "agl": round(agl, 2),
        "total_interventions": total_interv,
    }
    save_csv_summary(summary_file, metrics)
    print("[MMMU A-OSP Metrics]")
    print(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="A-OSP MMMU Evaluation")
    p.add_argument("--model_path", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct")
    p.add_argument("--v_matrix", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--output_dir", type=str,
                    default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, default="aosp_qwen2vl7b_mmmu_val")
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  A-OSP Evaluation Pipeline — MMMU with A-OSP Intervention")
    print(f"  Model  : {args.model_path}")
    print(f"  V_matrix: {args.v_matrix}")
    print(f"  Hyperparams: alpha={args.alpha} mu={args.mu} beta={args.beta}")
    print("=" * 60)
    run_aosp_mmmu_eval(args)
