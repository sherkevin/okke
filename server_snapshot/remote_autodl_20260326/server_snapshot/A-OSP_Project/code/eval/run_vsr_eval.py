#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import (
    append_jsonl,
    load_completed_ids,
    save_csv_summary,
    log_gpu_memory,
    load_qwen2vl,
)

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "benchmarks" / "vsr" / "images"

def get_image(image_link: str, image_name: str) -> Image.Image:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    img_path = IMAGE_DIR / image_name
    if not img_path.exists():
        urllib.request.urlretrieve(image_link, str(img_path))
    return Image.open(img_path).convert("RGB")

def build_vsr_prompt(caption: str) -> str:
    prompt = f"Based on the image, is the following statement true or false?\nStatement: {caption}\nAnswer exactly 'True' or 'False'."
    return (
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def infer_vsr(
    model, processor, image: Image.Image, prompt: str, aosp_handle=None, max_new_tokens: int = 10
) -> dict:
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

def run_eval(args):
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{args.tag}_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    completed_ids = load_completed_ids(result_file)
    
    print(f"[Dataset] Loading VSR (Visual Spatial Reasoning) from HuggingFace...")
    ds = list(load_dataset('cambridgeltl/vsr_zeroshot', split='test', streaming=True).take(args.limit))
    
    for idx, s in enumerate(ds):
        s["question_id"] = idx

    pending = [s for s in ds if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(ds)} | Pending={len(pending)}")

    if not pending:
        print("[Done] All samples evaluated.")
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

    errors = 0
    correct = 0
    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        
        try:
            image = get_image(sample["image_link"], sample["image"])
            prompt = build_vsr_prompt(sample["caption"])
            result = infer_vsr(
                model, processor, image, prompt,
                aosp_handle=aosp_handle,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            result = {"text": "error", "gen_len": 0, "latency": 0.0, "interventions": 0}
            errors += 1

        gt_str = "True" if sample["label"] == 1 else "False"
        pred_clean = result["text"].strip().lower().strip('.')
        is_correct = (gt_str.lower() in pred_clean)
        if is_correct:
            correct += 1

        record = {
            "question_id": qid,
            "image": sample["image"],
            "caption": sample["caption"],
            "prediction": result["text"],
            "ground_truth": gt_str,
            "is_correct": is_correct,
            "generation_length": result["gen_len"],
            "latency_s": round(result["latency"], 4),
            "intervention_count": result["interventions"],
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(pending):
            print(f"[{idx+1}/{len(pending)}] qid={qid} pred='{result['text']}' gt='{gt_str}' interv={result['interventions']}")

        if 'image' in locals():
            del image
        gc.collect()
        torch.cuda.empty_cache()

    if aosp_handle is not None:
        aosp_handle.remove()

    print(f"\n[Done] Accuracy: {correct}/{len(pending)} ({correct/len(pending)*100:.2f}%)")
    log_gpu_memory("final")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--method", type=str, default="base", choices=["base", "aosp"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
