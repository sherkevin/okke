#!/usr/bin/env python3
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

from eval.eval_utils import (
    append_jsonl,
    load_completed_ids,
    load_jsonl,
    save_csv_summary,
    log_gpu_memory,
    load_qwen2vl,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def build_mmbench_prompt(sample: dict) -> str:
    question = sample.get("question", "")
    hint = sample.get("hint", "")
    options_text = ""
    for opt in ["A", "B", "C", "D"]:
        if sample.get(opt):
            options_text += f"{opt}. {sample[opt]}\n"
    
    prompt = ""
    if hint and hint.strip():
        prompt += f"{hint}\n"
    prompt += f"{question}\n{options_text}Answer with the option's letter from the given choices directly."
    
    return (
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def infer_mmbench(
    model, processor, image: Image.Image, prompt: str, aosp_handle=None, max_new_tokens: int = 16
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
    dataset = load_jsonl(args.dataset_file)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    
    # In MMBench, we use 'index' as question_id
    for s in dataset:
        if "question_id" not in s:
            s["question_id"] = s["index"]

    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(dataset)} | Pending={len(pending)}")

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
    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        image_path = PROJECT_ROOT / sample["image_path"]
        
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = build_mmbench_prompt(sample)
            result = infer_mmbench(
                model, processor, image, prompt,
                aosp_handle=aosp_handle,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            result = {"text": "error", "gen_len": 0, "latency": 0.0, "interventions": 0}
            errors += 1

        record = {
            "question_id": qid,
            "prediction": result["text"],
            "ground_truth": sample.get("answer", ""),
            "generation_length": result["gen_len"],
            "latency_s": round(result["latency"], 4),
            "intervention_count": result["interventions"],
        }
        append_jsonl(result_file, record)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(pending):
            print(f"[{idx+1}/{len(pending)}] qid={qid} pred='{result['text']}' gt='{sample.get('answer')}' interv={result['interventions']}")

        if 'image' in locals():
            del image
        gc.collect()
        torch.cuda.empty_cache()

    if aosp_handle is not None:
        aosp_handle.remove()

    log_gpu_memory("final")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--dataset_file", type=str, default="/root/autodl-tmp/A-OSP_Project/data/benchmarks/mmbench/mmbench_manifest.jsonl")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--method", type=str, default="base", choices=["base", "aosp"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
