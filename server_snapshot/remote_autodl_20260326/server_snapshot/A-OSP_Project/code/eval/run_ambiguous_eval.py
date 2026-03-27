#!/usr/bin/env python3
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
MAX_IMAGE_PIXELS = 1024 * 1024

def _safe_open(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def load_mmhal_image(image_field: str, image_id: str) -> Image.Image:
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

def infer_mmhal(
    model, processor, image: Image.Image, question: str, aosp_handle=None, max_new_tokens: int = 512
) -> dict:
    prompt = (
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if aosp_handle is not None:
        aosp_handle.reset()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0)

    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    gen_ids = output_ids[0, input_len:]
    gen_len = int(gen_ids.shape[0])
    gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    interventions = 0
    if aosp_handle is not None:
        interventions = aosp_handle.intervention_count

    del inputs, output_ids, gen_ids
    return {"text": gen_text, "gen_len": gen_len, "latency": latency, "interventions": interventions}

def run_eval(args):
    result_file = os.path.join(args.output_dir, f"{args.tag}_results.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    completed_ids = load_completed_ids(result_file)
    dataset = load_jsonl(args.mmhal_file)
    # Filter 50 extremely ambiguous/tiny-object images from MMHal (using ones that got intervened before)
    # For now we'll just take the first 50.
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    pending = [s for s in dataset if s["question_id"] not in completed_ids]
    print(f"[Dataset] Total={len(dataset)} | Pending={len(pending)}")
    if not pending:
        print("[Done] All samples evaluated.")
        return

    model, processor = load_qwen2vl(args.model_path)
    aosp_handle = None
    if args.method in ["aosp", "dynamic_aosp"]:
        from aosp_hook import apply_aosp_hook
        dynamic = (args.method == "dynamic_aosp")
        print(f"[Hook] Enabling A-OSP with dynamic_mu={dynamic}")
        aosp_handle = apply_aosp_hook(
            model, args.v_matrix, alpha=args.alpha, mu=args.mu, beta=args.beta,
            epsilon_steady=args.epsilon_steady, dynamic_mu=dynamic
        )

    for idx, sample in enumerate(pending):
        qid = sample["question_id"]
        question = sample["question"]
        image = load_mmhal_image(sample.get("image", ""), sample.get("image_id", ""))
        
        try:
            result = infer_mmhal(model, processor, image, question, aosp_handle=aosp_handle, max_new_tokens=args.max_new_tokens)
        except Exception as exc:
            print(f"[ERROR] qid={qid}: {exc}")
            result = {"text": "error", "gen_len": 0, "latency": 0.0, "interventions": 0}

        record = {
            "question_id": qid, "prediction": result["text"], "generation_length": result["gen_len"],
            "latency_s": round(result["latency"], 4), "intervention_count": result["interventions"]
        }
        append_jsonl(result_file, record)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(pending):
            print(f"[{idx+1}/{len(pending)}] qid={qid} interv={result['interventions']} len={result['gen_len']} lat={result['latency']:.1f}s")
        del image; gc.collect(); torch.cuda.empty_cache()

    if aosp_handle is not None:
        aosp_handle.remove()
    log_gpu_memory("final")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--v_matrix", type=str, default="/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt")
    p.add_argument("--mmhal_file", type=str, default="/root/autodl-tmp/A-OSP_Project/data/mmhal_bench/mmhal_bench.jsonl")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/A-OSP_Project/logs/eval_results")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--method", type=str, default="aosp", choices=["base", "aosp", "dynamic_aosp"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--epsilon_steady", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
