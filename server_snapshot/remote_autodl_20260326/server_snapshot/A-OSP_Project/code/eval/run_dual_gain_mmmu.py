"""
Dual-Gain MMMU Mini-Batch Evaluation (30 hard samples)
Proves that A-OSP and Test-Time Compute (Self-Correction) are orthogonally complementary.

Configurations:
  1. Base                    — single greedy pass
  2. SelfCorrection          — 2-pass TTC: generate → reflect → correct
  3. AOSP+SelfCorrection     — A-OSP geometric intervention on pass-1, then TTC reflect on pass-2

Model    : Qwen3-VL-8B-Instruct
V_matrix : models/qwen3vl/V_text_only.pt  (EVR=0.879, Agent 1 official)
Dataset  : MMMU validation set (900 samples available; we take the 30 hardest by index-shuffle)
Output   : logs/eval_results/dual_gain_mmmu_minibatch.json
"""

import json
import argparse
import time
import torch
import gc
import sys
import os

sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from eval_utils import load_qwen2vl, log_gpu_memory
from aosp_hook import apply_aosp_hook

OUTPUT_PATH = "/root/autodl-tmp/A-OSP_Project/logs/eval_results/dual_gain_mmmu_minibatch.json"


# ---------------------------------------------------------------------------
# MCQ helpers
# ---------------------------------------------------------------------------

def format_options(options: list) -> str:
    labels = ["A", "B", "C", "D", "E"]
    return "\n".join(f"({labels[i]}) {opt}" for i, opt in enumerate(options))


def extract_letter(text: str, valid: list) -> str:
    """Return the first letter from valid that appears in the text (case-insensitive)."""
    text_upper = text.upper()
    for letter in valid:
        if f"({letter})" in text_upper or f" {letter} " in text_upper or \
           text_upper.strip().startswith(letter):
            return letter
    # fallback: find any valid letter mention
    for letter in valid:
        if letter in text_upper:
            return letter
    return valid[0]   # conservative default: A


def build_question_prompt(question: str, options: list) -> str:
    opts_str = format_options(options)
    return (
        f"{question}\n\n{opts_str}\n\n"
        "Answer with only the letter of the correct option (e.g., A, B, C, or D)."
    )


def build_reflection_prompt(question: str, options: list, first_answer: str) -> str:
    opts_str = format_options(options)
    return (
        f"{question}\n\n{opts_str}\n\n"
        f"Your previous answer was: {first_answer}\n"
        "Carefully review the image and reasoning. Is your answer correct? "
        "If you spot any error, provide the corrected letter. "
        "Reply with only the final answer letter (A, B, C, or D)."
    )


# ---------------------------------------------------------------------------
# Single-pass inference (Base / with hook)
# ---------------------------------------------------------------------------

def infer_single(model, processor, image_path, prompt_text, max_new_tokens=20):
    """One greedy forward pass. Returns (decoded_text, latency_ms, peak_vram_gb)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt_text},
        ],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = (time.time() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    del inputs, out, gen_ids
    gc.collect()
    torch.cuda.empty_cache()
    return text, latency, peak_vram


# ---------------------------------------------------------------------------
# Three evaluation configurations
# ---------------------------------------------------------------------------

def run_base(model, processor, image_path, question, options):
    prompt  = build_question_prompt(question, options)
    ans, lat, vram = infer_single(model, processor, image_path, prompt)
    valid   = ["A", "B", "C", "D", "E"][: len(options)]
    letter  = extract_letter(ans, valid)
    return {"answer": letter, "raw": ans, "latency_ms": round(lat, 1), "peak_vram_gb": round(vram, 3), "passes": 1}


def run_self_correction(model, processor, image_path, question, options):
    valid  = ["A", "B", "C", "D", "E"][: len(options)]
    total_lat = 0.0
    peak_v    = 0.0

    # Pass 1 — generate initial answer
    prompt1    = build_question_prompt(question, options)
    ans1, lat1, vram1 = infer_single(model, processor, image_path, prompt1)
    total_lat += lat1
    peak_v     = max(peak_v, vram1)
    letter1    = extract_letter(ans1, valid)

    # Pass 2 — reflect and correct
    prompt2    = build_reflection_prompt(question, options, letter1)
    ans2, lat2, vram2 = infer_single(model, processor, image_path, prompt2)
    total_lat += lat2
    peak_v     = max(peak_v, vram2)
    letter2    = extract_letter(ans2, valid)

    return {
        "answer": letter2, "raw": ans2, "pass1_answer": letter1,
        "latency_ms": round(total_lat, 1), "peak_vram_gb": round(peak_v, 3), "passes": 2,
    }


def run_aosp_self_correction(model, processor, image_path, question, options, v_matrix_path):
    valid  = ["A", "B", "C", "D", "E"][: len(options)]
    total_lat = 0.0
    peak_v    = 0.0

    # Pass 1 — A-OSP geometric intervention
    prompt1  = build_question_prompt(question, options)
    handle   = apply_aosp_hook(model, v_matrix_path)
    handle.reset()
    ans1, lat1, vram1 = infer_single(model, processor, image_path, prompt1)
    interventions = handle.intervention_count
    handle.remove()
    total_lat += lat1
    peak_v     = max(peak_v, vram1)
    letter1    = extract_letter(ans1, valid)

    gc.collect()
    torch.cuda.empty_cache()

    # Pass 2 — TTC reflection (no hook)
    prompt2    = build_reflection_prompt(question, options, letter1)
    ans2, lat2, vram2 = infer_single(model, processor, image_path, prompt2)
    total_lat += lat2
    peak_v     = max(peak_v, vram2)
    letter2    = extract_letter(ans2, valid)

    return {
        "answer": letter2, "raw": ans2, "pass1_answer": letter1,
        "latency_ms": round(total_lat, 1), "peak_vram_gb": round(peak_v, 3),
        "passes": 2, "aosp_interventions": interventions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--v_matrix",   type=str,
                        default="/root/autodl-tmp/A-OSP_Project/models/qwen3vl/V_text_only.pt")
    parser.add_argument("--manifest",   type=str,
                        default="/root/autodl-tmp/A-OSP_Project/data/benchmarks/mmmu/mmmu_manifest.jsonl")
    parser.add_argument("--n_samples",  type=int, default=30)
    args = parser.parse_args()

    model, processor = load_qwen2vl(args.model_path)
    log_gpu_memory("after_load")

    # Load manifest; take a fixed hard subset (last N to avoid duplicating earlier mini-batch)
    with open(args.manifest) as f:
        all_samples = [json.loads(l) for l in f]

    # Select 30 samples spread across the 900-sample pool (every 30th entry = diverse subjects)
    subset = all_samples[::len(all_samples) // args.n_samples][: args.n_samples]
    print(f"[INFO] Running {len(subset)} MMMU samples — 3 configurations each")

    results   = []
    cfg_stats = {"Base": [], "SelfCorrection": [], "AOSP+SC": []}

    for i, sample in enumerate(tqdm(subset, desc="MMMU Dual-Gain")):
        img_path = sample["image_path"]
        question = sample["question"].replace("<image 1>", "").strip()
        options  = sample["options"]
        gt       = sample["answer"]          # e.g. "B"
        valid    = ["A", "B", "C", "D", "E"][: len(options)]

        # --- Base ---
        b = run_base(model, processor, img_path, question, options)
        b["correct"] = (b["answer"] == gt)
        cfg_stats["Base"].append(b)

        # --- Self-Correction ---
        sc = run_self_correction(model, processor, img_path, question, options)
        sc["correct"] = (sc["answer"] == gt)
        cfg_stats["SelfCorrection"].append(sc)

        # --- A-OSP + Self-Correction ---
        ac = run_aosp_self_correction(model, processor, img_path, question, options, args.v_matrix)
        ac["correct"] = (ac["answer"] == gt)
        cfg_stats["AOSP+SC"].append(ac)

        results.append({
            "index":    sample["_index"],
            "id":       sample["id"],
            "gt":       gt,
            "Base":             b,
            "SelfCorrection":   sc,
            "AOSP+SelfCorrection": ac,
        })

        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n=== DUAL-GAIN SUMMARY ===")
    summary = {}
    for cfg, records in cfg_stats.items():
        acc      = sum(r["correct"] for r in records) / len(records)
        avg_lat  = sum(r["latency_ms"]    for r in records) / len(records)
        avg_vram = sum(r["peak_vram_gb"]  for r in records) / len(records)
        avg_pass = sum(r["passes"]        for r in records) / len(records)
        summary[cfg] = {
            "accuracy":      round(acc, 4),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_peak_vram_gb": round(avg_vram, 3),
            "avg_passes":    round(avg_pass, 2),
        }
        print(f"{cfg:22s}: Acc={acc:.2%} | Lat={avg_lat:.0f}ms | VRAM={avg_vram:.2f}GB")

    # Dual-Gain delta
    if "Base" in summary and "AOSP+SC" in summary:
        delta = summary["AOSP+SC"]["accuracy"] - summary["Base"]["accuracy"]
        print(f"\nDual-Gain over Base  : +{delta:.2%}")
        print(f"SC alone over Base   : +{summary['SelfCorrection']['accuracy']-summary['Base']['accuracy']:.2%}")
        print(f"AOSP+SC vs SC alone  : +{summary['AOSP+SC']['accuracy']-summary['SelfCorrection']['accuracy']:.2%}")

    output = {
        "model": args.model_path,
        "v_matrix": args.v_matrix,
        "n_samples": len(subset),
        "summary": summary,
        "per_sample": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
