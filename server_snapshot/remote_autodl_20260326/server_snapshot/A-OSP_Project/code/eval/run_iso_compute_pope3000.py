"""
Iso-Compute Full-Scale POPE Run (3000 samples)
Model: Qwen3-VL-8B-Instruct
Configurations: Base, Majority Voting (T=0.7, Top_p=0.9), A-OSP
Features:
  - Checkpoint every 200 samples (no data loss)
  - Strict per-sample VRAM cleanup
  - Skips already-completed samples (resume support)
"""

import json
import argparse
import time
import torch
import gc
import sys
import os
from collections import Counter
from pathlib import Path

sys.path.append("code")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from eval_utils import load_qwen2vl, compute_pope_metrics
from aosp_hook import apply_aosp_hook

CHECKPOINT_DIR = "/root/autodl-tmp/A-OSP_Project/logs/eval_results/iso_compute_checkpoints"
FINAL_OUTPUT   = "/root/autodl-tmp/A-OSP_Project/logs/eval_results/iso_compute_full_run_3000.json"
CHECKPOINT_INTERVAL = 200

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    t = text.lower()
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return "yes"   # conservative default


def prepare_inputs(processor, image_path, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": question},
        ],
    }]
    text_prompt   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return inputs


def infer_base(model, inputs):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    lat  = (time.time() - t0) * 1000
    vram = torch.cuda.max_memory_allocated() / 1024**3
    gen  = out[0][inputs["input_ids"].shape[1]:]
    return gen, lat, vram


def infer_majority_voting(model, processor, inputs, n_votes=3):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    all_gen = []
    for _ in range(n_votes):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        all_gen.append(out[0][inputs["input_ids"].shape[1]:])
        del out
        gc.collect()
        torch.cuda.empty_cache()
    lat  = (time.time() - t0) * 1000
    vram = torch.cuda.max_memory_allocated() / 1024**3
    answers = [extract_answer(processor.decode(g, skip_special_tokens=True)) for g in all_gen]
    final   = Counter(answers).most_common(1)[0][0]
    avg_agl = sum(len(processor.decode(g, skip_special_tokens=True).split()) for g in all_gen) / n_votes
    del all_gen
    return final, lat, vram, avg_agl


def infer_aosp(model, processor, inputs, v_matrix_path):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    handle = apply_aosp_hook(model, v_matrix_path)
    handle.reset()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    lat  = (time.time() - t0) * 1000
    vram = torch.cuda.max_memory_allocated() / 1024**3
    gen  = out[0][inputs["input_ids"].shape[1]:]
    interventions = handle.intervention_count
    handle.remove()
    del out
    return gen, lat, vram, interventions


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(cfg: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"pope_{cfg}_live.jsonl")


def load_completed(cfg: str) -> dict:
    """Return {question_id: record} for all already-saved records."""
    fpath = ckpt_path(cfg)
    completed = {}
    if not os.path.exists(fpath):
        return completed
    with open(fpath, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed[rec["question_id"]] = rec
            except Exception:
                pass
    return completed


def append_ckpt(cfg: str, record: dict):
    with open(ckpt_path(cfg), "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summarise a list of per-sample records
# ---------------------------------------------------------------------------

def summarise(records: list) -> dict:
    metrics = compute_pope_metrics(records)
    avg_lat  = sum(r["latency_ms"] for r in records) / len(records)
    avg_vram = sum(r["peak_vram_gb"] for r in records) / len(records)
    avg_agl  = sum(r["agl"] for r in records) / len(records)
    return {**metrics, "avg_latency_ms": round(avg_lat, 1), "avg_peak_vram_gb": round(avg_vram, 3), "avg_agl": round(avg_agl, 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--v_matrix",   type=str,
                        default="/root/autodl-tmp/A-OSP_Project/models/qwen3vl/V_text_only.pt")
    parser.add_argument("--pope_path",  type=str,
                        default="/root/autodl-tmp/A-OSP_Project/data/pope/pope_coco_popular.jsonl")
    parser.add_argument("--image_dir",  type=str,
                        default="/root/autodl-tmp/A-OSP_Project/data/coco_val2014")
    args = parser.parse_args()

    print(f"[INFO] Checkpoints directory : {CHECKPOINT_DIR}")
    print(f"[INFO] Final output           : {FINAL_OUTPUT}")
    print(f"[INFO] Checkpoint interval    : every {CHECKPOINT_INTERVAL} samples")
    print(f"[INFO] Model                  : {args.model_path}")
    print(f"[INFO] V_matrix               : {args.v_matrix}")

    model, processor = load_qwen2vl(args.model_path)

    # Load POPE data
    with open(args.pope_path) as f:
        all_samples = [json.loads(l) for l in f]
    print(f"[INFO] Loaded {len(all_samples)} POPE samples")

    # Load previously completed records (resume support)
    base_done = load_completed("Base")
    mv_done   = load_completed("MajorityVoting")
    aosp_done = load_completed("AOSP")
    print(f"[RESUME] Base={len(base_done)}, MV={len(mv_done)}, AOSP={len(aosp_done)} already done")

    for i, sample in enumerate(tqdm(all_samples, desc="POPE 3000")):
        qid       = sample["question_id"]
        img_path  = os.path.join(args.image_dir, sample["image"] + ".jpg")
        question  = sample["question"]
        gt        = sample["ground_truth"]

        if not os.path.exists(img_path):
            continue

        inputs = prepare_inputs(processor, img_path, question)
        inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        # ---- BASE ----
        if qid not in base_done:
            gen, lat, vram = infer_base(model, inputs)
            ans = processor.decode(gen, skip_special_tokens=True)
            rec = {
                "question_id": qid, "prediction": extract_answer(ans),
                "ground_truth": gt, "latency_ms": round(lat, 1),
                "peak_vram_gb": round(vram, 3), "agl": len(ans.split()),
            }
            append_ckpt("Base", rec)
            base_done[qid] = rec
            del gen, ans
            gc.collect()
            torch.cuda.empty_cache()

        # ---- MAJORITY VOTING ----
        if qid not in mv_done:
            final_ans, lat, vram, avg_agl = infer_majority_voting(model, processor, inputs)
            rec = {
                "question_id": qid, "prediction": final_ans,
                "ground_truth": gt, "latency_ms": round(lat, 1),
                "peak_vram_gb": round(vram, 3), "agl": round(avg_agl, 1),
            }
            append_ckpt("MajorityVoting", rec)
            mv_done[qid] = rec
            gc.collect()
            torch.cuda.empty_cache()

        # ---- A-OSP ----
        if qid not in aosp_done:
            gen, lat, vram, n_intv = infer_aosp(model, processor, inputs, args.v_matrix)
            ans = processor.decode(gen, skip_special_tokens=True)
            rec = {
                "question_id": qid, "prediction": extract_answer(ans),
                "ground_truth": gt, "latency_ms": round(lat, 1),
                "peak_vram_gb": round(vram, 3), "agl": len(ans.split()),
                "interventions": n_intv,
            }
            append_ckpt("AOSP", rec)
            aosp_done[qid] = rec
            del gen, ans
            gc.collect()
            torch.cuda.empty_cache()

        del inputs
        gc.collect()
        torch.cuda.empty_cache()

        # Periodic progress log every 200 samples
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            base_recs = list(base_done.values())
            mv_recs   = list(mv_done.values())
            aosp_recs = list(aosp_done.values())
            print(f"\n[CKPT {i+1}] Base={len(base_recs)}, MV={len(mv_recs)}, AOSP={len(aosp_recs)}")
            if base_recs:
                bm = summarise(base_recs)
                print(f"  Base F1={bm['f1']:.4f} | Lat={bm['avg_latency_ms']:.0f}ms | VRAM={bm['avg_peak_vram_gb']:.2f}GB")
            if mv_recs:
                mm = summarise(mv_recs)
                print(f"  MV   F1={mm['f1']:.4f} | Lat={mm['avg_latency_ms']:.0f}ms | VRAM={mm['avg_peak_vram_gb']:.2f}GB")
            if aosp_recs:
                am = summarise(aosp_recs)
                print(f"  AOSP F1={am['f1']:.4f} | Lat={am['avg_latency_ms']:.0f}ms | VRAM={am['avg_peak_vram_gb']:.2f}GB | Intv={sum(r.get('interventions',0) for r in aosp_recs)}")

    # ---- Merge and save final output ----
    base_recs = list(base_done.values())
    mv_recs   = list(mv_done.values())
    aosp_recs = list(aosp_done.values())

    final = {
        "model": args.model_path,
        "v_matrix": args.v_matrix,
        "n_samples": len(base_recs),
        "Base":           summarise(base_recs),
        "MajorityVoting": summarise(mv_recs),
        "AOSP":           summarise(aosp_recs),
    }

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n[DONE] Saved final output → {FINAL_OUTPUT}")
    print("\n=== FINAL SUMMARY ===")
    for cfg in ["Base", "MajorityVoting", "AOSP"]:
        m = final[cfg]
        print(f"{cfg:16s}: F1={m['f1']:.4f} | Acc={m['accuracy']:.4f} | "
              f"Lat={m['avg_latency_ms']:.0f}ms | AGL={m['avg_agl']:.1f} | VRAM={m['avg_peak_vram_gb']:.2f}GB")


if __name__ == "__main__":
    main()
