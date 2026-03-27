#!/usr/bin/env python3
"""
VisualWebBench Evaluation Pipeline for A-OSP
=============================================
Evaluates Qwen3-VL Base vs A-OSP on the VisualWebBench benchmark.

VisualWebBench tasks evaluated here:
  - Element Grounding (predict bounding box or element description)
  - Action Prediction (predict next action given a web screenshot)
  - WebQA (answer questions about web page content)

Dataset: visualwebbench/VisualWebBench (HuggingFace)
  Total: ~1536 samples across 7 tasks
  We run a 50-sample mini-batch per task subset.

Key Metrics:
  - Element Grounding: IoU ≥ 0.5 accuracy (coordinate prediction)
  - Action Prediction: Exact match accuracy
  - WebQA: F1 token overlap

Usage:
    python run_visualwebbench_eval.py --mode base  --limit 50
    python run_visualwebbench_eval.py --mode aosp  --limit 50
    python run_visualwebbench_eval.py --mode base  --task element_grounding --limit 50
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import (
    append_jsonl,
    load_completed_ids,
    load_jsonl,
    load_qwen2vl,
    aggressive_vram_cleanup,
    log_gpu_memory,
)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
DATA_DIR = PROJECT / "data" / "benchmarks" / "visualwebbench"
OUT_DIR = PROJECT / "logs" / "eval_results"

DEFAULT_MODEL = str(PROJECT / "models" / "Qwen3-VL-8B-Instruct")
DEFAULT_V_MATRIX = str(PROJECT / "models" / "V_matrix.pt")

# VisualWebBench task configs
TASK_CONFIGS = {
    "captioning": {
        "hf_split": "captioning",
        "prompt_template": "Briefly describe what you see on this webpage.",
        "metric": "f1",
    },
    "webqa": {
        "hf_split": "webqa",
        "prompt_template": "Answer this question about the webpage: {question}",
        "metric": "f1",
    },
    "heading_ocr": {
        "hf_split": "heading_ocr",
        "prompt_template": "What is the main heading or title on this webpage?",
        "metric": "exact_match",
    },
    "element_ocr": {
        "hf_split": "element_ocr",
        "prompt_template": "What text is displayed in the highlighted/boxed element?",
        "metric": "exact_match",
    },
    "element_grounding": {
        "hf_split": "element_grounding",
        "prompt_template": (
            "Locate the element described as: '{question}'. "
            "Provide its bounding box as [x1, y1, x2, y2] in pixel coordinates."
        ),
        "metric": "iou",
    },
    "action_prediction": {
        "hf_split": "action_prediction",
        "prompt_template": (
            "Given this webpage screenshot, what action would you take next to: {question}. "
            "Answer with one of: click, type, scroll, select, hover"
        ),
        "metric": "exact_match",
    },
    "action_grounding": {
        "hf_split": "action_grounding",
        "prompt_template": (
            "To perform the action '{question}', click on which element? "
            "Provide bounding box as [x1, y1, x2, y2]."
        ),
        "metric": "iou",
    },
}


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_visualwebbench(task: str, limit: int, output_dir: Path) -> list:
    """Download VisualWebBench samples for a given task split."""
    manifest_path = output_dir / f"vwb_{task}_manifest.jsonl"
    img_dir = output_dir / "images" / task

    if manifest_path.exists():
        records = load_jsonl(str(manifest_path))
        if records:
            print(f"[VWB] Loaded {len(records)} cached samples for task={task}")
            return records[:limit] if limit else records

    print(f"[VWB] Downloading task={task} from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(
        "visualwebbench/VisualWebBench",
        name=task,
        split="test",
    )

    img_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    n = min(limit, len(ds)) if limit else len(ds)

    for i, item in enumerate(ds):
        if i >= n:
            break
        img = item.get("image") or item.get("screenshot")
        img_path = img_dir / f"{task}_{i:05d}.png"
        if img is not None and not img_path.exists():
            img.save(str(img_path))

        record = {
            "question_id": i,
            "task": task,
            "image_path": str(img_path),
            "question": str(item.get("question", item.get("instruction", ""))),
            "ground_truth": str(item.get("answer", item.get("element_bbox", ""))),
            "image_size": list(img.size) if img else [],
        }
        records.append(record)
        with open(str(manifest_path), "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"[VWB] Saved {len(records)} samples → {manifest_path}")
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def token_f1(pred: str, gt: str) -> float:
    pred_toks = set(pred.lower().split())
    gt_toks = set(gt.lower().split())
    if not gt_toks:
        return float(pred == "")
    common = pred_toks & gt_toks
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(gt_toks)
    return 2 * prec * rec / (prec + rec)


def parse_bbox(text: str) -> list | None:
    """Extract [x1, y1, x2, y2] from model output."""
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= 4:
        return [float(x) for x in nums[:4]]
    return None


def compute_iou(box_pred: list, box_gt: list) -> float:
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_p = max(0, box_pred[2] - box_pred[0]) * max(0, box_pred[3] - box_pred[1])
    area_g = max(0, box_gt[2] - box_gt[0]) * max(0, box_gt[3] - box_gt[1])
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


def score_sample(pred: str, gt: str, metric: str) -> float:
    if metric == "exact_match":
        return float(pred.strip().lower() == gt.strip().lower())
    if metric == "f1":
        return token_f1(pred, gt)
    if metric == "iou":
        pred_box = parse_bbox(pred)
        gt_box = parse_bbox(gt)
        if pred_box is None or gt_box is None:
            return 0.0
        iou = compute_iou(pred_box, gt_box)
        # Element Grounding Accuracy = IoU ≥ 0.5
        return float(iou >= 0.5)
    return 0.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, records: list, output_path: Path,
                  task_cfg: dict, aosp_handle=None) -> None:
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    completed = load_completed_ids(str(output_path))
    if completed:
        print(f"[Resume] {len(completed)} already done.")
    pending = [r for r in records if r["question_id"] not in completed]
    print(f"[Inference] {len(pending)}/{len(records)} pending")

    metric_name = task_cfg["metric"]
    prompt_tmpl = task_cfg["prompt_template"]
    scores = []

    for idx, rec in enumerate(pending):
        img_path = rec["image_path"]
        question = rec["question"]

        try:
            img = Image.open(img_path).convert("RGB")
            # Resize if too large (>2MP)
            w, h = img.size
            if w * h > 2_000_000:
                scale = (2_000_000 / (w * h)) ** 0.5
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            prompt = prompt_tmpl.format(question=question) if "{question}" in prompt_tmpl else prompt_tmpl

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            if aosp_handle:
                aosp_handle.reset()

            t0 = time.time()
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            latency = time.time() - t0

            n_new = gen_ids.shape[1] - inputs["input_ids"].shape[1]
            prediction = processor.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            ic = aosp_handle.intervention_count if aosp_handle else 0
            s = score_sample(prediction, rec["ground_truth"], metric_name)
            scores.append(s)

            record = {
                "question_id": rec["question_id"],
                "task": rec["task"],
                "question": question,
                "prediction": prediction,
                "ground_truth": rec["ground_truth"],
                "score": round(s, 4),
                "metric": metric_name,
                "generation_length": n_new,
                "latency_s": round(latency, 3),
                "intervention_count": ic,
            }
            append_jsonl(str(output_path), record)

        except Exception as e:
            print(f"  [ERROR] qid={rec['question_id']}: {e}")
            record = {
                "question_id": rec["question_id"],
                "task": rec["task"],
                "score": 0.0,
                "metric": metric_name,
                "error": str(e),
            }
            append_jsonl(str(output_path), record)
        finally:
            try:
                del inputs, gen_ids
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pending):
            running_acc = sum(scores) / len(scores) if scores else 0
            print(f"  [{idx+1}/{len(pending)}] score={s:.3f} | "
                  f"running_acc={running_acc:.3f} | ic={ic}")

    all_done = load_jsonl(str(output_path))
    valid = [r for r in all_done if "score" in r and "error" not in r]
    if valid:
        acc = sum(r["score"] for r in valid) / len(valid)
        total_ic = sum(r.get("intervention_count", 0) for r in valid)
        avg_len = sum(r.get("generation_length", 0) for r in valid) / len(valid)
        print(f"\n[Result] task={rec['task']} | metric={metric_name} | "
              f"acc={acc:.4f} | n={len(valid)} | "
              f"total_interventions={total_ic} | avg_len={avg_len:.1f}")

        summary = {
            "task": rec["task"],
            "metric": metric_name,
            "accuracy": round(acc, 4),
            "n_samples": len(valid),
            "total_interventions": total_ic,
            "avg_interventions_per_sample": round(total_ic / len(valid), 3),
            "avg_generation_length": round(avg_len, 1),
        }
        summary_path = str(output_path).replace("_results.jsonl", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Result] Summary → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "aosp"], default="base")
    p.add_argument("--task", default="element_grounding",
                   choices=list(TASK_CONFIGS.keys()),
                   help="VisualWebBench task split to evaluate")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--v_matrix", default=DEFAULT_V_MATRIX)
    p.add_argument("--limit", type=int, default=50,
                   help="Max samples per task (50 = mini-batch)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--output_dir", default=str(OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    task_cfg = TASK_CONFIGS[args.task]
    tag = f"vwb_{args.task}_{args.mode}_q3_n{args.limit}"
    output_path = Path(args.output_dir) / f"{tag}_results.jsonl"

    print(f"\n{'='*60}")
    print(f"VisualWebBench | task={args.task} | mode={args.mode} | limit={args.limit}")
    print(f"Output → {output_path}")
    print(f"{'='*60}\n")

    records = download_visualwebbench(args.task, args.limit, DATA_DIR)
    if not records:
        print("[ERROR] No records downloaded. Check HF access.")
        sys.exit(1)

    print(f"[Model] Loading {args.model_path}")
    model, processor = load_qwen2vl(args.model_path)

    aosp_handle = None
    if args.mode == "aosp":
        sys.path.insert(0, str(PROJECT / "code"))
        from aosp_hook import apply_aosp_hook
        aosp_handle = apply_aosp_hook(
            model, args.v_matrix,
            alpha=args.alpha, mu=args.mu, beta=args.beta,
        )

    run_inference(model, processor, records, output_path, task_cfg, aosp_handle)

    if aosp_handle:
        aosp_handle.remove()
    del model, processor
    aggressive_vram_cleanup()


if __name__ == "__main__":
    main()
