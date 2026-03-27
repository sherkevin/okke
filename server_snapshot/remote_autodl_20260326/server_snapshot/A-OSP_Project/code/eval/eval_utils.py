"""
A-OSP Evaluation Pipeline — Shared Utilities
Provides: JSONL checkpoint I/O, metric computation, VRAM cleanup, model loading.
"""

import json
import os
import gc
import time
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# JSONL Append & Checkpoint
# ---------------------------------------------------------------------------

def append_jsonl(filepath: str, record: dict) -> None:
    """Atomic append of a single JSON record (one line) to *filepath*."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_ids(filepath: str) -> set:
    """Return the set of `question_id` values already persisted in *filepath*."""
    ids: set = set()
    if not os.path.exists(filepath):
        return ids
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ids.add(rec["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def load_jsonl(filepath: str) -> list[dict]:
    """Load a full JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_csv_summary(filepath: str, metrics: dict) -> None:
    """Write a flat dict of metrics as a single-row CSV (header + values)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import csv
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


# ---------------------------------------------------------------------------
# POPE Metrics
# ---------------------------------------------------------------------------

def compute_pope_metrics(results: list[dict]) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 and Yes-ratio from POPE results.
    Each record must have 'prediction' and 'ground_truth' with values 'yes'/'no'.
    """
    tp = fp = tn = fn = 0
    total_gen_len = 0
    count = 0

    for rec in results:
        pred_raw = rec.get("prediction", "").strip().lower()
        gt = rec.get("ground_truth", "").strip().lower()

        pred = "yes" if "yes" in pred_raw else "no"

        total_gen_len += rec.get("generation_length", 0)
        count += 1

        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    accuracy = (tp + tn) / max(count, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    yes_ratio = (tp + fp) / max(count, 1)
    agl = total_gen_len / max(count, 1)

    return {
        "total": count,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_ratio, 4),
        "agl": round(agl, 2),
    }


# ---------------------------------------------------------------------------
# VRAM Cleanup
# ---------------------------------------------------------------------------

def aggressive_vram_cleanup(*tensors_or_vars: Any) -> None:
    """Delete references, run Python GC, then flush CUDA cache."""
    for obj in tensors_or_vars:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(tag: str = "") -> None:
    """Print current GPU memory stats to stdout."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[GPU-MEM {tag}] allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")


# ---------------------------------------------------------------------------
# Unified VL Model Loader (BF16, Flash-Attn 2)
# Supports: Qwen2-VL, Qwen2.5-VL, Qwen3-VL
# ---------------------------------------------------------------------------

def load_qwen2vl(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load Qwen-VL family model in BF16.
    Auto-detects model generation (Qwen2-VL / Qwen2.5-VL / Qwen3-VL) from
    config.json and uses the correct class.
    Prefers Flash-Attention 2; falls back to SDPA if flash_attn is unavailable.
    Returns (model, processor).
    """
    from transformers import AutoProcessor, AutoConfig

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("[Loader] flash_attn not found — falling back to SDPA (PyTorch native)")

    print(f"[Loader] Loading model from {model_path} (attn={attn_impl}) ...")
    t0 = time.time()

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "").lower()

    if "qwen3_vl" in model_type or "qwen3-vl" in model_type:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
        print(f"[Loader] Detected Qwen3-VL (model_type={model_type})")
    elif "qwen2_5_vl" in model_type or "qwen2.5" in model_type:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
        print(f"[Loader] Detected Qwen2.5-VL (model_type={model_type})")
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model_cls = Qwen2VLForConditionalGeneration
        print(f"[Loader] Detected Qwen2-VL (model_type={model_type})")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        max_memory={0: "30GiB"},
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    n_layers = getattr(cfg, "num_hidden_layers", "?")
    hidden = getattr(cfg, "hidden_size", "?")
    print(f"[Loader] Model loaded in {time.time() - t0:.1f}s  "
          f"(layers={n_layers}, hidden={hidden})")
    log_gpu_memory("after_load")
    return model, processor


# ---------------------------------------------------------------------------
# Timing Utility
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.3f}s")
