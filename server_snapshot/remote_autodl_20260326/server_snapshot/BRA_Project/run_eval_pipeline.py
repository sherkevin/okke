#!/usr/bin/env python3
"""
Unified evaluation pipeline for the current internal BRA/TLRA route.

This runner still targets the model-coupled path implemented in
`bra_logits_processor.py`. The new strict universal sidecar route introduced in
`bra_universal_plugin.py` should be evaluated through a separate runtime that
does not depend on family-specific hidden-state adapters.

Usage:
    python run_eval_pipeline.py --method base --dataset pope --mini_test 50
    python run_eval_pipeline.py --method vcd  --dataset chair --mini_test 50
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from baseline_result_validator import (
    append_jsonl_record,
    compute_record_coverage,
    derive_artifacts,
    load_jsonl_records,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── paths ────────────────────────────────────────────────────────────
PROJECT   = Path("/root/autodl-tmp/BRA_Project")
MODEL_MAP = {
    "qwen2-vl-7b":  PROJECT / "models" / "Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b":  PROJECT / "models" / "Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b":  PROJECT / "models" / "Qwen3-VL-8B-Instruct",
    "qwen3-vl-4b":  PROJECT / "models" / "Qwen3-VL-4B-Instruct",
    "qwen3-vl-2b":  PROJECT / "models" / "Qwen3-VL-2B-Instruct",
    "llava-v1.5-7b": PROJECT / "models" / "llava-1.5-7b-hf",
    "instructblip-7b": PROJECT / "models" / "instructblip-vicuna-7b",
}
MODEL_FAMILY = {
    "qwen2-vl-7b": "qwen2vl",
    "qwen2.5-vl-7b": "qwen2.5vl",
    "qwen3-vl-8b": "qwen3vl",
    "qwen3-vl-4b": "qwen3vl",
    "qwen3-vl-2b": "qwen3vl",
    "llava-v1.5-7b": "llava",
    "instructblip-7b": "instructblip",
}
COCO_IMG  = PROJECT / "datasets" / "coco2014" / "val2014"
COCO_ANN  = PROJECT / "datasets" / "coco2014" / "annotations"
COCO_KARPATHY_TEST_FILE = COCO_ANN / "coco_karpathy_test.json"
COCO_KARPATHY_TEST_URL = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"
POPE_DIR  = PROJECT / "datasets" / "POPE" / "output" / "coco"
HALLUSIONBENCH_FILE = PROJECT / "datasets" / "HallusionBench_hf" / "data" / "image-00000-of-00001.parquet"
MMBENCH_FILE = PROJECT / "datasets" / "MMBench_EN_hf" / "data" / "dev-00000-of-00001-75b6649fb044d38b.parquet"
MME_DIR = PROJECT / "datasets" / "MME_hf" / "data"
LOG_DIR   = PROJECT / "logs" / "minitest"

DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_CHAIR_MAX_NEW_TOKENS = 512

# ─── arg parsing ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="qwen3-vl-8b",
        choices=[
            "qwen2-vl-7b",
            "qwen2.5-vl-7b",
            "qwen3-vl-8b",
            "qwen3-vl-4b",
            "qwen3-vl-2b",
            "llava-v1.5-7b",
            "instructblip-7b",
        ],
    )
    p.add_argument("--dataset",   default="pope", choices=["pope", "chair", "hallusionbench", "mmbench", "mme"])
    p.add_argument(
        "--method",
        default="ifcb",
        choices=[
            "base", "beam_search", "vcd", "opera", "ekko_legacy", "dola", "damo", "ifcb",
            "bra", "bra_v1", "bra_v2", "bra_zero", "bra_calib",
            "bra_meanpool", "bra_maxpool", "bra_max", "bra_no_vasm", "bra_v1_like",
            "tlra", "tlra_zero", "tlra_calib", "tlra_full",
            "tlra_meanpool", "tlra_max", "tlra_no_vasm", "tlra_v1_like", "tlra_adaptivetopk",
            "tlra_randomk",
            "bra_mask_binary", "bra_mask_entropy", "bra_mask_none",
        ],
    )
    p.add_argument("--mini_test", type=int, default=50)
    p.add_argument("--pope_split", default="random", choices=["random", "popular", "adversarial"])
    p.add_argument("--chair_prompt", default="Please describe this image in detail.")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--chair_max_new_tokens", type=int, default=DEFAULT_CHAIR_MAX_NEW_TOKENS)
    p.add_argument("--output_json", default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--vasm_artifact", default=None)
    p.add_argument("--projector_checkpoint", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_log_jsonl", default=None)
    p.add_argument("--sample_indices_json", default=None)
    return p.parse_args()

# ─── model loading ────────────────────────────────────────────────────

def load_model_and_processor(model_key: str, method: str):
    model_path = MODEL_MAP[model_key]
    logger.info(f"Loading model from {model_path} ...")

    attn_impl = "sdpa"
    if method in {"opera", "ekko_legacy"}:
        attn_impl = "eager"
        logger.info("%s requires output_attentions; using attn_implementation='eager'", method.upper())

    if model_key.startswith(("qwen2-vl", "qwen2.5-vl", "qwen3-vl")):
        from transformers import AutoProcessor

        if model_key.startswith("qwen3-vl"):
            from transformers import Qwen3VLForConditionalGeneration as QwenVLForConditionalGeneration
        elif model_key.startswith("qwen2.5-vl"):
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration

        model = QwenVLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        processor = AutoProcessor.from_pretrained(str(model_path))
    elif model_key == "llava-v1.5-7b":
        from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor

        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        try:
            processor = AutoProcessor.from_pretrained(str(model_path))
        except Exception as exc:
            logger.warning("Falling back to slow LLaVA processor load: %s", exc)
            image_processor = AutoImageProcessor.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
            processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    elif model_key == "instructblip-7b":
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        model = InstructBlipForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = InstructBlipProcessor.from_pretrained(str(model_path))
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    model.eval()
    return model, processor

# ─── input builder ────────────────────────────────────────────────────

def build_single_input(processor, image_path: str, question: str, device="cuda"):
    image = image_path if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
    processor_type = type(processor).__name__.lower()
    if "instructblip" in processor_type:
        inputs = processor(images=image, text=question, return_tensors="pt")
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "llava" in processor_type:
        if hasattr(processor, "apply_chat_template"):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            text = f"USER: <image>\n{question} ASSISTANT:"
        inputs = processor(images=image, text=text, return_tensors="pt")
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def extract_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    matches = re.findall(r"\b(yes|no)\b", t)
    return matches[0] if matches else "unknown"


def extract_letter(text: str) -> str:
    t = text.strip()
    match = re.match(r"^([A-D])", t)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-D])\b", t)
    if match:
        return match.group(1)
    return t[:1].upper() if t else ""


def _resolve_output_json_path(args, ts: str | None = None) -> Path:
    if args.output_json:
        return Path(args.output_json)
    stamp = args.run_id or ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.method, args.dataset]
    if args.dataset == "pope":
        parts.append(args.pope_split)
    return LOG_DIR / f"{'_'.join(parts)}_{stamp}.json"


def _resolve_sample_log_path(args, output_path: Path) -> Path:
    if args.sample_log_jsonl:
        return Path(args.sample_log_jsonl)
    return derive_artifacts(output_path).sample_jsonl


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _load_existing_output_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_existing_records(path: Path) -> dict[int, dict]:
    records_by_index: dict[int, dict] = {}
    for record in load_jsonl_records(path):
        try:
            sample_index = int(record["sample_index"])
        except Exception:
            continue
        records_by_index[sample_index] = record
    return records_by_index


def _reset_artifacts(output_path: Path, sample_log_path: Path) -> None:
    for path in (output_path, sample_log_path, derive_artifacts(output_path).validation_json):
        if path.exists():
            path.unlink()


def _runtime_profile(args) -> dict:
    return {
        "dataset": args.dataset,
        "method": args.method,
        "max_new_tokens": _resolve_max_new_tokens(args.dataset, args),
        "checkpoint_every": int(args.checkpoint_every),
        "resume": bool(args.resume),
        "answer_mode": "yes_no" if args.dataset in {"pope", "hallusionbench", "mme"} else "letter" if args.dataset == "mmbench" else "caption",
    }


def _get_requested_sample_indices(args) -> list[int] | None:
    indices = getattr(args, "sample_indices", None)
    if indices is not None:
        return [int(idx) for idx in indices]
    path = getattr(args, "sample_indices_json", None)
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("sample_indices_json must contain a JSON list of integers")
    return [int(idx) for idx in payload]


def _subset_with_sample_indices(items: list, sample_indices: list[int] | None, limit: int) -> list:
    if sample_indices:
        selected = []
        for idx in sample_indices:
            if 0 <= int(idx) < len(items):
                selected.append(items[int(idx)])
        return selected
    return items[:limit]


def _build_run_snapshot(
    args,
    *,
    output_path: Path,
    sample_log_path: Path,
    status: str,
    metrics: dict | None = None,
    completed_samples: int = 0,
    target_samples: int | None = None,
    attempted_samples: int | None = None,
    expected_n: int | None = None,
    complete: bool = False,
    started_at: str | None = None,
    ended_at: str | None = None,
    runtime_profile: dict | None = None,
    validation: dict | None = None,
    error: dict | None = None,
) -> dict:
    timestamp = datetime.now().isoformat()
    payload = {
        "status": status,
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model,
        "model_family": MODEL_FAMILY.get(args.model, "unknown"),
        "run_id": args.run_id,
        "output_json": str(output_path),
        "sample_log_jsonl": str(sample_log_path),
        "artifact_paths": {
            "output_json": str(output_path),
            "sample_log_jsonl": str(sample_log_path),
            "validation_json": str(derive_artifacts(output_path).validation_json),
        },
        "completed_samples": int(completed_samples),
        "target_samples": None if target_samples is None else int(target_samples),
        "attempted_n": None if attempted_samples is None else int(attempted_samples),
        "expected_n": None if expected_n is None else int(expected_n),
        "complete": bool(complete),
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": None,
        "timestamp": timestamp,
    }
    if started_at and ended_at:
        try:
            payload["elapsed_seconds"] = round(
                (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds(),
                3,
            )
        except Exception:
            payload["elapsed_seconds"] = None
    if args.dataset == "pope":
        payload["pope_split"] = args.pope_split
    if metrics:
        payload.update(metrics)
    if runtime_profile is not None:
        payload["runtime_profile"] = runtime_profile
    if validation is not None:
        payload["validation"] = validation
    if error is not None:
        payload["error"] = error
    return payload


def _persist_run_snapshot(
    args,
    *,
    output_path: Path,
    sample_log_path: Path,
    status: str,
    metrics: dict | None = None,
    completed_samples: int = 0,
    target_samples: int | None = None,
    attempted_samples: int | None = None,
    expected_n: int | None = None,
    complete: bool = False,
    started_at: str | None = None,
    ended_at: str | None = None,
    runtime_profile: dict | None = None,
    validation: dict | None = None,
    error: dict | None = None,
) -> None:
    payload = _build_run_snapshot(
        args,
        output_path=output_path,
        sample_log_path=sample_log_path,
        status=status,
        metrics=metrics,
        completed_samples=completed_samples,
        target_samples=target_samples,
        attempted_samples=attempted_samples,
        expected_n=expected_n,
        complete=complete,
        started_at=started_at,
        ended_at=ended_at,
        runtime_profile=runtime_profile,
        validation=validation,
        error=error,
    )
    _write_json_atomic(output_path, payload)
    validation_path = derive_artifacts(output_path).validation_json
    validation_payload = validation if validation is not None else {
        "status": status,
        "complete": bool(complete),
        "expected_n": expected_n,
        "attempted_n": attempted_samples,
    }
    _write_json_atomic(validation_path, validation_payload)

# ─── per-method setup (stateful objects that persist across samples) ──

_opera_instance = None
_ekko_legacy_instance = None
_dola_instance = None
_damo_instance = None

def setup_persistent(method: str, model, processor):
    """Install hooks that live across all samples. Called once."""
    global _opera_instance, _ekko_legacy_instance, _dola_instance, _damo_instance

    if method == "opera":
        from baseline_processors import OPERALogitsProcessor
        _opera_instance = OPERALogitsProcessor(model, penalty_weight=1.0, scale_factor=50.0)
        logger.info("OPERA processor installed.")
    elif method == "ekko_legacy":
        from baseline_processors import EKKOLegacyLogitsProcessor
        _ekko_legacy_instance = EKKOLegacyLogitsProcessor(model, penalty_weight=1.0, scale_factor=50.0)
        logger.info("EKKO legacy processor installed.")
    elif method == "dola":
        from baseline_processors import DoLaLogitsProcessor
        _dola_instance = DoLaLogitsProcessor(model)
        logger.info("DoLa processor installed.")
    elif method == "damo":
        from baseline_processors import DAMOLogitsProcessor
        _damo_instance = DAMOLogitsProcessor(model)
        logger.info("DAMO processor installed.")

def teardown_persistent(method: str):
    global _opera_instance, _ekko_legacy_instance, _dola_instance, _damo_instance
    if _opera_instance is not None:
        _opera_instance.cleanup()
        _opera_instance = None
    if _ekko_legacy_instance is not None:
        _ekko_legacy_instance.cleanup()
        _ekko_legacy_instance = None
    if _dola_instance is not None:
        _dola_instance.cleanup()
        _dola_instance = None
    if _damo_instance is not None:
        _damo_instance.cleanup()
        _damo_instance = None

def get_logits_processors(method: str, model, inputs):
    """
    Return list of LogitsProcessors for this sample.
    VCD creates a new processor per sample (each image is different).
    OPERA/DoLa use persistent hooks.
    """
    if method == "vcd":
        from baseline_processors import VCDLogitsProcessor
        vcd = VCDLogitsProcessor.prepare(model, inputs, cd_alpha=1.0, cd_beta=0.1, noise_step=500)
        return [vcd]
    elif method == "opera" and _opera_instance is not None:
        return [_opera_instance]
    elif method == "ekko_legacy" and _ekko_legacy_instance is not None:
        return [_ekko_legacy_instance]
    elif method == "dola" and _dola_instance is not None:
        return [_dola_instance]
    elif method == "damo" and _damo_instance is not None:
        return [_damo_instance]
    return []


def _make_ifcb_config(args):
    from ifcb import IFCBConfig

    frontier_k = 4 if args.dataset == "pope" else 5
    return IFCBConfig(alpha=6.0, frontier_k=frontier_k, probe_topk=frontier_k)


def _append_generation_inputs(inputs, next_token: torch.LongTensor):
    updated = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            updated[key] = value
            continue
        if key == "input_ids":
            updated[key] = torch.cat([value, next_token.to(value.device)], dim=1)
        elif key == "attention_mask":
            ones = torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)
            updated[key] = torch.cat([value, ones], dim=1)
        else:
            updated[key] = value
    if "attention_mask" not in updated:
        updated["attention_mask"] = torch.ones_like(updated["input_ids"])
    return updated


def _generate_with_ifcb(model, processor, image_path, question, timer, args):
    from ifcb import create_ifcb_processor

    inputs = build_single_input(processor, image_path, question)
    tokenizer = getattr(processor, "tokenizer", processor)
    ifcb_proc = create_ifcb_processor(
        model=model,
        tokenizer=tokenizer,
        model_family=MODEL_FAMILY.get(args.model, "unknown"),
        image_token_id=getattr(model.config, "image_token_index", 32000),
        config=_make_ifcb_config(args),
    )
    generated = []
    max_new_tokens = 1 if args.dataset in {"pope", "hallusionbench", "mmbench", "mme"} else _resolve_max_new_tokens(args.dataset, args)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    try:
        timer.start()
        for step_idx in range(max_new_tokens):
            next_token, _ = ifcb_proc.step(model_inputs=inputs, dataset=args.dataset, step_idx=step_idx)
            generated.append(int(next_token[0, 0].item()))
            inputs = _append_generation_inputs(inputs, next_token)
            if eos_token_id is not None and generated[-1] == int(eos_token_id):
                break
        elapsed = timer.stop()
        text = processor.decode(generated, skip_special_tokens=True).strip()
        return text, len(generated), elapsed, ifcb_proc.get_stats(), ifcb_proc.get_audit_log()
    finally:
        ifcb_proc.remove()

def cleanup_sample_processors(procs):
    for p in procs:
        if hasattr(p, 'cleanup') and type(p).__name__ == 'VCDLogitsProcessor':
            p.cleanup()


def _append_sample_record(sample_log_path: Path, payload: dict) -> None:
    append_jsonl_record(sample_log_path, payload)


def _recover_pope_state(records_by_index: dict[int, dict]) -> tuple[int, int, int, int, list[int], list[float], list[dict], list[dict], list[dict]]:
    tp = fp = fn = tn = 0
    gen_lengths: list[int] = []
    latencies: list[float] = []
    method_stats_all: list[dict] = []
    sample_audits: list[dict] = []
    errors: list[dict] = []
    for idx in sorted(records_by_index):
        record = records_by_index[idx]
        if record.get("status") != "ok":
            errors.append({"sample": idx, "error": record.get("error", "recorded_error"), "traceback": record.get("traceback", "")})
            continue
        label = str(record.get("label", "")).strip().lower()
        pred = str(record.get("normalized_prediction", "unknown")).strip().lower()
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        else:
            tn += 1
        gl = int(record.get("gen_length", 0) or 0)
        gen_lengths.append(gl)
        elapsed_ms = float(record.get("elapsed_ms", 0.0) or 0.0)
        if gl > 0 and elapsed_ms > 0:
            latencies.append(elapsed_ms / gl)
        if record.get("method_stats") is not None:
            method_stats_all.append(record["method_stats"])
        if record.get("method_audit") and len(sample_audits) < 3:
            sample_audits.append(
                {
                    "sample_index": idx,
                    "prediction": pred,
                    "label": label,
                    "text_preview": str(record.get("prediction_text", ""))[:160],
                    "audit": record["method_audit"][:5],
                }
            )
    return tp, fp, fn, tn, gen_lengths, latencies, method_stats_all, sample_audits, errors


def _recover_generic_state(records_by_index: dict[int, dict]) -> tuple[list[str], list[str], list[str], list[int], list[float], list[dict], list[dict], list[dict]]:
    predictions: list[str] = []
    answers: list[str] = []
    categories: list[str] = []
    gen_lengths: list[int] = []
    latencies: list[float] = []
    method_stats_all: list[dict] = []
    sample_audits: list[dict] = []
    errors: list[dict] = []
    for idx in sorted(records_by_index):
        record = records_by_index[idx]
        if record.get("status") != "ok":
            errors.append({"sample": idx, "error": record.get("error", "recorded_error"), "traceback": record.get("traceback", "")})
            continue
        predictions.append(str(record.get("prediction_text", "")))
        answers.append(str(record.get("answer", "")))
        categories.append(str(record.get("category", "unknown")))
        gl = int(record.get("gen_length", 0) or 0)
        gen_lengths.append(gl)
        elapsed_ms = float(record.get("elapsed_ms", 0.0) or 0.0)
        if gl > 0 and elapsed_ms > 0:
            latencies.append(elapsed_ms / gl)
        if record.get("method_stats") is not None:
            method_stats_all.append(record["method_stats"])
        if record.get("method_audit") and len(sample_audits) < 3:
            sample_audits.append(
                {
                    "sample_index": idx,
                    "prediction_preview": str(record.get("prediction_text", ""))[:160],
                    "answer": record.get("answer", ""),
                    "audit": record["method_audit"][:5],
                }
            )
    return predictions, answers, categories, gen_lengths, latencies, method_stats_all, sample_audits, errors


def _recover_chair_state(records_by_index: dict[int, dict]) -> tuple[list[dict], list[int], list[float], list[dict], list[dict], list[dict]]:
    captions: list[dict] = []
    gen_lengths: list[int] = []
    latencies: list[float] = []
    method_stats_all: list[dict] = []
    sample_audits: list[dict] = []
    errors: list[dict] = []
    for idx in sorted(records_by_index):
        record = records_by_index[idx]
        if record.get("status") != "ok":
            errors.append({"sample": idx, "error": record.get("error", "recorded_error"), "traceback": record.get("traceback", "")})
            continue
        captions.append({"image": record.get("image", ""), "caption": str(record.get("prediction_text", ""))})
        gl = int(record.get("gen_length", 0) or 0)
        gen_lengths.append(gl)
        elapsed_ms = float(record.get("elapsed_ms", 0.0) or 0.0)
        if gl > 0 and elapsed_ms > 0:
            latencies.append(elapsed_ms / gl)
        if record.get("method_stats") is not None:
            method_stats_all.append(record["method_stats"])
        if record.get("method_audit") and len(sample_audits) < 3:
            sample_audits.append(
                {
                    "image": record.get("image", ""),
                    "caption_preview": str(record.get("prediction_text", ""))[:160],
                    "audit": record["method_audit"][:5],
                }
            )
    return captions, gen_lengths, latencies, method_stats_all, sample_audits, errors

# ─── timing ───────────────────────────────────────────────────────────

class GPUTimer:
    def __init__(self):
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt   = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.start_evt.record()
    def stop(self):
        self.end_evt.record()
        torch.cuda.synchronize()
        return self.start_evt.elapsed_time(self.end_evt)

# ─── generate one sample ──────────────────────────────────────────────

def generate_one(model, processor, method, image_path, question, timer, args):
    """Returns (text, gen_length, elapsed_ms, bra_stats, bra_audit) or raises on error."""
    if method == "ifcb":
        return _generate_with_ifcb(model, processor, image_path, question, timer, args)

    inputs = build_single_input(processor, image_path, question)
    procs = get_logits_processors(method, model, inputs)
    bra_cleanup = None
    if method.startswith("bra") or method == "bra":
        from bra_logits_processor import create_bra_processor, make_bra_config
        from bra_operator_multi import detect_adapter

        adapter = detect_adapter(model)
        tokenizer = getattr(processor, "tokenizer", processor)
        video_grid_thw = inputs.get("video_grid_thw")
        cfg = _make_bra_eval_config(method, args)
        extractor, bra_proc = create_bra_processor(
            model,
            adapter,
            inputs["input_ids"],
            config=cfg,
            tokenizer=tokenizer,
            video_grid_thw=video_grid_thw,
        )
        procs = [bra_proc]
        bra_cleanup = (extractor, bra_proc)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=_resolve_max_new_tokens(args.dataset, args),
        do_sample=False,
    )
    if method == "beam_search":
        gen_kwargs["num_beams"] = 5
        gen_kwargs["early_stopping"] = True
    if procs:
        gen_kwargs["logits_processor"] = procs
    if method in {"opera", "ekko_legacy"}:
        gen_kwargs["output_attentions"] = True

    try:
        timer.start()
        with torch.no_grad():
            out_ids = model.generate(**gen_kwargs)
        elapsed = timer.stop()

        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True).strip()
        method_stats = bra_cleanup[1].get_stats() if bra_cleanup is not None else None
        method_audit = bra_cleanup[1].get_audit_log() if bra_cleanup is not None else None
        return text, len(gen_ids), elapsed, method_stats, method_audit
    finally:
        cleanup_sample_processors(procs)
        if bra_cleanup is not None:
            extractor, bra_proc = bra_cleanup
            extractor.remove()
            bra_proc.reset()


def _make_bra_eval_config(method: str, args):
    from bra_logits_processor import make_bra_config

    if method in {"bra", "bra_v2", "bra_zero"}:
        return make_bra_config("bra_zero", vasm_artifact_path=args.vasm_artifact)
    if method in {"tlra", "tlra_zero"}:
        return make_bra_config("tlra_zero", vasm_artifact_path=args.vasm_artifact)
    if method in {"bra_v1", "bra_v1_like", "tlra_v1_like"}:
        return make_bra_config(method, vasm_artifact_path=args.vasm_artifact)
    if method in {"bra_calib", "tlra_calib"}:
        return make_bra_config(
            method,
            vasm_artifact_path=args.vasm_artifact,
            projector_checkpoint=args.projector_checkpoint,
        )
    if method in {"tlra_full", "tlra_adaptivetopk", "tlra_randomk", "tlra_meanpool", "tlra_no_vasm", "tlra_max"}:
        return make_bra_config(
            method,
            vasm_artifact_path=args.vasm_artifact,
            projector_checkpoint=args.projector_checkpoint,
        )
    if method in {"bra_meanpool"}:
        return make_bra_config("ablation_meanpool", vasm_artifact_path=args.vasm_artifact)
    if method in {"bra_maxpool", "bra_max"}:
        return make_bra_config("ablation_maxpool", vasm_artifact_path=args.vasm_artifact)
    if method in {"bra_no_vasm"}:
        return make_bra_config("bra_no_vasm", vasm_artifact_path=args.vasm_artifact)
    if method == "bra_mask_binary":
        return make_bra_config("bra_zero", mask_variant="binary_mask", vasm_artifact_path=args.vasm_artifact)
    if method == "bra_mask_entropy":
        return make_bra_config("bra_zero", mask_variant="entropy_mask", vasm_artifact_path=args.vasm_artifact)
    if method == "bra_mask_none":
        return make_bra_config("bra_zero", mask_variant="no_mask", vasm_artifact_path=args.vasm_artifact)
    raise ValueError(f"Unsupported BRA method: {method}")


def _resolve_max_new_tokens(dataset: str, args) -> int:
    if dataset == "chair":
        return max(args.chair_max_new_tokens, args.max_new_tokens)
    return args.max_new_tokens


def _build_notes(dataset: str, agl: float, max_new_tokens: int, trigger_rate: float | None = None):
    notes = []
    if dataset == "chair" and agl >= 0.9 * max_new_tokens:
        notes.append("chair_agl_near_cap")
    if trigger_rate is not None and trigger_rate >= 0.95:
        notes.append("high_trigger_rate")
    return notes


def _collect_runtime_notes(method: str) -> list[str]:
    notes: list[str] = []
    if method == "opera" and _opera_instance is not None and getattr(_opera_instance, "_attn_error_reported", False):
        notes.append("opera_attention_unavailable")
    if method == "ekko_legacy" and _ekko_legacy_instance is not None and getattr(_ekko_legacy_instance, "_attn_error_reported", False):
        notes.append("ekko_legacy_attention_unavailable")
    return notes


def _normalize_text_value(text) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parquet_image(row, image_cols):
    from io import BytesIO

    for col in image_cols:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, dict) and "bytes" in val:
            return Image.open(BytesIO(val["bytes"])).convert("RGB")
        if isinstance(val, bytes):
            return Image.open(BytesIO(val)).convert("RGB")
    return None


def _load_hallusionbench_samples(limit: int, sample_indices: list[int] | None = None):
    import pandas as pd

    df = pd.read_parquet(HALLUSIONBENCH_FILE)
    if sample_indices:
        df = df.iloc[[idx for idx in sample_indices if 0 <= idx < len(df)]]
    else:
        df = df.head(limit)
    image_cols = [c for c in df.columns if "image" in c.lower()]
    samples = []
    for _, row in df.iterrows():
        image = _parquet_image(row, image_cols)
        if image is None:
            continue
        prompt = f"{str(row.get('question', ''))} Answer yes or no."
        raw_gt = str(row.get("gt_answer", "")).strip()
        answer = {"1": "yes", "0": "no"}.get(raw_gt, raw_gt.lower())
        samples.append({"image": image, "prompt": prompt, "answer": answer})
    return samples


def _load_mmbench_samples(limit: int, sample_indices: list[int] | None = None):
    import pandas as pd

    df = pd.read_parquet(MMBENCH_FILE)
    if sample_indices:
        df = df.iloc[[idx for idx in sample_indices if 0 <= idx < len(df)]]
    else:
        df = df.head(limit)
    image_cols = [c for c in df.columns if "image" in c.lower()]
    samples = []
    for _, row in df.iterrows():
        image = _parquet_image(row, image_cols)
        if image is None:
            continue
        hint = str(row.get("hint", "")).strip()
        question = str(row.get("question", "")).strip()
        choices = []
        for letter in ("A", "B", "C", "D"):
            value = row.get(letter)
            if value is not None and str(value).strip():
                choices.append(f"{letter}. {value}")
        prompt = "\n".join(part for part in [hint, question, *choices, "Answer with the letter only."] if part)
        samples.append({"image": image, "prompt": prompt, "answer": str(row.get("answer", "")).strip().upper()})
    return samples


def _load_mme_samples(limit: int, sample_indices: list[int] | None = None):
    import pandas as pd

    requested = list(sample_indices or [])
    requested_set = set(requested)
    global_index = 0
    samples = []
    for parquet_file in sorted(MME_DIR.glob("*.parquet")):
        if not requested and len(samples) >= limit:
            break
        df = pd.read_parquet(parquet_file)
        image_cols = [c for c in df.columns if "image" in c.lower()]
        for _, row in df.iterrows():
            if not requested and len(samples) >= limit:
                break
            if requested and global_index not in requested_set:
                global_index += 1
                continue
            image = _parquet_image(row, image_cols)
            if image is None:
                global_index += 1
                continue
            samples.append({
                "image": image,
                "prompt": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip().lower(),
                "category": str(row.get("category", "unknown")).strip(),
            })
            global_index += 1
        if requested and len(samples) >= len(requested):
            break
    return samples


def _compute_generic_accuracy(predictions, answers, answer_type: str):
    correct = 0
    for prediction, answer in zip(predictions, answers):
        if answer_type == "yes_no":
            pred_value = extract_yes_no(prediction)
            gold_value = str(answer).strip().lower()
        elif answer_type == "letter":
            pred_value = extract_letter(prediction).upper()
            gold_value = str(answer).strip().upper()
        else:
            pred_value = _normalize_text_value(prediction)
            gold_value = _normalize_text_value(answer)
        correct += int(pred_value == gold_value)
    total = len(predictions)
    return {"accuracy": round(correct / max(total, 1), 4)}


def _compute_mme_metrics(predictions, answers, categories):
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    for prediction, answer, category in zip(predictions, answers, categories):
        pred_value = extract_yes_no(prediction)
        gold_value = str(answer).strip().lower()
        category_total[category] += 1
        if pred_value == gold_value:
            category_correct[category] += 1
    total_correct = sum(category_correct.values())
    total_samples = sum(category_total.values())
    perception_score = 0.0
    for category, total in category_total.items():
        perception_score += (category_correct[category] / max(total, 1)) * 100.0
    return {
        "accuracy": round(total_correct / max(total_samples, 1), 4),
        "perception_score": round(perception_score, 4),
    }


def run_generic_benchmark(model, processor, method, args, *, records_by_index: dict[int, dict] | None = None, sample_log_path: Path | None = None, persist_callback=None):
    sample_indices = _get_requested_sample_indices(args)
    if args.dataset == "hallusionbench":
        samples = _load_hallusionbench_samples(args.mini_test, sample_indices=sample_indices)
        answer_type = "yes_no"
    elif args.dataset == "mmbench":
        samples = _load_mmbench_samples(args.mini_test, sample_indices=sample_indices)
        answer_type = "letter"
    elif args.dataset == "mme":
        samples = _load_mme_samples(args.mini_test, sample_indices=sample_indices)
        answer_type = "yes_no"
    else:
        raise ValueError(f"Unsupported generic dataset: {args.dataset}")

    records_by_index = records_by_index or {}
    predictions, answers, categories, gen_lengths, latencies, method_stats_all, sample_audits, errors = _recover_generic_state(records_by_index)
    timer = GPUTimer()
    torch.cuda.reset_peak_memory_stats()
    checkpoint_every = max(int(args.checkpoint_every), 1)
    expected_n = len(samples)

    if persist_callback is not None:
        initial_metrics = {
            "dataset": args.dataset,
            "method": method,
            "model": args.model,
            "model_family": MODEL_FAMILY.get(args.model, "unknown"),
            "n_samples": len(predictions),
            "sample_count": len(predictions),
            "n_errors": len(errors),
        }
        persist_callback(
            initial_metrics,
            completed_samples=len(predictions),
            attempted_samples=len(records_by_index),
            target_samples=expected_n,
            expected_n=expected_n,
            status="partial",
        )

    for i, sample in enumerate(samples):
        if i in records_by_index:
            continue
        try:
            text, gl, ms, method_stats, method_audit = generate_one(
                model,
                processor,
                method,
                sample["image"],
                sample["prompt"],
                timer,
                args,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{method}] sample {i} FAILED:\n{tb}")
            errors.append({"sample": i, "error": str(e), "traceback": tb})
            if sample_log_path is not None:
                _append_sample_record(
                    sample_log_path,
                    {
                        "sample_index": i,
                        "status": "error",
                        "dataset": args.dataset,
                        "answer": sample["answer"],
                        "category": sample.get("category", "unknown"),
                        "error": str(e),
                        "traceback": tb,
                    },
                )
            continue
        predictions.append(text)
        answers.append(sample["answer"])
        categories.append(sample.get("category", "unknown"))
        gen_lengths.append(gl)
        if gl > 0:
            latencies.append(ms / gl)
        if method_stats is not None:
            method_stats_all.append(method_stats)
        if method_audit and len(sample_audits) < 3:
            sample_audits.append({
                "sample_index": i,
                "prediction_preview": text[:160],
                "answer": sample["answer"],
                "audit": method_audit[:5],
            })
        if sample_log_path is not None:
            _append_sample_record(
                sample_log_path,
                {
                    "sample_index": i,
                    "status": "ok",
                    "dataset": args.dataset,
                    "prediction_text": text,
                    "answer": sample["answer"],
                    "category": sample.get("category", "unknown"),
                    "gen_length": gl,
                    "elapsed_ms": round(float(ms), 4),
                    "method_stats": method_stats,
                    "method_audit": method_audit[:5] if method_audit else [],
                },
            )
        if i < 3 or (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] len={gl}  {text[:80]}")
        if persist_callback is not None and (((i + 1) % checkpoint_every == 0) or (i + 1) == len(samples)):
            partial_metrics = {
                "dataset": args.dataset,
                "method": method,
                "model": args.model,
                "model_family": MODEL_FAMILY.get(args.model, "unknown"),
                "n_samples": len(predictions),
                "sample_count": len(predictions),
                "n_errors": len(errors),
            }
            persist_callback(
                partial_metrics,
                completed_samples=len(predictions),
                attempted_samples=len(_load_existing_records(sample_log_path)) if sample_log_path is not None else len(predictions) + len(errors),
                target_samples=expected_n,
                expected_n=expected_n,
                status="partial",
            )

    vram_peak = torch.cuda.max_memory_allocated()
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)
    metrics = {
        "dataset": args.dataset,
        "method": method,
        "model": args.model,
        "model_family": MODEL_FAMILY.get(args.model, "unknown"),
        "n_samples": len(predictions),
        "sample_count": len(predictions),
        "n_errors": len(errors),
        "agl": round(agl, 2),
        "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
        "itl_ms_per_token": round(itl, 2),
        "tpot_ms_per_token": round(itl, 2),
        "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
        "peak_vram_gb": round(vram_peak / 1e9, 3),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3] if errors else [],
        "expected_n": expected_n,
        "attempted_n": len(_load_existing_records(sample_log_path)) if sample_log_path is not None else len(predictions) + len(errors),
    }
    if args.dataset == "mme":
        metrics.update(_compute_mme_metrics(predictions, answers, categories))
    else:
        metrics.update(_compute_generic_accuracy(predictions, answers, answer_type))
    if method_stats_all:
        if method == "ifcb":
            metrics.update(_aggregate_ifcb_stats(method_stats_all))
        else:
            metrics.update(_aggregate_bra_stats(method_stats_all))
    metrics["notes"] = _build_notes(args.dataset, agl, _resolve_max_new_tokens(args.dataset, args), metrics.get("intervention_rate"))
    metrics["notes"].extend(_collect_runtime_notes(method))
    if sample_audits:
        metrics["sample_audits"] = sample_audits
    return metrics

# ─── POPE ─────────────────────────────────────────────────────────────

def _build_pope_metrics(
    *,
    args,
    method: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    gen_lengths: list[int],
    latencies: list[float],
    method_stats_all: list[dict],
    sample_audits: list[dict],
    errors: list[dict],
    vram_peak: int,
) -> dict:
    total = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(total, 1)
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)

    metrics = {
        "dataset": "pope", "pope_split": args.pope_split,
        "method": method, "model": args.model, "model_family": MODEL_FAMILY.get(args.model, "unknown"),
        "n_samples": total, "sample_count": total, "n_errors": len(errors),
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "agl": round(agl, 2),
        "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
        "itl_ms_per_token": round(itl, 2),
        "tpot_ms_per_token": round(itl, 2),
        "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
        "peak_vram_gb": round(vram_peak / 1e9, 3),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3] if errors else [],
    }
    if method_stats_all:
        if method == "ifcb":
            metrics.update(_aggregate_ifcb_stats(method_stats_all))
        else:
            metrics.update(_aggregate_bra_stats(method_stats_all))
    metrics["notes"] = _build_notes(
        "pope",
        agl,
        _resolve_max_new_tokens("pope", args),
        metrics.get("intervention_rate"),
    )
    metrics["notes"].extend(_collect_runtime_notes(method))
    if sample_audits:
        metrics["sample_audits"] = sample_audits
    return metrics


def run_pope(model, processor, method, args, *, records_by_index: dict[int, dict] | None = None, sample_log_path: Path | None = None, persist_callback=None):
    split_file = POPE_DIR / f"coco_pope_{args.pope_split}.json"
    logger.info(f"POPE split: {split_file}")

    raw_samples = [json.loads(l) for l in open(split_file)]
    samples = _subset_with_sample_indices(raw_samples, _get_requested_sample_indices(args), args.mini_test)

    records_by_index = records_by_index or {}
    tp, fp, fn, tn, gen_lengths, latencies, method_stats_all, sample_audits, errors = _recover_pope_state(records_by_index)
    timer = GPUTimer()
    torch.cuda.reset_peak_memory_stats()
    checkpoint_every = max(int(args.checkpoint_every), 1)
    expected_n = len(samples)

    if persist_callback is not None:
        initial_metrics = _build_pope_metrics(
            args=args,
            method=method,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            gen_lengths=gen_lengths,
            latencies=latencies,
            method_stats_all=method_stats_all,
            sample_audits=sample_audits,
            errors=errors,
            vram_peak=0,
        )
        persist_callback(
            initial_metrics,
            completed_samples=tp + tn + fp + fn,
            attempted_samples=len(records_by_index),
            target_samples=expected_n,
            expected_n=expected_n,
            status="partial",
        )

    for i, s in enumerate(samples):
        if i in records_by_index:
            continue
        img = str(COCO_IMG / s["image"])
        label = s["label"].strip().lower()
        try:
            text, gl, ms, method_stats, method_audit = generate_one(model, processor, method, img, s["text"], timer, args)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{method}] sample {i} FAILED:\n{tb}")
            errors.append({"sample": i, "error": str(e), "traceback": tb})
            if sample_log_path is not None:
                _append_sample_record(
                    sample_log_path,
                    {
                        "sample_index": i,
                        "status": "error",
                        "dataset": "pope",
                        "pope_split": args.pope_split,
                        "image": s["image"],
                        "label": label,
                        "error": str(e),
                        "traceback": tb,
                    },
                )
            continue

        gen_lengths.append(gl)
        if gl > 0:
            latencies.append(ms / gl)
        if method_stats is not None:
            method_stats_all.append(method_stats)

        pred = extract_yes_no(text)
        if method_audit and len(sample_audits) < 3:
            sample_audits.append({
                "sample_index": i,
                "prediction": pred,
                "label": label,
                "text_preview": text[:160],
                "audit": method_audit[:5],
            })
        if sample_log_path is not None:
            _append_sample_record(
                sample_log_path,
                {
                    "sample_index": i,
                    "status": "ok",
                    "dataset": "pope",
                    "pope_split": args.pope_split,
                    "image": s["image"],
                    "label": label,
                    "prediction_text": text,
                    "normalized_prediction": pred,
                    "gen_length": gl,
                    "elapsed_ms": round(float(ms), 4),
                    "method_stats": method_stats,
                    "method_audit": method_audit[:5] if method_audit else [],
                },
            )
        if label == "yes" and pred == "yes": tp += 1
        elif label == "no"  and pred == "yes": fp += 1
        elif label == "yes" and pred == "no":  fn += 1
        else: tn += 1

        if i < 5 or (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] pred={pred} label={label} len={gl}  {text[:60]}")

        if persist_callback is not None and ((i + 1) % checkpoint_every == 0 or (i + 1) == len(samples)):
            partial_metrics = _build_pope_metrics(
                args=args,
                method=method,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                gen_lengths=gen_lengths,
                latencies=latencies,
                method_stats_all=method_stats_all,
                sample_audits=sample_audits,
                errors=errors,
                vram_peak=torch.cuda.max_memory_allocated(),
            )
            persist_callback(
                partial_metrics,
                completed_samples=tp + tn + fp + fn,
                attempted_samples=len(_load_existing_records(sample_log_path)) if sample_log_path is not None else tp + tn + fp + fn + len(errors),
                target_samples=expected_n,
                expected_n=expected_n,
                status="partial",
            )

    vram_peak = torch.cuda.max_memory_allocated()
    metrics = _build_pope_metrics(
        args=args,
        method=method,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        gen_lengths=gen_lengths,
        latencies=latencies,
        method_stats_all=method_stats_all,
        sample_audits=sample_audits,
        errors=errors,
        vram_peak=vram_peak,
    )
    metrics["expected_n"] = expected_n
    metrics["attempted_n"] = len(_load_existing_records(sample_log_path)) if sample_log_path is not None else tp + tn + fp + fn + len(errors)
    return metrics

# ─── CHAIR ────────────────────────────────────────────────────────────

def load_coco_objects():
    inst_file = COCO_ANN / "instances_val2014.json"
    with open(inst_file) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    all_cats = set(cat_map.values())
    img_objs = defaultdict(set)
    for ann in data["annotations"]:
        img_objs[ann["image_id"]].add(cat_map[ann["category_id"]])
    id_from_file = {img["file_name"]: img["id"] for img in data["images"]}
    return all_cats, img_objs, id_from_file


def load_karpathy_test_images(limit: int, sample_indices: list[int] | None = None):
    if not COCO_KARPATHY_TEST_FILE.exists():
        logger.info("CHAIR: downloading Karpathy test split annotation to %s", COCO_KARPATHY_TEST_FILE)
        COCO_KARPATHY_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO_KARPATHY_TEST_URL, COCO_KARPATHY_TEST_FILE)

    with open(COCO_KARPATHY_TEST_FILE, encoding="utf-8") as f:
        data = json.load(f)

    images = []
    seen = set()
    for entry in data:
        rel_path = str(entry.get("image", "")).strip()
        if not rel_path:
            continue
        file_name = Path(rel_path).name
        if file_name in seen:
            continue
        image_path = COCO_IMG / file_name
        if not image_path.exists():
            logger.warning("CHAIR: Karpathy image missing locally, skipping %s", image_path)
            continue
        seen.add(file_name)
        images.append(image_path)
        if not sample_indices and len(images) >= limit:
            break
    return _subset_with_sample_indices(images, sample_indices, limit)

def compute_chair(captions, all_cats, img_objs, id_from_file):
    synonyms = {}
    for c in all_cats:
        synonyms[c] = {c}
        if " " in c:
            synonyms[c].add(c.replace(" ", ""))
        if c == "dining table":   synonyms[c].add("table")
        if c == "potted plant":   synonyms[c].add("plant")
        if c == "traffic light":  synonyms[c].add("light")
        if c == "cell phone":     synonyms[c].add("phone")
        if c == "wine glass":     synonyms[c].add("glass")
        if c == "hot dog":        synonyms[c].add("hotdog")

    total_s = hall_s = total_i = hall_i = 0
    for entry in captions:
        gt = img_objs.get(id_from_file.get(entry["image"]), set())
        for sent in re.split(r'[.!?]+', entry["caption"].lower()):
            sent = sent.strip()
            if not sent:
                continue
            total_s += 1
            sent_hall = False
            for cat in all_cats:
                for syn in synonyms.get(cat, {cat}):
                    if re.search(r'\b' + re.escape(syn) + r'(s|es)?\b', sent):
                        total_i += 1
                        if cat not in gt:
                            hall_i += 1
                            sent_hall = True
                        break
            if sent_hall:
                hall_s += 1
    return hall_s / max(total_s, 1), hall_i / max(total_i, 1)

def run_chair(model, processor, method, args, *, records_by_index: dict[int, dict] | None = None, sample_log_path: Path | None = None, persist_callback=None):
    all_cats, img_objs, id_from_file = load_coco_objects()
    logger.info(f"CHAIR: {len(all_cats)} categories, {len(img_objs)} annotated images")

    images = load_karpathy_test_images(args.mini_test, sample_indices=_get_requested_sample_indices(args))
    logger.info("CHAIR: %s Karpathy test images selected", len(images))

    records_by_index = records_by_index or {}
    captions, gen_lengths, latencies, method_stats_all, sample_audits, errors = _recover_chair_state(records_by_index)
    timer = GPUTimer()
    torch.cuda.reset_peak_memory_stats()
    checkpoint_every = max(int(args.checkpoint_every), 1)
    expected_n = len(images)

    if persist_callback is not None:
        initial_metrics = {
            "dataset": "chair",
            "method": method,
            "model": args.model,
            "model_family": MODEL_FAMILY.get(args.model, "unknown"),
            "n_samples": len(captions),
            "sample_count": len(captions),
            "n_errors": len(errors),
        }
        persist_callback(
            initial_metrics,
            completed_samples=len(captions),
            attempted_samples=len(records_by_index),
            target_samples=expected_n,
            expected_n=expected_n,
            status="partial",
        )

    for i, img in enumerate(images):
        if i in records_by_index:
            continue
        try:
            text, gl, ms, method_stats, method_audit = generate_one(model, processor, method, str(img), args.chair_prompt, timer, args)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{method}] sample {i} FAILED:\n{tb}")
            errors.append({"sample": i, "error": str(e), "traceback": tb})
            if sample_log_path is not None:
                _append_sample_record(
                    sample_log_path,
                    {
                        "sample_index": i,
                        "status": "error",
                        "dataset": "chair",
                        "image": img.name,
                        "error": str(e),
                        "traceback": tb,
                    },
                )
            continue
        gen_lengths.append(gl)
        if gl > 0:
            latencies.append(ms / gl)
        if method_stats is not None:
            method_stats_all.append(method_stats)
        if method_audit and len(sample_audits) < 3:
            sample_audits.append({
                "image": img.name,
                "caption_preview": text[:160],
                "audit": method_audit[:5],
            })
        captions.append({"image": img.name, "caption": text})
        if sample_log_path is not None:
            _append_sample_record(
                sample_log_path,
                {
                    "sample_index": i,
                    "status": "ok",
                    "dataset": "chair",
                    "image": img.name,
                    "prediction_text": text,
                    "gen_length": gl,
                    "elapsed_ms": round(float(ms), 4),
                    "method_stats": method_stats,
                    "method_audit": method_audit[:5] if method_audit else [],
                },
            )
        if i < 3 or (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(images)}] len={gl}  {text[:80]}")
        if persist_callback is not None and (((i + 1) % checkpoint_every == 0) or (i + 1) == len(images)):
            partial_metrics = {
                "dataset": "chair",
                "method": method,
                "model": args.model,
                "model_family": MODEL_FAMILY.get(args.model, "unknown"),
                "n_samples": len(captions),
                "sample_count": len(captions),
                "n_errors": len(errors),
            }
            persist_callback(
                partial_metrics,
                completed_samples=len(captions),
                attempted_samples=len(_load_existing_records(sample_log_path)) if sample_log_path is not None else len(captions) + len(errors),
                target_samples=expected_n,
                expected_n=expected_n,
                status="partial",
            )

    vram_peak = torch.cuda.max_memory_allocated()
    cs, ci = compute_chair(captions, all_cats, img_objs, id_from_file)
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)

    metrics = {
        "dataset": "chair", "method": method, "model": args.model, "model_family": MODEL_FAMILY.get(args.model, "unknown"),
        "n_samples": len(captions), "sample_count": len(captions), "n_errors": len(errors),
        "chair_s": round(cs, 4), "chair_i": round(ci, 4),
        "agl": round(agl, 2),
        "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
        "itl_ms_per_token": round(itl, 2),
        "tpot_ms_per_token": round(itl, 2),
        "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
        "peak_vram_gb": round(vram_peak / 1e9, 3),
        "timestamp": datetime.now().isoformat(),
        "sample_captions": [c["caption"][:150] for c in captions[:3]],
        "errors": errors[:3] if errors else [],
        "expected_n": expected_n,
        "attempted_n": len(_load_existing_records(sample_log_path)) if sample_log_path is not None else len(captions) + len(errors),
    }
    if method_stats_all:
        if method == "ifcb":
            metrics.update(_aggregate_ifcb_stats(method_stats_all))
        else:
            metrics.update(_aggregate_bra_stats(method_stats_all))
    metrics["notes"] = _build_notes(
        "chair",
        agl,
        _resolve_max_new_tokens("chair", args),
        metrics.get("intervention_rate"),
    )
    metrics["notes"].extend(_collect_runtime_notes(method))
    if sample_audits:
        metrics["sample_audits"] = sample_audits
    return metrics


def _aggregate_bra_stats(stats_list):
    if not stats_list:
        return {}
    keys = [
        "avg_candidate_window",
        "avg_visual_topk",
        "avg_resonance_time_ms",
        "avg_routing_time_ms",
        "avg_vasm_time_ms",
        "intervention_rate",
        "continuation_success_rate",
    ]
    out = {}
    for key in keys:
        vals = [float(s.get(key, 0.0)) for s in stats_list]
        out[key] = round(sum(vals) / max(len(vals), 1), 4)
    frame_hist = defaultdict(int)
    for s in stats_list:
        for k, v in s.get("selected_frame_histogram", {}).items():
            frame_hist[int(k)] += int(v)
    if frame_hist:
        out["selected_frame_histogram"] = dict(sorted(frame_hist.items()))
    for key in ("visual_state_provenance", "vasm_metadata"):
        for s in stats_list:
            if s.get(key):
                out[key] = s[key]
                break
    failure_examples = []
    for s in stats_list:
        failure_examples.extend(s.get("continuation_failure_examples", []))
    out["continuation_failure_examples"] = failure_examples[:10]
    out["suffix_collapse_failures"] = int(sum(int(s.get("suffix_collapse_failures", 0)) for s in stats_list))
    out["continuation_attempts"] = int(sum(int(s.get("continuation_attempts", 0)) for s in stats_list))
    return out


def _aggregate_ifcb_stats(stats_list):
    if not stats_list:
        return {}
    keys = [
        "decode_steps",
        "intervention_steps",
        "avg_candidate_count",
        "avg_risk",
        "avg_visual_participation",
        "max_risk",
    ]
    out = {}
    for key in keys:
        vals = [float(s.get(key, 0.0)) for s in stats_list]
        out[key] = round(sum(vals) / max(len(vals), 1), 4)
    return out


def _build_validation_summary(output_path: Path, sample_log_path: Path, expected_n: int) -> dict:
    records = load_jsonl_records(sample_log_path)
    coverage = compute_record_coverage(records, expected_n)
    return {
        "output_json": str(output_path),
        "sample_log_jsonl": str(sample_log_path),
        **coverage,
    }

# ─── main ─────────────────────────────────────────────────────────────

def execute_pipeline(args, *, model=None, processor=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = _resolve_output_json_path(args, ts)
    sample_log_path = _resolve_sample_log_path(args, out_file)
    runtime_profile = _runtime_profile(args)
    existing_payload = _load_existing_output_payload(out_file) if args.resume else None
    started_at = existing_payload.get("started_at") if existing_payload and existing_payload.get("started_at") else datetime.now().isoformat()
    if not args.resume:
        _reset_artifacts(out_file, sample_log_path)
    records_by_index = _load_existing_records(sample_log_path) if args.resume else {}
    metrics = None
    setup_done = False
    manage_model = model is None or processor is None
    if args.method == "ifcb" and args.model not in {"llava-v1.5-7b", "instructblip-7b"}:
        raise ValueError("IFCB is currently wired for llava-v1.5-7b and instructblip-7b")
    logger.info(
        f"Pipeline: method={args.method}  dataset={args.dataset}  "
        f"model={args.model}  mini_test={args.mini_test}  resume={args.resume}"
    )

    _persist_run_snapshot(
        args,
        output_path=out_file,
        sample_log_path=sample_log_path,
        status="partial",
        completed_samples=sum(1 for record in records_by_index.values() if record.get("status") == "ok"),
        attempted_samples=len(records_by_index),
        target_samples=args.mini_test,
        expected_n=args.mini_test,
        complete=False,
        started_at=started_at,
        ended_at=None,
        runtime_profile=runtime_profile,
        validation=_build_validation_summary(out_file, sample_log_path, args.mini_test) if sample_log_path.exists() else None,
    )
    try:
        if manage_model:
            model, processor = load_model_and_processor(args.model, args.method)
        setup_persistent(args.method, model, processor)
        setup_done = True

        persist_callback = lambda payload, *, completed_samples, attempted_samples, target_samples, expected_n, status: _persist_run_snapshot(
            args,
            output_path=out_file,
            sample_log_path=sample_log_path,
            status=status,
            metrics=payload,
            completed_samples=completed_samples,
            attempted_samples=attempted_samples,
            target_samples=target_samples,
            expected_n=expected_n,
            complete=False,
            started_at=started_at,
            ended_at=None,
            runtime_profile=runtime_profile,
            validation=_build_validation_summary(out_file, sample_log_path, expected_n) if sample_log_path.exists() else None,
        )

        if args.dataset == "pope":
            metrics = run_pope(
                model,
                processor,
                args.method,
                args,
                records_by_index=records_by_index,
                sample_log_path=sample_log_path,
                persist_callback=persist_callback,
            )
        elif args.dataset == "chair":
            metrics = run_chair(
                model,
                processor,
                args.method,
                args,
                records_by_index=records_by_index,
                sample_log_path=sample_log_path,
                persist_callback=persist_callback,
            )
        else:
            metrics = run_generic_benchmark(
                model,
                processor,
                args.method,
                args,
                records_by_index=records_by_index,
                sample_log_path=sample_log_path,
                persist_callback=persist_callback,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Pipeline failed:\n%s", tb)
        partial_count = 0 if metrics is None else int(metrics.get("sample_count", metrics.get("n_samples", 0)))
        expected_n = args.mini_test if metrics is None else int(metrics.get("expected_n", args.mini_test) or args.mini_test)
        attempted_n = len(_load_existing_records(sample_log_path))
        validation = _build_validation_summary(out_file, sample_log_path, expected_n) if sample_log_path.exists() else None
        _persist_run_snapshot(
            args,
            output_path=out_file,
            sample_log_path=sample_log_path,
            status="error",
            metrics=metrics,
            completed_samples=partial_count,
            attempted_samples=attempted_n,
            target_samples=expected_n,
            expected_n=expected_n,
            complete=False,
            started_at=started_at,
            ended_at=datetime.now().isoformat(),
            runtime_profile=runtime_profile,
            validation=validation,
            error={"message": str(exc), "traceback": tb},
        )
        raise
    finally:
        if setup_done:
            teardown_persistent(args.method)

    expected_n = int(metrics.get("expected_n", args.mini_test) or args.mini_test)
    attempted_n = int(metrics.get("attempted_n", len(_load_existing_records(sample_log_path))) or 0)
    completed_samples = int(metrics.get("sample_count", metrics.get("n_samples", 0)))
    validation = _build_validation_summary(out_file, sample_log_path, expected_n)
    complete = bool(validation.get("complete")) and int(metrics.get("n_errors", 0) or 0) == 0 and completed_samples == expected_n
    final_status = "final_complete" if complete else "failed_validated"
    ended_at = datetime.now().isoformat()
    _persist_run_snapshot(
        args,
        output_path=out_file,
        sample_log_path=sample_log_path,
        status=final_status,
        metrics=metrics,
        completed_samples=completed_samples,
        attempted_samples=attempted_n,
        target_samples=expected_n,
        expected_n=expected_n,
        complete=complete,
        started_at=started_at,
        ended_at=ended_at,
        runtime_profile=runtime_profile,
        validation=validation,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  Results -> {out_file}")
    logger.info(f"  Sample Log -> {sample_log_path}")
    logger.info(f"  Final Status -> {final_status}")
    logger.info(f"{'='*60}")
    for k, v in metrics.items():
        if k not in ("sample_captions", "timestamp", "errors"):
            logger.info(f"  {k:25s} = {v}")
    if metrics.get("errors"):
        logger.warning(f"  {len(metrics['errors'])} error(s) occurred — see JSON for tracebacks")
    logger.info(f"{'='*60}\n")
    return {"output_json": out_file, "sample_log_jsonl": sample_log_path, "status": final_status, "complete": complete, "metrics": metrics}


def main():
    args = parse_args()
    execute_pipeline(args)


if __name__ == "__main__":
    main()
