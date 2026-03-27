#!/usr/bin/env python3
"""
UniGround v2 evaluation entry.

This runner keeps benchmark task formatting outside the v2 runtime. It does not
inject benchmark-only evidence bonuses or binary-choice heuristics into the
method implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from bra_universal_plugin import build_universal_claim_manifest
from run_eval_pipeline import COCO_IMG, GPUTimer, MODEL_MAP, POPE_DIR, compute_chair, load_coco_objects
from uniground_runtime import FrozenExternalEncoder, load_universal_scorer
from uniground_v2.detector import GroundingDinoProposalProvider
from uniground_v2.observation import build_v2_observation_cache
from uniground_v2.regions import RegionRetriever
from uniground_v2.runtime import UniGroundV2LogitsProcessor
from uniground_v2.scorer import HardcodedUniversalScorer
from uniground_v2.task_adapter import ChairTaskAdapter, PopeTaskAdapter
from uniground_v2.trigger import EntropyMarginTrigger


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("/root/autodl-tmp/BRA_Project")
OUTPUT_DIR = PROJECT / "logs" / "uniground_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniGround v2 evaluation runner.")
    parser.add_argument(
        "--model",
        default="qwen3-vl-8b",
        choices=["qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b", "llava-v1.5-7b"],
    )
    parser.add_argument("--dataset", default="pope", choices=["pope", "chair"])
    parser.add_argument("--mini_test", type=int, default=32)
    parser.add_argument("--pope_split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--pope_controller_mode", default="verifier", choices=["verifier", "frontier"])
    parser.add_argument("--pope_min_verifier_delta", type=float, default=0.05)
    parser.add_argument("--pope_min_evidence_confidence", type=float, default=0.03)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--psi_mode", default="checkpoint", choices=["checkpoint", "hardcoded"])
    parser.add_argument("--psi_checkpoint", default=None)
    parser.add_argument("--external_encoder", default="/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14")
    parser.add_argument("--external_device", default="cpu")
    parser.add_argument("--observation_mode", default="grid_regions", choices=["grid_regions", "detector_regions"])
    parser.add_argument("--detector_model", default="/root/autodl-tmp/BRA_Project/models/grounding-dino-base")
    parser.add_argument("--detector_device", default="cpu")
    parser.add_argument("--detector_threshold", type=float, default=0.30)
    parser.add_argument("--detector_text_threshold", type=float, default=0.25)
    parser.add_argument("--detector_max_regions", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--bias_scale", type=float, default=0.30)
    parser.add_argument("--score_temperature", type=float, default=0.35)
    parser.add_argument("--lambda_con", type=float, default=1.0)
    parser.add_argument("--abstain_threshold", type=float, default=0.55)
    parser.add_argument("--trigger_top_k", type=int, default=10)
    parser.add_argument("--entropy_threshold", type=float, default=1.5)
    parser.add_argument("--margin_threshold", type=float, default=0.75)
    parser.add_argument("--min_content_ratio", type=float, default=0.25)
    parser.add_argument("--retrieval_top_r", type=int, default=1)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_meta(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu")
    contract = payload.get("contract", {}) if isinstance(payload, dict) else {}
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "contract_version": contract.get("contract_version"),
        "checkpoint_format": contract.get("checkpoint_format"),
        "frozen_vision_encoder_name": contract.get("frozen_vision_encoder_name"),
        "frozen_text_encoder_name": contract.get("frozen_text_encoder_name"),
    }


def load_host_model_and_processor(model_key: str):
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration

    model_path = MODEL_MAP[model_key]
    logger.info("Loading model from %s ...", model_path)
    if model_key.startswith("qwen3-vl"):
        if torch.cuda.is_available():
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation="sdpa",
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
    elif model_key == "llava-v1.5-7b":
        if torch.cuda.is_available():
            model = LlavaForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    return model, processor


def build_single_input(processor, image_path: str, question: str, device: str = "cuda"):
    image = Image.open(image_path).convert("RGB")
    processor_type = type(processor).__name__.lower()
    if "llava" in processor_type:
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
        inputs = processor(images=image, text=text, return_tensors="pt")
        return image, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    skip = {"mm_token_type_ids"}
    return image, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items() if k not in skip}


def build_batch_inputs(processor, image_paths: list[str], questions: list[str], device: str = "cuda"):
    processor_type = type(processor).__name__.lower()
    if "llava" in processor_type:
        raise NotImplementedError("Batch eval is not yet supported for LLaVA in run_uniground_v2_eval.py.")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    tokenizer = getattr(processor, "tokenizer", None)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
        for image, question in zip(images, questions)
    ]
    texts = [processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    if tokenizer is not None and original_padding_side is not None:
        tokenizer.padding_side = original_padding_side
    skip = {"mm_token_type_ids"}
    return images, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items() if k not in skip}


def build_v2_processor(
    args: argparse.Namespace,
    processor,
    image: Image.Image,
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels: list[str] | None,
    runtime_context: dict | None = None,
):
    from bra_universal_plugin import UniversalPluginConfig

    cfg = UniversalPluginConfig(
        top_k=args.top_k,
        bias_scale=args.bias_scale,
        score_temperature=args.score_temperature,
        lambda_con=args.lambda_con,
        abstain_threshold=args.abstain_threshold,
    )
    trigger = EntropyMarginTrigger(
        top_k=args.trigger_top_k,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        min_content_ratio=args.min_content_ratio,
    )
    retriever = RegionRetriever(encoder=encoder, top_r=args.retrieval_top_r)
    tokenizer = getattr(processor, "tokenizer", processor)
    proposals = None
    if args.observation_mode == "detector_regions" and detector_provider is not None and detector_labels:
        proposals = detector_provider.propose(image, detector_labels)
    cache_result = build_v2_observation_cache(encoder, image, proposals=proposals, mode=args.observation_mode)
    if runtime_context:
        merged_context = dict(runtime_context)
        cache_result.observation.metadata["runtime_context"] = merged_context
        cache_result.cache_info["runtime_context"] = merged_context
    processor_obj = UniGroundV2LogitsProcessor(
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
        scorer=scorer,
        observation=cache_result.observation,
        trigger=trigger,
        retriever=retriever,
    )
    return processor_obj, cache_result.cache_info


def build_v2_batch_processor(
    args: argparse.Namespace,
    processor,
    images: list[Image.Image],
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels_list: list[list[str] | None],
    runtime_contexts: list[dict] | None = None,
):
    from bra_universal_plugin import UniversalPluginConfig

    cfg = UniversalPluginConfig(
        top_k=args.top_k,
        bias_scale=args.bias_scale,
        score_temperature=args.score_temperature,
        lambda_con=args.lambda_con,
        abstain_threshold=args.abstain_threshold,
    )
    trigger = EntropyMarginTrigger(
        top_k=args.trigger_top_k,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        min_content_ratio=args.min_content_ratio,
    )
    retriever = RegionRetriever(encoder=encoder, top_r=args.retrieval_top_r)
    tokenizer = getattr(processor, "tokenizer", processor)
    observations = []
    cache_infos = []
    runtime_contexts = runtime_contexts or [{} for _ in images]
    for image, detector_labels, runtime_context in zip(images, detector_labels_list, runtime_contexts):
        proposals = None
        if args.observation_mode == "detector_regions" and detector_provider is not None and detector_labels:
            proposals = detector_provider.propose(image, detector_labels)
        cache_result = build_v2_observation_cache(encoder, image, proposals=proposals, mode=args.observation_mode)
        if runtime_context:
            merged_context = dict(runtime_context)
            cache_result.observation.metadata["runtime_context"] = merged_context
            cache_result.cache_info["runtime_context"] = merged_context
        observations.append(cache_result.observation)
        cache_infos.append(cache_result.cache_info)
    processor_obj = UniGroundV2LogitsProcessor(
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
        scorer=scorer,
        observation=observations,
        trigger=trigger,
        retriever=retriever,
    )
    return processor_obj, cache_infos


def generate_one(
    model,
    processor,
    args: argparse.Namespace,
    image_path: str,
    question: str,
    timer: GPUTimer,
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels: list[str] | None,
    runtime_context: dict | None = None,
):
    image, inputs = build_single_input(processor, image_path, question)
    runtime_context = dict(runtime_context or {})
    runtime_context["prompt_token_count"] = int(inputs["input_ids"].shape[1])
    logits_processor, cache_info = build_v2_processor(
        args,
        processor,
        image,
        encoder,
        scorer,
        detector_provider,
        detector_labels,
        runtime_context=runtime_context,
    )
    timer.start()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            logits_processor=[logits_processor],
        )
    elapsed = timer.stop()
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return text, len(gen_ids), elapsed, logits_processor.get_summary_stats(), logits_processor.get_audit_log(), cache_info


def generate_batch(
    model,
    processor,
    args: argparse.Namespace,
    image_paths: list[str],
    questions: list[str],
    timer: GPUTimer,
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels_list: list[list[str] | None],
    runtime_contexts: list[dict] | None = None,
):
    images, inputs = build_batch_inputs(processor, image_paths, questions)
    prompt_token_count = int(inputs["input_ids"].shape[1])
    runtime_contexts = [dict(context or {}) for context in (runtime_contexts or [{} for _ in image_paths])]
    for context in runtime_contexts:
        context["prompt_token_count"] = prompt_token_count
    logits_processor, cache_infos = build_v2_batch_processor(
        args, processor, images, encoder, scorer, detector_provider, detector_labels_list, runtime_contexts=runtime_contexts
    )
    timer.start()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            logits_processor=[logits_processor],
        )
    elapsed = timer.stop()
    prompt_len = inputs["input_ids"].shape[1]
    texts = []
    lengths = []
    for row_idx in range(out_ids.shape[0]):
        gen_ids = out_ids[row_idx, prompt_len:]
        texts.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        lengths.append(int(gen_ids.shape[0]))
    return texts, lengths, elapsed, logits_processor.get_summary_stats(), logits_processor.get_audit_log(), cache_infos


def run_pope(
    model,
    processor,
    args: argparse.Namespace,
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels: list[str] | None,
) -> dict:
    adapter = PopeTaskAdapter()
    split_file = POPE_DIR / f"coco_pope_{args.pope_split}.json"
    samples = [json.loads(line) for line in open(split_file, encoding="utf-8")][: args.mini_test]
    tp = fp = fn = tn = 0
    route_stats = []
    sample_outputs = []
    timer = GPUTimer()
    errors = []
    cache_infos = []
    diagnostics = {
        "object_extraction_success": 0,
        "sample_count": len(samples),
        "positive_labels": 0,
        "negative_labels": 0,
        "trigger_positive": 0,
        "trigger_negative": 0,
        "intervened_positive": 0,
        "intervened_negative": 0,
        "abstained_positive": 0,
        "abstained_negative": 0,
        "changed_choice_positive": 0,
        "changed_choice_negative": 0,
        "host_yes_to_no": 0,
        "host_no_to_yes": 0,
        "semantic_alias_hits": 0,
        "semantic_alias_total": 0,
        "selected_region_count_total": 0,
        "selected_region_samples": 0,
        "selected_region_score_total": 0.0,
        "selected_region_score_count": 0,
    }
    batch_size = max(int(args.eval_batch_size), 1)

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]
        questions = [adapter.format_question(sample["text"]) for sample in batch_samples]
        image_paths = [str(COCO_IMG / sample["image"]) for sample in batch_samples]
        labels = [sample["label"].strip().lower() for sample in batch_samples]
        detector_labels_list: list[list[str] | None] = []
        runtime_contexts: list[dict] = []
        for sample, label in zip(batch_samples, labels):
            sample_detector_labels = detector_labels
            object_label = adapter.extract_object_label(sample["text"])
            runtime_context = adapter.build_runtime_context(
                sample["text"],
                split=args.pope_split,
                controller_mode=args.pope_controller_mode,
                label=label,
            )
            runtime_context["pope_min_verifier_delta"] = float(args.pope_min_verifier_delta)
            runtime_context["pope_min_evidence_confidence"] = float(args.pope_min_evidence_confidence)
            runtime_contexts.append(runtime_context)
            if object_label:
                diagnostics["object_extraction_success"] += 1
            if args.observation_mode == "detector_regions":
                if object_label:
                    sample_detector_labels = [object_label]
            detector_labels_list.append(sample_detector_labels)
        try:
            if len(batch_samples) == 1:
                text, gl, ms, stat_block, audit, cache_info = generate_one(
                    model,
                    processor,
                    args,
                    image_paths[0],
                    questions[0],
                    timer,
                    encoder,
                    scorer,
                    detector_provider,
                    detector_labels_list[0],
                    runtime_context=runtime_contexts[0],
                )
                texts = [text]
                lengths = [gl]
                audits = [audit]
                cache_info_list = [cache_info]
            else:
                texts, lengths, ms, stat_block, audit, cache_info_list = generate_batch(
                    model,
                    processor,
                    args,
                    image_paths,
                    questions,
                    timer,
                    encoder,
                    scorer,
                    detector_provider,
                    detector_labels_list,
                    runtime_contexts=runtime_contexts,
                )
                audits = [[entry for entry in audit if entry.get("row_idx") == row_idx] for row_idx in range(len(batch_samples))]
        except Exception as exc:
            errors.append({"sample": start_idx, "error": str(exc), "traceback": traceback.format_exc()})
            continue
        if stat_block:
            stat_block = dict(stat_block)
            stat_block["sample_count"] = len(batch_samples)
            route_stats.append(stat_block)
        for local_idx, (question, label, text, gl, cache_info, runtime_context, sample_audit) in enumerate(
            zip(questions, labels, texts, lengths, cache_info_list, runtime_contexts, audits)
        ):
            idx = start_idx + local_idx
            pred = adapter.parse_prediction(text)
            cache_infos.append(cache_info)
            _update_pope_diagnostics(diagnostics, label, sample_audit)
            if len(sample_outputs) < 5:
                sample_outputs.append(
                    {
                        "sample_index": idx,
                        "prediction": pred,
                        "label": label,
                        "question": question,
                        "text_preview": text[:200],
                        "observation_cache": cache_info,
                        "runtime_context": runtime_context,
                    }
                )
            if label == "yes" and pred == "yes":
                tp += 1
            elif label == "no" and pred == "yes":
                fp += 1
            elif label == "yes" and pred == "no":
                fn += 1
            elif label == "no" and pred == "no":
                tn += 1
            else:
                errors.append(
                    {
                        "sample": idx,
                        "error": f"unknown_prediction:{pred}",
                        "prediction_text": text[:80],
                    }
                )
            if idx < 5 or (idx + 1) % 10 == 0:
                logger.info(
                    "[pope-v2] [%s/%s] pred=%s label=%s len=%s ms=%.2f",
                    idx + 1,
                    len(samples),
                    pred,
                    label,
                    gl,
                    ms / max(len(batch_samples), 1),
                )

    total = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(total, 1)
    metrics = {
        "route": "uniground_v2",
        "dataset": "pope",
        "pope_split": args.pope_split,
        "method": "uniground_v2",
        "host_model": args.model,
        "sample_count": total,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "hallucination_primary_metric": "f1",
        "hallucination_primary_value": round(f1, 4),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3],
        "sample_outputs": sample_outputs,
    }
    metrics.update(_aggregate_stats(route_stats))
    metrics["observation_cache"] = _aggregate_cache_info(cache_infos)
    metrics["pope_controller_mode"] = args.pope_controller_mode
    metrics["pope_diagnostics"] = _finalize_pope_diagnostics(diagnostics)
    return metrics


def _update_pope_diagnostics(diagnostics: dict, label: str, audit_entries: list[dict]) -> None:
    if label == "yes":
        diagnostics["positive_labels"] += 1
        split_key = "positive"
    else:
        diagnostics["negative_labels"] += 1
        split_key = "negative"
    triggered = False
    intervened = False
    abstained = False
    changed_choice = False
    host_yes_to_no = False
    host_no_to_yes = False
    for entry in audit_entries:
        trigger = entry.get("trigger") or {}
        triggered = triggered or bool(trigger.get("fire"))
        intervened = intervened or bool(entry.get("intervened"))
        abstained = abstained or bool(entry.get("aborted"))
        decision_audit = entry.get("decision_audit") or {}
        changed_choice = changed_choice or bool(decision_audit.get("changed_choice"))
        host_yes_to_no = host_yes_to_no or (
            decision_audit.get("host_choice") == "yes" and decision_audit.get("adjusted_choice") == "no"
        )
        host_no_to_yes = host_no_to_yes or (
            decision_audit.get("host_choice") == "no" and decision_audit.get("adjusted_choice") == "yes"
        )
        for candidate in entry.get("candidates") or []:
            diagnostics["semantic_alias_total"] += 1
            if candidate.get("semantic_alias"):
                diagnostics["semantic_alias_hits"] += 1
        retrieval = entry.get("retrieval") or {}
        selected = retrieval.get("selected_region_indices_per_candidate") or []
        if selected:
            diagnostics["selected_region_count_total"] += sum(len(row) for row in selected)
            diagnostics["selected_region_samples"] += len(selected)
        selected_scores = retrieval.get("selected_region_scores_per_candidate") or []
        for row in selected_scores:
            diagnostics["selected_region_score_total"] += sum(float(score) for score in row)
            diagnostics["selected_region_score_count"] += len(row)
    diagnostics[f"trigger_{split_key}"] += int(triggered)
    diagnostics[f"intervened_{split_key}"] += int(intervened)
    diagnostics[f"abstained_{split_key}"] += int(abstained)
    diagnostics[f"changed_choice_{split_key}"] += int(changed_choice)
    diagnostics["host_yes_to_no"] += int(host_yes_to_no)
    diagnostics["host_no_to_yes"] += int(host_no_to_yes)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _finalize_pope_diagnostics(diagnostics: dict) -> dict:
    sample_count = int(diagnostics.get("sample_count", 0))
    positive = int(diagnostics.get("positive_labels", 0))
    negative = int(diagnostics.get("negative_labels", 0))
    semantic_total = int(diagnostics.get("semantic_alias_total", 0))
    selected_region_samples = int(diagnostics.get("selected_region_samples", 0))
    selected_region_score_count = int(diagnostics.get("selected_region_score_count", 0))
    return {
        "object_extraction_success_rate": _safe_rate(int(diagnostics.get("object_extraction_success", 0)), sample_count),
        "positive_trigger_rate": _safe_rate(int(diagnostics.get("trigger_positive", 0)), positive),
        "negative_trigger_rate": _safe_rate(int(diagnostics.get("trigger_negative", 0)), negative),
        "positive_intervention_rate": _safe_rate(int(diagnostics.get("intervened_positive", 0)), positive),
        "negative_intervention_rate": _safe_rate(int(diagnostics.get("intervened_negative", 0)), negative),
        "positive_abstention_rate": _safe_rate(int(diagnostics.get("abstained_positive", 0)), positive),
        "negative_abstention_rate": _safe_rate(int(diagnostics.get("abstained_negative", 0)), negative),
        "positive_changed_choice_rate": _safe_rate(int(diagnostics.get("changed_choice_positive", 0)), positive),
        "negative_changed_choice_rate": _safe_rate(int(diagnostics.get("changed_choice_negative", 0)), negative),
        "host_yes_to_no": int(diagnostics.get("host_yes_to_no", 0)),
        "host_no_to_yes": int(diagnostics.get("host_no_to_yes", 0)),
        "semantic_alias_hit_rate": _safe_rate(int(diagnostics.get("semantic_alias_hits", 0)), semantic_total),
        "avg_selected_region_count": round(float(diagnostics.get("selected_region_count_total", 0)) / selected_region_samples, 4)
        if selected_region_samples else None,
        "avg_selected_region_score": round(float(diagnostics.get("selected_region_score_total", 0.0)) / selected_region_score_count, 4)
        if selected_region_score_count else None,
    }


def run_chair(
    model,
    processor,
    args: argparse.Namespace,
    encoder: FrozenExternalEncoder,
    scorer,
    detector_provider: GroundingDinoProposalProvider | None,
    detector_labels: list[str] | None,
) -> dict:
    adapter = ChairTaskAdapter()
    all_cats, img_objs, id_from_file = load_coco_objects()
    images = sorted(COCO_IMG.glob("*.jpg"))[: args.mini_test]
    captions = []
    route_stats = []
    timer = GPUTimer()
    errors = []
    cache_infos = []

    for idx, image_path in enumerate(images):
        try:
            runtime_context = adapter.build_runtime_context()
            text, _gl, _ms, stat_block, _audit, cache_info = generate_one(
                model,
                processor,
                args,
                str(image_path),
                adapter.format_question(),
                timer,
                encoder,
                scorer,
                detector_provider,
                detector_labels,
                runtime_context=runtime_context,
            )
        except Exception as exc:
            errors.append({"sample": idx, "error": str(exc), "traceback": traceback.format_exc()})
            continue
        captions.append({"image": image_path.name, "caption": text})
        cache_infos.append(cache_info)
        if stat_block:
            route_stats.append(stat_block)
        if idx < 3 or (idx + 1) % 10 == 0:
            logger.info("[chair-v2] [%s/%s]", idx + 1, len(images))

    chair_s, chair_i = compute_chair(captions, all_cats, img_objs, id_from_file)
    metrics = {
        "route": "uniground_v2",
        "dataset": "chair",
        "method": "uniground_v2",
        "host_model": args.model,
        "sample_count": len(captions),
        "chair_s": round(chair_s, 4),
        "chair_i": round(chair_i, 4),
        "hallucination_primary_metric": "chair_i",
        "hallucination_primary_value": round(chair_i, 4),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3],
        "sample_captions": [caption["caption"][:150] for caption in captions[:3]],
    }
    metrics.update(_aggregate_stats(route_stats))
    metrics["observation_cache"] = _aggregate_cache_info(cache_infos)
    return metrics


def _aggregate_stats(stats_blocks: list[dict]) -> dict:
    if not stats_blocks:
        return {
            "intervention_coverage": None,
            "trigger_fire_rate": None,
            "avg_active_candidates": None,
            "latency_split": None,
        }
    keys = ["intervention_coverage", "trigger_fire_rate", "avg_active_candidates", "candidate_construction_ms", "sidecar_scoring_ms", "bridge_redistribution_ms", "total_step_ms"]
    out = {}
    for key in keys:
        weighted_pairs = [(block[key], int(block.get("sample_count", 1))) for block in stats_blocks if block.get(key) is not None]
        total_weight = sum(weight for _value, weight in weighted_pairs)
        out[key] = round(sum(value * weight for value, weight in weighted_pairs) / total_weight, 4) if total_weight else None
    weighted_latency_blocks = [
        (block["latency_split"], int(block.get("sample_count", 1)))
        for block in stats_blocks
        if isinstance(block.get("latency_split"), dict)
    ]
    out["latency_split"] = {
        metric: round(sum(block[metric] * weight for block, weight in weighted_latency_blocks) / sum(weight for _block, weight in weighted_latency_blocks), 4)
        for metric in ("candidate_construction_ms", "sidecar_scoring_ms", "bridge_redistribution_ms", "total_step_ms")
    } if weighted_latency_blocks else None
    return out


def _aggregate_cache_info(cache_infos: list[dict]) -> dict | None:
    if not cache_infos:
        return None
    precompute_values = [float(info["precompute_ms"]) for info in cache_infos if info.get("precompute_ms") is not None]
    region_counts = [int(info["region_count"]) for info in cache_infos if info.get("region_count") is not None]
    return {
        "cache_stage": "pre_generate",
        "region_mode": cache_infos[0].get("region_mode"),
        "uses_detector_proposals": bool(cache_infos[0].get("uses_detector_proposals", False)),
        "avg_precompute_ms": round(sum(precompute_values) / len(precompute_values), 4) if precompute_values else None,
        "avg_region_count": round(sum(region_counts) / len(region_counts), 4) if region_counts else None,
    }


def output_path(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    return out_dir / f"uniground_v2_{args.model}_{args.dataset}{tag}_{ts}.json"


def main() -> None:
    args = parse_args()
    model, processor = load_host_model_and_processor(args.model)
    encoder = FrozenExternalEncoder(args.external_encoder, device=args.external_device)
    if args.psi_mode == "hardcoded":
        scorer = HardcodedUniversalScorer()
    else:
        if not args.psi_checkpoint:
            raise ValueError("--psi_checkpoint is required when --psi_mode checkpoint.")
        scorer = load_universal_scorer(args.psi_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    detector_provider = None
    detector_labels = None
    if args.observation_mode == "detector_regions":
        detector_provider = GroundingDinoProposalProvider(
            args.detector_model,
            device=args.detector_device,
            threshold=args.detector_threshold,
            text_threshold=args.detector_text_threshold,
            max_regions=args.detector_max_regions,
        )
        detector_labels = sorted(load_coco_objects()[0])
    if args.dataset == "pope":
        metrics = run_pope(model, processor, args, encoder, scorer, detector_provider, detector_labels)
    else:
        metrics = run_chair(model, processor, args, encoder, scorer, detector_provider, detector_labels)
    checkpoint_meta = load_checkpoint_meta(args.psi_checkpoint) if args.psi_mode == "checkpoint" and args.psi_checkpoint else {
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "contract_version": "hardcoded_runtime_probe",
        "checkpoint_format": "hardcoded_scorer",
        "frozen_vision_encoder_name": None,
        "frozen_text_encoder_name": None,
    }
    metrics["psi_mode"] = args.psi_mode
    metrics["psi_checkpoint"] = args.psi_checkpoint
    metrics["external_encoder"] = args.external_encoder
    metrics["detector_model"] = args.detector_model if args.observation_mode == "detector_regions" else None
    metrics["universal_claim_manifest"] = build_universal_claim_manifest(checkpoint_meta)
    metrics["psi_univ_checkpoint"] = checkpoint_meta
    out_path = output_path(args)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", out_path)
    print(out_path)


if __name__ == "__main__":
    main()
