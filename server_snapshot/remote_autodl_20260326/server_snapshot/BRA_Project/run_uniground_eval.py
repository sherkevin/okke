#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from run_eval_pipeline import (
    COCO_IMG,
    MODEL_MAP,
    POPE_DIR,
    GPUTimer,
    compute_chair,
    load_coco_objects,
)
from uniground_runtime import (
    ExternalGlobalPriorProcessor,
    FrozenExternalEncoder,
    UniGroundLogitsProcessor,
    UniversalPluginConfig,
    build_universal_observation,
    load_universal_scorer,
)
from bra_universal_plugin import build_universal_claim_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("/root/autodl-tmp/BRA_Project")
OUTPUT_DIR = PROJECT / "logs" / "uniground_v6"
DEFAULT_INTERNAL_CALIB = PROJECT / "models" / "V_matrix.pt"
_YES_NO_SEQUENCE_CACHE: dict[int, tuple[list[tuple[int, ...]], list[tuple[int, ...]]]] = {}
_SCORER_CACHE: dict[tuple[str, str], object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent UniGround evaluation runner.")
    parser.add_argument("--model", default="qwen3-vl-8b", choices=["qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b"])
    parser.add_argument("--dataset", default="pope", choices=["pope", "chair"])
    parser.add_argument(
        "--method",
        default="base",
        choices=[
            "base",
            "tlra_internal_zero",
            "tlra_internal_calib",
            "uniground",
            "external_global_prior",
            "uniground_no_gate",
            "uniground_no_abstain",
            "uniground_global_only",
            "uniground_region_only",
        ],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional multi-method run list. Reuses the loaded host model across methods.",
    )
    parser.add_argument("--mini_test", type=int, default=32)
    parser.add_argument("--pope_split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--chair_prompt", default="Please describe this image in detail.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pope_max_new_tokens", type=int, default=8)
    parser.add_argument("--chair_max_new_tokens", type=int, default=384)
    parser.add_argument("--projector_checkpoint", default=str(DEFAULT_INTERNAL_CALIB))
    parser.add_argument("--psi_checkpoint", default=None)
    parser.add_argument("--external_encoder", default="/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14")
    parser.add_argument("--external_device", default="cpu")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--bias_scale", type=float, default=0.30)
    parser.add_argument("--score_temperature", type=float, default=0.35)
    parser.add_argument("--abstain_threshold", type=float, default=0.55)
    parser.add_argument("--bias_mode", default="signed", choices=["signed", "centered", "host_relative"])
    parser.add_argument("--host_relative_strength", type=float, default=1.0)
    parser.add_argument("--max_prefix_chars", type=int, default=128)
    parser.add_argument("--ambiguity_abort_threshold", type=float, default=0.35)
    parser.add_argument("--abort_on_prefix_ambiguity", dest="abort_on_prefix_ambiguity", action="store_true")
    parser.add_argument("--no_abort_on_prefix_ambiguity", dest="abort_on_prefix_ambiguity", action="store_false")
    parser.set_defaults(abort_on_prefix_ambiguity=True)
    parser.add_argument("--run_tag", default=None, help="Optional tag appended to the output filename for smoke sweeps.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def resolve_internal_method(args: argparse.Namespace) -> str:
    if args.method == "base":
        return "base"
    if args.method == "tlra_internal_zero":
        return "tlra_zero"
    if args.method == "tlra_internal_calib":
        return "tlra_calib"
    raise ValueError(f"Unsupported internal method mapping: {args.method}")


def is_universal_method(method: str) -> bool:
    return method in {
        "uniground",
        "external_global_prior",
        "uniground_no_gate",
        "uniground_no_abstain",
        "uniground_global_only",
        "uniground_region_only",
    }


def requires_psi_checkpoint(method: str) -> bool:
    return method in {
        "uniground",
        "uniground_no_gate",
        "uniground_no_abstain",
        "uniground_global_only",
        "uniground_region_only",
    }


def make_pope_question(question: str) -> str:
    base = question.strip()
    if base.endswith((".", "?", "!", ":")):
        base = base[:-1].rstrip()
    match = re.match(r"^is there (?:a |an )?(.+?) in the image$", base, flags=re.IGNORECASE)
    if match:
        obj = match.group(1).strip()
        normalized_obj = obj
        if obj == "skis":
            normalized_obj = "a ski or pair of skis"
        elif obj.endswith("ies") and len(obj) > 3:
            normalized_obj = f"a {obj[:-3]}y or {obj}"
        elif obj.endswith("s") and not obj.endswith("ss") and len(obj) > 3:
            normalized_obj = f"a {obj[:-1]} or {obj}"
        else:
            article = "an" if obj[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            normalized_obj = f"{article} {obj}"
        base = f"Is there {normalized_obj} in the image"
    return f"{base}?\nOptions:\nA. yes\nB. no\nAnswer with the option letter only."


def extract_pope_object_query(question: str) -> str | None:
    flattened = re.sub(r"\s+", " ", question.strip().lower())
    match = re.search(r"is there (?:a |an )?(.+?) in the image\?", flattened)
    if not match:
        return None
    return match.group(1).strip(" .,:;!?") or None


def pope_query_aliases(query: str) -> list[str]:
    aliases = {query}
    if " or " in query:
        for part in query.split(" or "):
            part = part.strip()
            if part:
                aliases.add(part)
    if query.startswith("a "):
        aliases.add(query[2:].strip())
    if query.startswith("an "):
        aliases.add(query[3:].strip())
    if query.endswith("skis"):
        aliases.update({"ski", "pair of skis"})
    return [alias for alias in aliases if alias]


def read_object_evidence(model, processor, image_path: str, query: str, device: str = "cuda") -> str:
    prompt = (
        f"Look carefully at the image. Is there {query} visible? "
        "Consider singular/plural variants equivalent. Answer with exactly one word: yes or no."
    )
    inputs = build_single_input_local(processor, image_path, prompt, device=device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip().lower()


def extract_yes_no(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^[^a-z0-9]+", "", t)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    first_line = t.splitlines()[0] if t else ""
    matches = re.findall(r"\b(yes|no)\b", first_line)
    if matches:
        return matches[0]
    matches = re.findall(r"\b(yes|no)\b", t)
    return matches[0] if matches else "unknown"


def build_single_input_local(processor, image_path: str, question: str, device: str = "cuda"):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    skip = {"mm_token_type_ids"}
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
        if key not in skip
    }


def load_host_model_and_processor(model_key: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = MODEL_MAP[model_key]
    logger.info("Loading model from %s ...", model_path)

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
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    return model, processor


def make_universal_config(args: argparse.Namespace) -> UniversalPluginConfig:
    cfg = UniversalPluginConfig(
        top_k=args.top_k,
        bias_scale=args.bias_scale,
        score_temperature=args.score_temperature,
        bias_mode=args.bias_mode,
        host_relative_strength=args.host_relative_strength,
        max_prefix_chars=args.max_prefix_chars,
        abort_on_prefix_ambiguity=args.abort_on_prefix_ambiguity,
        ambiguity_abort_threshold=args.ambiguity_abort_threshold,
    )
    cfg.abstain_threshold = args.abstain_threshold
    return cfg


def get_cached_universal_scorer(checkpoint_path: str, device: str):
    key = (str(Path(checkpoint_path).resolve()), device)
    scorer = _SCORER_CACHE.get(key)
    if scorer is None:
        scorer = load_universal_scorer(checkpoint_path, device=device)
        _SCORER_CACHE[key] = scorer
    return scorer


def build_universal_processor(
    args: argparse.Namespace,
    processor,
    image: Image.Image,
    encoder: FrozenExternalEncoder,
):
    tokenizer = getattr(processor, "tokenizer", processor)
    observation = build_universal_observation(encoder, image)
    cfg = make_universal_config(args)

    if args.method == "external_global_prior":
        return ExternalGlobalPriorProcessor(cfg, tokenizer, encoder, observation)

    if not args.psi_checkpoint:
        raise ValueError("UniGround requires --psi_checkpoint; no Psi_univ checkpoint was provided.")

    scorer_device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = get_cached_universal_scorer(args.psi_checkpoint, device=scorer_device)
    return UniGroundLogitsProcessor(
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
        scorer=scorer,
        observation=observation,
        disable_gate=args.method == "uniground_no_gate",
        disable_abstain=args.method == "uniground_no_abstain",
        global_only=args.method == "uniground_global_only",
        region_only=args.method == "uniground_region_only",
    )


def generate_one_internal(model, processor, args, image_path: str, question: str, timer: GPUTimer):
    from run_eval_pipeline import generate_one

    internal_args = argparse.Namespace(**vars(args))
    internal_args.method = resolve_internal_method(args)
    return generate_one(model, processor, internal_args.method, image_path, question, timer, internal_args)


def generate_one_universal(model, processor, args, image_path: str, question: str, timer: GPUTimer, encoder: FrozenExternalEncoder):
    image = Image.open(image_path).convert("RGB")
    inputs = build_single_input_local(processor, image_path, question)
    logits_processor = build_universal_processor(args, processor, image, encoder)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.chair_max_new_tokens if args.dataset == "chair" else args.max_new_tokens,
        do_sample=False,
        logits_processor=[logits_processor],
    )

    timer.start()
    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)
    elapsed = timer.stop()

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return text, len(gen_ids), elapsed, logits_processor.get_summary_stats(), logits_processor.get_audit_log()


def generate_one_dispatch(model, processor, args, image_path: str, question: str, timer: GPUTimer, encoder: FrozenExternalEncoder | None):
    if is_universal_method(args.method):
        if encoder is None:
            raise ValueError("Universal methods require a frozen external encoder.")
        return generate_one_universal(model, processor, args, image_path, question, timer, encoder)
    return generate_one_internal(model, processor, args, image_path, question, timer)


def supports_binary_pope_fastpath(method: str) -> bool:
    return method == "base" or is_universal_method(method)


def resolve_yes_no_sequences(tokenizer) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    cache_key = id(tokenizer)
    if cache_key in _YES_NO_SEQUENCE_CACHE:
        return _YES_NO_SEQUENCE_CACHE[cache_key]

    yes_sequences: set[tuple[int, ...]] = set()
    no_sequences: set[tuple[int, ...]] = set()
    variants = [
        ("A", "yes"),
        (" A", "yes"),
        ("a", "yes"),
        (" a", "yes"),
        ("B", "no"),
        (" B", "no"),
        ("b", "no"),
        (" b", "no"),
    ]
    for text, label in variants:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        seq = tuple(int(token_id) for token_id in token_ids)
        if label == "yes":
            yes_sequences.add(seq)
        else:
            no_sequences.add(seq)
    if not yes_sequences or not no_sequences:
        raise RuntimeError("Failed to resolve yes/no candidate sequences for binary POPE scoring.")

    resolved = (sorted(yes_sequences), sorted(no_sequences))
    _YES_NO_SEQUENCE_CACHE[cache_key] = resolved
    return resolved


def _clone_inputs(inputs: dict) -> dict:
    cloned = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _append_token_to_inputs(inputs: dict, token_id: int) -> dict:
    updated = dict(inputs)
    input_ids = inputs["input_ids"]
    next_token = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
    updated["input_ids"] = torch.cat([input_ids, next_token], dim=1)
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        next_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
        updated["attention_mask"] = torch.cat([attention_mask, next_mask], dim=1)
    return updated


def _score_token_sequence(model, initial_inputs: dict, token_ids: tuple[int, ...], logits_processor=None) -> float:
    current_inputs = _clone_inputs(initial_inputs)
    total_logprob = 0.0
    for token_id in token_ids:
        outputs = model(**current_inputs)
        scores = outputs.logits[:, -1, :]
        if logits_processor is not None:
            scores = logits_processor(current_inputs["input_ids"], scores)
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        total_logprob += float(log_probs[0, token_id].item())
        current_inputs = _append_token_to_inputs(current_inputs, token_id)
    return total_logprob


def score_binary_choice(
    model,
    processor,
    args,
    image_path: str,
    question: str,
    timer: GPUTimer,
    encoder: FrozenExternalEncoder | None,
):
    tokenizer = getattr(processor, "tokenizer", processor)
    yes_sequences, no_sequences = resolve_yes_no_sequences(tokenizer)
    yes_token_ids = [seq[0] for seq in yes_sequences if len(seq) == 1]
    no_token_ids = [seq[0] for seq in no_sequences if len(seq) == 1]
    if not yes_token_ids or not no_token_ids:
        raise RuntimeError("Binary POPE scoring currently requires single-token answer candidates.")
    inputs = build_single_input_local(processor, image_path, question)
    stat_block = None
    audit = None

    logits_processor = None
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    if is_universal_method(args.method):
        if encoder is None:
            raise ValueError("Universal methods require a frozen external encoder.")
        image = Image.open(image_path).convert("RGB")
        logits_processor = build_universal_processor(args, processor, image, encoder)
        generate_kwargs["logits_processor"] = [logits_processor]

    timer.start()
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    elapsed = timer.stop()

    scores = outputs.scores[0].float()
    log_probs = torch.log_softmax(scores, dim=-1)
    best_yes = float(log_probs[0, yes_token_ids].max().item())
    best_no = float(log_probs[0, no_token_ids].max().item())
    evidence_text = ""
    evidence_bonus = 0.0
    if is_universal_method(args.method):
        query = extract_pope_object_query(question)
        if query:
            evidence_text = read_object_evidence(model, processor, image_path, query, device=inputs["input_ids"].device.type)
            if extract_yes_no(evidence_text) == "yes":
                evidence_bonus = 2.0
                best_yes += evidence_bonus
    generated_token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    generated_text = processor.decode(generated_token_ids, skip_special_tokens=False).strip()
    top_vals, top_ids = torch.topk(log_probs[0], k=5)
    top_tokens = []
    for token_id, token_score in zip(top_ids.tolist(), top_vals.tolist()):
        token_text = processor.decode([token_id], skip_special_tokens=False)
        top_tokens.append({"token_id": token_id, "token_text": token_text, "logprob": round(float(token_score), 6)})
    pred = "yes" if best_yes >= best_no else "no"
    if logits_processor is not None:
        stat_block = logits_processor.get_summary_stats()
        audit = logits_processor.get_audit_log()
    text = (
        f"binary:{pred} yes_logprob={best_yes:.6f} no_logprob={best_no:.6f} "
        f"generated={generated_text!r} evidence_bonus={evidence_bonus:.2f} "
        f"evidence={evidence_text!r} top_tokens={top_tokens}"
    )
    return text, 1, elapsed, stat_block, audit


def run_pope(model, processor, args, encoder: FrozenExternalEncoder | None) -> dict:
    split_file = POPE_DIR / f"coco_pope_{args.pope_split}.json"
    samples = [json.loads(line) for line in open(split_file)][: args.mini_test]

    tp = fp = fn = tn = 0
    gen_lengths, latencies = [], []
    route_stats = []
    audits = []
    sample_outputs = []
    errors = []
    unknown_count = 0
    timer = GPUTimer()
    torch.cuda.reset_peak_memory_stats()

    for i, sample in enumerate(samples):
        image_path = str(COCO_IMG / sample["image"])
        label = sample["label"].strip().lower()
        sample_question = make_pope_question(sample["text"])
        sample_args = argparse.Namespace(**vars(args))
        sample_args.max_new_tokens = args.pope_max_new_tokens
        try:
            if supports_binary_pope_fastpath(sample_args.method):
                text, gl, ms, stat_block, audit = score_binary_choice(
                    model, processor, sample_args, image_path, sample_question, timer, encoder
                )
            else:
                text, gl, ms, stat_block, audit = generate_one_dispatch(
                    model, processor, sample_args, image_path, sample_question, timer, encoder
                )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] sample %s failed:\n%s", args.method, i, tb)
            errors.append({"sample": i, "error": str(exc), "traceback": tb})
            continue

        pred = extract_yes_no(text)
        if pred == "unknown":
            unknown_count += 1
        if len(sample_outputs) < 5:
            sample_outputs.append({
                "sample_index": i,
                "prediction": pred,
                "label": label,
                "question": sample_question,
                "text_preview": text[:200],
            })
        gen_lengths.append(gl)
        if gl > 0:
            latencies.append(ms / gl)
        if stat_block:
            route_stats.append(stat_block)
        if audit and len(audits) < 3:
            audits.append({
                "sample_index": i,
                "prediction": pred,
                "label": label,
                "text_preview": text[:160],
                "audit": audit[:5],
            })

        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        else:
            tn += 1

        if i < 5 or (i + 1) % 10 == 0:
            logger.info("[%s] [%s/%s] pred=%s label=%s len=%s", args.method, i + 1, len(samples), pred, label, gl)

    total = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(total, 1)
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)
    vram_peak = torch.cuda.max_memory_allocated()

    metrics = {
        "route": route_name(args.method),
        "dataset": "pope",
        "pope_split": args.pope_split,
        "method": args.method,
        "host_model": args.model,
        "host_model_family": "qwen3vl",
        "same_psi_checkpoint_reused": bool(args.psi_checkpoint) if args.method.startswith("uniground") else None,
        "uses_host_hidden_states": args.method.startswith("tlra_internal"),
        "uses_lm_head_geometry": args.method.startswith("tlra_internal"),
        "sample_count": total,
        "n_errors": len(errors),
        "unknown_count": unknown_count,
        "unknown_rate": round(unknown_count / max(total, 1), 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "hallucination_primary_metric": "f1",
        "hallucination_primary_value": round(f1, 4),
        "agl": round(agl, 2),
        "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
        "itl_ms_per_token": round(itl, 2),
        "tpot_ms_per_token": round(itl, 2),
        "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
        "peak_vram_gb": round(vram_peak / 1e9, 3),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3],
        "sample_outputs": sample_outputs,
    }
    metrics.update(aggregate_runtime_stats(route_stats))
    if audits:
        metrics["sample_audits"] = audits
    return metrics


def run_chair(model, processor, args, encoder: FrozenExternalEncoder | None) -> dict:
    all_cats, img_objs, id_from_file = load_coco_objects()
    images = sorted(COCO_IMG.glob("*.jpg"))[: args.mini_test]

    captions, gen_lengths, latencies, errors = [], [], [], []
    route_stats = []
    audits = []
    timer = GPUTimer()
    torch.cuda.reset_peak_memory_stats()

    for i, image_path in enumerate(images):
        try:
            text, gl, ms, stat_block, audit = generate_one_dispatch(
                model, processor, args, str(image_path), args.chair_prompt, timer, encoder
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] sample %s failed:\n%s", args.method, i, tb)
            errors.append({"sample": i, "error": str(exc), "traceback": tb})
            continue

        captions.append({"image": image_path.name, "caption": text})
        gen_lengths.append(gl)
        if gl > 0:
            latencies.append(ms / gl)
        if stat_block:
            route_stats.append(stat_block)
        if audit and len(audits) < 3:
            audits.append({"image": image_path.name, "audit": audit[:5], "caption_preview": text[:160]})

        if i < 3 or (i + 1) % 10 == 0:
            logger.info("[%s] [%s/%s] len=%s", args.method, i + 1, len(images), gl)

    chair_s, chair_i = compute_chair(captions, all_cats, img_objs, id_from_file)
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)
    vram_peak = torch.cuda.max_memory_allocated()

    metrics = {
        "route": route_name(args.method),
        "dataset": "chair",
        "method": args.method,
        "host_model": args.model,
        "host_model_family": "qwen3vl",
        "same_psi_checkpoint_reused": bool(args.psi_checkpoint) if args.method.startswith("uniground") else None,
        "uses_host_hidden_states": args.method.startswith("tlra_internal"),
        "uses_lm_head_geometry": args.method.startswith("tlra_internal"),
        "sample_count": len(captions),
        "n_errors": len(errors),
        "chair_s": round(chair_s, 4),
        "chair_i": round(chair_i, 4),
        "hallucination_primary_metric": "chair_i",
        "hallucination_primary_value": round(chair_i, 4),
        "agl": round(agl, 2),
        "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
        "itl_ms_per_token": round(itl, 2),
        "tpot_ms_per_token": round(itl, 2),
        "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
        "peak_vram_gb": round(vram_peak / 1e9, 3),
        "timestamp": datetime.now().isoformat(),
        "errors": errors[:3],
        "sample_captions": [caption["caption"][:150] for caption in captions[:3]],
    }
    metrics.update(aggregate_runtime_stats(route_stats))
    if audits:
        metrics["sample_audits"] = audits
    return metrics


def route_name(method: str) -> str:
    if method in {"base", "tlra_internal_zero", "tlra_internal_calib"}:
        return "internal_control"
    if method == "external_global_prior":
        return "external_global_prior"
    return "uniground"


def aggregate_runtime_stats(stats_blocks: list[dict]) -> dict:
    empty_latency_split = {
        "candidate_construction_ms": None,
        "sidecar_scoring_ms": None,
        "bridge_redistribution_ms": None,
        "jitter_ms": None,
    }
    if not stats_blocks:
        return {
            "intervention_coverage": None,
            "prefix_ambiguity_rate": None,
            "span_collapse_errors": None,
            "suffix_stability_rate": None,
            "abstention_rate": None,
            "abort_trigger_rate": None,
            "abort_backoff_verified_steps": None,
            "candidate_construction_ms": None,
            "sidecar_scoring_ms": None,
            "bridge_redistribution_ms": None,
            "jitter_ms": None,
            "latency_split": empty_latency_split,
        }
    keys = [
        "intervention_coverage",
        "prefix_ambiguity_rate",
        "span_collapse_errors",
        "suffix_stability_rate",
        "abstention_rate",
        "abort_trigger_rate",
        "abort_backoff_verified_steps",
        "candidate_construction_ms",
        "sidecar_scoring_ms",
        "bridge_redistribution_ms",
        "jitter_ms",
        "avg_candidate_window",
    ]
    out = {}
    for key in keys:
        vals = [block[key] for block in stats_blocks if block.get(key) is not None]
        if not vals:
            out[key] = None
        else:
            out[key] = round(sum(vals) / len(vals), 4)
    latency_split_blocks = [block.get("latency_split") for block in stats_blocks if isinstance(block.get("latency_split"), dict)]
    if latency_split_blocks:
        out["latency_split"] = {
            metric: round(sum(float(block.get(metric, 0.0)) for block in latency_split_blocks) / len(latency_split_blocks), 4)
            for metric in ("candidate_construction_ms", "sidecar_scoring_ms", "bridge_redistribution_ms", "jitter_ms")
        }
    else:
        out["latency_split"] = empty_latency_split
    return out


def output_path(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    name = f"uniground_{args.model}_{args.dataset}_{args.method}{tag}_{ts}.json"
    return out_dir / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_meta(checkpoint_path: str | None) -> dict | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
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


def run_single_method(model, processor, shared_encoder: FrozenExternalEncoder | None, args: argparse.Namespace) -> Path:
    encoder = shared_encoder if is_universal_method(args.method) else None
    if args.dataset == "pope":
        metrics = run_pope(model, processor, args, encoder)
    else:
        metrics = run_chair(model, processor, args, encoder)

    if args.method == "tlra_internal_calib":
        metrics["projector_checkpoint"] = args.projector_checkpoint
    if args.psi_checkpoint:
        metrics["psi_checkpoint"] = args.psi_checkpoint
    if is_universal_method(args.method):
        metrics["external_encoder"] = args.external_encoder
        metrics["decode_time_intervention"] = True
        metrics["final_token_generated_by_host_mllm"] = True
        metrics["runtime_knobs"] = {
            "top_k": args.top_k,
            "bias_scale": args.bias_scale,
            "score_temperature": args.score_temperature,
            "abstain_threshold": args.abstain_threshold,
            "bias_mode": args.bias_mode,
            "host_relative_strength": args.host_relative_strength,
            "max_prefix_chars": args.max_prefix_chars,
            "ambiguity_abort_threshold": args.ambiguity_abort_threshold,
            "abort_on_prefix_ambiguity": args.abort_on_prefix_ambiguity,
            "run_tag": args.run_tag,
        }
        checkpoint_meta = load_checkpoint_meta(args.psi_checkpoint) if args.psi_checkpoint else None
        metrics["universal_claim_manifest"] = build_universal_claim_manifest(checkpoint_meta)
        if checkpoint_meta:
            metrics["psi_univ_checkpoint"] = checkpoint_meta

    out_path = output_path(args)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", out_path)
    print(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    method_list = args.methods or [args.method]
    invalid = [method for method in method_list if method not in {
        "base",
        "tlra_internal_zero",
        "tlra_internal_calib",
        "uniground",
        "external_global_prior",
        "uniground_no_gate",
        "uniground_no_abstain",
        "uniground_global_only",
        "uniground_region_only",
    }]
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}")

    if "tlra_internal_calib" in method_list and not Path(args.projector_checkpoint).exists():
        raise FileNotFoundError(
            f"Control checkpoint not found: {args.projector_checkpoint}. "
            "Pass --projector_checkpoint explicitly or keep this control disabled."
        )

    if any(requires_psi_checkpoint(method) for method in method_list) and not args.psi_checkpoint:
        raise ValueError(
            "UniGround methods require --psi_checkpoint. "
            "No Psi_univ checkpoint was provided."
        )

    model, processor = load_host_model_and_processor(args.model)

    shared_encoder = None
    if any(is_universal_method(method) for method in method_list):
        logger.info("Loading frozen external encoder: %s", args.external_encoder)
        shared_encoder = FrozenExternalEncoder(args.external_encoder, device=args.external_device)

    for method in method_list:
        run_args = argparse.Namespace(**vars(args))
        run_args.method = method
        run_single_method(model, processor, shared_encoder, run_args)


if __name__ == "__main__":
    main()
