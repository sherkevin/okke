#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from bra_logits_processor import BRAVisionExtractor, make_bra_config
from bra_operator_multi import detect_adapter
from bra_projector import ProjectorSpec, create_projector
from bra_vasm import normalize_token


PROJECT = Path("/root/autodl-tmp/BRA_Project")
MODEL_MAP = {
    "qwen3-vl-2b": PROJECT / "models" / "Qwen3-VL-2B-Instruct",
    "qwen3-vl-8b": PROJECT / "models" / "Qwen3-VL-8B-Instruct",
}
COCO_IMG = PROJECT / "datasets" / "coco2014" / "val2014"
COCO_ANN = PROJECT / "datasets" / "coco2014" / "annotations" / "instances_val2014.json"
LOG_DIR = PROJECT / "logs" / "minitest"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl-2b", choices=sorted(MODEL_MAP))
    parser.add_argument("--method", default="tlra_zero")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--candidate_window", type=int, default=50)
    parser.add_argument("--projector_checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_model_and_processor(model_key: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = MODEL_MAP[model_key]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    return model, processor


def load_coco_samples(n_samples: int):
    data = json.loads(COCO_ANN.read_text(encoding="utf-8"))
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_meta = {img["id"]: img["file_name"] for img in data["images"]}
    img_labels = defaultdict(set)
    for ann in data["annotations"]:
        img_labels[ann["image_id"]].add(cat_map[ann["category_id"]])

    samples = []
    for image_id, labels in img_labels.items():
        image_path = COCO_IMG / img_meta[image_id]
        if image_path.exists() and labels:
            samples.append((image_id, image_path, sorted(labels)))
        if len(samples) >= n_samples:
            break
    return samples


def build_input(processor, image_path: Path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": "Name the main objects in this image."},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def label_targets(labels, tokenizer):
    targets = set()
    for label in labels:
        targets.add(label.lower())
        for word in label.lower().split():
            clean = normalize_token(word)
            if clean:
                targets.add(clean)
        tokenized = tokenizer.tokenize(" " + label)
        for token in tokenized:
            clean = normalize_token(token)
            if clean:
                targets.add(clean)
    return targets


def build_projector(cfg, z_dim: int, out_dim: int, device: torch.device):
    spec = None
    if cfg.projector_checkpoint:
        spec = cfg.projector_checkpoint
    projector = create_projector(
        ProjectorSpec(kind=cfg.projector_kind, checkpoint_path=spec),
        input_dim=z_dim,
        output_dim=out_dim,
        device=device,
    )
    projector.eval()
    return projector


def evaluate_sample(model, processor, adapter, image_path: Path, labels, args):
    cfg = make_bra_config(args.method, projector_checkpoint=args.projector_checkpoint)
    extractor = BRAVisionExtractor(model, adapter)
    inputs = build_input(processor, image_path)
    image_token_id = adapter.get_image_token_id(model)
    video_token_id = adapter.get_video_token_id(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        with torch.no_grad():
            model(**inputs)
        z_all, _, _ = extractor.extract_vision_features(
            inputs["input_ids"],
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            video_grid_thw=inputs.get("video_grid_thw"),
            build_temporal_diff=False,
        )
    finally:
        extractor.remove()

    if z_all is None or z_all.shape[0] == 0:
        return None

    device = z_all.device
    lm_head = adapter.get_lm_head_weight(model).to(device).float()
    projector = build_projector(cfg, z_all.shape[-1], lm_head.shape[-1], device)

    with torch.no_grad():
        z_proj = projector(z_all.float())
        scores = z_proj @ lm_head.T
        patch_topk_vals, patch_topk_ids = scores.topk(args.topk, dim=-1)

    targets = label_targets(labels, tokenizer)
    patch_token_lists = []
    hits = {1: 0, 5: 0, 10: 0}
    image_hits = {1: False, 5: False, 10: False}

    for patch_idx in range(patch_topk_ids.shape[0]):
        token_ids = patch_topk_ids[patch_idx].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        normalized = [normalize_token(tok) for tok in tokens]
        patch_token_lists.append(normalized)
        for k in (1, 5, 10):
            if any(tok in targets for tok in normalized[: min(k, len(normalized))]):
                hits[k] += 1
                image_hits[k] = True

    candidate_scores = scores.max(dim=0).values
    candidate_topm_ids = candidate_scores.topk(min(args.candidate_window, candidate_scores.shape[0]), dim=-1).indices
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_topm_ids.tolist())
    candidate_overlap = any(normalize_token(tok) in targets for tok in candidate_tokens)

    return {
        "image": image_path.name,
        "labels": labels,
        "targets": sorted(targets),
        "n_patches": int(patch_topk_ids.shape[0]),
        "image_hit_top1": image_hits[1],
        "image_hit_top5": image_hits[5],
        "image_hit_top10": image_hits[10],
        "patch_hit_rate_top1": hits[1] / max(patch_topk_ids.shape[0], 1),
        "patch_hit_rate_top5": hits[5] / max(patch_topk_ids.shape[0], 1),
        "patch_hit_rate_top10": hits[10] / max(patch_topk_ids.shape[0], 1),
        "candidate_window_overlap": candidate_overlap,
        "sample_patch_tokens": patch_token_lists[:5],
        "sample_patch_scores": [
            [round(float(x), 4) for x in row.tolist()]
            for row in patch_topk_vals[:5]
        ],
    }


def summarize(results, args):
    total = len(results)
    if total == 0:
        return {
            "method": args.method,
            "model": args.model,
            "n_samples": 0,
            "notes": ["no_valid_samples"],
            "timestamp": datetime.now().isoformat(),
        }

    def avg(key):
        return sum(float(r[key]) for r in results) / total

    return {
        "method": args.method,
        "model": args.model,
        "n_samples": total,
        "top1_overlap": round(avg("image_hit_top1"), 4),
        "top5_overlap": round(avg("image_hit_top5"), 4),
        "top10_overlap": round(avg("image_hit_top10"), 4),
        "patch_top1_overlap": round(avg("patch_hit_rate_top1"), 4),
        "patch_top5_overlap": round(avg("patch_hit_rate_top5"), 4),
        "patch_top10_overlap": round(avg("patch_hit_rate_top10"), 4),
        "candidate_window_overlap": round(avg("candidate_window_overlap"), 4),
        "notes": [],
        "timestamp": datetime.now().isoformat(),
        "samples": results[:5],
    }


def main():
    args = parse_args()
    model, processor = load_model_and_processor(args.model)
    adapter = detect_adapter(model)
    samples = load_coco_samples(args.n_samples)

    results = []
    for idx, (_image_id, image_path, labels) in enumerate(samples):
        item = evaluate_sample(model, processor, adapter, image_path, labels, args)
        if item is None:
            continue
        results.append(item)
        print(
            f"[{idx + 1}/{len(samples)}] {image_path.name} "
            f"top1={item['image_hit_top1']} top5={item['image_hit_top5']} "
            f"candidate_overlap={item['candidate_window_overlap']}"
        )

    summary = summarize(results, args)
    if summary.get("top10_overlap", 0.0) < 0.05:
        summary["notes"].append("tlra_zero_near_random")
    if summary.get("candidate_window_overlap", 0.0) < 0.25:
        summary["notes"].append("high_out_of_candidate_risk")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else LOG_DIR / (
        f"tlra_semantic_validity_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
