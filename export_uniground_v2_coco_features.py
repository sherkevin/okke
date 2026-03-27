#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


CANONICAL_VISION_ENCODER = "openai/clip-vit-large-patch14::image"
CANONICAL_TEXT_ENCODER = "openai/clip-vit-large-patch14::text"
ABSTAIN_TERMS = [
    "emotion",
    "intention",
    "motivation",
    "plan",
    "reason",
    "theme",
    "story",
    "symbolism",
]
PREFIX_TEMPLATES = [
    "Image context: {caption}. A grounded next mention could be",
    "Visual summary: {caption}. The image clearly supports the entity",
    "Caption draft: {caption}. A visually justified noun phrase is",
    "Question: considering the image and caption '{caption}', should the next grounded entity be",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a real UniGround v2 feature payload from COCO-style annotations and captions."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config-dump", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=2048)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--max-regions-per-record", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listing_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _select_feature_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor):
            return value[:, 0] if attr == "last_hidden_state" else value
    raise TypeError(f"Unsupported encoder output type: {type(outputs)!r}")


def load_encoder(encoder_path: str, device: str):
    from transformers import AutoModel, AutoProcessor, CLIPProcessor

    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    try:
        processor = AutoProcessor.from_pretrained(encoder_path)
    except Exception:
        processor = CLIPProcessor.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path).to(device_obj)
    model.eval()
    return processor, model, device_obj


@torch.no_grad()
def encode_texts(texts: list[str], processor, model, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**batch)
        else:
            feats = _select_feature_tensor(model(**batch))
        outputs.append(F.normalize(_select_feature_tensor(feats).float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_image_paths(image_paths: list[Path], processor, model, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        batch = processor(images=images, return_tensors="pt")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**batch)
        else:
            feats = _select_feature_tensor(model(**batch))
        outputs.append(F.normalize(_select_feature_tensor(feats).float(), dim=-1).cpu())
        for image in images:
            image.close()
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_pil_images(images: list[Image.Image], processor, model, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch = processor(images=batch_images, return_tensors="pt")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**batch)
        else:
            feats = _select_feature_tensor(model(**batch))
        outputs.append(F.normalize(_select_feature_tensor(feats).float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def load_instances(instances_json: Path) -> tuple[dict[int, str], dict[int, str], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(instances_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]).strip().lower() for cat in payload["categories"]}
    image_paths = {int(image["id"]): str(image["file_name"]) for image in payload["images"]}
    image_to_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload["annotations"]:
        box = ann.get("bbox") or [0, 0, 0, 0]
        width = float(box[2])
        height = float(box[3])
        if width <= 1 or height <= 1:
            continue
        image_to_annotations[int(ann["image_id"])].append(
            {
                "category_id": int(ann["category_id"]),
                "bbox": [float(v) for v in box],
                "area": float(ann.get("area", width * height)),
            }
        )
    return categories, image_paths, image_to_annotations


def load_captions(captions_json: Path) -> dict[int, list[str]]:
    payload = json.loads(captions_json.read_text(encoding="utf-8"))
    image_to_captions: dict[int, list[str]] = defaultdict(list)
    for ann in payload["annotations"]:
        caption = str(ann.get("caption", "")).strip()
        if caption:
            image_to_captions[int(ann["image_id"])].append(caption)
    return image_to_captions


def choose_negative_category(all_category_ids: list[int], present: set[int], rng: random.Random) -> int:
    absent = [cat_id for cat_id in all_category_ids if cat_id not in present]
    return rng.choice(absent)


def fallback_caption(present_names: list[str]) -> str:
    preview = ", ".join(present_names[:3])
    return f"the scene contains {preview}"


def build_prefix(caption: str, template_index: int) -> str:
    normalized = " ".join(caption.strip().split())
    return PREFIX_TEMPLATES[template_index % len(PREFIX_TEMPLATES)].format(caption=normalized)


def build_query_text(object_name: str) -> str:
    return f"a photo of {object_name}"


def build_present_hypothesis(object_name: str) -> str:
    return f"a photo containing {object_name}"


def build_absent_hypothesis(object_name: str) -> str:
    return f"a photo without {object_name}"


def bbox_to_key(image_path: str, bbox: list[float]) -> str:
    return f"{image_path}::" + ",".join(f"{value:.2f}" for value in bbox)


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    left = max(0, min(width, int(round(bbox[0]))))
    top = max(0, min(height, int(round(bbox[1]))))
    right = max(0, min(width, int(round(bbox[0] + bbox[2]))))
    bottom = max(0, min(height, int(round(bbox[1] + bbox[3]))))
    if right - left < 2:
        if right < width:
            right = min(width, left + 2)
        else:
            left = max(0, right - 2)
    if bottom - top < 2:
        if bottom < height:
            bottom = min(height, top + 2)
        else:
            top = max(0, bottom - 2)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def pick_region_boxes(
    annotations: list[dict[str, Any]],
    *,
    target_category_id: int | None,
    max_regions: int,
) -> list[list[float]]:
    matching = [ann for ann in annotations if target_category_id is not None and ann["category_id"] == target_category_id]
    source = matching if matching else annotations
    ranked = sorted(source, key=lambda ann: ann["area"], reverse=True)
    return [ann["bbox"] for ann in ranked[:max_regions]]


def build_records(
    *,
    image_dir: Path,
    categories: dict[int, str],
    image_paths: dict[int, str],
    image_to_annotations: dict[int, list[dict[str, Any]]],
    image_to_captions: dict[int, list[str]],
    max_images: int,
    max_regions_per_record: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    rng = random.Random(seed)
    all_category_ids = sorted(categories)
    records: list[dict[str, Any]] = []
    selected_images: list[Path] = []

    for image_id in sorted(image_paths):
        if len(selected_images) >= max_images:
            break
        annotations = image_to_annotations.get(image_id, [])
        if not annotations:
            continue
        image_path = image_dir / image_paths[image_id]
        if not image_path.exists():
            continue

        present_ids = sorted({ann["category_id"] for ann in annotations})
        if len(present_ids) == len(all_category_ids):
            continue

        selected_images.append(image_path)
        present_names = [categories[cat_id] for cat_id in present_ids]
        caption_pool = image_to_captions.get(image_id) or [fallback_caption(present_names)]
        caption = rng.choice(caption_pool)
        positive_cat_id = rng.choice(present_ids)
        negative_cat_id = choose_negative_category(all_category_ids, set(present_ids), rng)
        abstain_term = rng.choice(ABSTAIN_TERMS)

        record_specs = [
            {
                "query_text": build_query_text(categories[positive_cat_id]),
                "hypothesis_text": build_present_hypothesis(categories[positive_cat_id]),
                "label": [1.0, 0.0, 0.0],
                "label_name": "support",
                "target_category_id": positive_cat_id,
                "hypothesis_family": "object_presence",
                "target_object_label": categories[positive_cat_id],
            },
            {
                "query_text": build_query_text(categories[positive_cat_id]),
                "hypothesis_text": build_absent_hypothesis(categories[positive_cat_id]),
                "label": [0.0, 1.0, 0.0],
                "label_name": "contradiction",
                "target_category_id": positive_cat_id,
                "hypothesis_family": "object_absence",
                "target_object_label": categories[positive_cat_id],
            },
            {
                "query_text": build_query_text(categories[negative_cat_id]),
                "hypothesis_text": build_present_hypothesis(categories[negative_cat_id]),
                "label": [0.0, 1.0, 0.0],
                "label_name": "contradiction",
                "target_category_id": None,
                "hypothesis_family": "object_presence",
                "target_object_label": categories[negative_cat_id],
            },
            {
                "query_text": build_query_text(categories[negative_cat_id]),
                "hypothesis_text": build_absent_hypothesis(categories[negative_cat_id]),
                "label": [1.0, 0.0, 0.0],
                "label_name": "support",
                "target_category_id": None,
                "hypothesis_family": "object_absence",
                "target_object_label": categories[negative_cat_id],
            },
            {
                "query_text": abstain_term,
                "hypothesis_text": abstain_term,
                "label": [0.0, 0.0, 1.0],
                "label_name": "abstain",
                "target_category_id": None,
                "hypothesis_family": "non_visual_abstain",
                "target_object_label": abstain_term,
            },
        ]

        for offset, spec in enumerate(record_specs):
            region_boxes = pick_region_boxes(
                annotations,
                target_category_id=spec["target_category_id"],
                max_regions=max_regions_per_record,
            )
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "candidate_text": spec["hypothesis_text"],
                    "prefix_text": spec["query_text"],
                    "query_text": spec["query_text"],
                    "hypothesis_text": spec["hypothesis_text"],
                    "caption_prompt_text": build_prefix(caption, len(records) + offset),
                    "caption_text": caption,
                    "label": spec["label"],
                    "label_name": spec["label_name"],
                    "hypothesis_family": spec["hypothesis_family"],
                    "target_object_label": spec["target_object_label"],
                    "region_boxes": region_boxes,
                    "present_categories": present_names,
                }
            )
    return records, selected_images


def encode_region_crops(
    *,
    records: list[dict[str, Any]],
    processor,
    model,
    device: torch.device,
    batch_size: int,
    image_dim: int,
    max_regions_per_record: int,
    image_embedding_map: dict[str, torch.Tensor],
) -> torch.Tensor:
    unique_region_keys: dict[str, tuple[str, list[float]]] = {}
    for record in records:
        for bbox in record["region_boxes"]:
            unique_region_keys.setdefault(bbox_to_key(record["image_path"], bbox), (record["image_path"], bbox))

    region_embedding_map: dict[str, torch.Tensor] = {}
    region_keys = sorted(unique_region_keys)
    for start in range(0, len(region_keys), batch_size):
        batch_keys = region_keys[start:start + batch_size]
        crops: list[Image.Image] = []
        valid_keys: list[str] = []
        opened_by_path: dict[str, Image.Image] = {}
        try:
            for key in batch_keys:
                image_path, bbox = unique_region_keys[key]
                base = opened_by_path.get(image_path)
                if base is None:
                    base = Image.open(image_path).convert("RGB")
                    opened_by_path[image_path] = base
                clamped = clamp_bbox(bbox, base.size[0], base.size[1])
                if clamped is None:
                    continue
                crops.append(base.crop(clamped))
                valid_keys.append(key)
            if valid_keys:
                embeddings = encode_pil_images(crops, processor, model, device, batch_size=len(crops))
                for key, embedding in zip(valid_keys, embeddings):
                    region_embedding_map[key] = embedding
        finally:
            for crop in crops:
                crop.close()
            for image in opened_by_path.values():
                image.close()

    fallback = torch.zeros(image_dim, dtype=torch.float32)
    stacked: list[torch.Tensor] = []
    for record in records:
        chosen: list[torch.Tensor] = []
        for bbox in record["region_boxes"][:max_regions_per_record]:
            key = bbox_to_key(record["image_path"], bbox)
            if key in region_embedding_map:
                chosen.append(region_embedding_map[key])
        if not chosen:
            chosen.append(image_embedding_map.get(record["image_path"], fallback))
        while len(chosen) < max_regions_per_record:
            chosen.append(chosen[-1])
        stacked.append(torch.stack(chosen[:max_regions_per_record], dim=0))
    return torch.stack(stacked, dim=0)


def build_payload_metadata(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    unique_image_paths: list[Path],
    unique_candidates: list[str],
    unique_prefixes: list[str],
    instances_json: Path,
    captions_json: Path,
) -> dict[str, Any]:
    export_script = Path(__file__)
    label_counts = {"support": 0, "contradiction": 0, "abstain": 0}
    captioned_records = 0
    for record in records:
        label_counts[record["label_name"]] = label_counts.get(record["label_name"], 0) + 1
        if record.get("caption_text"):
            captioned_records += 1
    return {
        "record_count": len(records),
        "image_count": len(unique_image_paths),
        "hypothesis_vocab_size": len(unique_candidates),
        "query_count": len(unique_prefixes),
        "label_counts": label_counts,
        "caption_coverage_rate": round(captioned_records / max(len(records), 1), 6),
        "label_schema": ["support", "contradiction", "abstain"],
        "training_semantics": {
            "query_embedding_key": "query_embeddings",
            "hypothesis_embedding_key": "hypothesis_embeddings",
            "runtime_alignment": "query_plus_hypothesis",
            "hypothesis_families": sorted({record.get("hypothesis_family", "unknown") for record in records}),
        },
        "source_paths": {
            "image_dir": str(Path(args.image_dir).resolve()),
            "instances_json": str(instances_json.resolve()),
            "captions_json": str(captions_json.resolve()),
        },
        "source_hashes": {
            "instances_json": file_sha256(instances_json),
            "captions_json": file_sha256(captions_json),
            "image_listing": listing_sha256(unique_image_paths),
            "export_uniground_v2_coco_features.py": file_sha256(export_script),
        },
        "encoder_contract": {
            "frozen_vision_encoder_name": CANONICAL_VISION_ENCODER,
            "frozen_text_encoder_name": CANONICAL_TEXT_ENCODER,
            "encoder_load_path": str(Path(args.encoder_path).resolve()),
            "device": args.device,
        },
        "region_mode": "gt_bbox_topr",
        "max_regions_per_record": args.max_regions_per_record,
        "augmentation_policy": {
            "llm_required": False,
            "llm_used": False,
            "mode": "rule_only_coco_instances_captions",
            "note": "First formal payload uses only human annotations and captions. LLM augmentation is optional for later hard negatives.",
        },
        "records_preview": records[:12],
    }


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    instances_json = Path(args.instances_json)
    captions_json = Path(args.captions_json)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    config_dump_path = Path(args.config_dump) if args.config_dump else output_path.with_suffix(".config.json")

    categories, image_paths, image_to_annotations = load_instances(instances_json)
    image_to_captions = load_captions(captions_json)
    records, selected_images = build_records(
        image_dir=image_dir,
        categories=categories,
        image_paths=image_paths,
        image_to_annotations=image_to_annotations,
        image_to_captions=image_to_captions,
        max_images=args.max_images,
        max_regions_per_record=args.max_regions_per_record,
        seed=args.seed,
    )
    if not records:
        raise SystemExit("No exportable records were built from the provided image dir and annotation assets.")

    processor, model, device = load_encoder(args.encoder_path, args.device)
    unique_image_paths = sorted({Path(record["image_path"]) for record in records})
    unique_candidates = sorted({record["hypothesis_text"] for record in records})
    unique_prefixes = sorted({record["query_text"] for record in records})

    image_embeddings = encode_image_paths(unique_image_paths, processor, model, device, args.image_batch_size)
    candidate_embeddings = encode_texts(unique_candidates, processor, model, device, args.text_batch_size)
    prefix_embeddings = encode_texts(unique_prefixes, processor, model, device, args.text_batch_size)

    image_index = {str(path): idx for idx, path in enumerate(unique_image_paths)}
    candidate_index = {text: idx for idx, text in enumerate(unique_candidates)}
    prefix_index = {text: idx for idx, text in enumerate(unique_prefixes)}
    image_embedding_map = {str(path): image_embeddings[idx] for idx, path in enumerate(unique_image_paths)}

    payload_image = torch.stack([image_embeddings[image_index[record["image_path"]]] for record in records], dim=0)
    payload_candidate = torch.stack([candidate_embeddings[candidate_index[record["hypothesis_text"]]] for record in records], dim=0)
    payload_prefix = torch.stack([prefix_embeddings[prefix_index[record["query_text"]]] for record in records], dim=0)
    payload_regions = encode_region_crops(
        records=records,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.region_batch_size,
        image_dim=int(payload_image.shape[-1]),
        max_regions_per_record=args.max_regions_per_record,
        image_embedding_map=image_embedding_map,
    )
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)

    metadata = build_payload_metadata(
        args=args,
        records=records,
        unique_image_paths=unique_image_paths,
        unique_candidates=unique_candidates,
        unique_prefixes=unique_prefixes,
        instances_json=instances_json,
        captions_json=captions_json,
    )
    payload = {
        "image_embeddings": payload_image,
        "hypothesis_embeddings": payload_candidate,
        "candidate_embeddings": payload_candidate,
        "query_embeddings": payload_prefix,
        "prefix_embeddings": payload_prefix,
        "retrieved_region_embeddings": payload_regions,
        "region_embeddings": payload_regions,
        "labels": labels,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    payload_sha256 = file_sha256(output_path)
    manifest_path.write_text(
        json.dumps(
            {
                "payload_path": str(output_path.resolve()),
                "payload_sha256": payload_sha256,
                "record_count": len(records),
                "region_shape": list(payload_regions.shape),
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_dump_path.write_text(
        json.dumps(
            {
                "output_path": str(output_path.resolve()),
                "output_sha256": payload_sha256,
                "metadata": metadata,
                "export_args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved payload to {output_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved config dump to {config_dump_path}")


if __name__ == "__main__":
    main()
