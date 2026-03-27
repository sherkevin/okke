#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


CANONICAL_VISION_ENCODER = "openai/clip-vit-large-patch14::image"
CANONICAL_TEXT_ENCODER = "openai/clip-vit-large-patch14::text"
ABSTAIN_TERMS = [
    "concept",
    "emotion",
    "intention",
    "mood",
    "plan",
    "reason",
    "strategy",
    "theme",
]
PREFIX_TEMPLATES = [
    "Question: Is there a {candidate} in the image? Answer:",
    "Verify visually whether the image contains {candidate}.",
    "Based on the image, should the span '{candidate}' be supported?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen external-space UniGround training payload from COCO-style annotations.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-dump", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=4096)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
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


def load_instances(instances_json: Path) -> tuple[dict[int, str], dict[int, set[int]], dict[int, str]]:
    payload = json.loads(instances_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]).strip().lower() for cat in payload["categories"]}
    image_paths = {int(image["id"]): str(image["file_name"]) for image in payload["images"]}
    image_to_categories: dict[int, set[int]] = defaultdict(set)
    for ann in payload["annotations"]:
        image_to_categories[int(ann["image_id"])].add(int(ann["category_id"]))
    return categories, image_to_categories, image_paths


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
        feats = _select_feature_tensor(feats)
        outputs.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_images(image_paths: list[Path], processor, model, device: torch.device, batch_size: int) -> torch.Tensor:
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
        feats = _select_feature_tensor(feats)
        outputs.append(F.normalize(feats.float(), dim=-1).cpu())
        for image in images:
            image.close()
    return torch.cat(outputs, dim=0)


def choose_negative_category(all_category_ids: list[int], present: set[int], rng: random.Random) -> int:
    absent = [cat_id for cat_id in all_category_ids if cat_id not in present]
    return rng.choice(absent)


def _select_feature_tensor(outputs) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    pooler_output = getattr(outputs, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    image_embeds = getattr(outputs, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds
    text_embeds = getattr(outputs, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]
    raise TypeError(f"Unsupported encoder output type: {type(outputs)!r}")


def build_records(
    image_dir: Path,
    categories: dict[int, str],
    image_to_categories: dict[int, set[int]],
    image_paths: dict[int, str],
    max_images: int,
    seed: int,
) -> tuple[list[dict], list[Path]]:
    rng = random.Random(seed)
    all_category_ids = sorted(categories)
    records: list[dict] = []
    selected_images: list[Path] = []

    for image_id in sorted(image_paths):
        if len(selected_images) >= max_images:
            break
        present = image_to_categories.get(image_id)
        if not present:
            continue
        image_path = image_dir / image_paths[image_id]
        if not image_path.exists():
            continue

        selected_images.append(image_path)
        positive_name = categories[rng.choice(sorted(present))]
        negative_name = categories[choose_negative_category(all_category_ids, present, rng)]
        abstain_name = rng.choice(ABSTAIN_TERMS)

        triples = [
            (positive_name, [1.0, 0.0, 0.0], "support"),
            (negative_name, [0.0, 1.0, 0.0], "contradiction"),
            (abstain_name, [0.0, 0.0, 1.0], "abstain"),
        ]
        for idx, (candidate, label, label_name) in enumerate(triples):
            template = PREFIX_TEMPLATES[(len(records) + idx) % len(PREFIX_TEMPLATES)]
            prefix_text = template.format(candidate=candidate)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "candidate_text": candidate,
                    "prefix_text": prefix_text,
                    "label": label,
                    "label_name": label_name,
                    "present_categories": sorted(categories[cat_id] for cat_id in present),
                }
            )
    return records, selected_images


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    instances_json = Path(args.instances_json)
    output_path = Path(args.output)
    config_dump_path = Path(args.config_dump) if args.config_dump else output_path.with_suffix(".config.json")

    categories, image_to_categories, image_paths = load_instances(instances_json)
    records, selected_images = build_records(
        image_dir=image_dir,
        categories=categories,
        image_to_categories=image_to_categories,
        image_paths=image_paths,
        max_images=args.max_images,
        seed=args.seed,
    )
    if not records:
        raise SystemExit("No exportable records were built from the provided image dir and instances json.")

    processor, model, device = load_encoder(args.encoder_path, args.device)
    unique_image_paths = sorted({Path(record["image_path"]) for record in records})
    unique_candidates = sorted({record["candidate_text"] for record in records})
    unique_prefixes = sorted({record["prefix_text"] for record in records})

    image_embeddings = encode_images(unique_image_paths, processor, model, device, args.image_batch_size)
    candidate_embeddings = encode_texts(unique_candidates, processor, model, device, args.text_batch_size)
    prefix_embeddings = encode_texts(unique_prefixes, processor, model, device, args.text_batch_size)

    image_index = {str(path): idx for idx, path in enumerate(unique_image_paths)}
    candidate_index = {text: idx for idx, text in enumerate(unique_candidates)}
    prefix_index = {text: idx for idx, text in enumerate(unique_prefixes)}

    payload_image = torch.stack([image_embeddings[image_index[record["image_path"]]] for record in records], dim=0)
    payload_candidate = torch.stack([candidate_embeddings[candidate_index[record["candidate_text"]]] for record in records], dim=0)
    payload_prefix = torch.stack([prefix_embeddings[prefix_index[record["prefix_text"]]] for record in records], dim=0)
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_script = Path(__file__)
    runtime_script = export_script.with_name("uniground_runtime.py")
    source_hashes = {
        "instances_json": file_sha256(instances_json),
        "image_listing": listing_sha256(unique_image_paths),
        "export_universal_coco_payload.py": file_sha256(export_script),
        "uniground_runtime.py": file_sha256(runtime_script),
    }
    metadata = {
        "record_count": len(records),
        "image_count": len(unique_image_paths),
        "candidate_vocab_size": len(unique_candidates),
        "prefix_count": len(unique_prefixes),
        "label_schema": ["support", "contradiction", "abstain"],
        "source_paths": {
            "image_dir": str(image_dir.resolve()),
            "instances_json": str(instances_json.resolve()),
        },
        "source_hashes": source_hashes,
        "encoder_contract": {
            "frozen_vision_encoder_name": CANONICAL_VISION_ENCODER,
            "frozen_text_encoder_name": CANONICAL_TEXT_ENCODER,
            "encoder_load_path": str(Path(args.encoder_path).resolve()),
            "device": str(device),
        },
        "records_preview": records[:12],
    }
    torch.save(
        {
            "image_embeddings": payload_image,
            "candidate_embeddings": payload_candidate,
            "prefix_embeddings": payload_prefix,
            "labels": labels,
            "metadata": metadata,
        },
        output_path,
    )
    payload_sha256 = file_sha256(output_path)
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
    print(f"Saved config dump to {config_dump_path}")


if __name__ == "__main__":
    main()
