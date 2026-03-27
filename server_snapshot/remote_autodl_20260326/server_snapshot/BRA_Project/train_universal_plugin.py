#!/usr/bin/env python3
"""
Train the universal plugin core `Psi_univ` on frozen external embeddings.

Expected training payload keys:
- `image_embeddings`: [N, D]
- `candidate_embeddings`: [N, D]
- `prefix_embeddings`: [N, D]
- `labels`: [N, 3] or [N]

This script intentionally avoids any dependency on model-native hidden states,
token IDs, or `lm_head.weight`. A checkpoint trained here is designed to be
portable across MLLM families, provided the same frozen external encoders are
used at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bra_universal_plugin import MLPUniversalScorer, UniversalObservation


CANONICAL_VISION_ENCODER = "openai/clip-vit-large-patch14::image"
CANONICAL_TEXT_ENCODER = "openai/clip-vit-large-patch14::text"
CHECKPOINT_FORMAT = "psi_univ_checkpoint_v1"
CONTRACT_VERSION = "uniground_train_contract_v1"


def load_training_payload(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Training payload must be a dict.")
    return payload


def load_training_tensors(payload: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {"image_embeddings", "candidate_embeddings", "prefix_embeddings", "labels"}
    if not required.issubset(payload):
        raise ValueError(
            "Training payload must contain image_embeddings, candidate_embeddings, "
            "prefix_embeddings, and labels."
        )

    image_embeddings = payload["image_embeddings"].float()
    candidate_embeddings = payload["candidate_embeddings"].float()
    prefix_embeddings = payload["prefix_embeddings"].float()
    labels = payload["labels"]

    if image_embeddings.ndim != 2 or candidate_embeddings.ndim != 2 or prefix_embeddings.ndim != 2:
        raise ValueError("Embedding tensors must be 2D.")
    if image_embeddings.shape != candidate_embeddings.shape or image_embeddings.shape != prefix_embeddings.shape:
        raise ValueError("All embedding tensors must share the same [N, D] shape.")
    if labels.shape[0] != image_embeddings.shape[0]:
        raise ValueError("Labels must align with the batch dimension.")

    return image_embeddings, candidate_embeddings, prefix_embeddings, labels


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        return F.cross_entropy(logits, labels.long())
    if labels.ndim == 2 and labels.shape[-1] == 3:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    raise ValueError("Labels must have shape [N] or [N, 3].")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to .pt feature payload")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--plugin-version", default=CONTRACT_VERSION)
    parser.add_argument("--vision-encoder-name", default=CANONICAL_VISION_ENCODER)
    parser.add_argument("--text-encoder-name", default=CANONICAL_TEXT_ENCODER)
    parser.add_argument("--region-features-enabled", action="store_true")
    parser.add_argument("--config-dump", default=None)
    args = parser.parse_args()

    features_payload = load_training_payload(args.features)
    image_embeddings, candidate_embeddings, prefix_embeddings, labels = load_training_tensors(features_payload)
    dataset = TensorDataset(image_embeddings, candidate_embeddings, prefix_embeddings, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = int(image_embeddings.shape[-1])
    model = MLPUniversalScorer(embed_dim=embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for image_batch, candidate_batch, prefix_batch, label_batch in loader:
            image_batch = image_batch.to(device)
            candidate_batch = candidate_batch.to(device)
            prefix_batch = prefix_batch.to(device)
            label_batch = label_batch.to(device)

            observation = UniversalObservation(image_embedding=image_batch)
            output = model(observation, candidate_batch, prefix_batch)
            logits = torch.stack([output.support, output.contradiction, output.abstain], dim=-1)
            loss = compute_loss(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * image_batch.shape[0]
            count += image_batch.shape[0]

        print(f"epoch={epoch} loss={running / max(count, 1):.6f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_files = {
        "train_universal_plugin.py": file_sha256(Path(__file__)),
        "bra_universal_plugin.py": file_sha256(Path(__file__).with_name("bra_universal_plugin.py")),
    }
    payload_metadata = features_payload.get("metadata", {}) if isinstance(features_payload.get("metadata"), dict) else {}
    payload_source_hashes = payload_metadata.get("source_hashes", {}) if isinstance(payload_metadata.get("source_hashes"), dict) else {}
    features_payload_sha256 = file_sha256(Path(args.features))
    contract = {
        "contract_version": args.plugin_version,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "plugin_route": "TLRA_univ",
        "learned_module": "Psi_univ",
        "frozen_vision_encoder_name": args.vision_encoder_name,
        "frozen_text_encoder_name": args.text_encoder_name,
        "embedding_dim": embed_dim,
        "region_features_enabled": bool(args.region_features_enabled),
        "uses_model_native_hidden_states": False,
        "uses_lm_head_geometry": False,
        "learned_per_model_adapter": False,
        "parameter_free_tokenizer_bridge": True,
        "source_hashes": {
            **source_files,
            **payload_source_hashes,
            "features_payload": features_payload_sha256,
        },
        "features_path": str(Path(args.features).resolve()),
        "training_source_paths": payload_metadata.get("source_paths", {}),
        "training_record_count": payload_metadata.get("record_count"),
        "training_image_count": payload_metadata.get("image_count"),
    }
    payload = {
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "hidden_dim": args.hidden_dim,
        "contract": contract,
    }
    torch.save(payload, out)
    checkpoint_sha256 = file_sha256(out)
    config_dump_path = Path(args.config_dump) if args.config_dump else out.with_suffix(".config.json")
    config_dump = {
        "checkpoint_path": str(out.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "contract": contract,
        "training_args": {
            "features": str(Path(args.features).resolve()),
            "features_payload_sha256": features_payload_sha256,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
        },
        "training_source_paths": payload_metadata.get("source_paths", {}),
        "training_source_hashes": payload_source_hashes,
    }
    config_dump_path.write_text(json.dumps(config_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved universal plugin checkpoint to {out}")
    print(f"Saved config dump to {config_dump_path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
