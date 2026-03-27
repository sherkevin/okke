#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bra_universal_plugin import MLPUniversalScorer, UniversalObservation
from uniground_v2.training_dataset import PsiUnivV2Dataset, load_feature_payload, load_training_tensors_v2


CANONICAL_VISION_ENCODER = "openai/clip-vit-large-patch14::image"
CANONICAL_TEXT_ENCODER = "openai/clip-vit-large-patch14::text"
CHECKPOINT_FORMAT = "psi_univ_checkpoint_v2"
CONTRACT_VERSION = "uniground_train_contract_v2"


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        return F.cross_entropy(logits, labels.long())
    if labels.ndim == 2 and labels.shape[-1] == 3:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    raise ValueError("Labels must have shape [N] or [N, 3].")


def _dataset_semantics(metadata: dict) -> dict:
    semantics = metadata.get("training_semantics")
    return semantics if isinstance(semantics, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract_v2(
    *,
    args: argparse.Namespace,
    embed_dim: int,
    features_path: Path,
    payload_metadata: dict,
) -> dict:
    source_files = {
        "train_universal_plugin_v2.py": file_sha256(Path(__file__)),
        "bra_universal_plugin.py": file_sha256(Path(__file__).with_name("bra_universal_plugin.py")),
        "uniground_v2/training_dataset.py": file_sha256(Path(__file__).with_name("uniground_v2").joinpath("training_dataset.py")),
    }
    payload_source_hashes = payload_metadata.get("source_hashes", {}) if isinstance(payload_metadata.get("source_hashes"), dict) else {}
    return {
        "contract_version": CONTRACT_VERSION,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "plugin_route": "TLRA_univ",
        "learned_module": "Psi_univ",
        "frozen_vision_encoder_name": args.vision_encoder_name,
        "frozen_text_encoder_name": args.text_encoder_name,
        "embedding_dim": embed_dim,
        "region_features_enabled": True,
        "region_mode": args.region_mode,
        "uses_model_native_hidden_states": False,
        "uses_lm_head_geometry": False,
        "learned_per_model_adapter": False,
        "parameter_free_tokenizer_bridge": True,
        "source_hashes": {
            **source_files,
            **payload_source_hashes,
            "features_payload": file_sha256(features_path),
        },
        "features_path": str(features_path.resolve()),
        "training_source_paths": payload_metadata.get("source_paths", {}),
        "training_record_count": payload_metadata.get("record_count"),
        "training_image_count": payload_metadata.get("image_count"),
        "training_semantics": _dataset_semantics(payload_metadata),
    }


def save_checkpoint_v2(
    *,
    model: MLPUniversalScorer,
    output_path: Path,
    embed_dim: int,
    hidden_dim: int,
    contract: dict,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "contract": contract,
    }
    torch.save(payload, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--vision-encoder-name", default=CANONICAL_VISION_ENCODER)
    parser.add_argument("--text-encoder-name", default=CANONICAL_TEXT_ENCODER)
    parser.add_argument("--region-mode", default="retrieved_topr")
    parser.add_argument("--config-dump", default=None)
    args = parser.parse_args()

    features_path = Path(args.features)
    features_payload = load_feature_payload(features_path)
    tensors = load_training_tensors_v2(features_payload)
    dataset = PsiUnivV2Dataset(tensors)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = int(tensors.image_embeddings.shape[-1])
    model = MLPUniversalScorer(embed_dim=embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for batch in loader:
            image_batch = batch["image_embeddings"].to(device)
            candidate_batch = batch["hypothesis_embeddings"].to(device)
            prefix_batch = batch["query_embeddings"].to(device)
            region_batch = batch["region_embeddings"].to(device)
            label_batch = batch["labels"].to(device)

            observation = UniversalObservation(image_embedding=image_batch, region_embeddings=region_batch)
            output = model(observation, candidate_batch, prefix_batch)
            logits = torch.stack([output.support, output.contradiction, output.abstain], dim=-1)
            loss = compute_loss(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * image_batch.shape[0]
            count += image_batch.shape[0]

        print(f"epoch={epoch} loss={running / max(count, 1):.6f}")

    contract = build_contract_v2(
        args=args,
        embed_dim=embed_dim,
        features_path=features_path,
        payload_metadata=tensors.metadata,
    )
    output_path = save_checkpoint_v2(
        model=model,
        output_path=Path(args.output),
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        contract=contract,
    )
    checkpoint_sha256 = file_sha256(output_path)

    config_dump_path = Path(args.config_dump) if args.config_dump else output_path.with_suffix(".config.json")
    config_dump = {
        "checkpoint_path": str(output_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "contract": contract,
        "training_args": {
            "features": str(features_path.resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "region_mode": args.region_mode,
        },
        "training_semantics": _dataset_semantics(tensors.metadata),
    }
    config_dump_path.write_text(json.dumps(config_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved universal plugin v2 checkpoint to {output_path}")
    print(f"Saved config dump to {config_dump_path}")


if __name__ == "__main__":
    main()
