from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def build_feature_payload(
    *,
    image_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    prefix_embeddings: torch.Tensor,
    region_embeddings: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "image_embeddings": image_embeddings.float(),
        "candidate_embeddings": candidate_embeddings.float(),
        "prefix_embeddings": prefix_embeddings.float(),
        "region_embeddings": region_embeddings.float(),
        "labels": labels,
        "metadata": metadata or {},
    }


def save_feature_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def payload_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_payload_manifest(path: str | Path, payload_path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "payload_path": str(Path(payload_path).resolve()),
        "payload_sha256": payload_sha256(payload_path),
        "record_count": int(payload["image_embeddings"].shape[0]),
        "image_dim": int(payload["image_embeddings"].shape[-1]),
        "region_shape": list(payload["region_embeddings"].shape),
        "metadata": payload.get("metadata", {}),
    }
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
