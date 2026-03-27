from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class V2TrainingTensors:
    image_embeddings: torch.Tensor
    hypothesis_embeddings: torch.Tensor
    query_embeddings: torch.Tensor
    region_embeddings: torch.Tensor
    labels: torch.Tensor
    metadata: dict[str, Any]


def load_feature_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Feature payload must be a dict.")
    return payload


def load_training_tensors_v2(payload: dict[str, Any]) -> V2TrainingTensors:
    required = {"image_embeddings", "labels"}
    if not required.issubset(payload):
        raise ValueError("Feature payload missing required embedding or label tensors.")

    region_key = "retrieved_region_embeddings" if "retrieved_region_embeddings" in payload else "region_embeddings"
    if region_key not in payload:
        raise ValueError("Feature payload must include region_embeddings or retrieved_region_embeddings.")

    image_embeddings = payload["image_embeddings"].float()
    hypothesis_key = "hypothesis_embeddings" if "hypothesis_embeddings" in payload else "candidate_embeddings"
    query_key = "query_embeddings" if "query_embeddings" in payload else "prefix_embeddings"
    if hypothesis_key not in payload or query_key not in payload:
        raise ValueError("Feature payload must include query/hypothesis embeddings or legacy prefix/candidate embeddings.")
    hypothesis_embeddings = payload[hypothesis_key].float()
    query_embeddings = payload[query_key].float()
    region_embeddings = payload[region_key].float()
    labels = payload["labels"]

    if image_embeddings.ndim != 2 or hypothesis_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Image, hypothesis, and query embeddings must be 2D [N, D].")
    if image_embeddings.shape != hypothesis_embeddings.shape or image_embeddings.shape != query_embeddings.shape:
        raise ValueError("Image, hypothesis, and query embeddings must share the same [N, D] shape.")
    if region_embeddings.ndim not in {2, 3}:
        raise ValueError("Region embeddings must be [N, D] or [N, R, D].")
    if region_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError("Region embeddings must share the same batch dimension as image embeddings.")
    if labels.shape[0] != image_embeddings.shape[0]:
        raise ValueError("Labels must align with the batch dimension.")

    return V2TrainingTensors(
        image_embeddings=image_embeddings,
        hypothesis_embeddings=hypothesis_embeddings,
        query_embeddings=query_embeddings,
        region_embeddings=region_embeddings,
        labels=labels,
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    )


class PsiUnivV2Dataset(Dataset):
    def __init__(self, tensors: V2TrainingTensors) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return int(self.tensors.image_embeddings.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image_embeddings": self.tensors.image_embeddings[index],
            "candidate_embeddings": self.tensors.hypothesis_embeddings[index],
            "hypothesis_embeddings": self.tensors.hypothesis_embeddings[index],
            "prefix_embeddings": self.tensors.query_embeddings[index],
            "query_embeddings": self.tensors.query_embeddings[index],
            "region_embeddings": self.tensors.region_embeddings[index],
            "labels": self.tensors.labels[index],
        }
