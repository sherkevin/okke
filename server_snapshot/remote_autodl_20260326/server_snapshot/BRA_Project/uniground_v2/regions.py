from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from bra_universal_plugin import UniversalObservation
from uniground_v2.contracts import RetrievalResult


@dataclass
class RegionProposal:
    left: int
    top: int
    right: int
    bottom: int
    score: float = 1.0
    label: str = ""


class RegionProposalCacheBuilder:
    def __init__(self, encoder: Any, grid_size: int = 2) -> None:
        self.encoder = encoder
        self.grid_size = grid_size

    def build(
        self,
        image: Image.Image,
        proposals: Optional[Iterable[RegionProposal]] = None,
        mode: str = "grid_regions",
    ) -> UniversalObservation:
        image_embedding = self.encoder.encode_image(image)
        if proposals:
            boxes = []
            crops = []
            width, height = image.size
            for proposal in proposals:
                box = (
                    max(0, min(width, proposal.left)),
                    max(0, min(height, proposal.top)),
                    max(0, min(width, proposal.right)),
                    max(0, min(height, proposal.bottom)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                boxes.append(box)
                crops.append(image.crop(box))
            region_embeddings = self._encode_crops(crops)
            metadata = {"region_mode": "detector_regions", "region_boxes": boxes}
            return UniversalObservation(image_embedding=image_embedding, region_embeddings=region_embeddings, metadata=metadata)

        region_embeddings, boxes = self._build_grid_regions(image)
        metadata = {"region_mode": mode, "region_boxes": boxes}
        return UniversalObservation(image_embedding=image_embedding, region_embeddings=region_embeddings, metadata=metadata)

    def _build_grid_regions(self, image: Image.Image) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        width, height = image.size
        boxes: list[tuple[int, int, int, int]] = []
        crops = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                left = int(round(col * width / self.grid_size))
                right = int(round((col + 1) * width / self.grid_size))
                top = int(round(row * height / self.grid_size))
                bottom = int(round((row + 1) * height / self.grid_size))
                box = (left, top, right, bottom)
                boxes.append(box)
                crops.append(image.crop(box))
        region_embeddings = self._encode_crops(crops)
        if region_embeddings is None:
            raise ValueError("Grid region construction must yield at least one crop.")
        return region_embeddings, boxes

    def _encode_crops(self, crops: list[Image.Image]) -> torch.Tensor | None:
        if not crops:
            return None
        if hasattr(self.encoder, "encode_images"):
            return self.encoder.encode_images(crops)
        return torch.stack([self.encoder.encode_image(crop) for crop in crops], dim=0)


class RegionRetriever:
    def __init__(self, encoder: Any, top_r: int = 2) -> None:
        self.encoder = encoder
        self.top_r = top_r

    def retrieve(
        self,
        observation: UniversalObservation,
        candidates: list[Any],
        device: torch.device,
    ) -> RetrievalResult:
        region_embeddings = observation.region_embeddings
        if region_embeddings is None or region_embeddings.numel() == 0:
            return RetrievalResult(
                observation=UniversalObservation(
                    image_embedding=observation.image_embedding.to(device),
                    region_embeddings=None,
                    metadata=dict(observation.metadata),
                ),
                metadata={"selected_region_indices": [], "region_mode": observation.metadata.get("region_mode", "none")},
            )

        runtime_context = observation.metadata.get("runtime_context", {}) if isinstance(observation.metadata, dict) else {}
        retrieval_scope = runtime_context.get("retrieval_scope") if isinstance(runtime_context, dict) else None
        retrieval_query_text = runtime_context.get("retrieval_query_text") if isinstance(runtime_context, dict) else None
        if retrieval_scope == "task_query" and retrieval_query_text:
            retrieval_texts = [str(retrieval_query_text)] * len(candidates)
        else:
            retrieval_texts = [candidate.span_text for candidate in candidates]

        candidate_embeddings = self.encoder.encode_texts(retrieval_texts).float()
        region_embeddings_cpu = region_embeddings.float()
        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
        region_embeddings_cpu = F.normalize(region_embeddings_cpu, dim=-1)
        similarity = candidate_embeddings @ region_embeddings_cpu.T
        top_r = min(self.top_r, region_embeddings_cpu.shape[0])
        selected_scores, selected_idx = torch.topk(similarity, k=top_r, dim=-1)
        selected_regions = region_embeddings_cpu[selected_idx].to(device)
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "selected_region_indices_per_candidate": selected_idx.tolist(),
                "selected_region_scores_per_candidate": [
                    [round(float(score), 6) for score in candidate_scores]
                    for candidate_scores in selected_scores.tolist()
                ],
            }
        )
        return RetrievalResult(
            observation=UniversalObservation(
                image_embedding=observation.image_embedding.to(device),
                region_embeddings=selected_regions,
                metadata=metadata,
            ),
            metadata={
                "region_mode": metadata.get("region_mode", "unknown"),
                "selected_region_indices_per_candidate": selected_idx.tolist(),
                "selected_region_scores_per_candidate": [
                    [round(float(score), 6) for score in candidate_scores]
                    for candidate_scores in selected_scores.tolist()
                ],
            },
        )
