from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from uniground_v2.regions import RegionProposal


class GroundingDinoProposalProvider:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        threshold: float = 0.30,
        text_threshold: float = 0.25,
        max_regions: int = 8,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.max_regions = max_regions
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def propose(self, image: Image.Image, labels: Sequence[str]) -> list[RegionProposal]:
        labels = [str(label).strip().lower() for label in labels if str(label).strip()]
        if not labels:
            return []
        text = " . ".join(labels) + " ."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )
        result = results[0]
        text_labels = result.get("text_labels") or result.get("labels") or []
        proposals: list[RegionProposal] = []
        for box, score, label in zip(result["boxes"], result["scores"], text_labels):
            left, top, right, bottom = [int(round(float(value))) for value in box.tolist()]
            proposals.append(
                RegionProposal(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    score=float(score),
                    label=str(label),
                )
            )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[: self.max_regions]
