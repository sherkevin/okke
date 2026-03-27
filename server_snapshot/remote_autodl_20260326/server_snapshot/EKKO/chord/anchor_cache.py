from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from chord.knowledge_kernel_evaluator import DetectedAnchor


def _normalize_image_path(image_path: str) -> str:
    return image_path.replace("\\", "/")


def build_anchor_cache_key(
    *,
    image_path: str,
    query: str,
    anchor_query: str | None = None,
    box_threshold: float,
    text_threshold: float,
    max_boxes: int,
) -> str:
    payload = {
        "image_path": _normalize_image_path(image_path),
        "query": query.strip(),
        "anchor_query": (anchor_query or query).strip(),
        "box_threshold": round(float(box_threshold), 6),
        "text_threshold": round(float(text_threshold), 6),
        "max_boxes": int(max_boxes),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class CachedAnchorEntry:
    image_path: str
    image_size: tuple[int, int]
    query: str
    box_threshold: float
    text_threshold: float
    max_boxes: int
    anchors: list[DetectedAnchor]
    anchor_query: str | None = None
    grid_size: tuple[int, int] | None = None
    membership: list[list[float]] | None = None
    relevance: list[float] | None = None
    confidence: list[float] | None = None

    def key(self) -> str:
        return build_anchor_cache_key(
            image_path=self.image_path,
            query=self.query,
            anchor_query=self.anchor_query,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            max_boxes=self.max_boxes,
        )

    def to_payload(self) -> dict:
        return {
            "image_path": _normalize_image_path(self.image_path),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "query": self.query,
            "anchor_query": self.anchor_query,
            "box_threshold": float(self.box_threshold),
            "text_threshold": float(self.text_threshold),
            "max_boxes": int(self.max_boxes),
            "grid_size": [int(self.grid_size[0]), int(self.grid_size[1])] if self.grid_size is not None else None,
            "membership": self.membership,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "anchors": [
                {
                    "box": [float(x) for x in anchor.box],
                    "confidence": float(anchor.confidence),
                    "phrase": anchor.phrase,
                }
                for anchor in self.anchors
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "CachedAnchorEntry":
        return cls(
            image_path=_normalize_image_path(str(payload["image_path"])),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            query=str(payload["query"]),
            anchor_query=(
                str(payload["anchor_query"])
                if payload.get("anchor_query") is not None
                else str(payload["query"])
            ),
            box_threshold=float(payload["box_threshold"]),
            text_threshold=float(payload["text_threshold"]),
            max_boxes=int(payload["max_boxes"]),
            grid_size=(
                (int(payload["grid_size"][0]), int(payload["grid_size"][1]))
                if payload.get("grid_size") is not None
                else None
            ),
            membership=payload.get("membership"),
            relevance=[float(x) for x in payload.get("relevance", [])] if payload.get("relevance") is not None else None,
            confidence=[float(x) for x in payload.get("confidence", [])] if payload.get("confidence") is not None else None,
            anchors=[
                DetectedAnchor(
                    box=tuple(float(x) for x in anchor["box"]),
                    confidence=float(anchor["confidence"]),
                    phrase=anchor.get("phrase"),
                )
                for anchor in payload.get("anchors", [])
            ],
        )

    def membership_tensor(self) -> torch.Tensor:
        if self.membership is None:
            raise ValueError("cached membership is missing; regenerate the knowledge-kernel cache")
        membership = torch.tensor(self.membership, dtype=torch.float32)
        if membership.ndim == 2:
            return membership
        if membership.numel() == 0 and self.grid_size is not None:
            grid_h, grid_w = self.grid_size
            return torch.zeros((0, int(grid_h) * int(grid_w)), dtype=torch.float32)
        raise ValueError("cached membership must be rank-2 or an empty payload with grid_size metadata")

    def relevance_tensor(self) -> torch.Tensor:
        if self.relevance is None:
            raise ValueError("cached relevance is missing; regenerate the knowledge-kernel cache")
        return torch.tensor(self.relevance, dtype=torch.float32)

    def confidence_tensor(self) -> torch.Tensor:
        if self.confidence is not None:
            return torch.tensor(self.confidence, dtype=torch.float32)
        return torch.tensor([anchor.confidence for anchor in self.anchors], dtype=torch.float32)


class AnchorCache:
    def __init__(self, entries: list[CachedAnchorEntry]) -> None:
        self._entries = {entry.key(): entry for entry in entries}

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        *,
        image_path: str,
        query: str,
        anchor_query: str | None = None,
        box_threshold: float,
        text_threshold: float,
        max_boxes: int,
    ) -> CachedAnchorEntry | None:
        return self._entries.get(
            build_anchor_cache_key(
                image_path=image_path,
                query=query,
                anchor_query=anchor_query,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_boxes=max_boxes,
            )
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AnchorCache":
        entries: list[CachedAnchorEntry] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entries.append(CachedAnchorEntry.from_payload(json.loads(line)))
        return cls(entries)


def dump_anchor_cache(entries: list[CachedAnchorEntry], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_payload(), ensure_ascii=False) + "\n")
