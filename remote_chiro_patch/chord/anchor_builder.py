from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DetectedAnchor:
    box: tuple[float, float, float, float]
    confidence: float
    phrase: str | None = None


@dataclass(frozen=True)
class AnchorWeightResult:
    token_weights: torch.Tensor
    membership: torch.Tensor
    relevance: torch.Tensor
    confidence: torch.Tensor
    used_fallback: bool


def uniform_visual_token_weights(
    num_visual_tokens: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.ones(num_visual_tokens, device=device, dtype=dtype)


def normalize_query_terms(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9]+", text.lower()) if token)


def score_anchor_relevance(query: str, anchor_phrase: str | None) -> float:
    query_terms = set(normalize_query_terms(query))
    anchor_terms = set(normalize_query_terms(anchor_phrase or ""))
    if not query_terms or not anchor_terms:
        return 0.0

    overlap = query_terms & anchor_terms
    if overlap:
        return float(len(overlap) / len(anchor_terms))

    # Conservative similarity-only fallback when hard overlap misses.
    similarities = []
    for query_term in query_terms:
        for anchor_term in anchor_terms:
            prefix = 0
            for q_char, a_char in zip(query_term, anchor_term):
                if q_char != a_char:
                    break
                prefix += 1
            similarities.append(prefix / max(len(query_term), len(anchor_term)))
    return float(max(similarities, default=0.0))


def boxes_to_visual_membership(
    anchors: list[DetectedAnchor],
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    grid_h, grid_w = grid_size
    image_h, image_w = image_size
    membership = torch.zeros((len(anchors), grid_h * grid_w), dtype=dtype)
    if not anchors:
        return membership

    cell_h = image_h / grid_h
    cell_w = image_w / grid_w
    centers = []
    for row in range(grid_h):
        for col in range(grid_w):
            centers.append(((col + 0.5) * cell_w, (row + 0.5) * cell_h))

    for anchor_idx, anchor in enumerate(anchors):
        x0, y0, x1, y1 = anchor.box
        for token_idx, (center_x, center_y) in enumerate(centers):
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                membership[anchor_idx, token_idx] = 1.0
    return membership


def build_visual_token_weights(
    membership: torch.Tensor,
    relevance: torch.Tensor,
    confidence: torch.Tensor,
    alpha_anchor: float,
) -> torch.Tensor:
    if membership.ndim != 2:
        raise ValueError("membership must have shape [num_anchors, num_visual_tokens]")
    if relevance.ndim != 1 or confidence.ndim != 1:
        raise ValueError("relevance and confidence must be rank-1 tensors")
    if membership.shape[0] != relevance.shape[0] or membership.shape[0] != confidence.shape[0]:
        raise ValueError("membership, relevance, and confidence must agree on the anchor dimension")

    num_visual_tokens = membership.shape[1]
    device = membership.device
    dtype = membership.dtype

    if membership.shape[0] == 0:
        return uniform_visual_token_weights(num_visual_tokens, device=device, dtype=dtype)

    positive_relevance = torch.clamp(relevance, min=0.0)
    positive_confidence = torch.clamp(confidence, min=0.0)
    if torch.count_nonzero(positive_relevance) == 0:
        return uniform_visual_token_weights(num_visual_tokens, device=device, dtype=dtype)

    support = membership * positive_relevance[:, None] * positive_confidence[:, None]
    max_support = support.max(dim=0).values if support.numel() else torch.zeros(num_visual_tokens, device=device, dtype=dtype)

    # First-version policy is enhance-only: unmatched or unanchored tokens remain at 1.
    return uniform_visual_token_weights(num_visual_tokens, device=device, dtype=dtype) + alpha_anchor * max_support


def build_anchor_weight_result(
    anchors: list[DetectedAnchor],
    query: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    alpha_anchor: float,
) -> AnchorWeightResult:
    membership = boxes_to_visual_membership(anchors, image_size=image_size, grid_size=grid_size)
    relevance = torch.tensor([score_anchor_relevance(query, anchor.phrase) for anchor in anchors], dtype=torch.float32)
    confidence = torch.tensor([anchor.confidence for anchor in anchors], dtype=torch.float32)
    token_weights = build_visual_token_weights(
        membership=membership,
        relevance=relevance,
        confidence=confidence,
        alpha_anchor=alpha_anchor,
    )
    used_fallback = bool(len(anchors) == 0 or torch.count_nonzero(relevance) == 0)
    return AnchorWeightResult(
        token_weights=token_weights,
        membership=membership,
        relevance=relevance,
        confidence=confidence,
        used_fallback=used_fallback,
    )


def compute_anchor_visual_support(attn_to_visual: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    if attn_to_visual.shape[-1] != token_weights.shape[-1]:
        raise ValueError("attention and token weights must share the visual-token dimension")
    return (attn_to_visual * token_weights).sum(dim=-1)
