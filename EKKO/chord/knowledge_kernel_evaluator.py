from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from chord.query_formulation import extract_anchor_query, normalize_object_phrase, normalize_object_terms

if TYPE_CHECKING:
    from chord.anchor_cache import CachedAnchorEntry


@dataclass(frozen=True)
class DetectedAnchor:
    box: tuple[float, float, float, float]
    confidence: float
    phrase: str | None = None


@dataclass(frozen=True)
class KnowledgeKernelResult:
    token_weights: torch.Tensor
    membership: torch.Tensor
    relevance: torch.Tensor
    confidence: torch.Tensor
    used_fallback: bool


# Backward-compatible alias while the codebase migrates from the old name.
AnchorWeightResult = KnowledgeKernelResult


def uniform_visual_token_weights(
    num_visual_tokens: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.ones(num_visual_tokens, device=device, dtype=dtype)


def normalize_query_terms(text: str) -> tuple[str, ...]:
    return normalize_object_terms(text)


def score_anchor_relevance(query: str, anchor_phrase: str | None) -> float:
    query_phrase = normalize_object_phrase(extract_anchor_query(query))
    anchor_phrase_norm = normalize_object_phrase(anchor_phrase or "")
    query_terms = set(normalize_query_terms(query_phrase))
    anchor_terms = set(normalize_query_terms(anchor_phrase_norm))
    if not query_phrase or not anchor_phrase_norm or not query_terms or not anchor_terms:
        return 0.0

    if query_phrase == anchor_phrase_norm:
        return 1.0

    overlap = query_terms & anchor_terms
    if overlap:
        query_head = query_phrase.split(" ")[-1]
        anchor_head = anchor_phrase_norm.split(" ")[-1]
        if query_terms == anchor_terms:
            return 0.98
        if query_head == anchor_head:
            return 0.9
        return 0.65 * float(len(overlap) / max(len(query_terms), len(anchor_terms)))

    similarities = []
    for query_term in query_terms:
        for anchor_term in anchor_terms:
            prefix = 0
            for q_char, a_char in zip(query_term, anchor_term):
                if q_char != a_char:
                    break
                prefix += 1
            similarities.append(prefix / max(len(query_term), len(anchor_term)))
    return 0.25 * float(max(similarities, default=0.0))


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
    return uniform_visual_token_weights(num_visual_tokens, device=device, dtype=dtype) + alpha_anchor * max_support


def build_knowledge_kernel_result(
    *,
    anchors: list[DetectedAnchor],
    query: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    alpha_anchor: float,
) -> KnowledgeKernelResult:
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
    return KnowledgeKernelResult(
        token_weights=token_weights,
        membership=membership,
        relevance=relevance,
        confidence=confidence,
        used_fallback=used_fallback,
    )


def build_knowledge_kernel_result_from_cache(
    *,
    cached_entry: "CachedAnchorEntry",
    alpha_anchor: float,
) -> KnowledgeKernelResult:
    membership = cached_entry.membership_tensor()
    relevance = cached_entry.relevance_tensor()
    confidence = cached_entry.confidence_tensor()
    token_weights = build_visual_token_weights(
        membership=membership,
        relevance=relevance,
        confidence=confidence,
        alpha_anchor=alpha_anchor,
    )
    used_fallback = bool(membership.shape[0] == 0 or torch.count_nonzero(relevance) == 0)
    return KnowledgeKernelResult(
        token_weights=token_weights,
        membership=membership,
        relevance=relevance,
        confidence=confidence,
        used_fallback=used_fallback,
    )


def build_anchor_weight_result(
    anchors: list[DetectedAnchor],
    query: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    alpha_anchor: float,
) -> AnchorWeightResult:
    return build_knowledge_kernel_result(
        anchors=anchors,
        query=query,
        image_size=image_size,
        grid_size=grid_size,
        alpha_anchor=alpha_anchor,
    )


def compute_knowledge_kernel_support(attn_to_visual: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    if attn_to_visual.shape[-1] != token_weights.shape[-1]:
        raise ValueError("attention and token weights must share the visual-token dimension")
    return (attn_to_visual * token_weights).sum(dim=-1)


def compute_knowledge_kernel_bonus(attn_to_visual: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    if attn_to_visual.shape[-1] != token_weights.shape[-1]:
        raise ValueError("attention and token weights must share the visual-token dimension")
    bonus_weights = torch.clamp(token_weights - 1.0, min=0.0)
    return (attn_to_visual * bonus_weights).sum(dim=-1)


def compute_anchor_visual_support(attn_to_visual: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    return compute_knowledge_kernel_support(attn_to_visual, token_weights)
