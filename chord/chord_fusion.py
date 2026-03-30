from __future__ import annotations

from dataclasses import dataclass

import torch

from chord.knowledge_kernel_evaluator import compute_knowledge_kernel_bonus


@dataclass(frozen=True)
class ChordRerankResult:
    ranked_candidate_tokens: torch.Tensor
    ranked_candidate_indices: torch.Tensor
    fused_scores: torch.Tensor | None
    rollback_triggered: bool
    used_chord: bool
    recompute_after_rollback: bool


def fuse_chord_scores(
    *,
    baseline_scores: torch.Tensor,
    v_anchor: torch.Tensor,
    f_future: torch.Tensor,
    lambda_cur: float,
    lambda_fut: float,
    zero_anchor_penalty: float = 0.0,
) -> torch.Tensor:
    if lambda_cur == 0.0 and lambda_fut == 0.0 and zero_anchor_penalty == 0.0:
        return baseline_scores.clone()
    no_anchor_mask = (v_anchor <= 0.0).to(baseline_scores.dtype)
    return baseline_scores + lambda_cur * v_anchor + lambda_fut * f_future - zero_anchor_penalty * no_anchor_mask


def apply_current_chord_score(
    *,
    baseline_scores: torch.Tensor,
    candidate_visual_attn: torch.Tensor,
    token_weights: torch.Tensor,
    lambda_cur: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lambda_cur == 0.0:
        zero_bonus = torch.zeros_like(baseline_scores)
        return baseline_scores.clone(), zero_bonus

    v_anchor = compute_knowledge_kernel_bonus(candidate_visual_attn, token_weights)
    return baseline_scores + lambda_cur * v_anchor, v_anchor


def apply_chord_rerank(
    *,
    candidate_tokens: torch.Tensor,
    baseline_scores: torch.Tensor,
    v_anchor: torch.Tensor,
    f_future: torch.Tensor,
    lambda_cur: float,
    lambda_fut: float,
    zero_anchor_penalty: float,
    rollback_triggered: bool,
) -> ChordRerankResult:
    if rollback_triggered:
        identity = torch.arange(candidate_tokens.shape[0], device=candidate_tokens.device)
        return ChordRerankResult(
            ranked_candidate_tokens=candidate_tokens,
            ranked_candidate_indices=identity,
            fused_scores=None,
            rollback_triggered=True,
            used_chord=False,
            recompute_after_rollback=True,
        )

    fused_scores = fuse_chord_scores(
        baseline_scores=baseline_scores,
        v_anchor=v_anchor,
        f_future=f_future,
        lambda_cur=lambda_cur,
        lambda_fut=lambda_fut,
        zero_anchor_penalty=zero_anchor_penalty,
    )
    order = torch.argsort(fused_scores, descending=True, stable=True)
    return ChordRerankResult(
        ranked_candidate_tokens=candidate_tokens[order],
        ranked_candidate_indices=order,
        fused_scores=fused_scores,
        rollback_triggered=False,
        used_chord=bool(lambda_cur != 0.0 or lambda_fut != 0.0),
        recompute_after_rollback=False,
    )
