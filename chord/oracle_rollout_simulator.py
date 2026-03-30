from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from chord.knowledge_kernel_evaluator import compute_anchor_visual_support


@dataclass(frozen=True)
class FutureRolloutResult:
    sum_visual: float
    sum_text: float
    f_future: float
    r_future: float
    failed: bool
    error_message: str | None = None


@dataclass(frozen=True)
class FutureRolloutTrace:
    generated_tokens: tuple[int, ...]
    step_visual_sums: tuple[float, ...]
    step_text_sums: tuple[float, ...]
    result: FutureRolloutResult


def reduce_attention_maps(
    attentions: Sequence[torch.Tensor],
    *,
    last_n_layers: int = 2,
    head_reduce: str = "mean",
) -> torch.Tensor:
    if not attentions:
        raise ValueError("attention tuple must not be empty")
    if last_n_layers < 1:
        raise ValueError("last_n_layers must be at least 1")
    if head_reduce != "mean":
        raise ValueError(f"Unsupported head_reduce policy: {head_reduce!r}")

    selected = tuple(attentions[-min(len(attentions), int(last_n_layers)) :])
    stacked = torch.stack([layer.to(torch.float32) for layer in selected], dim=0)
    reduced_layers = stacked.mean(dim=0)
    return reduced_layers.mean(dim=1)


def compute_visual_anchor_ratio(
    sum_visual: torch.Tensor,
    sum_text: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    return sum_visual.to(torch.float32) / (sum_visual.to(torch.float32) + sum_text.to(torch.float32) + float(eps))


def build_branch_text_token_indices(
    *,
    prefix_length: int,
    image_token_span: tuple[int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    image_start, image_end = image_token_span
    text_indices = [idx for idx in range(prefix_length) if not (image_start <= idx <= image_end)]
    return torch.tensor(text_indices, dtype=torch.long, device=device)


def compute_future_result_from_sums(
    *,
    sum_visual: float,
    sum_text: float,
    lambda_txt: float,
    eps: float = 1e-6,
) -> FutureRolloutResult:
    f_future = sum_visual - lambda_txt * sum_text
    r_future = sum_visual / (sum_visual + sum_text + eps)
    return FutureRolloutResult(
        sum_visual=sum_visual,
        sum_text=sum_text,
        f_future=f_future,
        r_future=r_future,
        failed=False,
    )


def summarize_rollout_step(
    *,
    attn_last_row: torch.Tensor,
    prefix_length: int,
    visual_token_indices: torch.Tensor,
    image_token_span: tuple[int, int],
    token_weights: torch.Tensor,
) -> tuple[float, float]:
    if attn_last_row.ndim != 1:
        raise ValueError("rollout attention vectors must be rank-1")
    if attn_last_row.shape[0] < prefix_length:
        raise ValueError("rollout attention length must cover the current branch prefix length")
    if visual_token_indices.numel() > 0 and int(visual_token_indices.max().item()) >= attn_last_row.shape[0]:
        raise ValueError("rollout attention length must cover the configured visual token indices")

    visual_attn = attn_last_row.index_select(0, visual_token_indices.to(attn_last_row.device))
    text_indices = build_branch_text_token_indices(
        prefix_length=prefix_length,
        image_token_span=image_token_span,
        device=attn_last_row.device,
    )
    text_attn = attn_last_row.index_select(0, text_indices)
    return (
        float(compute_anchor_visual_support(visual_attn, token_weights).item()),
        float(text_attn.sum().item()),
    )


def summarize_rollout_step_batch(
    *,
    attn_last_rows: torch.Tensor,
    prefix_length: int,
    visual_token_indices: torch.Tensor,
    image_token_span: tuple[int, int],
    token_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn_last_rows.ndim != 2:
        raise ValueError("attn_last_rows must have shape [batch, sequence_length]")
    if attn_last_rows.shape[1] < prefix_length:
        raise ValueError("rollout attention length must cover the current branch prefix length")
    if visual_token_indices.numel() > 0 and int(visual_token_indices.max().item()) >= attn_last_rows.shape[1]:
        raise ValueError("rollout attention length must cover the configured visual token indices")

    visual_attn = attn_last_rows.index_select(1, visual_token_indices.to(attn_last_rows.device))
    text_indices = build_branch_text_token_indices(
        prefix_length=prefix_length,
        image_token_span=image_token_span,
        device=attn_last_rows.device,
    )
    text_attn = attn_last_rows.index_select(1, text_indices)
    sum_visual = compute_anchor_visual_support(visual_attn, token_weights).to(torch.float32)
    sum_text = text_attn.sum(dim=-1).to(torch.float32)
    return sum_visual, sum_text


def compute_future_trajectory_score(
    visual_attentions: Sequence[torch.Tensor],
    text_attentions: Sequence[torch.Tensor],
    token_weights: torch.Tensor,
    lambda_txt: float,
    eps: float = 1e-6,
) -> FutureRolloutResult:
    if len(visual_attentions) != len(text_attentions):
        raise ValueError("visual_attentions and text_attentions must have the same length")

    sum_visual = 0.0
    sum_text = 0.0

    for visual_step, text_step in zip(visual_attentions, text_attentions):
        sum_visual += float(compute_anchor_visual_support(visual_step, token_weights).item())
        sum_text += float(text_step.sum().item())

    return compute_future_result_from_sums(
        sum_visual=sum_visual,
        sum_text=sum_text,
        lambda_txt=lambda_txt,
        eps=eps,
    )


def greedy_future_rollout(
    *,
    prefix_ids: torch.Tensor,
    step_fn: Callable[[torch.Tensor], tuple[int, torch.Tensor]],
    visual_token_indices: torch.Tensor,
    image_token_span: tuple[int, int],
    token_weights: torch.Tensor,
    horizon: int,
    lambda_txt: float,
    eps: float = 1e-6,
) -> FutureRolloutTrace:
    if prefix_ids.ndim != 1:
        raise ValueError("prefix_ids must be a rank-1 tensor for greedy future rollout")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    running_prefix = prefix_ids.clone()
    generated_tokens: list[int] = []
    step_visual_sums: list[float] = []
    step_text_sums: list[float] = []

    for _ in range(horizon):
        next_token_id, attn_last_row = step_fn(running_prefix)
        step_visual, step_text = summarize_rollout_step(
            attn_last_row=attn_last_row,
            prefix_length=running_prefix.shape[0],
            visual_token_indices=visual_token_indices,
            image_token_span=image_token_span,
            token_weights=token_weights,
        )
        step_visual_sums.append(step_visual)
        step_text_sums.append(step_text)
        generated_tokens.append(int(next_token_id))
        running_prefix = torch.cat(
            [running_prefix, torch.tensor([next_token_id], dtype=running_prefix.dtype, device=running_prefix.device)]
        )

    result = compute_future_result_from_sums(
        sum_visual=sum(step_visual_sums),
        sum_text=sum(step_text_sums),
        lambda_txt=lambda_txt,
        eps=eps,
    )
    return FutureRolloutTrace(
        generated_tokens=tuple(generated_tokens),
        step_visual_sums=tuple(step_visual_sums),
        step_text_sums=tuple(step_text_sums),
        result=result,
    )


def greedy_future_rollout_from_bootstrap(
    *,
    prefix_ids: torch.Tensor,
    bootstrap_next_token_id: int,
    bootstrap_attn_last_row: torch.Tensor,
    continuation_step_fn: Callable[[torch.Tensor], tuple[int, torch.Tensor]],
    visual_token_indices: torch.Tensor,
    image_token_span: tuple[int, int],
    token_weights: torch.Tensor,
    horizon: int,
    lambda_txt: float,
    eps: float = 1e-6,
) -> FutureRolloutTrace:
    if prefix_ids.ndim != 1:
        raise ValueError("prefix_ids must be a rank-1 tensor for greedy future rollout")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if bootstrap_attn_last_row.ndim != 1:
        raise ValueError("bootstrap_attn_last_row must be rank-1")
    if bootstrap_attn_last_row.shape[0] < prefix_ids.shape[0]:
        raise ValueError("bootstrap attention length must cover the candidate-appended prefix length")

    running_prefix = prefix_ids.clone()
    generated_tokens: list[int] = []
    step_visual_sums: list[float] = []
    step_text_sums: list[float] = []

    step_visual, step_text = summarize_rollout_step(
        attn_last_row=bootstrap_attn_last_row,
        prefix_length=running_prefix.shape[0],
        visual_token_indices=visual_token_indices,
        image_token_span=image_token_span,
        token_weights=token_weights,
    )
    step_visual_sums.append(step_visual)
    step_text_sums.append(step_text)
    generated_tokens.append(int(bootstrap_next_token_id))
    running_prefix = torch.cat(
        [running_prefix, torch.tensor([bootstrap_next_token_id], dtype=running_prefix.dtype, device=running_prefix.device)]
    )

    for _ in range(1, horizon):
        next_token_id, attn_last_row = continuation_step_fn(running_prefix)
        step_visual, step_text = summarize_rollout_step(
            attn_last_row=attn_last_row,
            prefix_length=running_prefix.shape[0],
            visual_token_indices=visual_token_indices,
            image_token_span=image_token_span,
            token_weights=token_weights,
        )
        step_visual_sums.append(step_visual)
        step_text_sums.append(step_text)
        generated_tokens.append(int(next_token_id))
        running_prefix = torch.cat(
            [running_prefix, torch.tensor([next_token_id], dtype=running_prefix.dtype, device=running_prefix.device)]
        )

    result = compute_future_result_from_sums(
        sum_visual=sum(step_visual_sums),
        sum_text=sum(step_text_sums),
        lambda_txt=lambda_txt,
        eps=eps,
    )
    return FutureRolloutTrace(
        generated_tokens=tuple(generated_tokens),
        step_visual_sums=tuple(step_visual_sums),
        step_text_sums=tuple(step_text_sums),
        result=result,
    )


def safe_rollout_future(
    rollout_fn: Callable[..., FutureRolloutResult],
    *args,
    **kwargs,
) -> FutureRolloutResult | FutureRolloutTrace:
    try:
        return rollout_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - safe-fallback behavior is required by spec.
        return FutureRolloutResult(
            sum_visual=0.0,
            sum_text=0.0,
            f_future=0.0,
            r_future=0.0,
            failed=True,
            error_message=str(exc),
        )
