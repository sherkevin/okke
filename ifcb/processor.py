from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from baseline_processors import _get_decoder_layers, _get_decoder_root, _get_lm_head


def _normalize_token_text(text: str) -> str:
    return text.replace("▁", "").replace("Ġ", "").strip().lower()


def is_semantic_token(tokenizer: Any, token_id: int) -> bool:
    token = tokenizer.decode([int(token_id)], skip_special_tokens=True) if tokenizer is not None else str(token_id)
    normalized = _normalize_token_text(token)
    if normalized.startswith("<") and normalized.endswith(">"):
        return False
    return bool(normalized) and any(ch.isalpha() for ch in normalized)


def build_modal_masks(input_ids: torch.LongTensor, image_token_id: int, hidden_seq_len: int) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    flat_ids = input_ids[0]
    image_positions = (flat_ids == int(image_token_id)).nonzero(as_tuple=False).flatten().tolist()
    visual_mask = torch.zeros(hidden_seq_len, dtype=torch.bool, device=input_ids.device)
    if not image_positions:
        return visual_mask, ~visual_mask
    if hidden_seq_len == flat_ids.shape[0]:
        visual_mask = flat_ids == int(image_token_id)
        return visual_mask, ~visual_mask

    placeholder_count = len(image_positions)
    expanded_visual_tokens = hidden_seq_len - (flat_ids.shape[0] - placeholder_count)
    expanded_visual_tokens = max(expanded_visual_tokens, placeholder_count)
    base = expanded_visual_tokens // placeholder_count
    remainder = expanded_visual_tokens % placeholder_count
    cursor = 0
    image_index = 0
    for token_index in range(flat_ids.shape[0]):
        if token_index in image_positions:
            span = base + (1 if image_index < remainder else 0)
            visual_mask[cursor:cursor + span] = True
            cursor += span
            image_index += 1
        else:
            cursor += 1
    return visual_mask, ~visual_mask


def build_query_token_masks(query_token_count: int, hidden_seq_len: int, device: torch.device) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    visual_mask = torch.zeros(hidden_seq_len, dtype=torch.bool, device=device)
    visual_span = min(max(int(query_token_count), 0), hidden_seq_len)
    visual_mask[:visual_span] = True
    return visual_mask, ~visual_mask


def resolve_probe_layers(num_layers: int, normalized_depths: Sequence[float] | None = None) -> list[int]:
    if num_layers < 2:
        return [0]
    depths = normalized_depths or (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)
    final_idx = num_layers - 1
    resolved: set[int] = set()
    for depth in depths:
        if depth >= 1.0:
            idx = final_idx - 1
        else:
            idx = int(math.floor(depth * final_idx))
        idx = max(0, min(idx, final_idx - 1))
        resolved.add(idx)
    return sorted(resolved)


def compute_commitment_risks(
    *,
    candidate_ids: torch.LongTensor,
    probe_logprobs: Sequence[torch.Tensor],
    final_logprobs: torch.Tensor,
    probe_topk_ids: Sequence[torch.LongTensor],
    visual_participation: dict[int, float],
) -> dict[int, dict[str, float]]:
    if not probe_logprobs:
        return {}

    risks: dict[int, dict[str, float]] = {}
    for token_id in candidate_ids.tolist():
        token_supports = [float(lp[0, token_id].item()) for lp in probe_logprobs]
        support = sum(token_supports) / max(len(token_supports), 1)
        final_commitment = float(final_logprobs[0, token_id].item())
        surge = max(0.0, final_commitment - support)
        persistence = sum(float((top_ids == token_id).any().item()) for top_ids in probe_topk_ids) / max(len(probe_topk_ids), 1)
        visual = float(visual_participation.get(token_id, 0.0))
        risk = surge * (1.0 - persistence) * (1.0 - visual)
        risks[int(token_id)] = {
            "support": support,
            "final_commitment": final_commitment,
            "late_surge": surge,
            "persistence": persistence,
            "visual_participation": visual,
            "risk": risk,
        }
    return risks


def apply_ifcb_penalty(
    scores: torch.Tensor,
    candidate_ids: torch.LongTensor,
    risks: dict[int, dict[str, float]],
    alpha: float,
) -> torch.Tensor:
    adjusted = scores.clone()
    if candidate_ids.numel() == 0:
        return adjusted
    penalties = torch.tensor(
        [alpha * float(risks[int(token_id)]["risk"]) for token_id in candidate_ids.tolist()],
        device=scores.device,
        dtype=scores.dtype,
    )
    adjusted[0, candidate_ids] = adjusted[0, candidate_ids] - penalties
    return adjusted


def build_pope_binary_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    variants = {
        "yes": ["yes", " yes", "Yes", " Yes"],
        "no": ["no", " no", "No", " No"],
    }
    out: dict[str, list[int]] = {}
    for label, texts in variants.items():
        ids = set()
        for text in texts:
            pieces = tokenizer.encode(text, add_special_tokens=False)
            if len(pieces) == 1:
                ids.add(int(pieces[0]))
        if not ids:
            raise ValueError(f"Could not find a single-token encoding for '{label}'")
        out[label] = sorted(ids)
    return out


def build_choice_token_ids(tokenizer: Any, labels: Sequence[str] = ("A", "B", "C", "D")) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label in labels:
        ids = set()
        for text in (label, f" {label}"):
            pieces = tokenizer.encode(text, add_special_tokens=False)
            if len(pieces) == 1:
                ids.add(int(pieces[0]))
        if ids:
            out[label] = sorted(ids)
    if len(out) != len(labels):
        missing = [label for label in labels if label not in out]
        raise ValueError(f"Could not find single-token encodings for choices: {missing}")
    return out


@dataclass
class IFCBConfig:
    alpha: float = 6.0
    frontier_k: int = 5
    probe_topk: int = 5
    normalized_probe_depths: tuple[float, ...] = (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)
    epsilon: float = 1e-8
    audit_max_steps: int = 32


@dataclass
class IFCBStats:
    decode_steps: int = 0
    intervention_steps: int = 0
    candidate_count_sum: int = 0
    total_risk_sum: float = 0.0
    max_risk: float = 0.0
    visual_participation_sum: float = 0.0

    def to_dict(self) -> dict[str, float]:
        denom = max(self.intervention_steps, 1)
        return {
            **asdict(self),
            "avg_candidate_count": self.candidate_count_sum / max(self.decode_steps, 1),
            "avg_risk": self.total_risk_sum / denom,
            "avg_visual_participation": self.visual_participation_sum / denom,
        }


class _LayerCapture:
    def __init__(self, model: Any, layer_indices: Iterable[int]):
        self._layers = _get_decoder_layers(model)
        self._indices = sorted(set(int(idx) for idx in layer_indices))
        self.hidden_by_layer: dict[int, torch.Tensor] = {}
        self._hooks = [self._layers[idx].register_forward_hook(self._make_hook(idx)) for idx in self._indices]

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.hidden_by_layer[idx] = hidden
            return output
        return hook

    def clear(self) -> None:
        self.hidden_by_layer.clear()

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class _BaseIFCBProcessor:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        image_token_id: int,
        config: IFCBConfig | None = None,
        *,
        model_family: str,
        query_token_count: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.cfg = config or IFCBConfig()
        self.model_family = model_family
        self.query_token_count = int(query_token_count)
        self.lm_head = _get_lm_head(model)
        self.final_norm = self._get_final_norm(model)
        self.num_layers = len(_get_decoder_layers(model))
        self.probe_layers = resolve_probe_layers(self.num_layers, self.cfg.normalized_probe_depths)
        self.fusion_layer = min(self.num_layers - 1, max(0, self.num_layers // 2))
        self._capture = _LayerCapture(model, self.probe_layers + [self.fusion_layer])
        self._pope_binary_ids = build_pope_binary_token_ids(tokenizer)
        self._choice_ids = build_choice_token_ids(tokenizer)
        self._stats = IFCBStats()
        self._audit_log: list[dict[str, Any]] = []

    def remove(self) -> None:
        self._capture.remove()

    def get_stats(self) -> dict[str, float]:
        return self._stats.to_dict()

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)

    def step(
        self,
        *,
        model_inputs: dict[str, torch.Tensor],
        dataset: str,
        step_idx: int,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        self._capture.clear()
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(**model_inputs, use_cache=False)
        scores = outputs.logits[:, -1, :]
        adjusted_scores = scores

        if self._should_intervene(dataset, step_idx):
            adjusted_scores = self._apply_ifcb(dataset, model_inputs["input_ids"], scores)

        next_token = self._select_next_token(dataset, adjusted_scores)
        self._stats.decode_steps += 1
        return next_token, adjusted_scores

    def _should_intervene(self, dataset: str, step_idx: int) -> bool:
        if dataset in {"pope", "hallusionbench", "mme", "mmbench"}:
            return step_idx == 0
        return True

    def _frontier_ids(self, scores: torch.Tensor) -> torch.LongTensor:
        top_k = min(self.cfg.frontier_k, scores.shape[-1])
        return scores[0].topk(top_k).indices

    def _binary_answer_ids(self) -> torch.LongTensor:
        ids = sorted(set(self._pope_binary_ids["yes"] + self._pope_binary_ids["no"]))
        return torch.tensor(ids, dtype=torch.long)

    def _candidate_ids(self, dataset: str, scores: torch.Tensor) -> torch.LongTensor:
        if dataset in {"pope", "hallusionbench", "mme"}:
            return self._binary_answer_ids().to(device=scores.device)
        if dataset == "mmbench":
            return self._frontier_ids(scores)
        candidate_ids = self._frontier_ids(scores)
        semantic_ids = [token_id for token_id in candidate_ids.tolist() if is_semantic_token(self.tokenizer, token_id)]
        if not semantic_ids:
            semantic_ids = candidate_ids.tolist()
        return torch.tensor(semantic_ids, device=scores.device, dtype=torch.long)

    def _select_next_token(self, dataset: str, adjusted_scores: torch.Tensor) -> torch.LongTensor:
        if dataset in {"pope", "hallusionbench", "mme"}:
            binary_ids = sorted(set(self._pope_binary_ids["yes"] + self._pope_binary_ids["no"]))
            binary_scores = adjusted_scores[0, binary_ids]
            best_index = int(binary_scores.argmax().item())
            return torch.tensor([[binary_ids[best_index]]], device=adjusted_scores.device, dtype=torch.long)
        if dataset == "mmbench":
            choice_ids: list[int] = []
            for label in ("A", "B", "C", "D"):
                choice_ids.extend(self._choice_ids[label])
            choice_ids = sorted(set(choice_ids))
            choice_scores = adjusted_scores[0, choice_ids]
            best_index = int(choice_scores.argmax().item())
            return torch.tensor([[choice_ids[best_index]]], device=adjusted_scores.device, dtype=torch.long)
        return adjusted_scores.argmax(dim=-1, keepdim=True)

    def _apply_ifcb(self, dataset: str, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        candidate_ids = self._candidate_ids(dataset, scores)
        if candidate_ids.numel() == 0:
            return scores

        final_logprobs = F.log_softmax(scores.float(), dim=-1)
        probe_logprobs: list[torch.Tensor] = []
        probe_topk_ids: list[torch.LongTensor] = []
        top_k = max(self.cfg.probe_topk, candidate_ids.numel())
        for layer_idx in self.probe_layers:
            hidden = self._capture.hidden_by_layer[layer_idx][:, -1:, :]
            logits = self.lm_head(self.final_norm(hidden))[:, -1, :]
            logprobs = F.log_softmax(logits.float(), dim=-1)
            probe_logprobs.append(logprobs)
            probe_topk_ids.append(logprobs.topk(min(top_k, logprobs.shape[-1]), dim=-1).indices[0])

        visual_participation = self._compute_visual_participation(candidate_ids, input_ids, scores)
        risks = compute_commitment_risks(
            candidate_ids=candidate_ids,
            probe_logprobs=probe_logprobs,
            final_logprobs=final_logprobs,
            probe_topk_ids=probe_topk_ids,
            visual_participation=visual_participation,
        )
        adjusted = apply_ifcb_penalty(scores, candidate_ids, risks, self.cfg.alpha)
        self._record_step(candidate_ids, risks)
        return adjusted

    def _compute_visual_participation(
        self,
        candidate_ids: torch.LongTensor,
        input_ids: torch.LongTensor,
        scores: torch.Tensor,
    ) -> dict[int, float]:
        fusion_hidden = self._capture.hidden_by_layer[self.fusion_layer]
        if not fusion_hidden.requires_grad or not scores.requires_grad:
            raise RuntimeError("IFCB requires gradient-enabled fusion hidden states and final logits to compute G_t")
        if self.model_family == "instructblip":
            visual_mask, text_mask = build_query_token_masks(
                self.query_token_count,
                fusion_hidden.shape[1],
                fusion_hidden.device,
            )
        else:
            visual_mask, text_mask = build_modal_masks(input_ids, self.image_token_id, fusion_hidden.shape[1])
        values: dict[int, float] = {}
        candidate_list = candidate_ids.tolist()
        for idx, token_id in enumerate(candidate_list):
            grad = torch.autograd.grad(
                scores[0, token_id],
                fusion_hidden,
                retain_graph=idx < (len(candidate_list) - 1),
                allow_unused=False,
            )[0]
            if grad is None:
                raise RuntimeError(f"IFCB could not compute G_t for candidate token {token_id}")
            grad_norm = grad.abs().sum(dim=-1)[0]
            visual = float(grad_norm[visual_mask].sum().item()) if visual_mask.any() else 0.0
            text = float(grad_norm[text_mask].sum().item()) if text_mask.any() else 0.0
            values[int(token_id)] = visual / (visual + text + self.cfg.epsilon)
        return values

    def _record_step(self, candidate_ids: torch.LongTensor, risks: dict[int, dict[str, float]]) -> None:
        self._stats.intervention_steps += 1
        self._stats.candidate_count_sum += int(candidate_ids.numel())
        risk_values = [float(risks[token_id]["risk"]) for token_id in candidate_ids.tolist()]
        visual_values = [float(risks[token_id]["visual_participation"]) for token_id in candidate_ids.tolist()]
        self._stats.total_risk_sum += sum(risk_values) / max(len(risk_values), 1)
        self._stats.visual_participation_sum += sum(visual_values) / max(len(visual_values), 1)
        self._stats.max_risk = max(self._stats.max_risk, max(risk_values, default=0.0))
        if len(self._audit_log) < self.cfg.audit_max_steps:
            self._audit_log.append(
                {
                    "candidate_ids": candidate_ids.tolist(),
                    "risks": {
                        str(token_id): {
                            key: round(float(value), 6)
                            for key, value in risks[token_id].items()
                        }
                        for token_id in candidate_ids.tolist()
                    },
                }
            )

    def _get_final_norm(self, model: Any) -> torch.nn.Module:
        root = _get_decoder_root(model)
        if hasattr(root, "norm"):
            return root.norm
        if hasattr(root, "decoder") and hasattr(root.decoder, "norm"):
            return root.decoder.norm
        raise AttributeError(f"Could not locate final norm for {type(model).__name__}")


class LLaVAIFCBProcessor(_BaseIFCBProcessor):
    def __init__(self, model: Any, tokenizer: Any, image_token_id: int, config: IFCBConfig | None = None):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_token_id=image_token_id,
            config=config,
            model_family="llava",
        )


class InstructBLIPIFCBProcessor(_BaseIFCBProcessor):
    def __init__(self, model: Any, tokenizer: Any, image_token_id: int, config: IFCBConfig | None = None):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_token_id=image_token_id,
            config=config,
            model_family="instructblip",
            query_token_count=getattr(model.config, "num_query_tokens", 32),
        )


def create_ifcb_processor(model: Any, tokenizer: Any, model_family: str, image_token_id: int, config: IFCBConfig | None = None):
    family = (model_family or "").lower()
    if family == "llava":
        return LLaVAIFCBProcessor(model=model, tokenizer=tokenizer, image_token_id=image_token_id, config=config)
    if family == "instructblip":
        return InstructBLIPIFCBProcessor(model=model, tokenizer=tokenizer, image_token_id=image_token_id, config=config)
    raise ValueError(f"IFCB is not implemented for model family '{model_family}'")
