"""
Universal decode-time sidecar primitives for train-once, run-anywhere grounding
control.

Unlike `bra_logits_processor.py`, this module never consumes model-native hidden
states or `lm_head.weight`. It operates on decoded candidate strings, frozen
external embeddings, and a parameter-free tokenizer bridge. That makes it the
correct architectural home for a strict universal plugin claim.

During the v2 refactor, this file remains the source of reusable universal data
types and scorer definitions. The legacy `UniversalSidecarProcessor` is retained
as a v1 reference, but new runtime behavior should move into the dedicated v2
runtime stack.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn as nn


@dataclass
class UniversalPluginConfig:
    top_k: int = 50
    bias_scale: float = 0.30
    score_temperature: float = 0.35
    lambda_con: float = 1.0
    bias_mode: str = "signed"
    host_relative_strength: float = 1.0
    max_prefix_chars: int = 128
    max_audit_steps: int = 32
    min_groundable_chars: int = 2
    allow_numeric_candidates: bool = True
    abstain_threshold: float = 0.60
    abort_on_prefix_ambiguity: bool = True
    ambiguity_abort_threshold: float = 0.35
    suffix_protect_continuations: bool = True
    runtime_version: str = "uniground_v1"


@dataclass
class UniversalPluginStats:
    eligible_steps: int = 0
    intervention_steps: int = 0
    abstention_steps: int = 0
    abort_trigger_steps: int = 0
    abort_backoff_verified_steps: int = 0
    prefix_ambiguity_events: int = 0
    span_collapse_errors: int = 0
    suffix_stability_checks: int = 0
    suffix_stability_successes: int = 0
    candidate_construction_time_ms_sum: float = 0.0
    sidecar_scoring_time_ms_sum: float = 0.0
    bridge_redistribution_time_ms_sum: float = 0.0
    jitter_time_ms_sum: float = 0.0

    def reset(self) -> None:
        self.__dict__.update(type(self)().__dict__)

    def to_dict(self) -> dict[str, Any]:
        eligible = max(self.eligible_steps, 1)
        suffix_checks = max(self.suffix_stability_checks, 1)
        return {
            "runtime_version": "uniground_v1",
            "intervention_rate": self.intervention_steps / eligible,
            "abstention_rate": self.abstention_steps / eligible,
            "abort_trigger_rate": self.abort_trigger_steps / eligible,
            "abort_backoff_verified_rate": self.abort_backoff_verified_steps / eligible,
            "abort_backoff_verified_steps": self.abort_backoff_verified_steps,
            "prefix_ambiguity_rate": self.prefix_ambiguity_events / eligible,
            "span_collapse_errors": int(self.span_collapse_errors),
            "suffix_stability_rate": self.suffix_stability_successes / suffix_checks,
            "latency_split": {
                "candidate_construction_ms": self.candidate_construction_time_ms_sum / eligible,
                "sidecar_scoring_ms": self.sidecar_scoring_time_ms_sum / eligible,
                "bridge_redistribution_ms": self.bridge_redistribution_time_ms_sum / eligible,
                "jitter_ms": self.jitter_time_ms_sum / eligible,
            },
        }


@dataclass
class UniversalObservation:
    image_embedding: torch.Tensor
    region_embeddings: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalCandidate:
    token_id: int
    token_text: str
    span_text: str
    normalized_text: str
    is_groundable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalPluginOutput:
    support: torch.Tensor
    contradiction: torch.Tensor
    abstain: torch.Tensor


class FrozenTextEncoder(Protocol):
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return one embedding per input string."""


class UniversalScorer(Protocol):
    def __call__(
        self,
        observation: UniversalObservation,
        candidate_embeddings: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> UniversalPluginOutput:
        """Return support / contradiction / abstain scores per candidate."""


class StringStructuralGate:
    """
    Parameter-free gate over decoded strings.

    This is the universal counterpart to vocab-indexed VASM. It operates on the
    decoded candidate string rather than token IDs, which keeps the learned path
    tokenizer-agnostic.
    """

    DEFAULT_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
        "with", ".", ",", ":", ";", "?", "!", "'", '"', "-", "(", ")", "[", "]",
    }

    def __init__(self, min_groundable_chars: int = 2, allow_numeric_candidates: bool = True):
        self.min_groundable_chars = min_groundable_chars
        self.allow_numeric_candidates = allow_numeric_candidates

    def is_groundable(self, text: str) -> bool:
        normalized = normalize_span(text)
        if not normalized:
            return False
        if normalized in self.DEFAULT_STOPWORDS:
            return False
        if len(normalized) < self.min_groundable_chars:
            return False
        if normalized.isdigit() and not self.allow_numeric_candidates:
            return False
        return any(ch.isalnum() for ch in normalized)


class PrefixSpanMapper:
    """
    Deterministic tokenizer bridge.

    The universal plugin acts on decoded strings, then maps span scores back to
    next-token logits for the currently available token prefixes. No learned,
    model-specific adapter is allowed here.
    """

    def build_candidates(
        self,
        tokenizer: Any,
        top_ids: torch.Tensor,
        gate: StringStructuralGate,
        prefix_text: str = "",
    ) -> list[UniversalCandidate]:
        candidates: list[UniversalCandidate] = []
        option_semantics = extract_option_letter_semantics(prefix_text)
        for token_id in top_ids.tolist():
            token_text = decode_token(tokenizer, token_id)
            span_text = canonicalize_token_text(token_text)
            normalized = normalize_span(span_text)
            option_key = option_letter_key(span_text)
            semantic_span = option_semantics.get(option_key)
            candidate_span = semantic_span or span_text
            if semantic_span is not None:
                normalized = semantic_span
            binary_focus = bool(option_semantics)
            is_binary_target = (
                semantic_span is not None
                or normalized in {"yes", "no"}
                or option_key in option_semantics
            )
            candidates.append(
                UniversalCandidate(
                    token_id=token_id,
                    token_text=token_text,
                    span_text=candidate_span,
                    normalized_text=normalized,
                    is_groundable=gate.is_groundable(candidate_span) and (not binary_focus or is_binary_target),
                    metadata={
                        "raw_token_text": token_text,
                        "raw_span_text": span_text,
                        "semantic_alias": semantic_span,
                        "binary_option_focus": binary_focus,
                        "binary_option_target": is_binary_target,
                        "is_continuation_fragment": is_continuation_fragment(token_text),
                    },
                )
            )
        return candidates

    def scatter_bias(
        self,
        original_scores: torch.Tensor,
        top_ids: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        updated = original_scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(original_scores.dtype))
        return updated


class MLPUniversalScorer(nn.Module):
    """
    Small learned universal core `Psi_univ`.

    Inputs live entirely in the frozen external embedding space:
    - global image embedding
    - optional pooled region embedding
    - prefix embedding
    - candidate span embedding
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        fused_dim = embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        observation: UniversalObservation,
        candidate_embeddings: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> UniversalPluginOutput:
        image_embedding = observation.image_embedding
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.unsqueeze(0)
        if image_embedding.shape[0] == 1:
            image_embedding = image_embedding.expand(candidate_embeddings.shape[0], -1)

        if observation.region_embeddings is None:
            region_summary = image_embedding
        else:
            region_embeddings = observation.region_embeddings
            if region_embeddings.ndim == 3:
                region_summary = region_embeddings.mean(dim=1)
            elif region_embeddings.ndim == 2 and region_embeddings.shape[0] == candidate_embeddings.shape[0]:
                region_summary = region_embeddings
            else:
                region_summary = region_embeddings.mean(dim=0, keepdim=True)
                region_summary = region_summary.expand(candidate_embeddings.shape[0], -1)

        if prefix_embedding.shape[0] == 1:
            prefix_embedding = prefix_embedding.expand(candidate_embeddings.shape[0], -1)

        fused = torch.cat(
            [image_embedding, region_summary, prefix_embedding, candidate_embeddings],
            dim=-1,
        )
        logits = self.net(fused)
        return UniversalPluginOutput(
            support=logits[:, 0],
            contradiction=logits[:, 1],
            abstain=logits[:, 2],
        )


class UniversalSidecarProcessor:
    """
    Decode-time logits processor for the universal route.

    This processor only assumes access to:
    - current next-token logits
    - tokenizer decoding
    - frozen external image/text embeddings
    - the universal learned scorer `Psi_univ`
    """

    def __init__(
        self,
        config: UniversalPluginConfig,
        tokenizer: Any,
        text_encoder: FrozenTextEncoder,
        scorer: UniversalScorer,
        observation: UniversalObservation,
        gate: Optional[StringStructuralGate] = None,
        mapper: Optional[PrefixSpanMapper] = None,
    ):
        self.cfg = config
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scorer = scorer
        self.observation = observation
        self.gate = gate or StringStructuralGate(
            min_groundable_chars=config.min_groundable_chars,
            allow_numeric_candidates=config.allow_numeric_candidates,
        )
        self.mapper = mapper or PrefixSpanMapper()
        self.audit_log: list[dict[str, Any]] = []
        self.stats = UniversalPluginStats()

    def reset(self) -> None:
        self.audit_log.clear()
        self.stats.reset()

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self.audit_log)

    def get_stats(self) -> dict[str, Any]:
        return self.stats.to_dict()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() != 2 or scores.shape[0] != 1:
            return scores

        total_start = time.perf_counter()
        top_k = min(self.cfg.top_k, scores.shape[-1])
        top_vals, top_ids = scores.topk(top_k, dim=-1)
        prefix_text = self._decode_prefix(input_ids)
        construction_start = time.perf_counter()
        candidates = self.mapper.build_candidates(self.tokenizer, top_ids[0], self.gate, prefix_text=prefix_text)
        ambiguity_rate, span_collapse_errors = summarize_prefix_ambiguity(candidates)
        self.stats.eligible_steps += 1
        self.stats.prefix_ambiguity_events += int(ambiguity_rate > 0.0)
        self.stats.span_collapse_errors += span_collapse_errors
        construction_ms = (time.perf_counter() - construction_start) * 1000.0
        self.stats.candidate_construction_time_ms_sum += construction_ms

        active = [candidate for candidate in candidates if candidate.is_groundable]
        if not active:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.stats.jitter_time_ms_sum += max(elapsed_ms - construction_ms, 0.0)
            self._record(
                prefix_text="",
                candidates=candidates,
                bias=None,
                aborted=False,
                abort_reason=None,
                abstain_max=0.0,
                prefix_ambiguity_rate=ambiguity_rate,
                span_collapse_errors=span_collapse_errors,
                latency_split_ms={
                    "candidate_construction_ms": round(construction_ms, 6),
                    "sidecar_scoring_ms": 0.0,
                    "bridge_redistribution_ms": 0.0,
                    "jitter_ms": round(max(elapsed_ms - construction_ms, 0.0), 6),
                },
            )
            return scores

        scoring_start = time.perf_counter()
        prefix_embedding = self.text_encoder.encode_texts([prefix_text]).to(scores.device)
        candidate_embeddings = self.text_encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)

        output = self.scorer(
            observation=_move_observation(self.observation, scores.device),
            candidate_embeddings=candidate_embeddings,
            prefix_embedding=prefix_embedding,
        )
        scoring_ms = (time.perf_counter() - scoring_start) * 1000.0
        self.stats.sidecar_scoring_time_ms_sum += scoring_ms
        bias = self._compute_bias(output).to(scores.device)
        abstain_probs = output.abstain.sigmoid()
        abstain_max = float(abstain_probs.max().item()) if abstain_probs.numel() else 0.0
        should_abort, abort_reason = self._should_abort(abstain_max, ambiguity_rate)
        self.stats.abstention_steps += int(abstain_max >= self.cfg.abstain_threshold)
        self.stats.abort_trigger_steps += int(should_abort)

        redistribution_start = time.perf_counter()
        full_bias = torch.zeros_like(top_vals[0], dtype=torch.float32)
        active_pos = {candidate.token_id: idx for idx, candidate in enumerate(active)}
        for i, token_id in enumerate(top_ids[0].tolist()):
            idx = active_pos.get(token_id)
            if idx is not None:
                full_bias[i] = bias[idx]

        bias_would_intervene = bool(torch.any(full_bias.abs() > 1e-8).item())
        updated_scores = scores
        if should_abort:
            self.stats.abort_backoff_verified_steps += int(bias_would_intervene)
        else:
            updated_scores = self.mapper.scatter_bias(scores, top_ids[0], full_bias)
            self.stats.intervention_steps += int(bias_would_intervene)

        redistribution_ms = (time.perf_counter() - redistribution_start) * 1000.0
        self.stats.bridge_redistribution_time_ms_sum += redistribution_ms
        continuation_candidates = [candidate for candidate in candidates if candidate.metadata.get("is_continuation_fragment")]
        if continuation_candidates:
            self.stats.suffix_stability_checks += 1
            continuation_ids = {candidate.token_id for candidate in continuation_candidates}
            continuation_bias_ok = True
            for i, token_id in enumerate(top_ids[0].tolist()):
                if token_id in continuation_ids and abs(float(full_bias[i].item())) > 1e-8:
                    continuation_bias_ok = False
                    break
            self.stats.suffix_stability_successes += int(continuation_bias_ok or should_abort)

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        jitter_ms = max(total_elapsed_ms - construction_ms - scoring_ms - redistribution_ms, 0.0)
        self.stats.jitter_time_ms_sum += jitter_ms
        self._record(
            prefix_text=prefix_text,
            candidates=candidates,
            bias=full_bias,
            aborted=should_abort,
            abort_reason=abort_reason,
            abstain_max=abstain_max,
            prefix_ambiguity_rate=ambiguity_rate,
            span_collapse_errors=span_collapse_errors,
            latency_split_ms={
                "candidate_construction_ms": round(construction_ms, 6),
                "sidecar_scoring_ms": round(scoring_ms, 6),
                "bridge_redistribution_ms": round(redistribution_ms, 6),
                "jitter_ms": round(jitter_ms, 6),
            },
        )
        return updated_scores

    def _decode_prefix(self, input_ids: torch.LongTensor) -> str:
        try:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            text = str(input_ids[0].tolist())
        return text[-self.cfg.max_prefix_chars :]

    def _compute_bias(self, output: UniversalPluginOutput) -> torch.Tensor:
        support = output.support / max(self.cfg.score_temperature, 1e-6)
        contradiction = output.contradiction / max(self.cfg.score_temperature, 1e-6)
        abstain = output.abstain.sigmoid()
        signed = (support - contradiction).tanh()
        return self.cfg.bias_scale * signed * (1.0 - abstain)

    def _should_abort(self, abstain_max: float, ambiguity_rate: float) -> tuple[bool, Optional[str]]:
        if abstain_max >= self.cfg.abstain_threshold:
            return True, "abstention_threshold"
        if ambiguity_rate >= self.cfg.ambiguity_abort_threshold:
            return True, "prefix_ambiguity"
        return False, None

    def _record(
        self,
        prefix_text: str,
        candidates: list[UniversalCandidate],
        bias: Optional[torch.Tensor],
        aborted: bool,
        abort_reason: Optional[str],
        abstain_max: float,
        prefix_ambiguity_rate: float,
        span_collapse_errors: int,
        latency_split_ms: dict[str, float],
    ) -> None:
        if len(self.audit_log) >= self.cfg.max_audit_steps:
            return
        entry = {
            "prefix_tail": prefix_text,
            "candidates": [
                {
                    "token_id": candidate.token_id,
                    "token_text": candidate.token_text,
                    "span_text": candidate.span_text,
                    "normalized_text": candidate.normalized_text,
                    "is_groundable": candidate.is_groundable,
                }
                for candidate in candidates[:10]
            ],
            "aborted": aborted,
            "abort_reason": abort_reason,
            "abstain_max": round(abstain_max, 6),
            "prefix_ambiguity_rate": round(prefix_ambiguity_rate, 6),
            "span_collapse_errors": span_collapse_errors,
            "latency_split_ms": latency_split_ms,
        }
        if bias is not None:
            entry["bias"] = [round(float(x), 6) for x in bias[:10].tolist()]
        self.audit_log.append(entry)


def normalize_span(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def option_letter_key(text: str) -> str:
    raw = text.strip()
    raw = raw.replace("Ġ", "").replace("▁", "")
    raw = raw.replace("ï¼¡", "A").replace("ï¼¢", "B")
    match = re.match(r"^[\W_]*([AB])[\W_]*$", raw)
    if match:
        return match.group(1).lower()
    return ""


def extract_option_letter_semantics(prefix_text: str) -> dict[str, str]:
    lowered = prefix_text.lower()
    if "answer with the option letter only" not in lowered:
        return {}
    if "a. yes" in lowered and "b. no" in lowered:
        return {"a": "yes", "b": "no"}
    if "a. no" in lowered and "b. yes" in lowered:
        return {"a": "no", "b": "yes"}
    return {}


def canonicalize_token_text(token_text: str) -> str:
    text = token_text.replace("Ġ", " ").replace("▁", " ")
    text = text.replace("</w>", "")
    text = text.replace("##", "")
    return text.strip()


def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        token = tokenizer.convert_ids_to_tokens(token_id)
    return token or str(token_id)


def is_continuation_fragment(token_text: str) -> bool:
    stripped = token_text.strip()
    if not stripped:
        return False
    if token_text.startswith("##"):
        return True
    if token_text.startswith(("Ġ", "▁")):
        return False
    if stripped[0].isalnum() and token_text[:1].isalnum():
        return True
    return False


def summarize_prefix_ambiguity(candidates: list[UniversalCandidate]) -> tuple[float, int]:
    normalized_counts: dict[str, int] = {}
    semantic_alias_groups: dict[str, set[str]] = {}
    active = [candidate for candidate in candidates if candidate.is_groundable and candidate.normalized_text]
    for candidate in active:
        normalized = candidate.normalized_text
        alias = candidate.metadata.get("semantic_alias")
        raw_span = candidate.metadata.get("raw_span_text", candidate.span_text)
        if alias is not None:
            semantic_alias_groups.setdefault(normalized, set()).add(str(raw_span))
            normalized_counts.setdefault(normalized, 1)
        else:
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
    ambiguous = 0
    collapse_errors = 0
    for normalized, count in normalized_counts.items():
        alias_variants = semantic_alias_groups.get(normalized)
        if alias_variants:
            continue
        if count > 1:
            ambiguous += count
            collapse_errors += count - 1
    rate = ambiguous / max(len(active), 1)
    return rate, collapse_errors


def build_universal_claim_manifest(checkpoint_meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload = {
        "method_route": "TLRA_univ",
        "learned_module": "Psi_univ",
        "uses_model_native_hidden_states": False,
        "uses_lm_head_geometry": False,
        "learned_per_model_adapter": False,
        "parameter_free_tokenizer_bridge": True,
        "abstention_controls_behavior": True,
    }
    if checkpoint_meta:
        payload["psi_univ_checkpoint"] = checkpoint_meta
    return payload


def build_universal_result_payload(
    processor: UniversalSidecarProcessor,
    checkpoint_meta: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "universal_claim_manifest": build_universal_claim_manifest(checkpoint_meta),
        **processor.get_stats(),
    }
    if extra:
        payload.update(extra)
    return payload


def _move_observation(observation: UniversalObservation, device: torch.device) -> UniversalObservation:
    image_embedding = observation.image_embedding.to(device)
    region_embeddings = observation.region_embeddings.to(device) if observation.region_embeddings is not None else None
    return UniversalObservation(
        image_embedding=image_embedding,
        region_embeddings=region_embeddings,
        metadata=dict(observation.metadata),
    )
