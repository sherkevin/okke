from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

import torch

from bra_universal_plugin import (
    StringStructuralGate,
    UniversalObservation,
    UniversalPluginConfig,
    UniversalPluginOutput,
)
from uniground_v2.contracts import RetrievalResult
from uniground_v2.bridge import CandidateFrontierBuilder
from uniground_v2.regions import RegionRetriever
from uniground_v2.trigger import EntropyMarginTrigger, TriggerDecision


@dataclass
class RuntimeStepRecord:
    trigger_fired: bool
    trigger_reason: str
    active_candidates: int
    intervened: bool
    abstention_score: float
    abstained: bool
    aborted: bool
    abort_backoff_verified: bool
    prefix_text: str
    candidate_construction_ms: float
    sidecar_scoring_ms: float
    bridge_redistribution_ms: float
    total_step_ms: float


class TriggerProtocol(Protocol):
    def decide(
        self,
        scores: torch.FloatTensor,
        input_ids: torch.LongTensor,
        tokenizer: Any,
    ) -> TriggerDecision:
        ...


class CandidateBuilderProtocol(Protocol):
    def build(
        self,
        tokenizer: Any,
        top_ids: torch.Tensor,
        gate: StringStructuralGate,
        prefix_text: str,
    ) -> list[Any]:
        ...


class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        observation: UniversalObservation,
        candidates: list[Any],
        device: torch.device,
    ) -> RetrievalResult:
        ...


class _IdentityRetriever:
    def retrieve(
        self,
        observation: UniversalObservation,
        candidates: list[Any],
        device: torch.device,
    ) -> RetrievalResult:
        return RetrievalResult(observation=_move_observation(observation, device), metadata={})


class _RuntimeAudit:
    def __init__(self) -> None:
        self.audit_log: list[dict[str, Any]] = []
        self.step_records: list[RuntimeStepRecord] = []

    def reset(self) -> None:
        self.audit_log.clear()
        self.step_records.clear()

    def append(self, entry: dict[str, Any], step_record: RuntimeStepRecord) -> None:
        self.audit_log.append(entry)
        self.step_records.append(step_record)

    def summary(self) -> dict[str, Any]:
        if not self.step_records:
            return {
                "intervention_coverage": 0.0,
                "trigger_fire_rate": 0.0,
                "abstention_rate": 0.0,
                "abort_trigger_rate": 0.0,
                "abort_backoff_verified_steps": 0,
                "avg_active_candidates": 0.0,
                "candidate_construction_ms": 0.0,
                "sidecar_scoring_ms": 0.0,
                "bridge_redistribution_ms": 0.0,
                "total_step_ms": 0.0,
                "latency_split": {
                    "candidate_construction_ms": 0.0,
                    "sidecar_scoring_ms": 0.0,
                    "bridge_redistribution_ms": 0.0,
                    "total_step_ms": 0.0,
                },
            }

        total = len(self.step_records)
        mean = lambda values: round(sum(values) / max(len(values), 1), 4)
        candidate_ms = mean([record.candidate_construction_ms for record in self.step_records])
        scoring_ms = mean([record.sidecar_scoring_ms for record in self.step_records])
        bridge_ms = mean([record.bridge_redistribution_ms for record in self.step_records])
        total_ms = mean([record.total_step_ms for record in self.step_records])
        return {
            "intervention_coverage": round(sum(1 for record in self.step_records if record.intervened) / total, 4),
            "trigger_fire_rate": round(sum(1 for record in self.step_records if record.trigger_fired) / total, 4),
            "abstention_rate": round(sum(1 for record in self.step_records if record.abstained) / total, 4),
            "abort_trigger_rate": round(sum(1 for record in self.step_records if record.aborted) / total, 4),
            "abort_backoff_verified_steps": int(sum(1 for record in self.step_records if record.abort_backoff_verified)),
            "avg_active_candidates": mean([record.active_candidates for record in self.step_records]),
            "candidate_construction_ms": candidate_ms,
            "sidecar_scoring_ms": scoring_ms,
            "bridge_redistribution_ms": bridge_ms,
            "total_step_ms": total_ms,
            "latency_split": {
                "candidate_construction_ms": candidate_ms,
                "sidecar_scoring_ms": scoring_ms,
                "bridge_redistribution_ms": bridge_ms,
                "total_step_ms": total_ms,
            },
        }


class UniGroundV2LogitsProcessor:
    """
    Single runtime kernel for the v2 universal path.

    The processor intentionally delegates trigger, bridge, and retrieval logic
    to injectable components so later phases can evolve each subsystem without
    creating multiple competing logits processors.
    """

    def __init__(
        self,
        config: UniversalPluginConfig,
        tokenizer: Any,
        encoder: Any,
        scorer: Any,
        observation: UniversalObservation,
        *,
        trigger: Optional[TriggerProtocol] = None,
        candidate_builder: Optional[CandidateBuilderProtocol] = None,
        retriever: Optional[RetrieverProtocol] = None,
        gate: Optional[StringStructuralGate] = None,
    ) -> None:
        self.cfg = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.scorer = scorer
        self.base_observation = observation
        self.trigger = trigger or EntropyMarginTrigger(top_k=config.top_k)
        self.candidate_builder = candidate_builder or CandidateFrontierBuilder()
        self.retriever = retriever or RegionRetriever(encoder=encoder)
        self.gate = gate or StringStructuralGate(
            min_groundable_chars=config.min_groundable_chars,
            allow_numeric_candidates=config.allow_numeric_candidates,
        )
        self.audit = _RuntimeAudit()

    def reset(self) -> None:
        self.audit.reset()
        if hasattr(self.candidate_builder, "reset"):
            self.candidate_builder.reset()

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self.audit.audit_log)

    def get_summary_stats(self) -> dict[str, Any]:
        return self.audit.summary()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() != 2 or scores.shape[0] != 1:
            return scores

        total_start = time.perf_counter()
        prefix_text = self._decode_prefix(input_ids)
        if hasattr(self.candidate_builder, "sync_prefix_state"):
            self.candidate_builder.sync_prefix_state(self.tokenizer, input_ids)
        trigger_decision = self.trigger.decide(scores, input_ids, self.tokenizer)

        if not trigger_decision.fire:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "prefix_text": prefix_text,
                    "trigger": asdict(trigger_decision),
                    "candidates": [],
                    "retrieval": {},
                    "bias": [],
                    "intervened": False,
                },
                step_record=RuntimeStepRecord(
                    trigger_fired=False,
                    trigger_reason=trigger_decision.reason,
                    active_candidates=0,
                    intervened=False,
                    abstention_score=-1.0,
                    abstained=False,
                    aborted=False,
                    abort_backoff_verified=False,
                    prefix_text=prefix_text,
                    candidate_construction_ms=0.0,
                    sidecar_scoring_ms=0.0,
                    bridge_redistribution_ms=0.0,
                    total_step_ms=elapsed_ms,
                ),
            )
            return scores

        top_k = min(self.cfg.top_k, scores.shape[-1])
        _top_vals, top_ids = scores.topk(top_k, dim=-1)

        construction_start = time.perf_counter()
        candidates = self.candidate_builder.build(self.tokenizer, top_ids[0], self.gate, prefix_text=prefix_text)
        active = [candidate for candidate in candidates if getattr(candidate, "is_groundable", False)]
        construction_ms = (time.perf_counter() - construction_start) * 1000.0

        if not active:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "prefix_text": prefix_text,
                    "trigger": asdict(trigger_decision),
                    "candidates": [],
                    "retrieval": {},
                    "bias": [],
                    "intervened": False,
                },
                step_record=RuntimeStepRecord(
                    trigger_fired=True,
                    trigger_reason=trigger_decision.reason,
                    active_candidates=0,
                    intervened=False,
                    abstention_score=-1.0,
                    abstained=False,
                    aborted=False,
                    abort_backoff_verified=False,
                    prefix_text=prefix_text,
                    candidate_construction_ms=construction_ms,
                    sidecar_scoring_ms=0.0,
                    bridge_redistribution_ms=0.0,
                    total_step_ms=elapsed_ms,
                ),
            )
            return scores

        scoring_start = time.perf_counter()
        retrieved = self.retriever.retrieve(self.base_observation, active, scores.device)
        prefix_embedding = self.encoder.encode_texts([prefix_text]).to(scores.device)
        candidate_embeddings = self.encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)
        scorer_output = self.scorer(retrieved.observation, candidate_embeddings, prefix_embedding)
        abstain = scorer_output.abstain.sigmoid()
        bias = self._compose_bias(scorer_output).to(scores.device)
        scoring_ms = (time.perf_counter() - scoring_start) * 1000.0

        abstention_score = float(abstain.mean().item()) if abstain.numel() else -1.0
        if abstention_score >= self.cfg.abstain_threshold:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "prefix_text": prefix_text,
                    "trigger": asdict(trigger_decision),
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
                    "retrieval": dict(retrieved.metadata),
                    "bias": [],
                    "intervened": False,
                    "aborted": True,
                    "abstention_score": round(abstention_score, 6),
                },
                step_record=RuntimeStepRecord(
                    trigger_fired=True,
                    trigger_reason=trigger_decision.reason,
                    active_candidates=len(active),
                    intervened=False,
                    abstention_score=abstention_score,
                    abstained=True,
                    aborted=True,
                    abort_backoff_verified=True,
                    prefix_text=prefix_text,
                    candidate_construction_ms=construction_ms,
                    sidecar_scoring_ms=scoring_ms,
                    bridge_redistribution_ms=0.0,
                    total_step_ms=elapsed_ms,
                ),
            )
            return scores

        bridge_start = time.perf_counter()
        full_bias = torch.zeros_like(top_ids[0], dtype=torch.float32)
        active_positions = {candidate.token_id: idx for idx, candidate in enumerate(active)}
        for i, token_id in enumerate(top_ids[0].tolist()):
            idx = active_positions.get(token_id)
            if idx is not None:
                full_bias[i] = bias[idx]
        updated_scores = self.candidate_builder.scatter_bias(scores, top_ids[0], full_bias)
        bridge_ms = (time.perf_counter() - bridge_start) * 1000.0

        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        intervened = bool((full_bias.abs() > 1e-8).any().item())
        self.audit.append(
            entry={
                "prefix_text": prefix_text,
                "trigger": asdict(trigger_decision),
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
                "retrieval": dict(retrieved.metadata),
                "bias": [round(float(value), 6) for value in full_bias[:10].tolist()],
                "intervened": intervened,
                "aborted": False,
                "abstention_score": round(abstention_score, 6),
            },
            step_record=RuntimeStepRecord(
                trigger_fired=True,
                trigger_reason=trigger_decision.reason,
                active_candidates=len(active),
                intervened=intervened,
                abstention_score=abstention_score,
                abstained=abstention_score >= self.cfg.abstain_threshold,
                aborted=False,
                abort_backoff_verified=False,
                prefix_text=prefix_text,
                candidate_construction_ms=construction_ms,
                sidecar_scoring_ms=scoring_ms,
                bridge_redistribution_ms=bridge_ms,
                total_step_ms=elapsed_ms,
            ),
        )
        return updated_scores

    def _decode_prefix(self, input_ids: torch.LongTensor) -> str:
        try:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            text = str(input_ids[0].tolist())
        return text[-self.cfg.max_prefix_chars :]

    def _compose_bias(self, output: UniversalPluginOutput) -> torch.Tensor:
        support = output.support.sigmoid()
        contradiction = output.contradiction.sigmoid()
        abstain = output.abstain.sigmoid()
        resonance = (1.0 - abstain) * (support - self.cfg.lambda_con * contradiction)
        return self.cfg.bias_scale * torch.tanh(resonance / max(self.cfg.score_temperature, 1e-6))


def _move_observation(observation: UniversalObservation, device: torch.device) -> UniversalObservation:
    image_embedding = observation.image_embedding.to(device)
    region_embeddings = observation.region_embeddings
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(device)
    return UniversalObservation(
        image_embedding=image_embedding,
        region_embeddings=region_embeddings,
        metadata=dict(observation.metadata),
    )
