from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

import torch

from bra_universal_plugin import (
    StringStructuralGate,
    UniversalCandidate,
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
        context: Optional[dict[str, Any]] = None,
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
        self._row_candidate_builders: dict[int, Any] = {}

    def reset(self) -> None:
        self.audit.reset()
        if hasattr(self.candidate_builder, "reset"):
            self.candidate_builder.reset()
        for builder in self._row_candidate_builders.values():
            if hasattr(builder, "reset"):
                builder.reset()
        self._row_candidate_builders.clear()

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self.audit.audit_log)

    def get_summary_stats(self) -> dict[str, Any]:
        return self.audit.summary()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() != 2:
            return scores

        updated_scores = scores.clone()
        for row_idx in range(scores.shape[0]):
            row_input_ids = input_ids[row_idx : row_idx + 1]
            row_scores = updated_scores[row_idx : row_idx + 1]
            updated_scores[row_idx : row_idx + 1] = self._process_row(
                row_idx=row_idx,
                input_ids=row_input_ids,
                scores=row_scores,
                observation=self._observation_for_row(row_idx, row_scores.device),
            )
        return updated_scores

    def _process_row(
        self,
        *,
        row_idx: int,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        observation: UniversalObservation,
    ) -> torch.FloatTensor:
        candidate_builder = self._candidate_builder_for_row(row_idx)
        runtime_context = self._runtime_context(observation)

        total_start = time.perf_counter()
        prefix_text = self._decode_prefix(input_ids)
        if hasattr(candidate_builder, "sync_prefix_state"):
            candidate_builder.sync_prefix_state(self.tokenizer, input_ids)
        trigger_decision = self._trigger_decide(scores, input_ids, runtime_context)

        if not trigger_decision.fire:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "row_idx": row_idx,
                    "prefix_text": prefix_text,
                    "runtime_context": self._audit_context(runtime_context),
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

        if self._uses_explicit_answer_control(runtime_context):
            return self._process_explicit_answer_control(
                row_idx=row_idx,
                scores=scores,
                observation=observation,
                prefix_text=prefix_text,
                trigger_decision=trigger_decision,
                runtime_context=runtime_context,
                total_start=total_start,
            )

        top_k = min(self.cfg.top_k, scores.shape[-1])
        _top_vals, top_ids = scores.topk(top_k, dim=-1)

        construction_start = time.perf_counter()
        candidates = self._build_candidates(candidate_builder, top_ids[0], prefix_text, runtime_context)
        active = [candidate for candidate in candidates if getattr(candidate, "is_groundable", False)]
        construction_ms = (time.perf_counter() - construction_start) * 1000.0

        if not active:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "row_idx": row_idx,
                    "prefix_text": prefix_text,
                    "runtime_context": self._audit_context(runtime_context),
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
        retrieved = self.retriever.retrieve(observation, active, scores.device)
        prefix_query_text = self._prefix_query_text(prefix_text, runtime_context)
        prefix_embedding = self.encoder.encode_texts([prefix_query_text]).to(scores.device)
        candidate_embeddings = self.encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)
        scorer_output = self.scorer(retrieved.observation, candidate_embeddings, prefix_embedding)
        abstain = scorer_output.abstain.sigmoid()
        binary_host_choice = self._binary_host_choice(active, top_ids[0], scores[0], runtime_context)
        bias, verifier_meta = self._compose_bias(
            active,
            scorer_output,
            runtime_context,
            binary_host_choice=binary_host_choice,
        )
        bias = bias.to(scores.device)
        scoring_ms = (time.perf_counter() - scoring_start) * 1000.0

        abstention_score = float(abstain.mean().item()) if abstain.numel() else -1.0
        if abstention_score >= self.cfg.abstain_threshold:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "row_idx": row_idx,
                    "prefix_text": prefix_text,
                    "runtime_context": self._audit_context(runtime_context),
                    "trigger": asdict(trigger_decision),
                    "candidates": [
                        {
                            "token_id": candidate.token_id,
                            "token_text": candidate.token_text,
                            "span_text": candidate.span_text,
                            "normalized_text": candidate.normalized_text,
                            "is_groundable": candidate.is_groundable,
                            "semantic_alias": candidate.metadata.get("semantic_alias"),
                            "binary_label": candidate.metadata.get("binary_label"),
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
        updated_scores = candidate_builder.scatter_bias(scores, top_ids[0], full_bias)
        bridge_ms = (time.perf_counter() - bridge_start) * 1000.0
        decision_audit = self._binary_decision_audit(
            active,
            top_ids[0],
            scores[0],
            full_bias,
            scorer_output,
            runtime_context,
            verifier_meta=verifier_meta,
        )

        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        intervened = bool((full_bias.abs() > 1e-8).any().item())
        self.audit.append(
            entry={
                "row_idx": row_idx,
                "prefix_text": prefix_text,
                "runtime_context": self._audit_context(runtime_context),
                "trigger": asdict(trigger_decision),
                "candidates": [
                    {
                        "token_id": candidate.token_id,
                        "token_text": candidate.token_text,
                        "span_text": candidate.span_text,
                        "normalized_text": candidate.normalized_text,
                        "is_groundable": candidate.is_groundable,
                        "semantic_alias": candidate.metadata.get("semantic_alias"),
                        "binary_label": candidate.metadata.get("binary_label"),
                    }
                    for candidate in candidates[:10]
                ],
                "retrieval": dict(retrieved.metadata),
                "bias": [round(float(value), 6) for value in full_bias[:10].tolist()],
                "intervened": intervened,
                "aborted": False,
                "abstention_score": round(abstention_score, 6),
                "decision_audit": decision_audit,
                "verifier_meta": verifier_meta,
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

    def _process_explicit_answer_control(
        self,
        *,
        row_idx: int,
        scores: torch.FloatTensor,
        observation: UniversalObservation,
        prefix_text: str,
        trigger_decision: TriggerDecision,
        runtime_context: dict[str, Any],
        total_start: float,
    ) -> torch.FloatTensor:
        construction_start = time.perf_counter()
        candidates, answer_token_ids = self._build_answer_label_candidates(runtime_context)
        answer_token_ids = {
            label: [token_id for token_id in token_ids if 0 <= token_id < scores.shape[-1]]
            for label, token_ids in answer_token_ids.items()
        }
        construction_ms = (time.perf_counter() - construction_start) * 1000.0
        active = [candidate for candidate in candidates if answer_token_ids.get(str(candidate.metadata.get("binary_label") or ""))]
        if not active:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "row_idx": row_idx,
                    "prefix_text": prefix_text,
                    "runtime_context": self._audit_context(runtime_context),
                    "trigger": asdict(trigger_decision),
                    "candidates": [],
                    "retrieval": {},
                    "bias": [],
                    "intervened": False,
                    "decision_audit": {
                        "bias_source": "answer_labels",
                        "evidence_gate_passed": False,
                        "evidence_gate_reason": "missing_answer_token_ids",
                    },
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
        retrieved = self.retriever.retrieve(observation, active, scores.device)
        prefix_query_text = self._prefix_query_text(prefix_text, runtime_context)
        prefix_embedding = self.encoder.encode_texts([prefix_query_text]).to(scores.device)
        candidate_embeddings = self.encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)
        scorer_output = self.scorer(retrieved.observation, candidate_embeddings, prefix_embedding)
        abstain = scorer_output.abstain.sigmoid()
        binary_host_choice = self._answer_label_host_choice(scores[0], answer_token_ids, runtime_context)
        bias, verifier_meta = self._compose_bias(
            active,
            scorer_output,
            runtime_context,
            binary_host_choice=binary_host_choice,
        )
        bias = bias.to(scores.device)
        scoring_ms = (time.perf_counter() - scoring_start) * 1000.0

        abstention_score = float(abstain.mean().item()) if abstain.numel() else -1.0
        if abstention_score >= self.cfg.abstain_threshold:
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            self.audit.append(
                entry={
                    "row_idx": row_idx,
                    "prefix_text": prefix_text,
                    "runtime_context": self._audit_context(runtime_context),
                    "trigger": asdict(trigger_decision),
                    "candidates": [
                        {
                            "token_id": candidate.token_id,
                            "token_text": candidate.token_text,
                            "span_text": candidate.span_text,
                            "normalized_text": candidate.normalized_text,
                            "is_groundable": candidate.is_groundable,
                            "semantic_alias": candidate.metadata.get("semantic_alias"),
                            "binary_label": candidate.metadata.get("binary_label"),
                            "answer_token_ids": answer_token_ids.get(str(candidate.metadata.get("binary_label") or ""), []),
                        }
                        for candidate in active
                    ],
                    "retrieval": dict(retrieved.metadata),
                    "bias": [],
                    "intervened": False,
                    "aborted": True,
                    "abstention_score": round(abstention_score, 6),
                    "decision_audit": verifier_meta,
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
        updated_scores = scores.clone()
        label_bias_map: dict[str, float] = {}
        for idx, candidate in enumerate(active):
            label = str(candidate.metadata.get("binary_label") or "")
            directed_bias = float(bias[idx].item())
            label_bias_map[label] = directed_bias
            for token_id in answer_token_ids.get(label, []):
                updated_scores[0, token_id] += bias[idx].to(updated_scores.dtype)
        bridge_ms = (time.perf_counter() - bridge_start) * 1000.0
        decision_audit = self._answer_label_decision_audit(
            active,
            original_scores=scores[0],
            updated_scores=updated_scores[0],
            output=scorer_output,
            answer_token_ids=answer_token_ids,
            verifier_meta=verifier_meta,
        )

        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        intervened = any(abs(value) > 1e-8 for value in label_bias_map.values())
        self.audit.append(
            entry={
                "row_idx": row_idx,
                "prefix_text": prefix_text,
                "runtime_context": self._audit_context(runtime_context),
                "trigger": asdict(trigger_decision),
                "candidates": [
                    {
                        "token_id": candidate.token_id,
                        "token_text": candidate.token_text,
                        "span_text": candidate.span_text,
                        "normalized_text": candidate.normalized_text,
                        "is_groundable": candidate.is_groundable,
                        "semantic_alias": candidate.metadata.get("semantic_alias"),
                        "binary_label": candidate.metadata.get("binary_label"),
                        "answer_token_ids": answer_token_ids.get(str(candidate.metadata.get("binary_label") or ""), []),
                    }
                    for candidate in active
                ],
                "retrieval": dict(retrieved.metadata),
                "bias": [round(float(value), 6) for value in label_bias_map.values()],
                "intervened": intervened,
                "aborted": False,
                "abstention_score": round(abstention_score, 6),
                "decision_audit": decision_audit,
                "verifier_meta": verifier_meta,
            },
            step_record=RuntimeStepRecord(
                trigger_fired=True,
                trigger_reason=trigger_decision.reason,
                active_candidates=len(active),
                intervened=intervened,
                abstention_score=abstention_score,
                abstained=False,
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

    def _uses_explicit_answer_control(self, runtime_context: dict[str, Any]) -> bool:
        return runtime_context.get("decision_mode") == "answer_labels"

    def _build_answer_label_candidates(
        self,
        runtime_context: dict[str, Any],
    ) -> tuple[list[Any], dict[str, list[int]]]:
        answer_labels = runtime_context.get("answer_labels") or []
        hypotheses = runtime_context.get("hypothesis_text_by_label") or {}
        choice_texts = runtime_context.get("answer_choice_texts") or {}
        candidates = []
        answer_token_ids: dict[str, list[int]] = {}
        for idx, label in enumerate(answer_labels):
            label = str(label)
            token_ids = self._resolve_answer_token_ids(label, choice_texts.get(label))
            answer_token_ids[label] = token_ids
            span_text = str(hypotheses.get(label) or label)
            candidates.append(
                {
                    "token_id": -1000 - idx,
                    "token_text": label,
                    "span_text": span_text,
                    "normalized_text": span_text,
                    "is_groundable": bool(token_ids),
                    "metadata": {
                        "semantic_alias": span_text,
                        "binary_label": label,
                    },
                }
            )
        universal_candidates = []
        for candidate in candidates:
            universal_candidates.append(
                UniversalCandidate(
                    token_id=candidate["token_id"],
                    token_text=candidate["token_text"],
                    span_text=candidate["span_text"],
                    normalized_text=candidate["normalized_text"],
                    is_groundable=candidate["is_groundable"],
                    metadata=candidate["metadata"],
                )
            )
        return universal_candidates, answer_token_ids

    def _resolve_answer_token_ids(self, label: str, explicit_texts: Any) -> list[int]:
        surfaces: list[str] = []
        if isinstance(explicit_texts, list):
            surfaces.extend(str(text) for text in explicit_texts if str(text).strip())
        elif explicit_texts:
            surfaces.append(str(explicit_texts))
        if not surfaces:
            surfaces.append(label)
        prefixed = []
        for surface in surfaces:
            cleaned = surface.strip()
            if not cleaned:
                continue
            prefixed.extend([cleaned, f" {cleaned}"])
        surfaces = list(dict.fromkeys(prefixed))
        token_ids: list[int] = []
        for surface in surfaces:
            token_ids.extend(self._encode_single_token_ids(surface))
        return list(dict.fromkeys(token_ids))

    def _encode_single_token_ids(self, surface: str) -> list[int]:
        tokenizer = self.tokenizer
        token_ids: list[int] = []
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                ids = encode(surface, add_special_tokens=False)
            except TypeError:
                ids = encode(surface)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, list) and len(ids) == 1:
                token_ids.append(int(ids[0]))
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            candidate_id = convert(surface)
            if isinstance(candidate_id, int) and candidate_id >= 0:
                token_ids.append(int(candidate_id))
        return token_ids

    def _answer_label_host_choice(
        self,
        original_scores: torch.Tensor,
        answer_token_ids: dict[str, list[int]],
        runtime_context: dict[str, Any],
    ) -> str | None:
        yes_label, no_label = self._binary_label_pair(runtime_context)
        label_scores: dict[str, float] = {}
        for label in (yes_label, no_label):
            token_ids = answer_token_ids.get(label, [])
            if token_ids:
                label_scores[label] = float(torch.max(original_scores[token_ids]).item())
        if yes_label in label_scores and no_label in label_scores:
            return max(label_scores.items(), key=lambda item: item[1])[0]
        return None

    def _answer_label_decision_audit(
        self,
        candidates: list[Any],
        *,
        original_scores: torch.Tensor,
        updated_scores: torch.Tensor,
        output: UniversalPluginOutput,
        answer_token_ids: dict[str, list[int]],
        verifier_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        binary_rows: dict[str, dict[str, Any]] = {}
        for idx, candidate in enumerate(candidates):
            label = str(candidate.metadata.get("binary_label") or "")
            token_ids = answer_token_ids.get(label, [])
            if not token_ids:
                continue
            original_score = float(torch.max(original_scores[token_ids]).item())
            adjusted_score = float(torch.max(updated_scores[token_ids]).item())
            binary_rows[label] = {
                "token_ids": token_ids,
                "original_score": original_score,
                "adjusted_score": adjusted_score,
                "bias": round(adjusted_score - original_score, 6),
                "support_sigmoid": float(output.support.sigmoid()[idx].item()),
                "contradiction_sigmoid": float(output.contradiction.sigmoid()[idx].item()),
                "abstain_sigmoid": float(output.abstain.sigmoid()[idx].item()),
            }
        if not binary_rows:
            return dict(verifier_meta or {})
        host_choice = max(binary_rows.items(), key=lambda item: item[1]["original_score"])[0]
        adjusted_choice = max(binary_rows.items(), key=lambda item: item[1]["adjusted_score"])[0]
        audit = {
            "binary_candidates": binary_rows,
            "host_choice": host_choice,
            "adjusted_choice": adjusted_choice,
            "changed_choice": host_choice != adjusted_choice,
        }
        if verifier_meta:
            audit.update(verifier_meta)
        return audit

    def _observation_for_row(self, row_idx: int, device: torch.device) -> UniversalObservation:
        if isinstance(self.base_observation, list):
            observation = self.base_observation[row_idx]
            return _move_observation(observation, device)
        return _move_observation(self.base_observation, device)

    def _candidate_builder_for_row(self, row_idx: int):
        if row_idx == 0:
            return self.candidate_builder
        builder = self._row_candidate_builders.get(row_idx)
        if builder is not None:
            return builder
        try:
            builder = copy.deepcopy(self.candidate_builder)
        except Exception:
            builder = self.candidate_builder.__class__()
        if hasattr(builder, "reset"):
            builder.reset()
        self._row_candidate_builders[row_idx] = builder
        return builder

    def _decode_prefix(self, input_ids: torch.LongTensor) -> str:
        try:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            text = str(input_ids[0].tolist())
        return text[-self.cfg.max_prefix_chars :]

    def _trigger_decide(
        self,
        scores: torch.FloatTensor,
        input_ids: torch.LongTensor,
        runtime_context: dict[str, Any],
    ) -> TriggerDecision:
        try:
            return self.trigger.decide(scores, input_ids, self.tokenizer, context=runtime_context)
        except TypeError:
            return self.trigger.decide(scores, input_ids, self.tokenizer)

    def _build_candidates(
        self,
        candidate_builder: Any,
        top_ids: torch.Tensor,
        prefix_text: str,
        runtime_context: dict[str, Any],
    ) -> list[Any]:
        try:
            return candidate_builder.build(
                self.tokenizer,
                top_ids,
                self.gate,
                prefix_text=prefix_text,
                context=runtime_context,
            )
        except TypeError:
            return candidate_builder.build(self.tokenizer, top_ids, self.gate, prefix_text=prefix_text)

    def _runtime_context(self, observation: UniversalObservation) -> dict[str, Any]:
        metadata = getattr(observation, "metadata", None)
        if not isinstance(metadata, dict):
            return {}
        context = metadata.get("runtime_context")
        return dict(context) if isinstance(context, dict) else {}

    def _audit_context(self, runtime_context: dict[str, Any]) -> dict[str, Any]:
        if not runtime_context:
            return {}
        keys = (
            "task_name",
            "task_family",
            "controller_mode",
            "decision_mode",
            "decision_scope",
            "retrieval_scope",
            "pope_split",
            "object_label",
            "label",
            "answer_mode",
            "answer_labels",
            "yes_hypothesis",
            "no_hypothesis",
        )
        return {key: runtime_context.get(key) for key in keys if runtime_context.get(key) is not None}

    def _prefix_query_text(self, prefix_text: str, runtime_context: dict[str, Any]) -> str:
        scorer_query = runtime_context.get("scorer_query_text") if runtime_context else None
        if scorer_query:
            return str(scorer_query)
        return prefix_text

    def _binary_decision_audit(
        self,
        candidates: list[Any],
        top_ids: torch.Tensor,
        original_scores: torch.Tensor,
        full_bias: torch.Tensor,
        output: UniversalPluginOutput,
        runtime_context: dict[str, Any],
        verifier_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        binary_rows: dict[str, dict[str, float]] = {}
        candidate_by_id = {int(candidate.token_id): candidate for candidate in candidates}
        candidate_index = {int(candidate.token_id): idx for idx, candidate in enumerate(candidates)}
        for idx, token_id in enumerate(top_ids.tolist()):
            candidate = candidate_by_id.get(int(token_id))
            if candidate is None:
                continue
            label = str(candidate.metadata.get("binary_label") or "")
            if label not in {"yes", "no", "a", "b"}:
                continue
            cand_idx = candidate_index.get(int(token_id))
            binary_rows[label] = {
                "token_id": int(token_id),
                "original_score": float(original_scores[idx].item()),
                "adjusted_score": float((original_scores[idx] + full_bias[idx]).item()),
                "bias": float(full_bias[idx].item()),
                "support_sigmoid": float(output.support.sigmoid()[cand_idx].item()) if cand_idx is not None else None,
                "contradiction_sigmoid": float(output.contradiction.sigmoid()[cand_idx].item()) if cand_idx is not None else None,
                "abstain_sigmoid": float(output.abstain.sigmoid()[cand_idx].item()) if cand_idx is not None else None,
            }
        if not binary_rows:
            return dict(verifier_meta or {})
        host_choice = max(binary_rows.items(), key=lambda item: item[1]["original_score"])[0]
        adjusted_choice = max(binary_rows.items(), key=lambda item: item[1]["adjusted_score"])[0]
        audit = {
            "binary_candidates": binary_rows,
            "host_choice": host_choice,
            "adjusted_choice": adjusted_choice,
            "changed_choice": host_choice != adjusted_choice,
        }
        if verifier_meta:
            audit.update(verifier_meta)
        if runtime_context.get("decision_mode") == "answer_labels":
            yes_label, no_label = self._binary_label_pair(runtime_context)
            if yes_label in binary_rows and no_label in binary_rows:
                audit["verifier_margin"] = round(
                    binary_rows[yes_label]["bias"] - binary_rows[no_label]["bias"],
                    6,
                )
        return audit

    def _compose_bias(
        self,
        candidates: list[Any],
        output: UniversalPluginOutput,
        runtime_context: dict[str, Any],
        *,
        binary_host_choice: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        support = output.support.sigmoid()
        contradiction = output.contradiction.sigmoid()
        abstain = output.abstain.sigmoid()
        resonance = (1.0 - abstain) * (support - self.cfg.lambda_con * contradiction)
        base_bias = self.cfg.bias_scale * torch.tanh(resonance / max(self.cfg.score_temperature, 1e-6))
        if runtime_context.get("decision_mode") != "answer_labels":
            return base_bias, {}

        yes_label, no_label = self._binary_label_pair(runtime_context)
        label_to_idx = {
            str(candidate.metadata.get("binary_label") or ""): idx
            for idx, candidate in enumerate(candidates)
            if str(candidate.metadata.get("binary_label") or "") in {yes_label, no_label}
        }
        yes_idx = label_to_idx.get(yes_label)
        no_idx = label_to_idx.get(no_label)
        if yes_idx is None or no_idx is None:
            return torch.zeros_like(base_bias), {
                "bias_source": "answer_labels",
                "evidence_gate_passed": False,
                "evidence_gate_reason": "missing_binary_pair",
            }

        verifier_delta_tensor = resonance[yes_idx] - resonance[no_idx]
        pair_abstain = torch.maximum(abstain[yes_idx], abstain[no_idx])
        evidence_confidence_tensor = (1.0 - pair_abstain) * verifier_delta_tensor.abs()
        min_delta = float(runtime_context.get("pope_min_verifier_delta", 0.0) or 0.0)
        min_confidence = float(runtime_context.get("pope_min_evidence_confidence", 0.0) or 0.0)
        verifier_delta = float(verifier_delta_tensor.item())
        evidence_confidence = float(evidence_confidence_tensor.item())
        gate_reason = "passed"
        if abs(verifier_delta) < min_delta:
            gate_reason = "delta_below_threshold"
        elif evidence_confidence < min_confidence:
            gate_reason = "confidence_below_threshold"
        verifier_meta = {
            "bias_source": "answer_labels",
            "verifier_delta": round(verifier_delta, 6),
            "evidence_confidence": round(evidence_confidence, 6),
            "pope_min_verifier_delta": min_delta,
            "pope_min_evidence_confidence": min_confidence,
            "evidence_gate_passed": gate_reason == "passed",
            "evidence_gate_reason": gate_reason,
        }
        if gate_reason != "passed":
            return torch.zeros_like(base_bias), verifier_meta

        directed = self.cfg.bias_scale * 1.35 * torch.tanh(
            verifier_delta_tensor / max(self.cfg.score_temperature, 1e-6)
        )
        directed = self._apply_verifier_asymmetry(directed, binary_host_choice, runtime_context)
        verifier_bias = torch.zeros_like(base_bias)
        verifier_bias[yes_idx] = directed
        verifier_bias[no_idx] = -directed
        return verifier_bias, verifier_meta

    def _binary_label_pair(self, runtime_context: dict[str, Any]) -> tuple[str, str]:
        labels = runtime_context.get("answer_labels")
        if isinstance(labels, list) and len(labels) >= 2:
            return str(labels[0]), str(labels[1])
        answer_mode = runtime_context.get("answer_mode")
        if answer_mode == "option_ab":
            return "a", "b"
        return "yes", "no"

    def _binary_host_choice(
        self,
        candidates: list[Any],
        top_ids: torch.Tensor,
        original_scores: torch.Tensor,
        runtime_context: dict[str, Any],
    ) -> str | None:
        yes_label, no_label = self._binary_label_pair(runtime_context)
        candidate_by_id = {int(candidate.token_id): candidate for candidate in candidates}
        binary_scores: dict[str, float] = {}
        for idx, token_id in enumerate(top_ids.tolist()):
            candidate = candidate_by_id.get(int(token_id))
            if candidate is None:
                continue
            label = str(candidate.metadata.get("binary_label") or "")
            if label in {yes_label, no_label}:
                binary_scores[label] = float(original_scores[idx].item())
        if yes_label in binary_scores and no_label in binary_scores:
            return max(binary_scores.items(), key=lambda item: item[1])[0]
        return None

    def _apply_verifier_asymmetry(
        self,
        directed: torch.Tensor,
        binary_host_choice: str | None,
        runtime_context: dict[str, Any],
    ) -> torch.Tensor:
        if binary_host_choice is None:
            return directed
        yes_label, no_label = self._binary_label_pair(runtime_context)
        if binary_host_choice == no_label and directed.item() > 0:
            return directed * 1.35
        if binary_host_choice == yes_label and directed.item() < 0:
            return directed * 0.65
        return directed


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
