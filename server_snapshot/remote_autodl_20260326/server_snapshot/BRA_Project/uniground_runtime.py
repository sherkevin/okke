"""
Legacy UniGround v1 runtime.

This module remains in the repository as the reference implementation for the
old universal route and for compatibility with existing v1 checkpoints,
validators, and result manifests. New method development should target the v2
runtime instead of extending this file with additional benchmark-specific
heuristics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import re

import torch
import torch.nn.functional as F
from PIL import Image

from bra_universal_plugin import (
    MLPUniversalScorer,
    PrefixSpanMapper,
    StringStructuralGate,
    UniversalObservation,
    UniversalPluginConfig,
    decode_token,
)


@dataclass
class UniGroundStepStat:
    active_candidates: int
    prefix_ambiguity: bool
    span_collapse_errors: int
    suffix_stability_ok: bool
    abstention_score: float
    aborted: bool
    abort_backoff_verified: bool
    intervened: bool
    candidate_construction_ms: float
    sidecar_scoring_ms: float
    bridge_redistribution_ms: float
    total_step_ms: float


class FrozenExternalEncoder:
    """
    Frozen public encoder bundle for the universal path.

    The model runs on CPU by default to avoid colliding with the host MLLM GPU.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModel

        self.device = torch.device(device)
        self.processor = load_frozen_processor(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batch = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if hasattr(self.model, "get_text_features"):
            feats = self.model.get_text_features(**batch)
        else:
            outputs = self.model(**batch)
            feats = _select_feature_tensor(outputs)
        feats = _select_feature_tensor(feats)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        batch = self.processor(images=image, return_tensors="pt")
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**batch)
        else:
            outputs = self.model(**batch)
            feats = _select_feature_tensor(outputs)
        feats = _select_feature_tensor(feats)
        return F.normalize(feats.float(), dim=-1).squeeze(0)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("encode_images requires at least one image.")
        batch = self.processor(images=images, return_tensors="pt")
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**batch)
        else:
            outputs = self.model(**batch)
            feats = _select_feature_tensor(outputs)
        feats = _select_feature_tensor(feats)
        return F.normalize(feats.float(), dim=-1)


def load_frozen_processor(model_name: str):
    from transformers import AutoProcessor, CLIPProcessor

    try:
        return AutoProcessor.from_pretrained(model_name)
    except Exception:
        return CLIPProcessor.from_pretrained(model_name)


def load_universal_scorer(checkpoint_path: str | Path, device: str = "cpu") -> MLPUniversalScorer:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    embed_dim = int(payload["embed_dim"])
    hidden_dim = int(payload.get("hidden_dim", 512))
    scorer = MLPUniversalScorer(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    scorer.load_state_dict(state_dict, strict=True)
    scorer.eval()
    return scorer


def build_universal_observation(
    encoder: FrozenExternalEncoder,
    image: Image.Image,
    region_embeddings: Optional[torch.Tensor] = None,
) -> UniversalObservation:
    if region_embeddings is None:
        width, height = image.size
        mid_x = max(width // 2, 1)
        mid_y = max(height // 2, 1)
        crops = [
            image.crop((0, 0, mid_x, mid_y)),
            image.crop((mid_x, 0, width, mid_y)),
            image.crop((0, mid_y, mid_x, height)),
            image.crop((mid_x, mid_y, width, height)),
        ]
        region_embeddings = torch.stack([encoder.encode_image(crop) for crop in crops], dim=0)
    return UniversalObservation(
        image_embedding=encoder.encode_image(image),
        region_embeddings=region_embeddings,
        metadata={},
    )


class _AuditMixin:
    def reset(self) -> None:
        self.audit_log: list[dict[str, Any]] = []
        self.step_stats: list[UniGroundStepStat] = []

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self.audit_log)

    def get_summary_stats(self) -> dict[str, float]:
        if not self.step_stats:
            return {
                "intervention_coverage": 0.0,
                "prefix_ambiguity_rate": 0.0,
                "span_collapse_errors": 0.0,
                "suffix_stability_rate": 0.0,
                "abstention_rate": 0.0,
                "abort_trigger_rate": 0.0,
                "abort_backoff_verified_steps": 0,
                "candidate_construction_ms": 0.0,
                "sidecar_scoring_ms": 0.0,
                "bridge_redistribution_ms": 0.0,
                "jitter_ms": 0.0,
                "avg_candidate_window": 0.0,
                "latency_split": {
                    "candidate_construction_ms": 0.0,
                    "sidecar_scoring_ms": 0.0,
                    "bridge_redistribution_ms": 0.0,
                    "jitter_ms": 0.0,
                },
            }
        tensor = lambda vals: torch.tensor(vals, dtype=torch.float32)
        total = len(self.step_stats)
        candidate_ms = round(float(tensor([s.candidate_construction_ms for s in self.step_stats]).mean().item()), 4)
        scoring_ms = round(float(tensor([s.sidecar_scoring_ms for s in self.step_stats]).mean().item()), 4)
        bridge_ms = round(float(tensor([s.bridge_redistribution_ms for s in self.step_stats]).mean().item()), 4)
        jitter_ms = round(float(tensor([s.total_step_ms for s in self.step_stats]).std(unbiased=False).item()), 4)
        return {
            "intervention_coverage": round(sum(1 for s in self.step_stats if s.intervened) / total, 4),
            "prefix_ambiguity_rate": round(sum(1 for s in self.step_stats if s.prefix_ambiguity) / total, 4),
            "span_collapse_errors": round(float(sum(s.span_collapse_errors for s in self.step_stats)), 4),
            "suffix_stability_rate": round(sum(1 for s in self.step_stats if s.suffix_stability_ok) / total, 4),
            "abstention_rate": round(sum(1 for s in self.step_stats if s.abstention_score >= self.cfg.abstain_threshold) / total, 4),
            "abort_trigger_rate": round(sum(1 for s in self.step_stats if s.aborted) / total, 4),
            "abort_backoff_verified_steps": int(sum(1 for s in self.step_stats if s.abort_backoff_verified)),
            "candidate_construction_ms": candidate_ms,
            "sidecar_scoring_ms": scoring_ms,
            "bridge_redistribution_ms": bridge_ms,
            "jitter_ms": jitter_ms,
            "avg_candidate_window": round(float(tensor([s.active_candidates for s in self.step_stats]).mean().item()), 4),
            "latency_split": {
                "candidate_construction_ms": candidate_ms,
                "sidecar_scoring_ms": scoring_ms,
                "bridge_redistribution_ms": bridge_ms,
                "jitter_ms": jitter_ms,
            },
        }


class ExternalGlobalPriorProcessor(_AuditMixin):
    """
    Static global visual prior baseline.

    This reads only frozen image/text embeddings and applies a fixed global bias.
    No candidate-local learned scoring is used.
    """

    def __init__(
        self,
        config: UniversalPluginConfig,
        tokenizer: Any,
        encoder: FrozenExternalEncoder,
        observation: UniversalObservation,
        gate: Optional[StringStructuralGate] = None,
        mapper: Optional[PrefixSpanMapper] = None,
    ) -> None:
        self.cfg = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.observation = observation
        self.gate = gate or StringStructuralGate(
            min_groundable_chars=config.min_groundable_chars,
            allow_numeric_candidates=config.allow_numeric_candidates,
        )
        self.mapper = mapper or PrefixSpanMapper()
        self.reset()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() != 2 or scores.shape[0] != 1:
            return scores

        step_start = time.perf_counter()
        top_k = min(self.cfg.top_k, scores.shape[-1])
        _top_vals, top_ids = scores.topk(top_k, dim=-1)

        prefix_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        t0 = time.perf_counter()
        candidates = self.mapper.build_candidates(self.tokenizer, top_ids[0], self.gate, prefix_text=prefix_text)
        active = [candidate for candidate in candidates if candidate.is_groundable]
        collapse_errors = _count_span_collapses(active)
        ambiguity_rate = collapse_errors / max(len(active), 1)
        ambiguity = ambiguity_rate > 0.0
        candidate_ms = (time.perf_counter() - t0) * 1000.0

        if not active:
            self._record(candidates, None)
            self.step_stats.append(
                UniGroundStepStat(0, ambiguity, collapse_errors, True, 0.0, False, False, False, candidate_ms, 0.0, 0.0, (time.perf_counter() - step_start) * 1000.0)
            )
            return scores

        t1 = time.perf_counter()
        cand_emb = self.encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)
        image_emb = self.observation.image_embedding.to(scores.device).unsqueeze(0).expand(cand_emb.shape[0], -1)
        sim = F.cosine_similarity(image_emb, cand_emb, dim=-1)
        score_ms = (time.perf_counter() - t1) * 1000.0

        bias = self.cfg.bias_scale * sim
        binary_bias = _compute_binary_option_bias(
            encoder=self.encoder,
            observation=self.observation,
            prefix_text=prefix_text,
            candidates=active,
            device=scores.device,
            bias_scale=self.cfg.bias_scale,
            temperature=self.cfg.score_temperature,
        )
        if binary_bias is not None:
            bias = torch.where(binary_bias.abs() > 0, binary_bias, bias)
        full_bias = torch.zeros_like(top_ids[0], dtype=torch.float32)
        active_pos = {candidate.token_id: idx for idx, candidate in enumerate(active)}
        for i, token_id in enumerate(top_ids[0].tolist()):
            idx = active_pos.get(token_id)
            if idx is not None:
                full_bias[i] = bias[idx]

        t2 = time.perf_counter()
        updated = self.mapper.scatter_bias(scores, top_ids[0], full_bias)
        bridge_ms = (time.perf_counter() - t2) * 1000.0
        intervened = bool((full_bias.abs() > 1e-6).any().item())
        self._record(candidates, full_bias)
        self.step_stats.append(
            UniGroundStepStat(
                active_candidates=len(active),
                prefix_ambiguity=ambiguity,
                span_collapse_errors=collapse_errors,
                    suffix_stability_ok=_suffix_stability_ok(candidates, full_bias),
                abstention_score=0.0,
                aborted=False,
                    abort_backoff_verified=False,
                intervened=intervened,
                candidate_construction_ms=candidate_ms,
                sidecar_scoring_ms=score_ms,
                bridge_redistribution_ms=bridge_ms,
                total_step_ms=(time.perf_counter() - step_start) * 1000.0,
            )
        )
        return updated

    def _record(self, candidates: list[Any], bias: Optional[torch.Tensor]) -> None:
        if len(self.audit_log) >= self.cfg.max_audit_steps:
            return
        entry = {
            "candidates": [
                {
                    "token_id": candidate.token_id,
                    "token_text": candidate.token_text,
                    "span_text": candidate.span_text,
                    "normalized_text": candidate.normalized_text,
                    "is_groundable": candidate.is_groundable,
                }
                for candidate in candidates[:10]
            ]
        }
        if bias is not None:
            entry["bias"] = [round(float(x), 6) for x in bias[:10].tolist()]
        debug = getattr(_compute_binary_option_bias, "last_debug", None)
        if isinstance(debug, dict):
            entry["binary_debug"] = debug
        self.audit_log.append(entry)


class UniGroundLogitsProcessor(_AuditMixin):
    def __init__(
        self,
        config: UniversalPluginConfig,
        tokenizer: Any,
        encoder: FrozenExternalEncoder,
        scorer: MLPUniversalScorer,
        observation: UniversalObservation,
        gate: Optional[StringStructuralGate] = None,
        mapper: Optional[PrefixSpanMapper] = None,
        disable_gate: bool = False,
        disable_abstain: bool = False,
        global_only: bool = False,
        region_only: bool = False,
    ) -> None:
        self.cfg = config
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.scorer = scorer
        self.observation = observation
        self.disable_gate = disable_gate
        self.disable_abstain = disable_abstain
        self.global_only = global_only
        self.region_only = region_only
        self.gate = gate or StringStructuralGate(
            min_groundable_chars=config.min_groundable_chars,
            allow_numeric_candidates=config.allow_numeric_candidates,
        )
        self.mapper = mapper or PrefixSpanMapper()
        self.reset()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() != 2 or scores.shape[0] != 1:
            return scores

        step_start = time.perf_counter()
        top_k = min(self.cfg.top_k, scores.shape[-1])
        _top_vals, top_ids = scores.topk(top_k, dim=-1)

        prefix_text = self._decode_prefix(input_ids)
        t0 = time.perf_counter()
        gate = self.gate if not self.disable_gate else _AlwaysGroundableGate()
        candidates = self.mapper.build_candidates(self.tokenizer, top_ids[0], gate, prefix_text=prefix_text)
        active = [candidate for candidate in candidates if candidate.is_groundable]
        collapse_errors = _count_span_collapses(active)
        ambiguity_rate = collapse_errors / max(len(active), 1)
        ambiguity = ambiguity_rate > 0.0
        candidate_ms = (time.perf_counter() - t0) * 1000.0

        if not active:
            self._record("", candidates, None, aborted=True, abort_reason="no_groundable")
            self.step_stats.append(
                UniGroundStepStat(
                    active_candidates=0,
                    prefix_ambiguity=ambiguity,
                    span_collapse_errors=collapse_errors,
                    suffix_stability_ok=True,
                    abstention_score=1.0,
                    aborted=True,
                    abort_backoff_verified=True,
                    intervened=False,
                    candidate_construction_ms=candidate_ms,
                    sidecar_scoring_ms=0.0,
                    bridge_redistribution_ms=0.0,
                    total_step_ms=(time.perf_counter() - step_start) * 1000.0,
                )
            )
            return scores

        t1 = time.perf_counter()
        prefix_embedding = self.encoder.encode_texts([prefix_text]).to(scores.device)
        candidate_embeddings = self.encoder.encode_texts([candidate.span_text for candidate in active]).to(scores.device)
        observation = self._maybe_filter_observation(scores.device)
        if hasattr(self.scorer, "parameters"):
            try:
                scorer_device = next(self.scorer.parameters()).device
            except StopIteration:
                scorer_device = scores.device
            if scorer_device != scores.device:
                self.scorer = self.scorer.to(scores.device)
        output = self.scorer(observation, candidate_embeddings, prefix_embedding)
        abstain = torch.sigmoid(output.abstain)
        if self.disable_abstain:
            abstain = torch.zeros_like(abstain)
        score_ms = (time.perf_counter() - t1) * 1000.0

        max_abstain = float(abstain.max().item()) if abstain.numel() else 0.0
        abort_reason = None
        if self.cfg.abort_on_prefix_ambiguity and ambiguity_rate >= self.cfg.ambiguity_abort_threshold:
            abort_reason = "prefix_ambiguity"
        elif max_abstain >= self.cfg.abstain_threshold:
            abort_reason = "abstain"

        if abort_reason is not None:
            self._record(prefix_text, candidates, None, aborted=True, abort_reason=abort_reason)
            self.step_stats.append(
                UniGroundStepStat(
                    active_candidates=len(active),
                    prefix_ambiguity=ambiguity,
                    span_collapse_errors=collapse_errors,
                    suffix_stability_ok=True,
                    abstention_score=max_abstain,
                    aborted=True,
                    abort_backoff_verified=True,
                    intervened=False,
                    candidate_construction_ms=candidate_ms,
                    sidecar_scoring_ms=score_ms,
                    bridge_redistribution_ms=0.0,
                    total_step_ms=(time.perf_counter() - step_start) * 1000.0,
                )
            )
            return scores

        scaled_delta = (output.support - output.contradiction) / max(self.cfg.score_temperature, 1e-6)
        full_bias = torch.zeros_like(top_ids[0], dtype=torch.float32)
        token_rank = {token_id: idx for idx, token_id in enumerate(top_ids[0].tolist())}
        active_pos = {candidate.token_id: idx for idx, candidate in enumerate(active)}
        active_rank_indices = [token_rank[candidate.token_id] for candidate in active]
        host_active_scores = scores[0, top_ids[0][active_rank_indices]].float()

        if self.cfg.bias_mode == "centered":
            centered_delta = scaled_delta - scaled_delta.mean()
            bias = self.cfg.bias_scale * torch.tanh(centered_delta) * (1.0 - abstain)
        elif self.cfg.bias_mode == "host_relative":
            centered_delta = scaled_delta - scaled_delta.mean()
            controller_prior = F.softmax(centered_delta, dim=0)
            host_prior = F.softmax(host_active_scores, dim=0)
            bias = (
                self.cfg.bias_scale
                * self.cfg.host_relative_strength
                * (controller_prior - host_prior)
                * (1.0 - abstain)
            )
        else:
            signed = torch.tanh(scaled_delta)
            bias = self.cfg.bias_scale * signed * (1.0 - abstain)

        binary_bias = _compute_binary_option_bias(
            encoder=self.encoder,
            observation=self.observation,
            prefix_text=prefix_text,
            candidates=active,
            device=scores.device,
            bias_scale=self.cfg.bias_scale * max(self.cfg.host_relative_strength, 1.0),
            temperature=self.cfg.score_temperature,
        )
        if binary_bias is not None:
            bias = torch.where(binary_bias.abs() > 0, binary_bias, bias)

        for i, token_id in enumerate(top_ids[0].tolist()):
            idx = active_pos.get(token_id)
            if idx is not None:
                full_bias[i] = bias[idx]

        t2 = time.perf_counter()
        updated = self.mapper.scatter_bias(scores, top_ids[0], full_bias)
        bridge_ms = (time.perf_counter() - t2) * 1000.0
        intervened = bool((full_bias.abs() > 1e-6).any().item())
        self._record(prefix_text, candidates, full_bias, aborted=False, abort_reason=None)
        self.step_stats.append(
            UniGroundStepStat(
                active_candidates=len(active),
                prefix_ambiguity=ambiguity,
                span_collapse_errors=collapse_errors,
                suffix_stability_ok=_suffix_stability_ok(candidates, full_bias),
                abstention_score=max_abstain,
                aborted=False,
                abort_backoff_verified=False,
                intervened=intervened,
                candidate_construction_ms=candidate_ms,
                sidecar_scoring_ms=score_ms,
                bridge_redistribution_ms=bridge_ms,
                total_step_ms=(time.perf_counter() - step_start) * 1000.0,
            )
        )
        return updated

    def _decode_prefix(self, input_ids: torch.LongTensor) -> str:
        try:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            text = str(input_ids[0].tolist())
        return text[-self.cfg.max_prefix_chars :]

    def _maybe_filter_observation(self, device: torch.device) -> UniversalObservation:
        image_embedding = self.observation.image_embedding.to(device)
        region_embeddings = self.observation.region_embeddings
        if self.global_only:
            region_embeddings = None
        if self.region_only and region_embeddings is not None:
            image_embedding = region_embeddings.mean(dim=0).to(device)
        elif self.region_only:
            image_embedding = image_embedding
        if region_embeddings is not None:
            region_embeddings = region_embeddings.to(device)
        return UniversalObservation(
            image_embedding=image_embedding,
            region_embeddings=region_embeddings,
            metadata=dict(self.observation.metadata),
        )

    def _record(
        self,
        prefix_text: str,
        candidates: list[Any],
        bias: Optional[torch.Tensor],
        aborted: bool,
        abort_reason: Optional[str],
    ) -> None:
        if len(self.audit_log) >= self.cfg.max_audit_steps:
            return
        entry = {
            "prefix_tail": prefix_text,
            "aborted": aborted,
            "abort_reason": abort_reason,
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
        }
        if bias is not None:
            entry["bias"] = [round(float(x), 6) for x in bias[:10].tolist()]
        debug = getattr(_compute_binary_option_bias, "last_debug", None)
        if isinstance(debug, dict):
            entry["binary_debug"] = debug
        self.audit_log.append(entry)


class _AlwaysGroundableGate:
    def is_groundable(self, text: str) -> bool:
        return bool(text.strip())


def _count_span_collapses(candidates: list[Any]) -> int:
    seen: dict[str, int] = {}
    alias_norms: set[str] = set()
    collapse_errors = 0
    for candidate in candidates:
        norm = candidate.normalized_text
        if not norm:
            continue
        if candidate.metadata.get("semantic_alias") is not None:
            alias_norms.add(norm)
            seen.setdefault(norm, 1)
            continue
        seen[norm] = seen.get(norm, 0) + 1
    for norm, count in seen.items():
        if norm in alias_norms:
            continue
        if count > 1:
            collapse_errors += count - 1
    return collapse_errors


def _extract_pope_object_query(prefix_text: str) -> Optional[str]:
    flattened = re.sub(r"\s+", " ", prefix_text.strip().lower())
    match = re.search(r"is there (?:a |an )?(.+?) in the image\?", flattened)
    if not match:
        return None
    phrase = match.group(1).strip(" .,:;!?")
    return phrase or None


def _singularize_token(token: str) -> str:
    irregular = {
        "skis": "ski",
        "men": "man",
        "women": "woman",
        "people": "person",
        "children": "child",
        "teeth": "tooth",
        "feet": "foot",
        "geese": "goose",
        "mice": "mouse",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _query_aliases(query: str) -> list[str]:
    query = query.strip()
    aliases = {query}
    if " or " in query:
        for part in query.split(" or "):
            part = part.strip()
            if part:
                aliases.add(part)
    if query.startswith("a "):
        aliases.add(query[2:].strip())
    if query.startswith("an "):
        aliases.add(query[3:].strip())
    tokens = query.split()
    if tokens:
        singular_last = _singularize_token(tokens[-1])
        if singular_last != tokens[-1]:
            aliases.add(" ".join(tokens[:-1] + [singular_last]).strip())
        if tokens[-1] == "skis":
            aliases.add("pair of skis")
            aliases.add("ski")
    return [alias for alias in aliases if alias]


def _compute_binary_option_bias(
    encoder: FrozenExternalEncoder,
    observation: UniversalObservation,
    prefix_text: str,
    candidates: list[Any],
    device: torch.device,
    bias_scale: float,
    temperature: float,
) -> Optional[torch.Tensor]:
    _compute_binary_option_bias.last_debug = None
    query = _extract_pope_object_query(prefix_text)
    if not query:
        return None
    yes_idx = [i for i, c in enumerate(candidates) if c.normalized_text == "yes"]
    no_idx = [i for i, c in enumerate(candidates) if c.normalized_text == "no"]
    if not yes_idx and not no_idx:
        return None

    aliases = _query_aliases(query)
    yes_prompts: list[str] = []
    no_prompts: list[str] = []
    for alias in aliases:
        yes_prompts.extend(
            [
                f"a photo containing {alias}",
                f"an image with {alias}",
                alias,
            ]
        )
        no_prompts.extend(
            [
                f"a photo without {alias}",
                f"an image without {alias}",
                f"no {alias}",
            ]
        )
    prompt_embeddings = encoder.encode_texts(yes_prompts + no_prompts).to(device)
    yes_emb = prompt_embeddings[: len(yes_prompts)].mean(dim=0, keepdim=True)
    no_emb = prompt_embeddings[len(yes_prompts) :].mean(dim=0, keepdim=True)
    yes_emb = F.normalize(yes_emb, dim=-1)
    no_emb = F.normalize(no_emb, dim=-1)
    image_bank = [observation.image_embedding.to(device).unsqueeze(0)]
    if observation.region_embeddings is not None:
        image_bank.append(observation.region_embeddings.to(device))
    image_emb = torch.cat(image_bank, dim=0)
    yes_sim = F.cosine_similarity(image_emb, yes_emb.expand(image_emb.shape[0], -1), dim=-1).max()
    no_sim = F.cosine_similarity(image_emb, no_emb.expand(image_emb.shape[0], -1), dim=-1).max()
    delta = torch.tanh((yes_sim - no_sim) / max(temperature, 1e-6))
    _compute_binary_option_bias.last_debug = {
        "query": query,
        "aliases": aliases,
        "yes_sim": round(float(yes_sim.item()), 6),
        "no_sim": round(float(no_sim.item()), 6),
        "delta": round(float(delta.item()), 6),
    }

    bias = torch.zeros(len(candidates), dtype=torch.float32, device=device)
    if yes_idx:
        bias[yes_idx] = bias_scale * delta
    if no_idx:
        bias[no_idx] = -bias_scale * delta
    return bias


def _select_feature_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    pooler_output = getattr(outputs, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    image_embeds = getattr(outputs, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds
    text_embeds = getattr(outputs, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]
    raise TypeError(f"Unsupported encoder output type: {type(outputs)!r}")


def _suffix_stability_ok(candidates: list[Any], bias: torch.Tensor) -> bool:
    continuation_positions = []
    for idx, candidate in enumerate(candidates[: len(bias)]):
        raw_token = candidate.metadata.get("raw_token_text", "")
        if _is_continuation_fragment(raw_token):
            continuation_positions.append(idx)
    if not continuation_positions:
        return True
    for idx in continuation_positions:
        if abs(float(bias[idx].item())) > 1e-6:
            return False
    return True


def _is_continuation_fragment(token_text: str) -> bool:
    stripped = token_text.strip()
    if not stripped:
        return False
    if token_text.startswith("##"):
        return True
    if token_text.startswith(("Ġ", "▁")):
        return False
    return token_text[:1].isalnum() and stripped[:1].isalnum()
