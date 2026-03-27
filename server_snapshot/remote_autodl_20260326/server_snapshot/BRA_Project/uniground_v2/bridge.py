from __future__ import annotations

import re
from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Optional

from bra_universal_plugin import (
    StringStructuralGate,
    UniversalCandidate,
    canonicalize_token_text,
    decode_token,
    normalize_span,
)


@dataclass
class BridgeResolution:
    status: str
    span_text: str
    normalized_text: str
    metadata: dict[str, Any]


@dataclass
class SubwordBridgeState:
    pending_text: str = ""

    def reset(self) -> None:
        self.pending_text = ""


class BinaryChoiceTaskAdapter:
    OBJECT_QUERY_PATTERNS = (
        re.compile(r"is there\s+(?P<object>.+?)\s+in the image", re.IGNORECASE),
        re.compile(r"are there\s+(?P<object>.+?)\s+in the image", re.IGNORECASE),
        re.compile(r"does (?:the|this) image contain\s+(?P<object>.+?)(?:\?|\.|\n|$)", re.IGNORECASE),
    )

    def extract_semantics(self, prefix_text: str, context: Optional[dict[str, Any]] = None) -> dict[str, str]:
        lowered = prefix_text.lower()
        object_query = self.object_query(prefix_text, context=context)
        explicit_hypotheses = context.get("hypothesis_text_by_label", {}) if context else {}
        yes_semantic = explicit_hypotheses.get("yes") or (
            self._present_hypothesis(object_query) if object_query else "yes"
        )
        no_semantic = explicit_hypotheses.get("no") or (
            self._absent_hypothesis(object_query) if object_query else "no"
        )

        semantics: dict[str, str] = {}
        if self._is_yes_no_mode(lowered, context=context):
            semantics["yes"] = yes_semantic
            semantics["no"] = no_semantic

        if self._is_option_mode(lowered, context=context):
            if "a. yes" in lowered and "b. no" in lowered:
                semantics.update({"a": yes_semantic, "b": no_semantic})
            elif "a. no" in lowered and "b. yes" in lowered:
                semantics.update({"a": no_semantic, "b": yes_semantic})
            elif context and context.get("decision_mode") == "answer_labels":
                semantics.update({"a": yes_semantic, "b": no_semantic})

        return semantics

    def extract_object_query(self, prefix_text: str) -> str:
        for pattern in self.OBJECT_QUERY_PATTERNS:
            match = pattern.search(prefix_text)
            if not match:
                continue
            phrase = normalize_span(match.group("object"))
            if phrase:
                return re.sub(r"^(a|an|the)\s+", "", phrase)
        return ""

    def object_query(self, prefix_text: str, context: Optional[dict[str, Any]] = None) -> str:
        if context:
            explicit = normalize_span(str(context.get("object_label") or ""))
            if explicit:
                return explicit
        return self.extract_object_query(prefix_text)

    def normalize_binary_label(self, token_text: str) -> str:
        raw = token_text.strip()
        raw = raw.replace("Ġ", "").replace("▁", "")
        raw = raw.replace("ï¼¡", "A").replace("ï¼¢", "B")
        normalized = normalize_span(raw)
        if normalized in {"yes", "no", "a", "b"}:
            return normalized
        return ""

    def option_key(self, token_text: str) -> str:
        raw = token_text.strip()
        raw = raw.replace("Ġ", "").replace("▁", "")
        raw = raw.replace("ï¼¡", "A").replace("ï¼¢", "B")
        match = re.match(r"^[\W_]*([AB])[\W_]*$", raw)
        if not match:
            return ""
        return match.group(1).lower()

    def token_key(self, token_text: str) -> str:
        normalized = self.normalize_binary_label(token_text)
        return normalized if normalized in {"yes", "no"} else ""

    def _is_yes_no_mode(self, lowered_prefix: str, context: Optional[dict[str, Any]] = None) -> bool:
        if context and context.get("answer_mode") == "yes_no":
            return True
        return "answer with exactly one word: yes or no." in lowered_prefix

    def _is_option_mode(self, lowered_prefix: str, context: Optional[dict[str, Any]] = None) -> bool:
        if context and context.get("answer_mode") == "option_ab":
            return True
        return "answer with the option letter only" in lowered_prefix

    def _present_hypothesis(self, object_query: str) -> str:
        return f"a photo containing {object_query}"

    def _absent_hypothesis(self, object_query: str) -> str:
        return f"a photo without {object_query}"


class MorphBoundaryResolver:
    def resolve(self, state: SubwordBridgeState, token_text: str) -> BridgeResolution:
        raw = token_text.strip()
        if not raw:
            state.reset()
            return BridgeResolution("skip", "", "", {"reason": "empty"})

        canonical = canonicalize_token_text(token_text)
        normalized = normalize_span(canonical)
        if not normalized or all(not ch.isalnum() for ch in normalized):
            state.reset()
            return BridgeResolution("skip", canonical, normalized, {"reason": "symbol"})

        if token_text.startswith("##"):
            continuation = token_text[2:].strip()
            joined = f"{state.pending_text}{continuation}"
            state.reset()
            return BridgeResolution("complete", joined, normalize_span(joined), {"continued": True})

        if token_text.startswith(("Ġ", "▁")) or token_text[:1].isspace():
            state.reset()
            return BridgeResolution("complete", canonical, normalized, {"word_start": True})

        if len(normalized) <= 2 and normalized.isalpha():
            state.pending_text = normalized
            return BridgeResolution("pending", normalized, normalized, {"reason": "short_fragment"})

        if state.pending_text:
            joined = f"{state.pending_text}{normalized}"
            state.reset()
            return BridgeResolution("complete", joined, normalize_span(joined), {"continued": True})

        return BridgeResolution("complete", canonical, normalized, {})


class CandidateFrontierBuilder:
    def __init__(
        self,
        *,
        resolver: Optional[MorphBoundaryResolver] = None,
        task_adapter: Optional[BinaryChoiceTaskAdapter] = None,
    ) -> None:
        self.resolver = resolver or MorphBoundaryResolver()
        self.task_adapter = task_adapter or BinaryChoiceTaskAdapter()
        self.prefix_state = SubwordBridgeState()
        self._prefix_token_count = 0

    def reset(self) -> None:
        self.prefix_state.reset()
        self._prefix_token_count = 0

    def sync_prefix_state(self, tokenizer: Any, input_ids) -> None:
        if hasattr(input_ids, "dim"):
            if input_ids.dim() == 2:
                token_ids = input_ids[0].tolist()
            else:
                token_ids = input_ids.tolist()
        else:
            token_ids = list(input_ids)
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]

        if len(token_ids) < self._prefix_token_count:
            self.reset()

        new_token_ids = token_ids[self._prefix_token_count :]
        for token_id in new_token_ids:
            token_text = self._decode_candidate(tokenizer, int(token_id))
            self.resolver.resolve(self.prefix_state, token_text)
        self._prefix_token_count = len(token_ids)

    def build(
        self,
        tokenizer: Any,
        top_ids,
        gate: StringStructuralGate,
        prefix_text: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[UniversalCandidate]:
        option_semantics = self.task_adapter.extract_semantics(prefix_text, context=context)
        answer_label_only = bool(context and context.get("decision_mode") == "answer_labels")
        candidates: list[UniversalCandidate] = []
        for token_id in top_ids.tolist():
            token_text = self._decode_candidate(tokenizer, token_id)
            candidate_state = replace(self.prefix_state)
            resolution = self.resolver.resolve(candidate_state, token_text)
            option_key = self.task_adapter.option_key(token_text)
            binary_label = option_key or self.task_adapter.token_key(token_text)
            semantic_span = option_semantics.get(option_key) or option_semantics.get(binary_label)
            span_text = semantic_span or resolution.span_text
            normalized = semantic_span or resolution.normalized_text
            is_pending = resolution.status == "pending" and semantic_span is None
            is_skip = resolution.status == "skip" and semantic_span is None
            is_groundable = (
                not is_pending
                and not is_skip
                and gate.is_groundable(span_text)
                and (not answer_label_only or semantic_span is not None)
            )
            candidates.append(
                UniversalCandidate(
                    token_id=token_id,
                    token_text=token_text,
                    span_text=span_text,
                    normalized_text=normalized,
                    is_groundable=is_groundable,
                    metadata={
                        **resolution.metadata,
                        "status": resolution.status,
                        "semantic_alias": semantic_span,
                        "option_key": option_key,
                        "binary_label": binary_label,
                        "task_name": context.get("task_name") if context else None,
                        "decision_mode": context.get("decision_mode") if context else None,
                    },
                )
            )
        return candidates

    def _decode_candidate(self, tokenizer: Any, token_id: int) -> str:
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return decode_token(tokenizer, token_id)
        try:
            return tokenizer.decode(int(token_id), skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated
