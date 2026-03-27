from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bra_universal_plugin import decode_token, normalize_span


@dataclass
class TriggerDecision:
    fire: bool
    reason: str
    entropy: float = 0.0
    margin: float = 0.0
    content_ratio: float = 0.0


class EntropyMarginTrigger:
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
        "with",
    }
    BINARY_OBJECT_PREFIX_MARKERS = (
        "answer with exactly one word: yes or no.",
        "answer with the option letter only",
    )
    BINARY_OBJECT_QUERY_MARKERS = (
        "is there ",
        "are there ",
        "does the image contain ",
        "does this image contain ",
    )
    BINARY_LABELS = {"yes", "no", "a", "b"}

    def __init__(
        self,
        *,
        top_k: int = 10,
        entropy_threshold: float = 1.5,
        margin_threshold: float = 0.75,
        min_content_ratio: float = 0.25,
    ) -> None:
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.min_content_ratio = min_content_ratio

    def decide(
        self,
        scores: torch.FloatTensor,
        input_ids: torch.LongTensor,
        tokenizer: Any,
    ) -> TriggerDecision:
        logits = scores[0].float()
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        prefix_text = self._decode_prefix(tokenizer, input_ids).lower()

        top_vals, top_ids = torch.topk(logits, k=min(self.top_k, logits.shape[-1]))
        margin = float((top_vals[0] - top_vals[1]).item()) if top_vals.numel() > 1 else float("inf")
        content_ratio = self._content_candidate_ratio(tokenizer, top_ids)

        if self._is_binary_object_query(prefix_text) and self._contains_binary_label(tokenizer, top_ids):
            return TriggerDecision(True, "binary_object_query", entropy, margin, content_ratio)
        if content_ratio < self.min_content_ratio:
            return TriggerDecision(False, "content_low", entropy, margin, content_ratio)
        if entropy < self.entropy_threshold:
            return TriggerDecision(False, "entropy_low", entropy, margin, content_ratio)
        if margin > self.margin_threshold:
            return TriggerDecision(False, "margin_high", entropy, margin, content_ratio)
        return TriggerDecision(True, "entropy_margin_fire", entropy, margin, content_ratio)

    def _content_candidate_ratio(self, tokenizer: Any, top_ids: torch.Tensor) -> float:
        if top_ids.numel() == 0:
            return 0.0
        content_count = 0
        for token_id in top_ids.tolist():
            token_text = self._decode_candidate(tokenizer, token_id)
            normalized = normalize_span(token_text)
            if not normalized:
                continue
            if normalized in self.STOPWORDS:
                continue
            if not any(ch.isalnum() for ch in normalized):
                continue
            if len(normalized) <= 1:
                continue
            content_count += 1
        return content_count / max(int(top_ids.numel()), 1)

    def _decode_candidate(self, tokenizer: Any, token_id: int) -> str:
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return decode_token(tokenizer, token_id)
        try:
            return tokenizer.decode(int(token_id), skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def _decode_prefix(self, tokenizer: Any, input_ids: torch.LongTensor) -> str:
        try:
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            return ""

    def _is_binary_object_query(self, prefix_text: str) -> bool:
        if not prefix_text:
            return False
        has_prompt = any(marker in prefix_text for marker in self.BINARY_OBJECT_PREFIX_MARKERS)
        has_query = any(marker in prefix_text for marker in self.BINARY_OBJECT_QUERY_MARKERS)
        return has_prompt and has_query

    def _contains_binary_label(self, tokenizer: Any, top_ids: torch.Tensor) -> bool:
        for token_id in top_ids.tolist():
            token_text = self._decode_candidate(tokenizer, token_id)
            normalized = normalize_span(token_text)
            if normalized in self.BINARY_LABELS:
                return True
        return False
