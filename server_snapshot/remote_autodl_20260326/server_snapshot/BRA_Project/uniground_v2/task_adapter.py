from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PopeTaskAdapter:
    prompt_suffix: str = "Answer with exactly one word: yes or no."
    OBJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"is there\s+(?P<object>.+?)\s+in the image", re.IGNORECASE),
        re.compile(r"are there\s+(?P<object>.+?)\s+in the image", re.IGNORECASE),
        re.compile(r"does (?:the|this) image contain\s+(?P<object>.+?)(?:\?|\.|$)", re.IGNORECASE),
    )

    def format_question(self, raw_question: str) -> str:
        base = raw_question.strip()
        if base.endswith((".", "?", "!", ":")):
            base = base[:-1].rstrip()
        return f"{base}?\n{self.prompt_suffix}"

    def extract_object_label(self, raw_question: str) -> str:
        for pattern in self.OBJECT_PATTERNS:
            match = pattern.search(raw_question.strip())
            if not match:
                continue
            label = match.group("object").strip().lower()
            label = re.sub(r"^(a|an|the)\s+", "", label)
            label = re.sub(r"[?.!,;:]+$", "", label).strip()
            if label:
                return label
        return ""

    def build_runtime_context(
        self,
        raw_question: str,
        *,
        split: str,
        controller_mode: str = "verifier",
        prompt_token_count: int | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        object_label = self.extract_object_label(raw_question)
        query_text = f"a photo of {object_label}" if object_label else raw_question.strip()
        yes_hypothesis = f"a photo containing {object_label}" if object_label else "yes"
        no_hypothesis = f"a photo without {object_label}" if object_label else "no"
        decision_mode = "answer_labels" if controller_mode == "verifier" else "frontier_candidates"
        decision_scope = "answer_step_only" if controller_mode == "verifier" else "frontier_steps"
        retrieval_scope = "task_query" if controller_mode == "verifier" else "candidate_span"
        return {
            "task_name": "pope",
            "task_family": "binary_verification",
            "controller_mode": controller_mode,
            "dataset": "pope",
            "pope_split": split,
            "raw_question": raw_question.strip(),
            "object_label": object_label,
            "task_query_text": query_text,
            "retrieval_query_text": query_text,
            "scorer_query_text": query_text,
            "prompt_token_count": int(prompt_token_count or 0),
            "label": label.strip().lower() if isinstance(label, str) else None,
            "answer_mode": "yes_no",
            "answer_labels": ["yes", "no"],
            "answer_choice_texts": {
                "yes": ["yes"],
                "no": ["no"],
            },
            "hypothesis_text_by_label": {
                "yes": yes_hypothesis,
                "no": no_hypothesis,
            },
            "yes_hypothesis": yes_hypothesis,
            "no_hypothesis": no_hypothesis,
            "decision_mode": decision_mode,
            "decision_scope": decision_scope,
            "retrieval_scope": retrieval_scope,
        }

    def parse_prediction(self, text: str) -> str:
        lowered = text.strip().lower()
        lowered = re.sub(r"^[^a-z0-9]+", "", lowered)
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
        match = re.search(r"\b(yes|no)\b", lowered)
        return match.group(1) if match else "unknown"


@dataclass
class ChairTaskAdapter:
    prompt: str = "Please describe this image in detail."

    def format_question(self) -> str:
        return self.prompt

    def build_runtime_context(
        self,
        *,
        prompt_token_count: int | None = None,
    ) -> dict[str, Any]:
        return {
            "task_name": "chair",
            "task_family": "caption_generation",
            "dataset": "chair",
            "task_query_text": self.prompt,
            "retrieval_query_text": self.prompt,
            "scorer_query_text": self.prompt,
            "prompt_token_count": int(prompt_token_count or 0),
            "decision_mode": "frontier_candidates",
            "decision_scope": "frontier_steps",
            "retrieval_scope": "candidate_span",
        }
