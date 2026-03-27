from __future__ import annotations

import re
from dataclasses import dataclass


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
