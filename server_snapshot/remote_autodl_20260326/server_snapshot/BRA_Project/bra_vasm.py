from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "of", "to", "and", "or", "but", "for", "nor", "so", "yet",
    "at", "by", "from", "with", "without", "into", "onto", "over", "under",
    "after", "before", "during", "through", "than", "that", "this", "these",
    "those", "it", "its", "their", "his", "her", "our", "my", "your",
}

LOGIC_WORDS = {
    "therefore", "thus", "however", "hence", "conversely", "because",
    "although", "unless", "if", "then", "else", "while", "whereas",
    "equals", "equal", "integrate", "derive", "prove", "assume", "suppose",
}

_TABLE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def detect_tokenizer_family(tokenizer: Any) -> str:
    cls_name = type(tokenizer).__name__.lower()
    name_or_path = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in cls_name or "qwen" in name_or_path or "tiktoken" in cls_name:
        return "qwen"
    if "llama" in cls_name or "sentencepiece" in cls_name or "vicuna" in name_or_path:
        return "llama"
    return "generic"


def token_has_leading_boundary(token: str) -> bool:
    return token.startswith("\u2581") or token.startswith("\u0120")


def normalize_token(token: str) -> str:
    return token.lower().lstrip("\u2581\u0120").strip()


def is_continuation_subword(token: str, family: str) -> bool:
    if not token:
        return False
    if token_has_leading_boundary(token):
        return False
    stripped = token.strip()
    if not stripped:
        return False
    alpha_chars = sum(ch.isalpha() for ch in stripped)
    if family == "qwen":
        return alpha_chars > 0 and not token_has_leading_boundary(token)
    if family == "llama":
        return alpha_chars > 0 and not token_has_leading_boundary(token)
    return alpha_chars > 0 and not token_has_leading_boundary(token)


def infer_gamma(token: str, family: str) -> float:
    clean = normalize_token(token)
    if not clean:
        return 0.0
    if clean in FUNCTION_WORDS or clean in LOGIC_WORDS:
        return 0.0
    if all(ch in string.punctuation for ch in clean):
        return 0.0
    if clean.isdigit():
        return 1.0
    if all(ch.isdigit() or ch in ".,:-/%" for ch in clean):
        return 1.0
    if any(ch.isalpha() for ch in clean):
        return 1.0
    return 0.5


def build_vasm_table(tokenizer: Any) -> dict[str, Any]:
    family = detect_tokenizer_family(tokenizer)
    vocab_size = len(tokenizer)
    cache_key = (family, vocab_size)
    if cache_key in _TABLE_CACHE:
        return _TABLE_CACHE[cache_key]

    tokens = []
    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        tokens.append(token or "")

    root_gamma_by_norm: dict[str, float] = {}
    for token in tokens:
        clean = normalize_token(token)
        if not clean:
            continue
        if token_has_leading_boundary(token) or not is_continuation_subword(token, family):
            root_gamma_by_norm.setdefault(clean, infer_gamma(token, family))

    gamma = []
    for token in tokens:
        clean = normalize_token(token)
        if clean and is_continuation_subword(token, family):
            gamma.append(root_gamma_by_norm.get(clean, infer_gamma(token, family)))
        else:
            gamma.append(infer_gamma(token, family))

    table = {
        "meta": {
            "family": family,
            "vocab_size": vocab_size,
            "source": "heuristic_root_inherit_vasm",
        },
        "gamma": gamma,
    }
    _TABLE_CACHE[cache_key] = table
    return table


def save_vasm_table(tokenizer: Any, output_path: str | Path) -> Path:
    table = build_vasm_table(tokenizer)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


@dataclass
class ProbabilisticVASM:
    gamma_by_id: list[float]
    family: str
    source: str

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "ProbabilisticVASM":
        table = build_vasm_table(tokenizer)
        meta = table["meta"]
        return cls(
            gamma_by_id=list(table["gamma"]),
            family=meta["family"],
            source=meta["source"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ProbabilisticVASM":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        return cls(
            gamma_by_id=list(data["gamma"]),
            family=meta.get("family", "unknown"),
            source=meta.get("source", "file"),
        )

    def lookup(self, token_ids: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        flat = token_ids.view(-1).tolist()
        vals = [self.gamma_by_id[idx] if 0 <= idx < len(self.gamma_by_id) else 0.0 for idx in flat]
        out = torch.tensor(vals, dtype=torch.float32, device=device or token_ids.device)
        return out.view_as(token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "source": self.source,
            "vocab_size": len(self.gamma_by_id),
        }
