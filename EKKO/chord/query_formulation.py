from __future__ import annotations

import re

_ANCHOR_QUERY_PATTERNS = (
    re.compile(r"^\s*is there\s+(?:a|an|any|the)?\s*(?P<object>.+?)\s+in the image\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*are there\s+(?:any|the)?\s*(?P<object>.+?)\s+in the image\??\s*$", re.IGNORECASE),
)

_STOPWORDS = {
    "a",
    "an",
    "any",
    "are",
    "do",
    "does",
    "image",
    "in",
    "is",
    "of",
    "on",
    "the",
    "there",
}

_PHRASE_CANONICALS = {
    "aeroplane": "airplane",
    "air plane": "airplane",
    "aircraft": "airplane",
    "cell phone": "phone",
    "cellphone": "phone",
    "couch": "sofa",
    "mobile phone": "phone",
    "motorbike": "motorcycle",
    "tv": "television",
    "tv monitor": "television",
}

_IRREGULAR_SINGULARS = {
    "skis": "ski",
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def singularize_token(token: str) -> str:
    if token in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and not token.endswith("ss"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_object_phrase(text: str, *, drop_stopwords: bool = True) -> str:
    lowered = _normalize_spaces(text.lower())
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = _normalize_spaces(lowered)
    lowered = _PHRASE_CANONICALS.get(lowered, lowered)
    tokens = []
    for token in re.findall(r"[a-z0-9]+", lowered):
        if drop_stopwords and token in _STOPWORDS:
            continue
        tokens.append(singularize_token(token))
    return " ".join(tokens)


def normalize_object_terms(text: str) -> tuple[str, ...]:
    normalized = normalize_object_phrase(text)
    if not normalized:
        return ()
    return tuple(token for token in normalized.split(" ") if token)


def extract_anchor_query(query: str, *, mode: str = "object_phrase") -> str:
    raw_query = _normalize_spaces(query)
    if mode == "raw":
        return raw_query
    if mode != "object_phrase":
        raise ValueError(f"unsupported anchor query mode: {mode}")

    for pattern in _ANCHOR_QUERY_PATTERNS:
        match = pattern.match(raw_query)
        if match:
            return _normalize_spaces(match.group("object"))

    stripped = raw_query.rstrip("?.! ")
    return _normalize_spaces(stripped)


def build_model_query(query: str, *, suffix: str = "") -> str:
    base_query = _normalize_spaces(query)
    clean_suffix = _normalize_spaces(suffix)
    if not clean_suffix:
        return base_query
    return f"{base_query} {clean_suffix}".strip()
