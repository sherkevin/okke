#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def normalize_entity(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def singularize_entity(text: str) -> str:
    if len(text) > 3 and text.endswith("ies"):
        return text[:-3] + "y"
    if len(text) > 3 and text.endswith("es"):
        return text[:-2]
    if len(text) > 2 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def canonicalize_entity(text: Any) -> str:
    return singularize_entity(normalize_entity(text))


def flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return flatten_strings(value.tolist())
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for nested in value.values():
            out.extend(flatten_strings(nested))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for nested in value:
            out.extend(flatten_strings(nested))
        return out
    return [str(value)]


def load_catalog_entities(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    raw_values: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("entries", payload.get("items", payload))
        raw_values.extend(flatten_strings(payload))
    elif suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw_values.extend(flatten_strings(json.loads(line)))
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_values.extend(flatten_strings(row))
    elif suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        raw_values.extend(flatten_strings(payload))
    else:
        raw_values.extend(flatten_strings(path.read_text(encoding="utf-8")))

    entities = {canonicalize_entity(v) for v in raw_values if canonicalize_entity(v)}
    return entities


def load_benchmark_entities(path: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark entity file not found: {path}")

    suffix = path.suffix.lower()
    category_to_entities: dict[str, set[str]] = defaultdict(set)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("entries", payload) if isinstance(payload, dict) else payload
        if isinstance(records, dict):
            records = [{"category": key, "entity": value} for key, value in records.items()]
        for row in records:
            if not isinstance(row, dict):
                continue
            category = str(row.get("category") or row.get("item") or row.get("label") or "").strip()
            entity = row.get("entity", row.get("answer_text", row.get("value", category)))
            if category:
                category_to_entities[category].add(canonicalize_entity(entity))
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = str(row.get("category") or row.get("item") or row.get("label") or "").strip()
                entity = row.get("entity", row.get("answer_text", row.get("value", category)))
                if category:
                    category_to_entities[category].add(canonicalize_entity(entity))
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            category = str(row.get("category", row.get("item", row.get("label", "")))).strip()
            entity = row.get("entity", row.get("answer_text", row.get("ground_truth", category)))
            if category:
                category_to_entities[category].add(canonicalize_entity(entity))
    else:
        raise ValueError(f"Unsupported benchmark entity file type: {path}")
    return {k: {v for v in vals if v} for k, vals in category_to_entities.items()}


def build_leakage_summary(catalog_entities: set[str], benchmark_entities_by_category: dict[str, set[str]]) -> dict[str, Any]:
    per_category = []
    seen = unseen = 0
    overlapping_entities = 0
    total_entities = 0

    for category, entities in sorted(benchmark_entities_by_category.items()):
        clean_entities = {canonicalize_entity(v) for v in entities if canonicalize_entity(v)}
        overlaps = sorted(clean_entities & catalog_entities)
        status = "seen" if overlaps else "unseen"
        if status == "seen":
            seen += 1
        else:
            unseen += 1
        overlapping_entities += len(overlaps)
        total_entities += len(clean_entities)
        per_category.append(
            {
                "category": category,
                "status": status,
                "entity_count": len(clean_entities),
                "overlap_count": len(overlaps),
                "overlap_entities": overlaps[:20],
            }
        )

    category_count = len(per_category)
    return {
        "seen_category_count": seen,
        "unseen_category_count": unseen,
        "benchmark_category_count": category_count,
        "catalog_entity_count": len(catalog_entities),
        "overlap_ratio": seen / category_count if category_count else 0.0,
        "entity_overlap_ratio": overlapping_entities / total_entities if total_entities else 0.0,
        "per_category_summary": per_category,
    }


def build_freak_entity_index(ground_truths: list[dict[str, Any]]) -> dict[str, set[str]]:
    category_to_entities: dict[str, set[str]] = defaultdict(set)
    for gt in ground_truths:
        category = str(gt.get("item") or gt.get("subset") or gt.get("answer_text") or "").strip()
        if not category:
            continue
        category_to_entities[category].add(canonicalize_entity(gt.get("answer_text") or category))
        category_to_entities[category].add(canonicalize_entity(category))
    return category_to_entities


def main():
    parser = argparse.ArgumentParser(description="Run category leakage audit.")
    parser.add_argument("--catalog", required=True, help="Calibrator pseudo-label catalog path.")
    parser.add_argument("--benchmark-entities", required=True, help="Benchmark entity list path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    catalog_entities = load_catalog_entities(args.catalog)
    benchmark_entities = load_benchmark_entities(args.benchmark_entities)
    payload = {
        "catalog_path": str(Path(args.catalog).resolve()),
        "benchmark_entities_path": str(Path(args.benchmark_entities).resolve()),
        **build_leakage_summary(catalog_entities, benchmark_entities),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved category leakage audit to {output}")


if __name__ == "__main__":
    main()
