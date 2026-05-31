#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobArtifacts:
    output_json: Path
    sample_jsonl: Path
    validation_json: Path


def derive_artifacts(output_json: str | Path) -> JobArtifacts:
    output_path = Path(output_json)
    base = output_path.with_suffix("")
    return JobArtifacts(
        output_json=output_path,
        sample_jsonl=base.with_name(base.name + ".samples.jsonl"),
        validation_json=base.with_name(base.name + ".validation.json"),
    )


def append_jsonl_record(path: str | Path, payload: dict[str, Any]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_record_coverage(records: list[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    seen: dict[int, int] = {}
    for record in records:
        sample_index = int(record["sample_index"])
        seen[sample_index] = seen.get(sample_index, 0) + 1
    duplicates = sorted(idx for idx, count in seen.items() if count > 1)
    missing = [idx for idx in range(expected_n) if idx not in seen]
    attempted_n = len(seen)
    ok_n = sum(1 for record in records if record.get("status") == "ok")
    error_n = sum(1 for record in records if record.get("status") != "ok")
    return {
        "expected_n": int(expected_n),
        "attempted_n": attempted_n,
        "ok_n": ok_n,
        "error_n": error_n,
        "missing_indices": missing,
        "duplicate_indices": duplicates,
        "complete": attempted_n == int(expected_n) and not missing and not duplicates and error_n == 0,
    }


def validate_job_output(output_json: str | Path) -> dict[str, Any]:
    artifacts = derive_artifacts(output_json)
    result = {
        "output_json_exists": artifacts.output_json.exists(),
        "sample_jsonl_exists": artifacts.sample_jsonl.exists(),
        "valid_json": False,
        "status": None,
        "complete": False,
        "expected_n": None,
        "sample_count": None,
        "n_errors": None,
        "issues": [],
    }
    if not artifacts.output_json.exists():
        result["issues"].append("missing_output_json")
        return result
    try:
        payload = json.loads(artifacts.output_json.read_text(encoding="utf-8"))
    except Exception as exc:
        result["issues"].append(f"invalid_output_json:{exc}")
        return result
    result["valid_json"] = True
    result["status"] = payload.get("status")
    result["complete"] = bool(payload.get("complete"))
    result["expected_n"] = payload.get("expected_n")
    result["sample_count"] = payload.get("sample_count", payload.get("n_samples"))
    result["n_errors"] = payload.get("n_errors")
    if not artifacts.sample_jsonl.exists():
        result["issues"].append("missing_sample_jsonl")
        return result
    try:
        coverage = compute_record_coverage(load_jsonl_records(artifacts.sample_jsonl), int(payload.get("expected_n") or 0))
    except Exception as exc:
        result["issues"].append(f"sample_jsonl_validation_failed:{exc}")
        return result
    result["coverage"] = coverage
    if not coverage["complete"]:
        result["issues"].append("incomplete_sample_coverage")
    if payload.get("status") != "final_complete":
        result["issues"].append("status_not_final_complete")
    if not payload.get("complete"):
        result["issues"].append("payload_not_marked_complete")
    if int(payload.get("sample_count", payload.get("n_samples", -1)) or -1) != int(payload.get("expected_n") or -1):
        result["issues"].append("sample_count_mismatch")
    if int(payload.get("n_errors", 0) or 0) != 0:
        result["issues"].append("nonzero_errors")
    return result
