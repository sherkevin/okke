#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_matrix_result(payload: Any, dataset: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "dataset" in payload:
            return payload
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("dataset") == dataset:
                        return item
    raise ValueError(f"Could not find dataset='{dataset}' in payload")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_docvqa(result: dict[str, Any], expected_bra_method: str | None) -> tuple[bool, dict[str, Any]]:
    baseline = result.get("baseline", {})
    bra = result.get("bra", {})
    sample_audits = result.get("sample_audits")
    checks = {
        "bra_method_present": isinstance(result.get("bra_method"), str),
        "bra_method_matches": result.get("bra_method") == expected_bra_method if expected_bra_method else True,
        "baseline_normalized_exact_match": _is_number(baseline.get("normalized_exact_match")),
        "bra_normalized_exact_match": _is_number(bra.get("normalized_exact_match")),
        "baseline_anls": _is_number(baseline.get("anls")),
        "bra_anls": _is_number(bra.get("anls")),
        "bra_intervention_rate": _is_number(bra.get("intervention_rate")),
        "sample_audits_present": isinstance(sample_audits, list),
        "sample_audits_nonempty": isinstance(sample_audits, list) and len(sample_audits) > 0,
    }
    return all(checks.values()), checks


def _check_mmmu(result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    bra = result.get("bra", {})
    provenance = bra.get("visual_state_provenance")
    checks = {
        "bra_intervention_rate": _is_number(bra.get("intervention_rate")),
        "visual_state_provenance_present": isinstance(provenance, dict),
        "visual_state_provenance_nonempty": isinstance(provenance, dict) and len(provenance) > 0,
        "avg_vasm_time_ms": _is_number(bra.get("avg_vasm_time_ms")),
        "avg_routing_time_ms": _is_number(bra.get("avg_routing_time_ms")),
    }
    return all(checks.values()), checks


def _check_chaina(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    checks = {
        "agl_stddev": _is_number(payload.get("agl_stddev")),
        "peak_vram_gb": _is_number(payload.get("peak_vram_gb")),
    }
    return all(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark export contract fields.")
    parser.add_argument("--kind", choices=["docvqa", "mmmu", "chaina"], required=True)
    parser.add_argument("--file", required=True, help="Result JSON path")
    parser.add_argument("--expected-bra-method", default=None)
    args = parser.parse_args()

    path = Path(args.file)
    payload = _load_json(path)

    if args.kind == "docvqa":
        result = _pick_matrix_result(payload, "docvqa")
        ok, checks = _check_docvqa(result, args.expected_bra_method)
    elif args.kind == "mmmu":
        result = _pick_matrix_result(payload, "mmmu")
        ok, checks = _check_mmmu(result)
    else:
        ok, checks = _check_chaina(payload)

    summary = {
        "ok": ok,
        "kind": args.kind,
        "file": str(path),
        "checks": checks,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
