#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_VISION_ENCODER = "openai/clip-vit-large-patch14::image"
EXPECTED_TEXT_ENCODER = "openai/clip-vit-large-patch14::text"
EXPECTED_ROUTE = "TLRA_univ"
EXPECTED_MODULE = "Psi_univ"
EXPECTED_FORMAT = "psi_univ_checkpoint_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def validate_checkpoint(path: Path) -> tuple[bool, dict[str, bool], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    contract = payload.get("contract", {}) if isinstance(payload, dict) else {}
    checks = {
        "checkpoint_format": contract.get("checkpoint_format") == EXPECTED_FORMAT,
        "method_route": contract.get("plugin_route") == EXPECTED_ROUTE,
        "learned_module": contract.get("learned_module") == EXPECTED_MODULE,
        "vision_encoder_name": contract.get("frozen_vision_encoder_name") == EXPECTED_VISION_ENCODER,
        "text_encoder_name": contract.get("frozen_text_encoder_name") == EXPECTED_TEXT_ENCODER,
        "embedding_dim_present": _is_number(contract.get("embedding_dim")),
        "region_feature_flag_present": isinstance(contract.get("region_features_enabled"), bool),
        "uses_model_native_hidden_states_false": contract.get("uses_model_native_hidden_states") is False,
        "uses_lm_head_geometry_false": contract.get("uses_lm_head_geometry") is False,
        "learned_per_model_adapter_false": contract.get("learned_per_model_adapter") is False,
        "parameter_free_tokenizer_bridge_true": contract.get("parameter_free_tokenizer_bridge") is True,
        "source_hashes_present": isinstance(contract.get("source_hashes"), dict) and bool(contract.get("source_hashes")),
    }
    info = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "contract": contract,
    }
    return all(checks.values()), checks, info


def validate_result(path: Path, checkpoint_info: dict[str, Any] | None) -> tuple[bool, dict[str, bool], dict[str, Any]]:
    payload = _load_json(path)
    manifest = _find_key(payload, "universal_claim_manifest") or {}
    latency_split = _find_key(payload, "latency_split") or {}
    prefix_ambiguity_rate = _find_key(payload, "prefix_ambiguity_rate")
    span_collapse_errors = _find_key(payload, "span_collapse_errors")
    abstention_rate = _find_key(payload, "abstention_rate")
    abort_trigger_rate = _find_key(payload, "abort_trigger_rate")
    abort_backoff_verified_steps = _find_key(payload, "abort_backoff_verified_steps")
    suffix_stability_rate = _find_key(payload, "suffix_stability_rate")

    checkpoint_sha = None if checkpoint_info is None else checkpoint_info["checkpoint_sha256"]
    result_checkpoint_meta = manifest.get("psi_univ_checkpoint", {})
    result_checkpoint_sha = result_checkpoint_meta.get("checkpoint_sha256")

    checks = {
        "manifest_present": bool(manifest),
        "method_route": manifest.get("method_route") == EXPECTED_ROUTE,
        "learned_module": manifest.get("learned_module") == EXPECTED_MODULE,
        "uses_model_native_hidden_states_false": manifest.get("uses_model_native_hidden_states") is False,
        "uses_lm_head_geometry_false": manifest.get("uses_lm_head_geometry") is False,
        "learned_per_model_adapter_false": manifest.get("learned_per_model_adapter") is False,
        "parameter_free_tokenizer_bridge_true": manifest.get("parameter_free_tokenizer_bridge") is True,
        "abstention_controls_behavior_true": manifest.get("abstention_controls_behavior") is True,
        "same_checkpoint_sha": (checkpoint_sha is None) or (result_checkpoint_sha == checkpoint_sha),
        "prefix_ambiguity_rate": _is_number(prefix_ambiguity_rate),
        "span_collapse_errors": _is_number(span_collapse_errors),
        "suffix_stability_rate": _is_number(suffix_stability_rate),
        "abstention_rate": _is_number(abstention_rate),
        "abort_trigger_rate": _is_number(abort_trigger_rate),
        "abort_backoff_verified_steps": _is_number(abort_backoff_verified_steps),
        "latency_split_present": isinstance(latency_split, dict),
        "latency_candidate_construction": _is_number(latency_split.get("candidate_construction_ms")),
        "latency_sidecar_scoring": _is_number(latency_split.get("sidecar_scoring_ms")),
        "latency_bridge_redistribution": _is_number(latency_split.get("bridge_redistribution_ms")),
        "latency_jitter": _is_number(latency_split.get("jitter_ms")),
        "abstention_abort_coupled": _validate_abort_backoff_logic(
            abstention_rate=abstention_rate,
            abort_trigger_rate=abort_trigger_rate,
            abort_backoff_verified_steps=abort_backoff_verified_steps,
        ),
    }
    info = {
        "result_path": str(path),
        "result_checkpoint_sha256": result_checkpoint_sha,
        "prefix_ambiguity_rate": prefix_ambiguity_rate,
        "span_collapse_errors": span_collapse_errors,
        "suffix_stability_rate": suffix_stability_rate,
        "abstention_rate": abstention_rate,
        "abort_trigger_rate": abort_trigger_rate,
        "abort_backoff_verified_steps": abort_backoff_verified_steps,
        "latency_split": latency_split,
    }
    return all(checks.values()), checks, info


def _validate_abort_backoff_logic(
    abstention_rate: Any,
    abort_trigger_rate: Any,
    abort_backoff_verified_steps: Any,
) -> bool:
    if not _is_number(abstention_rate) or not _is_number(abort_trigger_rate) or not _is_number(abort_backoff_verified_steps):
        return False
    if abort_trigger_rate == 0:
        return abort_backoff_verified_steps == 0
    return abort_backoff_verified_steps > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate UniGround universality contract.")
    parser.add_argument("--checkpoint", default=None, help="Psi_univ checkpoint path")
    parser.add_argument("--result-json", default=None, help="Runtime result JSON path")
    args = parser.parse_args()

    if not args.checkpoint and not args.result_json:
        raise SystemExit("Provide at least one of --checkpoint or --result-json")

    summary: dict[str, Any] = {"ok": True}
    checkpoint_info = None

    if args.checkpoint:
        ok, checks, info = validate_checkpoint(Path(args.checkpoint))
        summary["checkpoint"] = {"ok": ok, "checks": checks, "info": info}
        summary["ok"] = summary["ok"] and ok
        checkpoint_info = info

    if args.result_json:
        ok, checks, info = validate_result(Path(args.result_json), checkpoint_info)
        summary["result"] = {"ok": ok, "checks": checks, "info": info}
        summary["ok"] = summary["ok"] and ok

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
