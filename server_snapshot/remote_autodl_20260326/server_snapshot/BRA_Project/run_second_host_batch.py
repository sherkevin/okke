#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from validate_uniground_universality import validate_result


PROJECT = Path("/root/autodl-tmp/BRA_Project")
RELEASE_FEED = PROJECT / "experiment_logs" / "uniground_v6" / "checkpoint_release_feed_20260322.md"
BATCH_READY_FEED = PROJECT / "experiment_logs" / "uniground_v6" / "second_host_batch_ready_20260322.md"
PASS_FAIL_FEED = PROJECT / "experiment_logs" / "uniground_v6" / "audit_pass_fail_feed_20260322.md"
AUDIT_SUMMARY = PROJECT / "experiment_logs" / "uniground_v6" / "universality_audit_latest.json"
AUDIT_STATUS = PROJECT / "experiment_logs" / "uniground_v6" / "universality_audit_latest.md"
SECOND_HOST_RUNNER = PROJECT / "run_second_host_uniground.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current second-host UniGround batch.")
    parser.add_argument("--dataset", default="pope", choices=["pope", "chair"])
    parser.add_argument("--mini_test", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=["uniground", "uniground_no_gate", "uniground_global_only"])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_release_feed(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    checkpoint_matches = re.findall(r"checkpoint path:\s*`(/root/autodl-tmp/BRA_Project/models/uniground_v6/[^`]+\.pt)`", text)
    sha_matches = re.findall(r"(?m)^- sha256:\s*`([0-9a-f]{64})`", text)
    if not checkpoint_matches or not sha_matches:
        raise ValueError(f"Unable to parse canonical checkpoint from {path}")
    return checkpoint_matches[-1], sha_matches[-1]


def choose_host() -> str:
    if (PROJECT / "models" / "Qwen3-VL-4B-Instruct").exists():
        return "qwen3-vl-4b"
    if (PROJECT / "models" / "Qwen3-VL-2B-Instruct").exists():
        return "qwen3-vl-2b"
    raise FileNotFoundError("No second-host model is available.")


def output_dir_for(model_key: str) -> Path:
    if model_key == "qwen3-vl-4b":
        return PROJECT / "logs" / "uniground_v6" / "second_host_qwen4b"
    return PROJECT / "logs" / "uniground_v6" / "second_host_qwen2b"


def run_method(method: str, model_key: str, checkpoint_path: str, dataset: str, mini_test: int) -> Path:
    cmd = [
        sys.executable,
        str(SECOND_HOST_RUNNER),
        "--force-model",
        model_key,
        "--dataset",
        dataset,
        "--method",
        method,
        "--mini_test",
        str(mini_test),
        "--psi_checkpoint",
        checkpoint_path,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"{method} failed")
    lines = [line.strip() for line in (proc.stdout + "\n" + proc.stderr).splitlines() if line.strip()]
    pattern = re.compile(r"(/root/autodl-tmp/BRA_Project/logs/uniground_v6/[^\s]+\.json)")
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return Path(match.group(1))
    raise RuntimeError(f"Could not locate result JSON for {method}")


def append_batch_ready(
    timestamp: str,
    active_host: str,
    methods_completed: list[str],
    result_paths: list[Path],
    audit_pending: str,
    host_switch_note: str | None,
    failure_note: str | None = None,
) -> None:
    lines = [
        f"## {timestamp}",
        f"- timestamp: `{timestamp}`",
        f"- active host: `{active_host}`",
        f"- methods completed: `{', '.join(methods_completed) if methods_completed else 'none'}`",
        "- exact JSON filenames:",
    ]
    if result_paths:
        for path in result_paths:
            lines.append(f"  - `{path.name}`")
    else:
        lines.append("  - none")
    lines.append(f"- audit pending: `{audit_pending}`")
    if host_switch_note:
        lines.append(f"- host switch note: `{host_switch_note}`")
    if failure_note:
        lines.append(f"- failure note: `{failure_note}`")
    lines.append("")
    with open(BATCH_READY_FEED, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def audit_current_batch(result_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    for path in result_paths:
        ok, checks, info = validate_result(path, checkpoint_info=None)
        item = {
            "path": str(path),
            "filename": path.name,
            "checks": checks,
            "info": info,
        }
        if ok:
            accepted.append(item)
        else:
            rejected.append(item)
    return accepted, rejected


def reject_reasons(item: dict) -> list[str]:
    checks = item["checks"]
    info = item["info"]
    reasons = []
    required = {
        "universal_claim_manifest": checks.get("manifest_present", False),
        "psi_univ checkpoint sha": bool(info.get("result_checkpoint_sha256")),
        "prefix_ambiguity_rate": checks.get("prefix_ambiguity_rate", False),
        "span_collapse_errors": checks.get("span_collapse_errors", False),
        "suffix_stability_rate": checks.get("suffix_stability_rate", False),
        "abstention_rate": checks.get("abstention_rate", False),
        "abort_trigger_rate": checks.get("abort_trigger_rate", False),
        "abort_backoff_verified_steps": checks.get("abort_backoff_verified_steps", False),
        "latency_split": checks.get("latency_split_present", False),
    }
    for key, ok in required.items():
        if not ok:
            reasons.append(f"missing_or_invalid:{key}")
    if not checks.get("abstention_abort_coupled", False):
        reasons.append(
            f"abort_rule_failure:abort_trigger_rate={info.get('abort_trigger_rate')},"
            f"abort_backoff_verified_steps={info.get('abort_backoff_verified_steps')}"
        )
    return reasons


def append_pass_fail_feed(timestamp: str, result_paths: list[Path], accepted: list[dict], rejected: list[dict]) -> None:
    lines = [
        f"## {timestamp}",
        "- batch source:",
    ]
    for path in result_paths:
        lines.append(f"  - `{path.name}`")
    lines.append("- pass list:")
    if accepted:
        for item in accepted:
            lines.append(f"  - `{item['filename']}`")
    else:
        lines.append("  - none")
    lines.append("- fail list:")
    if rejected:
        for item in rejected:
            lines.append(f"  - `{item['filename']}`")
    else:
        lines.append("  - none")
    lines.append("- fail reasons:")
    if rejected:
        for item in rejected:
            lines.append(f"  - `{item['filename']}`: {', '.join(reject_reasons(item))}")
    else:
        lines.append("  - none")
    lines.append("")
    with open(PASS_FAIL_FEED, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    checkpoint_path_str, expected_sha = parse_release_feed(RELEASE_FEED)
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Canonical checkpoint missing: {checkpoint_path}")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise ValueError(f"Checkpoint SHA mismatch: expected {expected_sha}, got {actual_sha}")

    host_model = choose_host()
    host_switch_note = None
    completed_methods: list[str] = []
    result_paths: list[Path] = []

    for index, method in enumerate(args.methods):
        try:
            result_path = run_method(method, host_model, str(checkpoint_path), args.dataset, args.mini_test)
            completed_methods.append(method)
            result_paths.append(result_path)
        except Exception as exc:
            if host_model == "qwen3-vl-4b" and (PROJECT / "models" / "Qwen3-VL-2B-Instruct").exists():
                host_model = "qwen3-vl-2b"
                host_switch_note = f"Switched from qwen3-vl-4b to qwen3-vl-2b while running `{method}`."
                result_path = run_method(method, host_model, str(checkpoint_path), args.dataset, args.mini_test)
                completed_methods.append(method)
                result_paths.append(result_path)
            else:
                timestamp = datetime.now().isoformat()
                append_batch_ready(
                    timestamp,
                    host_model,
                    completed_methods,
                    result_paths,
                    audit_pending="no",
                    host_switch_note=host_switch_note,
                    failure_note=f"{method} failed: {exc}",
                )
                raise

    timestamp = datetime.now().isoformat()
    append_batch_ready(timestamp, host_model, completed_methods, result_paths, audit_pending="yes", host_switch_note=host_switch_note)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "audit_uniground_batch.py"),
            "--summary-json",
            str(AUDIT_SUMMARY),
            "--status-md",
            str(AUDIT_STATUS),
        ],
        cwd=str(PROJECT),
        env=env,
        check=False,
    )

    accepted, rejected = audit_current_batch(result_paths)
    append_pass_fail_feed(timestamp, result_paths, accepted, rejected)
    append_batch_ready(timestamp + "::audit_done", host_model, completed_methods, result_paths, audit_pending="no", host_switch_note=host_switch_note)
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
