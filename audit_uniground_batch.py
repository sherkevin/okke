#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from validate_uniground_universality import validate_result


REQUIRED_RESULT_KEYS = [
    "universal_claim_manifest",
    "psi_univ checkpoint sha",
    "prefix_ambiguity_rate",
    "span_collapse_errors",
    "suffix_stability_rate",
    "abstention_rate",
    "abort_trigger_rate",
    "abort_backoff_verified_steps",
    "latency_split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-audit UniGround result JSON files.")
    parser.add_argument(
        "--result-glob",
        default="logs/uniground_v6/**/*.json",
        help="Glob for TLRA_univ result JSON files.",
    )
    parser.add_argument(
        "--result-json",
        action="append",
        default=[],
        help="Optional explicit result JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--status-md",
        default="experiment_logs/uniground_v6/universality_audit_latest.md",
        help="Markdown status output path.",
    )
    parser.add_argument(
        "--summary-json",
        default="experiment_logs/uniground_v6/universality_audit_latest.json",
        help="JSON summary output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(item).resolve() for item in args.result_json]
    globbed = [path.resolve() for path in Path(".").glob(args.result_glob)]
    seen: dict[str, Path] = {}
    for path in explicit + globbed:
        if path.suffix.lower() != ".json" or not path.exists():
            continue
        seen[str(path)] = path
    return sorted(seen.values())


def is_tlra_univ_result(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    manifest = payload.get("universal_claim_manifest") or {}
    return manifest.get("method_route") == "TLRA_univ"


def classify_missing_fields(payload: dict[str, Any], checks: dict[str, bool], info: dict[str, Any]) -> list[str]:
    missing = []
    if not payload.get("universal_claim_manifest"):
        missing.append("universal_claim_manifest")
    manifest = payload.get("universal_claim_manifest") or {}
    psi_meta = manifest.get("psi_univ_checkpoint") or {}
    if not psi_meta.get("checkpoint_sha256"):
        missing.append("psi_univ checkpoint sha")
    if not checks.get("prefix_ambiguity_rate", False):
        missing.append("prefix_ambiguity_rate")
    if not checks.get("span_collapse_errors", False):
        missing.append("span_collapse_errors")
    if not checks.get("suffix_stability_rate", False):
        missing.append("suffix_stability_rate")
    if not checks.get("abstention_rate", False):
        missing.append("abstention_rate")
    if not checks.get("abort_trigger_rate", False):
        missing.append("abort_trigger_rate")
    if not checks.get("abort_backoff_verified_steps", False):
        missing.append("abort_backoff_verified_steps")
    if not checks.get("latency_split_present", False):
        missing.append("latency_split")
    return missing


def audit_one(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    ok, checks, info = validate_result(path, checkpoint_info=None)
    missing = classify_missing_fields(payload, checks, info)
    rejected_by_abort_rule = not checks.get("abstention_abort_coupled", False)
    return {
        "result_path": str(path),
        "ok": ok,
        "missing_required_fields": missing,
        "rejected_by_abort_rule": rejected_by_abort_rule,
        "abort_trigger_rate": info.get("abort_trigger_rate"),
        "abort_backoff_verified_steps": info.get("abort_backoff_verified_steps"),
        "prefix_ambiguity_rate": info.get("prefix_ambiguity_rate"),
        "span_collapse_errors": info.get("span_collapse_errors"),
        "suffix_stability_rate": info.get("suffix_stability_rate"),
        "abstention_rate": info.get("abstention_rate"),
        "latency_split": info.get("latency_split"),
        "checks": checks,
    }


def build_summary(results: list[dict[str, Any]], scanned_count: int) -> dict[str, Any]:
    accepted = [item for item in results if item["ok"]]
    rejected = [item for item in results if not item["ok"]]
    return {
        "generated_at": datetime.now().isoformat(),
        "required_fields": REQUIRED_RESULT_KEYS,
        "scanned_tlra_univ_results": scanned_count,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "overall_ok": len(rejected) == 0,
        "acceptance_rule": "If abort_trigger_rate > 0 then abort_backoff_verified_steps must be > 0; otherwise reject the result.",
    }


def render_status(summary: dict[str, Any]) -> str:
    lines = [
        "# UniGround Universality Audit Latest",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Scanned TLRA_univ results: `{summary['scanned_tlra_univ_results']}`",
        f"- Accepted: `{summary['accepted_count']}`",
        f"- Rejected: `{summary['rejected_count']}`",
        "",
        "## Acceptance Rule",
        "",
        f"- {summary['acceptance_rule']}",
        "",
        "## Required Fields",
        "",
    ]
    for field in summary["required_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Latest Conclusion", ""])
    if summary["scanned_tlra_univ_results"] == 0:
        lines.append("- No TLRA_univ result JSON has been received in this batch yet; acceptance remains blocked pending the first real result.")
    elif summary["rejected_count"] == 0:
        lines.append("- All scanned TLRA_univ results passed the current universality validator and abort back-off coupling rule.")
    else:
        lines.append("- At least one TLRA_univ result failed universality audit; those results are not acceptable for table use or claim support.")
    lines.extend(["", "## Rejected Results", ""])
    if not summary["rejected"]:
        lines.append("- None.")
    else:
        for item in summary["rejected"]:
            lines.append(f"- `{item['result_path']}`")
            if item["missing_required_fields"]:
                lines.append(f"  missing: {', '.join(item['missing_required_fields'])}")
            if item["rejected_by_abort_rule"]:
                lines.append(
                    f"  abort rule failure: abort_trigger_rate={item['abort_trigger_rate']} "
                    f"abort_backoff_verified_steps={item['abort_backoff_verified_steps']}"
                )
    lines.extend(["", "## Accepted Results", ""])
    if not summary["accepted"]:
        lines.append("- None.")
    else:
        for item in summary["accepted"]:
            lines.append(f"- `{item['result_path']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    candidate_paths = collect_results(args)
    tlra_univ_paths = []
    for path in candidate_paths:
        try:
            payload = load_json(path)
        except Exception:
            continue
        if is_tlra_univ_result(payload):
            tlra_univ_paths.append(path)

    results = [audit_one(path) for path in tlra_univ_paths]
    summary = build_summary(results, scanned_count=len(tlra_univ_paths))

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    status_path = Path(args.status_md)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(render_status(summary), encoding="utf-8")

    print(summary_path)
    print(status_path)
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
