#!/usr/bin/env python3
"""
Queue the still-needed POPE decoding baselines on the remote server (GPU0 only).

Skips:
  - llava-v1.5-7b (matrix already complete)
  - qwen3-vl-8b + random + base (JSON already exists)

Default runs (sequential, same shell, CUDA_VISIBLE_DEVICES=0):
  - qwen3-vl-8b: 11 cells
  - instructblip-7b: 12 cells
Default total: 23 jobs × 3000 POPE samples each.

Models intentionally excluded from the default queue:
  - qwen3-vl-4b
  - qwen3-vl-2b

Outputs on remote:
  - Parent log: logs/pope_full_matrix/pope_baseline_pending_gpu0_<ts>.log
  - Runner:     logs/pope_full_matrix/pope_baseline_pending_gpu0_<ts>.sh
  - Manifest:   logs/pope_full_matrix/pope_baseline_pending_gpu0_<ts>.manifest.tsv
    (one row per job: time, model, split, method, exit_code, guessed_json_path)

Result JSONs (unchanged convention): logs/minitest/{method}_pope_<timestamp>.json

Usage (from repo root, Windows with SSH key configured in launch_pope_chair_main_table):
  python launch_pope_baseline_pending_gpu0.py --mode dry-run
  python launch_pope_baseline_pending_gpu0.py --mode render
  python launch_pope_baseline_pending_gpu0.py --mode launch
"""
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/pope_full_matrix"
REMOTE_MINITEST = f"{REMOTE_PROJECT}/logs/minitest"

METHODS = ("base", "beam_search", "dola", "opera")
SPLITS = ("random", "popular", "adversarial")
POPE_COUNT = 3000


DEFAULT_MODELS = ("qwen3-vl-8b", "instructblip-7b")


def build_job_list(models: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Return (model, pope_split, method) tuples in execution order."""

    jobs: list[tuple[str, str, str]] = []

    for model in models:
        if model == "qwen3-vl-8b":
            # Qwen3-VL-8B already has random+base.
            for split in SPLITS:
                for method in METHODS:
                    if split == "random" and method == "base":
                        continue
                    jobs.append((model, split, method))
            continue

        for split in SPLITS:
            for method in METHODS:
                jobs.append((model, split, method))

    return jobs


def eval_command(model: str, split: str, method: str) -> str:
    parts = [
        REMOTE_PYTHON,
        f"{REMOTE_PROJECT}/run_eval_pipeline.py",
        "--model",
        model,
        "--dataset",
        "pope",
        "--method",
        method,
        "--mini_test",
        str(POPE_COUNT),
        "--pope_split",
        split,
    ]
    return " ".join(shlex.quote(p) for p in parts)


def build_runner_body(
    *,
    matrix_log: str,
    manifest_tsv: str,
    jobs: list[tuple[str, str, str]],
) -> str:
    lines: list[str] = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {REMOTE_PROJECT}",
        "source /etc/network_turbo >/dev/null 2>&1 || true",
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "export PYTHONUNBUFFERED=1",
        "export CUDA_VISIBLE_DEVICES=0",
        f'MATRIX_LOG="{matrix_log}"',
        f'MANIFEST="{manifest_tsv}"',
        f'MINITEST_DIR="{REMOTE_MINITEST}"',
        'echo "[MATRIX] pope_baseline_pending_gpu0 start $(date -Iseconds)"',
        'echo "[MATRIX] parent_log=${MATRIX_LOG}"',
        'printf "iso_time\\tmodel\\tpope_split\\tmethod\\texit_code\\tlatest_minitest_json_guess\\tmatrix_parent_log\\n" > "$MANIFEST"',
    ]
    for model, split, method in jobs:
        tag = f"{model}__{split}__{method}"
        cmd = eval_command(model, split, method)
        lines.append(f'echo "[START] {tag} $(date -Iseconds)"')
        lines.append(f"{cmd}")
        lines.append("rc=$?")
        # Newest file for this method; serial execution => this job's output under normal conditions.
        lines.append(f'guess=$(ls -t "$MINITEST_DIR"/{method}_pope_*.json 2>/dev/null | head -1 || true)')
        lines.append(
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{model}" "{split}" "{method}" '
            '"$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'echo "[DONE] {tag} rc=$rc guess=$guess $(date -Iseconds)"')
        lines.append('if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
    lines.append('echo "[MATRIX] pope_baseline_pending_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")

    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch pending POPE baseline matrix on remote GPU0.")
    parser.add_argument("--mode", choices=["dry-run", "render", "launch"], default="dry-run")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen3-vl-8b", "instructblip-7b", "qwen3-vl-4b", "qwen3-vl-2b"],
        default=list(DEFAULT_MODELS),
        help="Models to include in the queue. Defaults exclude qwen3-vl-4b and qwen3-vl-2b.",
    )
    args = parser.parse_args()

    jobs = build_job_list(tuple(args.models))
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/pope_baseline_pending_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/pope_baseline_pending_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/pope_baseline_pending_gpu0_{ts}.manifest.tsv"

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "models": args.models,
                    "job_count": len(jobs),
                    "first": jobs[:3],
                    "last": jobs[-3:],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print("\nRemote paths (preview):")
        print(matrix_log)
        print(runner_sh)
        print(manifest_tsv)
        return 0

    body = build_runner_body(matrix_log=matrix_log, manifest_tsv=manifest_tsv, jobs=jobs)

    local_backup_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_backup_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_backup_dir / f"pope_baseline_pending_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")
    print(f"Wrote local runner copy: {local_copy}")

    if args.mode == "render":
        print(body)
        return 0

    script = remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log)
    proc = run_remote_bash(script, timeout=300, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\n--- Suggested registry rows (Related Baseline / Matrix Logs) ---\n")
    parent = f"pope_baseline_pending_gpu0_{ts}"
    print(
        f"| `{parent}` | `running` | `{matrix_log}` | "
        f"`see manifest {manifest_tsv}` | GPU0 serial queue: {len(jobs)} POPE jobs for {', '.join(args.models)}; manifest maps each cell to minitest JSON. |"
    )
    print(
        f"| `{parent}_child_template` | `pending` | `{matrix_log}` | "
        f"`fill from manifest rows` | One registry child row per manifest line after completion. |"
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
