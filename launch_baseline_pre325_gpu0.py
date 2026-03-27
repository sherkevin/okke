#!/usr/bin/env python3
"""
Launch the baseline-only experiments still required before the 3.25 deadline.

Scope:
  - llava-v1.5-7b
  - instructblip-7b
Methods:
  - base
  - vcd
  - damo
Datasets:
  - POPE (three official splits)
  - CHAIR (LLaVA only in the baseline-only pre-3.25 package)
"""
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/baseline_pre325"
REMOTE_MINITEST = f"{REMOTE_PROJECT}/logs/minitest"


@dataclass(frozen=True)
class EvalJob:
    model: str
    dataset: str
    method: str
    split: str
    mini_test: int
    extra_args: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"


def build_jobs() -> list[EvalJob]:
    jobs: list[EvalJob] = []

    for split in ("random", "popular", "adversarial"):
        for method in ("vcd", "damo"):
            jobs.append(
                EvalJob(
                    model="llava-v1.5-7b",
                    dataset="pope",
                    method=method,
                    split=split,
                    mini_test=3000,
                    extra_args=("--pope_split", split),
                )
            )

    for method in ("base", "vcd", "damo"):
        jobs.append(
            EvalJob(
                model="llava-v1.5-7b",
                dataset="chair",
                method=method,
                split="default",
                mini_test=5000,
                extra_args=("--chair_max_new_tokens", "384"),
            )
        )

    for split in ("random", "popular", "adversarial"):
        for method in ("base", "vcd", "damo"):
            jobs.append(
                EvalJob(
                    model="instructblip-7b",
                    dataset="pope",
                    method=method,
                    split=split,
                    mini_test=3000,
                    extra_args=("--pope_split", split),
                )
            )

    return jobs


def eval_command(job: EvalJob) -> str:
    parts = [
        REMOTE_PYTHON,
        f"{REMOTE_PROJECT}/run_eval_pipeline.py",
        "--model",
        job.model,
        "--dataset",
        job.dataset,
        "--method",
        job.method,
        "--mini_test",
        str(job.mini_test),
        *job.extra_args,
    ]
    return " ".join(shlex.quote(p) for p in parts)


def build_runner_body(*, matrix_log: str, manifest_tsv: str, jobs: list[EvalJob]) -> str:
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
        'echo "[MATRIX] baseline_pre325_gpu0 start $(date -Iseconds)"',
        'printf "iso_time\\tmodel\\tdataset\\tsplit\\tmethod\\texit_code\\tlatest_minitest_json_guess\\tmatrix_parent_log\\n" > "$MANIFEST"',
    ]
    for job in jobs:
        cmd = eval_command(job)
        json_glob = f'{job.method}_{job.dataset}_*.json'
        lines.append(f'echo "[START] {job.tag} $(date -Iseconds)"')
        lines.append(cmd)
        lines.append("rc=$?")
        lines.append(f'guess=$(ls -t "$MINITEST_DIR"/{json_glob} 2>/dev/null | head -1 || true)')
        lines.append(
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" '
            '"$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'echo "[DONE] {job.tag} rc=$rc guess=$guess $(date -Iseconds)"')
        lines.append('if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
    lines.append('echo "[MATRIX] baseline_pre325_gpu0 finished $(date -Iseconds)"')
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
    parser = argparse.ArgumentParser(description="Launch the baseline-only pre-3.25 GPU0 queue.")
    parser.add_argument("--mode", choices=["dry-run", "render", "launch"], default="dry-run")
    args = parser.parse_args()

    jobs = build_jobs()
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/baseline_pre325_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/baseline_pre325_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/baseline_pre325_gpu0_{ts}.manifest.tsv"

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "job_count": len(jobs),
                    "first": [job.tag for job in jobs[:5]],
                    "last": [job.tag for job in jobs[-5:]],
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
    local_copy = local_backup_dir / f"baseline_pre325_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")
    print(f"Wrote local runner copy: {local_copy}")

    if args.mode == "render":
        print(body)
        return 0

    proc = run_remote_bash(
        remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log),
        timeout=300,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\n--- Suggested registry rows ---\n")
    parent = f"baseline_pre325_gpu0_{ts}"
    print(
        f"| `{parent}` | `running` | `{matrix_log}` | "
        f"`see manifest {manifest_tsv}` | 18-cell baseline-only queue for the 3.25 deadline (`LLaVA POPE VCD+DAMO`, `LLaVA CHAIR Base+VCD+DAMO`, `InstructBLIP POPE Base+VCD+DAMO`). |"
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
