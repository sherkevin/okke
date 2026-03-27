#!/usr/bin/env python3
"""
Queue the still-needed CHAIR decoding baselines on the remote server (GPU0 only).

Default models:
  - llava-v1.5-7b
  - qwen3-vl-8b
  - instructblip-7b

Excluded by default:
  - qwen3-vl-4b
  - qwen3-vl-2b

Default methods:
  - base
  - beam_search
  - dola
  - opera

Outputs on remote:
  - Parent log: logs/chair_full_matrix/chair_baseline_pending_gpu0_<ts>.log
  - Runner:     logs/chair_full_matrix/chair_baseline_pending_gpu0_<ts>.sh
  - Manifest:   logs/chair_full_matrix/chair_baseline_pending_gpu0_<ts>.manifest.tsv
  - Watcher:    logs/chair_full_matrix/chair_baseline_pending_gpu0_<ts>.watch.sh (when waiting for another PID)

Result JSONs (unchanged convention): logs/minitest/{method}_chair_<timestamp>.json
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

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/chair_full_matrix"
REMOTE_MINITEST = f"{REMOTE_PROJECT}/logs/minitest"

DEFAULT_MODELS = ("llava-v1.5-7b", "qwen3-vl-8b", "instructblip-7b")
METHODS = ("base", "beam_search", "dola", "opera")
CHAIR_COUNT = 5000
CHAIR_MAX_NEW_TOKENS = 384


def build_job_list(models: tuple[str, ...], methods: tuple[str, ...]) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for model in models:
        for method in methods:
            jobs.append((model, method))
    return jobs


def eval_command(model: str, method: str) -> str:
    parts = [
        REMOTE_PYTHON,
        f"{REMOTE_PROJECT}/run_eval_pipeline.py",
        "--model",
        model,
        "--dataset",
        "chair",
        "--method",
        method,
        "--mini_test",
        str(CHAIR_COUNT),
        "--chair_max_new_tokens",
        str(CHAIR_MAX_NEW_TOKENS),
    ]
    return " ".join(shlex.quote(p) for p in parts)


def build_runner_body(*, matrix_log: str, manifest_tsv: str, jobs: list[tuple[str, str]]) -> str:
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
        'echo "[MATRIX] chair_baseline_pending_gpu0 start $(date -Iseconds)"',
        'echo "[MATRIX] parent_log=${MATRIX_LOG}"',
        'printf "iso_time\\tmodel\\tmethod\\texit_code\\tlatest_minitest_json_guess\\tmatrix_parent_log\\n" > "$MANIFEST"',
    ]
    for model, method in jobs:
        tag = f"{model}__chair__{method}"
        cmd = eval_command(model, method)
        lines.append(f'echo "[START] {tag} $(date -Iseconds)"')
        lines.append(f"{cmd}")
        lines.append("rc=$?")
        lines.append(f'guess=$(ls -t "$MINITEST_DIR"/{method}_chair_*.json 2>/dev/null | head -1 || true)')
        lines.append(
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{model}" "{method}" '
            '"$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'echo "[DONE] {tag} rc=$rc guess=$guess $(date -Iseconds)"')
        lines.append('if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
    lines.append('echo "[MATRIX] chair_baseline_pending_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str, wait_for_pid: int | None) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")

    watch_sh = runner_sh.replace(".sh", ".watch.sh")
    watch_log = matrix_log.replace(".log", ".watch.log")

    launcher = (
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
    )
    if wait_for_pid is not None:
        launcher = (
            f"cat > \"{watch_sh}\" <<'EOF'\n"
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"WAIT_PID={wait_for_pid}\n"
            f"RUNNER={runner_sh}\n"
            f"MATRIX_LOG={matrix_log}\n"
            f"WATCH_LOG={watch_log}\n"
            "echo \"[WATCH] installed $(date -Iseconds)\" >> \"$WATCH_LOG\"\n"
            "while kill -0 \"$WAIT_PID\" 2>/dev/null; do sleep 15; done\n"
            "echo \"[WATCH] dependency cleared $(date -Iseconds); launching CHAIR queue\" | tee -a \"$WATCH_LOG\" >> \"$MATRIX_LOG\"\n"
            "nohup /bin/bash \"$RUNNER\" >> \"$MATRIX_LOG\" 2>&1 < /dev/null &\n"
            "echo \"[WATCH] launch_pid=$! $(date -Iseconds)\" | tee -a \"$WATCH_LOG\" >> \"$MATRIX_LOG\"\n"
            "EOF\n"
            f"chmod +x \"{watch_sh}\"\n"
            f"nohup /bin/bash \"{watch_sh}\" > /dev/null 2>&1 < /dev/null &\n"
            "echo \"WATCH_PID=$!\"\n"
            f"echo \"WATCHER={watch_sh}\"\n"
            f"echo \"WATCH_LOG={watch_log}\"\n"
        )

    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"{launcher}"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch pending CHAIR baseline matrix on remote GPU0.")
    parser.add_argument("--mode", choices=["dry-run", "render", "launch"], default="dry-run")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["llava-v1.5-7b", "qwen3-vl-8b", "instructblip-7b", "qwen3-vl-4b", "qwen3-vl-2b"],
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    parser.add_argument("--wait-for-pid", type=int, default=None, help="If set, install a remote watcher that launches after this PID exits.")
    args = parser.parse_args()

    jobs = build_job_list(tuple(args.models), tuple(args.methods))
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/chair_baseline_pending_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/chair_baseline_pending_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/chair_baseline_pending_gpu0_{ts}.manifest.tsv"

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "models": args.models,
                    "methods": args.methods,
                    "wait_for_pid": args.wait_for_pid,
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
    local_copy = local_backup_dir / f"chair_baseline_pending_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")
    print(f"Wrote local runner copy: {local_copy}")

    if args.mode == "render":
        print(body)
        return 0

    script = remote_launch_script(
        runner_body=body,
        runner_sh=runner_sh,
        matrix_log=matrix_log,
        wait_for_pid=args.wait_for_pid,
    )
    proc = run_remote_bash(script, timeout=300, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    parent = f"chair_baseline_pending_gpu0_{ts}"
    state = "scheduled-after-pid" if args.wait_for_pid is not None else "running"
    note = f"GPU0 serial CHAIR queue: {len(jobs)} jobs for {', '.join(args.models)}"
    if args.wait_for_pid is not None:
        note += f"; will launch after PID {args.wait_for_pid} exits"
    print("\n--- Suggested registry rows (Related Baseline / Matrix Logs) ---\n")
    print(
        f"| `{parent}` | `{state}` | `{matrix_log}` | "
        f"`see manifest {manifest_tsv}` | {note}. |"
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
