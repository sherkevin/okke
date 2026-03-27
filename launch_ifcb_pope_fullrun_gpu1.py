#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/pope_ifcb_gpu1"
REMOTE_JSON_DIR = f"{REMOTE_PROJECT}/logs/pope_ifcb_gpu1/results"

MODEL = "llava-v1.5-7b"
METHOD = "ifcb"
DATASET = "pope"
SPLITS = ("random", "popular", "adversarial")
POPE_COUNT = 3000


def output_json_path(run_id: str, split: str) -> str:
    return f"{REMOTE_JSON_DIR}/{METHOD}_{DATASET}_{split}_{run_id}.json"


def eval_command(*, split: str, run_id: str, checkpoint_every: int) -> tuple[str, str]:
    output_json = output_json_path(run_id, split)
    parts = [
        REMOTE_PYTHON,
        f"{REMOTE_PROJECT}/run_eval_pipeline.py",
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--method",
        METHOD,
        "--mini_test",
        str(POPE_COUNT),
        "--pope_split",
        split,
        "--run_id",
        run_id,
        "--output_json",
        output_json,
        "--checkpoint_every",
        str(checkpoint_every),
    ]
    return " ".join(shlex.quote(part) for part in parts), output_json


def build_runner_body(
    *,
    matrix_log: str,
    manifest_tsv: str,
    run_id: str,
    checkpoint_every: int,
    gpu_id: int,
) -> str:
    lines: list[str] = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {REMOTE_PROJECT}",
        "source /etc/network_turbo >/dev/null 2>&1 || true",
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "export PYTHONUNBUFFERED=1",
        f"export CUDA_VISIBLE_DEVICES={gpu_id}",
        f'MATRIX_LOG="{matrix_log}"',
        f'MANIFEST="{manifest_tsv}"',
        f'RUN_ID="{run_id}"',
        f'RESULT_DIR="{REMOTE_JSON_DIR}"',
        'mkdir -p "$RESULT_DIR"',
        'echo "[MATRIX] pope_ifcb_gpu1 start $(date -Iseconds)"',
        'echo "[MATRIX] parent_log=${MATRIX_LOG}"',
        'printf "iso_time\\tpope_split\\tmethod\\texit_code\\toutput_json\\tjson_exists\\tmatrix_parent_log\\n" > "$MANIFEST"',
    ]
    for split in SPLITS:
        cmd, output_json = eval_command(split=split, run_id=run_id, checkpoint_every=checkpoint_every)
        tag = f"{MODEL}__{split}__{METHOD}"
        lines.append(f'echo "[START] {tag} $(date -Iseconds) -> {output_json}"')
        lines.append(cmd)
        lines.append("rc=$?")
        lines.append(f'json_path="{output_json}"')
        lines.append('if [ -f "$json_path" ]; then json_exists=1; else json_exists=0; fi')
        lines.append(
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{split}" "{METHOD}" '
            '"$rc" "$json_path" "$json_exists" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'echo "[DONE] {tag} rc=$rc json=$json_path exists=$json_exists $(date -Iseconds)"')
        lines.append('if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
    lines.append('echo "[MATRIX] pope_ifcb_gpu1 finished $(date -Iseconds)"')
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
        f"mkdir -p {REMOTE_JSON_DIR}\n"
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
    parser = argparse.ArgumentParser(description="Launch LLaVA + IFCB + POPE full run on remote GPU1.")
    parser.add_argument("--mode", choices=["dry-run", "render", "launch"], default="dry-run")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--run-id", default=None, help="Stable suffix used in deterministic per-split output JSON paths.")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"llava_ifcb_gpu{args.gpu_id}_{ts}"
    matrix_log = f"{REMOTE_LOG_DIR}/pope_ifcb_gpu{args.gpu_id}_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/pope_ifcb_gpu{args.gpu_id}_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/pope_ifcb_gpu{args.gpu_id}_{ts}.manifest.tsv"

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "model": MODEL,
                    "method": METHOD,
                    "dataset": DATASET,
                    "gpu_id": args.gpu_id,
                    "run_id": run_id,
                    "checkpoint_every": args.checkpoint_every,
                    "splits": list(SPLITS),
                    "expected_outputs": {split: output_json_path(run_id, split) for split in SPLITS},
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

    body = build_runner_body(
        matrix_log=matrix_log,
        manifest_tsv=manifest_tsv,
        run_id=run_id,
        checkpoint_every=args.checkpoint_every,
        gpu_id=args.gpu_id,
    )

    local_backup_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_backup_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_backup_dir / f"pope_ifcb_gpu{args.gpu_id}_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")
    print(f"Wrote local runner copy: {local_copy}")

    if args.mode == "render":
        print(body)
        return 0

    script = remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log)
    proc = run_remote_bash(script, timeout=300, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\nExpected result JSONs:")
    for split in SPLITS:
        print(output_json_path(run_id, split))
    print("\nManifest:")
    print(manifest_tsv)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
