#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, SSH_HOST, SSH_KEY, SSH_PORT, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/baseline_delivery"
FILES_TO_SYNC = (
    "run_eval_pipeline.py",
    "baseline_processors.py",
    "baseline_result_validator.py",
    "baseline_manifest_tools.py",
    "baseline_delivery_runner.py",
)


def _scp_cmd(local_path: Path, remote_path: str) -> list[str]:
    return [
        "scp",
        "-i",
        SSH_KEY,
        "-P",
        SSH_PORT,
        "-o",
        "StrictHostKeyChecking=no",
        str(local_path),
        f"{SSH_HOST}:{remote_path}",
    ]


def sync_files() -> None:
    run_remote_bash(
        f"set -euo pipefail\nmkdir -p {REMOTE_PROJECT} {REMOTE_LOG_DIR}\n",
        timeout=120,
        check=True,
    )
    root = Path(__file__).resolve().parent
    for relative_path in FILES_TO_SYNC:
        local_path = root / relative_path
        proc = subprocess.run(
            _scp_cmd(local_path, f"{REMOTE_PROJECT}/{relative_path}"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to sync {relative_path}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def build_launch_script(*, log_path: str, stop_old_queues: bool) -> str:
    stop_block = ""
    if stop_old_queues:
        stop_block = """
pkill -f 'baseline_full_matrix' >/dev/null 2>&1 || true
pkill -f 'baseline_delivery_runner.py' >/dev/null 2>&1 || true
"""
    return f"""set -euo pipefail
cd {REMOTE_PROJECT}
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p {REMOTE_LOG_DIR}
{stop_block}
nohup {REMOTE_PYTHON} {REMOTE_PROJECT}/baseline_delivery_runner.py >> "{log_path}" 2>&1 < /dev/null &
echo "LAUNCH_PID=$!"
echo "LOG={log_path}"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync and launch the grouped baseline delivery runner on remote GPU0.")
    parser.add_argument("--mode", choices=["sync", "launch"], default="launch")
    parser.add_argument("--no-stop-old-queues", action="store_true")
    args = parser.parse_args()

    sync_files()
    if args.mode == "sync":
        print(json.dumps({"synced_files": list(FILES_TO_SYNC), "remote_project": REMOTE_PROJECT}, ensure_ascii=False, indent=2))
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"{REMOTE_LOG_DIR}/baseline_delivery_gpu0_{ts}.log"
    proc = run_remote_bash(
        build_launch_script(log_path=log_path, stop_old_queues=not args.no_stop_old_queues),
        timeout=120,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print(json.dumps({"remote_log": log_path}, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
