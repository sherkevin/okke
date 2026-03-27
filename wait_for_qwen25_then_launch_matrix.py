#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from launch_pope_chair_main_table import run_remote_bash


REMOTE_CHECK_SCRIPT = """set -euo pipefail
if [ -f /root/autodl-tmp/BRA_Project/models/Qwen2.5-VL-7B-Instruct/config.json ]; then
  echo READY
else
  echo WAITING
fi
"""


def remote_ready() -> bool:
    proc = run_remote_bash(REMOTE_CHECK_SCRIPT, timeout=120, check=False)
    return proc.returncode == 0 and "READY" in proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for remote Qwen2.5-VL-7B, then launch the full baseline matrix.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    launch_script = root / "launch_full_baseline_matrix_gpu0.py"

    while True:
        if remote_ready():
            break
        print("WAITING_FOR_QWEN25")
        sys.stdout.flush()
        time.sleep(args.poll_seconds)

    print("QWEN25_READY")
    sys.stdout.flush()
    proc = subprocess.run(
        [sys.executable, str(launch_script), "--mode", "launch"],
        cwd=str(root),
        text=True,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
