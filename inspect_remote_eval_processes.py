#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from launch_pope_chair_main_table import run_remote_bash


SCRIPT = """set -e
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ps -eo pid,etimes,%cpu,%mem,cmd
"""


def main() -> int:
    proc = run_remote_bash(SCRIPT, timeout=120, check=False)
    lines = proc.stdout.splitlines()
    keys = ("run_eval_pipeline.py", "run_uniground_v2_eval.py", "probe_psi_univ_v2_effect.py", "Qwen3", "python")
    filtered = [line for line in lines if any(key in line for key in keys)]
    json.dump(
        {
            "returncode": proc.returncode,
            "matches": filtered,
            "stderr": proc.stderr,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
