#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import sys

from launch_pope_chair_main_table import run_remote_bash


def build_remote_script(*, parent_pid: int, parent_log: str, manifest_tsv: str, target_completed: int) -> str:
    guard_log = parent_log.replace(".log", ".guard.log")
    guard_sh = parent_log.replace(".log", ".guard.sh")
    delim = "GUARD_" + secrets.token_hex(16)
    body = f"""#!/bin/bash
set -euo pipefail
PARENT={parent_pid}
LOG={parent_log}
MAN={manifest_tsv}
GUARD={guard_log}
TARGET_COMPLETED={target_completed}
printf '[GUARD] installed %s\\n' "$(date -Iseconds)" >> "$GUARD"
while kill -0 "$PARENT" 2>/dev/null; do
  lines=$(wc -l < "$MAN" 2>/dev/null || echo 0)
  completed=$(( lines > 0 ? lines - 1 : 0 ))
  if ps -eo pid,cmd | grep 'run_eval_pipeline.py --model qwen3-vl-4b' | grep -v grep >/tmp/pope_guard_child.txt 2>/dev/null; then
    child_pid=$(awk 'NR==1{{print $1}}' /tmp/pope_guard_child.txt)
    printf '[GUARD] qwen3-vl-4b started unexpectedly; killing child=%s parent=%s at %s\\n' "$child_pid" "$PARENT" "$(date -Iseconds)" | tee -a "$GUARD" >> "$LOG"
    kill "$child_pid" 2>/dev/null || true
    kill "$PARENT" 2>/dev/null || true
    exit 0
  fi
  if ps -eo pid,cmd | grep 'run_eval_pipeline.py --model qwen3-vl-2b' | grep -v grep >/tmp/pope_guard_child.txt 2>/dev/null; then
    child_pid=$(awk 'NR==1{{print $1}}' /tmp/pope_guard_child.txt)
    printf '[GUARD] qwen3-vl-2b started unexpectedly; killing child=%s parent=%s at %s\\n' "$child_pid" "$PARENT" "$(date -Iseconds)" | tee -a "$GUARD" >> "$LOG"
    kill "$child_pid" 2>/dev/null || true
    kill "$PARENT" 2>/dev/null || true
    exit 0
  fi
  if [ "$completed" -ge "$TARGET_COMPLETED" ]; then
    printf '[GUARD] desired scope finished (completed=%s); stopping parent=%s at %s\\n' "$completed" "$PARENT" "$(date -Iseconds)" | tee -a "$GUARD" >> "$LOG"
    kill "$PARENT" 2>/dev/null || true
    exit 0
  fi
  sleep 5
done
printf '[GUARD] parent already exited at %s\\n' "$(date -Iseconds)" >> "$GUARD"
"""
    if delim in body:
        raise RuntimeError("heredoc delimiter collision")
    return (
        "set -euo pipefail\n"
        f"cat > \"{guard_sh}\" <<'{delim}'\n"
        f"{body}"
        f"{delim}\n"
        f"chmod +x \"{guard_sh}\"\n"
        f"nohup /bin/bash \"{guard_sh}\" > /dev/null 2>&1 < /dev/null &\n"
        "echo \"GUARD_PID=$!\"\n"
        f"echo \"GUARD_SCRIPT={guard_sh}\"\n"
        f"echo \"GUARD_LOG={guard_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a remote guard to stop the POPE baseline queue before excluded models start.")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-log", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-completed", type=int, default=23)
    args = parser.parse_args()

    proc = run_remote_bash(
        build_remote_script(
            parent_pid=args.parent_pid,
            parent_log=args.parent_log,
            manifest_tsv=args.manifest,
            target_completed=args.target_completed,
        ),
        timeout=120,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
