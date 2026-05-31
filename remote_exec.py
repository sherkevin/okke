#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from launch_pope_chair_main_table import run_remote_bash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a bash script on the remote server via SSH.")
    parser.add_argument("--script-file", help="Local path to a bash script. If omitted, read from stdin.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--check", action="store_true", help="Raise on non-zero exit code.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.script_file:
        script = Path(args.script_file).read_text(encoding="utf-8")
    else:
        script = sys.stdin.read()
    proc = run_remote_bash(script, timeout=args.timeout, check=args.check)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
