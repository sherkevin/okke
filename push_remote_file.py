#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from launch_pope_chair_main_table import run_remote_bash


def build_script(remote_path: str, content: str) -> str:
    delim = "REMOTE_FILE_" + secrets.token_hex(16)
    if delim in content:
        raise RuntimeError("heredoc delimiter collision")
    return (
        "set -euo pipefail\n"
        f"mkdir -p \"$(dirname \"{remote_path}\")\"\n"
        f"cat > \"{remote_path}\" <<'{delim}'\n"
        f"{content}"
        f"{delim}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Push a local file to the remote AutoDL project via SSH heredoc.")
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()

    content = Path(args.local).read_text(encoding="utf-8")
    proc = run_remote_bash(build_script(args.remote, content), timeout=300, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
