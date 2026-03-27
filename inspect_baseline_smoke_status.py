#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from launch_pope_chair_main_table import run_remote_bash


SCRIPT = """set -e
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/autodl-tmp/BRA_Project
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import json

log_dir = Path('/root/autodl-tmp/BRA_Project/logs/baseline_main_table')
latest_log = sorted(log_dir.glob('baseline_fullrun_*.log'), key=lambda p: p.stat().st_mtime)[-1]
latest_runner = sorted(log_dir.glob('baseline_fullrun_*.sh'), key=lambda p: p.stat().st_mtime)[-1]
minitest_dir = Path('/root/autodl-tmp/BRA_Project/logs/minitest')
outputs = sorted(minitest_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)[-12:]

payload = {
    'latest_log': str(latest_log),
    'latest_runner': str(latest_runner),
    'log_tail': latest_log.read_text(encoding='utf-8', errors='replace').splitlines()[-120:],
    'recent_outputs': [str(path) for path in outputs],
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    proc = run_remote_bash(SCRIPT, timeout=120, check=False)
    sys.stdout.buffer.write(proc.stdout.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(proc.stderr.encode("utf-8", errors="replace"))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
