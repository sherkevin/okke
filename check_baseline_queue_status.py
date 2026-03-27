#!/usr/bin/env python3
from __future__ import annotations

import sys

from launch_pope_chair_main_table import run_remote_bash


SCRIPT = r"""set -euo pipefail
PY=/root/miniconda3/bin/python
POPE_LOG=/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.log
POPE_MAN=/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.manifest.tsv
POPE_GUARD=/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.guard.log
CHAIR_LOG=/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.log
CHAIR_MAN=/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.manifest.tsv
CHAIR_WATCH=/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.watch.log

echo __PS__
ps -eo pid,etimes,cmd | grep -E 'pope_baseline_pending_gpu0_20260323_202913|chair_baseline_pending_gpu0_20260324_101840' | grep -v grep || true

echo __POPE_MANIFEST__
if [ -f "$POPE_MAN" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.manifest.tsv')
lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
print('line_count=', len(lines))
for line in lines[-8:]:
    print(line)
PYEOF
else echo MISSING; fi

echo __POPE_GUARD__
if [ -f "$POPE_GUARD" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.guard.log')
for line in p.read_text(encoding='utf-8', errors='replace').splitlines()[-20:]:
    print(line)
PYEOF
else echo MISSING; fi

echo __POPE_LOG_TAIL__
if [ -f "$POPE_LOG" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.log')
for line in p.read_text(encoding='utf-8', errors='replace').splitlines()[-20:]:
    print(line)
PYEOF
else echo MISSING; fi

echo __CHAIR_WATCH__
if [ -f "$CHAIR_WATCH" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.watch.log')
for line in p.read_text(encoding='utf-8', errors='replace').splitlines()[-20:]:
    print(line)
PYEOF
else echo MISSING; fi

echo __CHAIR_MANIFEST__
if [ -f "$CHAIR_MAN" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.manifest.tsv')
lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
print('line_count=', len(lines))
for line in lines[-8:]:
    print(line)
PYEOF
else echo MISSING; fi

echo __CHAIR_LOG_TAIL__
if [ -f "$CHAIR_LOG" ]; then "$PY" - <<'PYEOF'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.log')
for line in p.read_text(encoding='utf-8', errors='replace').splitlines()[-20:]:
    print(line)
PYEOF
else echo MISSING; fi
"""


def main() -> int:
    proc = run_remote_bash(SCRIPT, timeout=120, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
