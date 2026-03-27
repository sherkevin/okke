#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from launch_pope_chair_main_table import run_remote_bash


GPU_AND_PROCESS_SCRIPT = """set -e
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ====
ps -eo pid,etimes,cmd | /root/miniconda3/bin/python -c "import sys; keys=['export_uniground_v2_coco_features.py','inspect_uniground_v2_payload.py','train_universal_plugin_v2.py']; [print(line.rstrip()) for line in sys.stdin if any(k in line for k in keys) and 'python -c' not in line]"
"""


LOG_SCRIPT = """set -e
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ls -lah /root/autodl-tmp/BRA_Project/logs/uniground_v2_fullrun
echo ====
if [ -f /root/autodl-tmp/BRA_Project/logs/uniground_v2_fullrun/full_export_train.log ]; then
  /root/miniconda3/bin/python - <<'PY'
from pathlib import Path
p = Path('/root/autodl-tmp/BRA_Project/logs/uniground_v2_fullrun/full_export_train.log')
lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
for line in lines[-120:]:
    print(line)
PY
else
  echo MISSING_LOG
fi
"""


def parse_elapsed_seconds(stdout: str) -> int | None:
    for line in stdout.splitlines():
        if "export_uniground_v2_coco_features.py" in line or "inspect_uniground_v2_payload.py" in line or "train_universal_plugin_v2.py" in line:
            match = re.match(r"\s*\d+\s+(\d+)\s+", line)
            if match:
                return int(match.group(1))
    return None


def main() -> int:
    proc_a = run_remote_bash(GPU_AND_PROCESS_SCRIPT, timeout=120, check=False)
    proc_b = run_remote_bash(LOG_SCRIPT, timeout=120, check=False)
    payload = {
        "gpu_and_process_returncode": proc_a.returncode,
        "gpu_and_process_stdout": proc_a.stdout,
        "gpu_and_process_stderr": proc_a.stderr,
        "log_returncode": proc_b.returncode,
        "log_stdout": proc_b.stdout,
        "log_stderr": proc_b.stderr,
        "elapsed_seconds": parse_elapsed_seconds(proc_a.stdout),
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0 if proc_a.returncode == 0 and proc_b.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
