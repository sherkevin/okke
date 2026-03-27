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
summary = {
    "coco_val_images": len(list(Path('/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014').glob('*.jpg'))),
    "pope_random": sum(1 for _ in open('/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/coco_pope_random.json', encoding='utf-8')),
    "pope_popular": sum(1 for _ in open('/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/coco_pope_popular.json', encoding='utf-8')),
    "pope_adversarial": sum(1 for _ in open('/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/coco_pope_adversarial.json', encoding='utf-8')),
    "qwen3_vl_8b_exists": Path('/root/autodl-tmp/BRA_Project/models/Qwen3-VL-8B-Instruct').exists(),
    "qwen3_vl_4b_exists": Path('/root/autodl-tmp/BRA_Project/models/Qwen3-VL-4B-Instruct').exists(),
    "qwen3_vl_2b_exists": Path('/root/autodl-tmp/BRA_Project/models/Qwen3-VL-2B-Instruct').exists(),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    proc = run_remote_bash(SCRIPT, timeout=120, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
