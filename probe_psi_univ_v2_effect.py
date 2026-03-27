#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys

from launch_pope_chair_main_table import run_remote_bash


REMOTE_PROJECT = "/root/autodl-tmp/BRA_Project"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_ENCODER = f"{REMOTE_PROJECT}/models/clip-vit-large-patch14"
REMOTE_PSI = f"{REMOTE_PROJECT}/logs/uniground_v2_fullrun/psi_univ_v2_train2014_full.pt"
REMOTE_OUTPUT = f"{REMOTE_PROJECT}/logs/uniground_v2_probe"


def build_script(model: str, pope_count: int, chair_count: int, gpu: int) -> str:
    return f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd {REMOTE_PROJECT}
mkdir -p {REMOTE_OUTPUT}
export CUDA_VISIBLE_DEVICES={gpu}

{REMOTE_PYTHON} run_eval_pipeline.py --model {model} --dataset pope --method base --mini_test {pope_count} --pope_split random
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
paths = sorted(Path('/root/autodl-tmp/BRA_Project/logs/minitest').glob('base_pope_*.json'), key=lambda p: p.stat().st_mtime)
print(f'BASE_POPE_JSON={{paths[-1]}}')
PY

{REMOTE_PYTHON} run_uniground_v2_eval.py --model {model} --dataset pope --mini_test {pope_count} --pope_split random --psi_mode checkpoint --psi_checkpoint {REMOTE_PSI} --external_encoder {REMOTE_ENCODER} --external_device cpu --output_dir {REMOTE_OUTPUT} --run_tag checkpoint_probe
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
paths = sorted(Path('/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe').glob('uniground_v2_*_pope_checkpoint_probe_*.json'), key=lambda p: p.stat().st_mtime)
print(f'V2_POPE_JSON={{paths[-1]}}')
PY

{REMOTE_PYTHON} run_eval_pipeline.py --model {model} --dataset chair --method base --mini_test {chair_count} --chair_max_new_tokens 384
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
paths = sorted(Path('/root/autodl-tmp/BRA_Project/logs/minitest').glob('base_chair_*.json'), key=lambda p: p.stat().st_mtime)
print(f'BASE_CHAIR_JSON={{paths[-1]}}')
PY

{REMOTE_PYTHON} run_uniground_v2_eval.py --model {model} --dataset chair --mini_test {chair_count} --max_new_tokens 384 --psi_mode checkpoint --psi_checkpoint {REMOTE_PSI} --external_encoder {REMOTE_ENCODER} --external_device cpu --output_dir {REMOTE_OUTPUT} --run_tag checkpoint_probe
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
paths = sorted(Path('/root/autodl-tmp/BRA_Project/logs/uniground_v2_probe').glob('uniground_v2_*_chair_checkpoint_probe_*.json'), key=lambda p: p.stat().st_mtime)
print(f'V2_CHAIR_JSON={{paths[-1]}}')
PY
"""


def extract_paths(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        match = re.match(r"^(BASE_POPE_JSON|V2_POPE_JSON|BASE_CHAIR_JSON|V2_CHAIR_JSON)=(.+)$", line.strip())
        if match:
            out[match.group(1)] = match.group(2)
    return out


def build_summary_script(paths: dict[str, str]) -> str:
    return f"""set -euo pipefail
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
import json
paths = {paths!r}
payload = {{}}
for key, path in paths.items():
    with open(path, 'r', encoding='utf-8') as f:
        payload[key] = json.load(f)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl-8b", choices=["qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b"])
    parser.add_argument("--pope-count", type=int, default=100)
    parser.add_argument("--chair-count", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_proc = run_remote_bash(build_script(args.model, args.pope_count, args.chair_count, args.gpu), timeout=7200, check=False)
    sys.stdout.write(run_proc.stdout)
    sys.stderr.write(run_proc.stderr)
    if run_proc.returncode != 0:
        return run_proc.returncode

    paths = extract_paths(run_proc.stdout)
    required = {"BASE_POPE_JSON", "V2_POPE_JSON", "BASE_CHAIR_JSON", "V2_CHAIR_JSON"}
    if set(paths) != required:
        print(f"Missing result paths: expected {required}, got {set(paths)}", file=sys.stderr)
        return 1

    summary_proc = run_remote_bash(build_summary_script(paths), timeout=120, check=False)
    sys.stdout.write(summary_proc.stdout)
    sys.stderr.write(summary_proc.stderr)
    return summary_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
