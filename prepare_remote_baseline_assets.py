#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

KEEP_MODEL_DIRS = {
    "Qwen3-VL-8B-Instruct",
    "Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "llava-1.5-7b-hf",
    "instructblip-vicuna-7b",
}
KEEP_DATASETS = {"POPE", "MMBench_EN_hf", "coco2014"}


def build_remote_script(*, apply: bool) -> str:
    action = "True" if apply else "False"
    return f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
from pathlib import Path
import json
import shutil
import urllib.request

APPLY = {action}
PROJECT = Path("{REMOTE_PROJECT}")
MODELS_DIR = PROJECT / "models"
DATASETS_DIR = PROJECT / "datasets"
KEEP_MODEL_DIRS = {sorted(KEEP_MODEL_DIRS)!r}
KEEP_DATASETS = {sorted(KEEP_DATASETS)!r}
COCO_DIR = DATASETS_DIR / "coco2014"
VAL_DIR = COCO_DIR / "val2014"
ANN_DIR = COCO_DIR / "annotations"
POPE_DIR = DATASETS_DIR / "POPE" / "output" / "coco"
KARPATHY_FILE = ANN_DIR / "coco_karpathy_test.json"
KARPATHY_URL = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"

report = {{
    "apply": APPLY,
    "deleted_models": [],
    "deleted_datasets": [],
    "deleted_coco_annotation_files": [],
    "deleted_coco_images": [],
    "kept_model_dirs": sorted(KEEP_MODEL_DIRS),
    "kept_datasets": sorted(KEEP_DATASETS),
}}

def remove_path(path: Path):
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)

if MODELS_DIR.exists():
    for path in sorted(MODELS_DIR.iterdir()):
        if path.name in KEEP_MODEL_DIRS:
            continue
        report["deleted_models"].append(path.name)
        if APPLY:
            remove_path(path)

if DATASETS_DIR.exists():
    for path in sorted(DATASETS_DIR.iterdir()):
        if path.name in KEEP_DATASETS:
            continue
        report["deleted_datasets"].append(path.name)
        if APPLY:
            remove_path(path)

if not KARPATHY_FILE.exists():
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(KARPATHY_URL, KARPATHY_FILE)

keep_images = set()
for split in ("random", "popular", "adversarial"):
    split_file = POPE_DIR / f"coco_pope_{{split}}.json"
    if not split_file.exists():
        continue
    with open(split_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_name = str(row.get("image", "")).strip()
            if image_name:
                keep_images.add(Path(image_name).name)

with open(KARPATHY_FILE, encoding="utf-8") as fh:
    karpathy = json.load(fh)
for entry in karpathy:
    rel = str(entry.get("image", "")).strip()
    if not rel:
        continue
    keep_images.add(Path(rel).name)
    if len(keep_images) >= 5000 + 3000:
        # soft guard only; union size can exceed this because POPE and CHAIR overlap unpredictably
        pass

keep_ann = {{"instances_val2014.json", "captions_val2014.json", "coco_karpathy_test.json"}}
if ANN_DIR.exists():
    for path in sorted(ANN_DIR.iterdir()):
        if path.name in keep_ann:
            continue
        report["deleted_coco_annotation_files"].append(path.name)
        if APPLY:
            remove_path(path)

if VAL_DIR.exists():
    for path in sorted(VAL_DIR.iterdir()):
        if path.name in keep_images:
            continue
        report["deleted_coco_images"].append(path.name)
        if APPLY:
            remove_path(path)

report["kept_image_count"] = len(keep_images)
report["remaining_model_entries"] = sorted([p.name for p in MODELS_DIR.iterdir()]) if MODELS_DIR.exists() else []
report["remaining_dataset_entries"] = sorted([p.name for p in DATASETS_DIR.iterdir()]) if DATASETS_DIR.exists() else []
report["remaining_coco_val_count"] = len(list(VAL_DIR.glob("*.jpg"))) if VAL_DIR.exists() else 0
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune remote models and datasets down to the baseline-only set.")
    parser.add_argument("--mode", choices=["plan", "apply"], default="plan")
    args = parser.parse_args()

    proc = run_remote_bash(build_remote_script(apply=args.mode == "apply"), timeout=7200, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
