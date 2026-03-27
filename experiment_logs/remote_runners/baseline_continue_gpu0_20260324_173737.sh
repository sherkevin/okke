#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173737.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/baseline_continue/baseline_continue_gpu0_20260324_173737.manifest.tsv"
MINITEST_DIR="/root/autodl-tmp/BRA_Project/logs/minitest"
echo "[MATRIX] baseline_continue_gpu0 start $(date -Iseconds)"
printf "iso_time\tmodel\tdataset\tsplit\tmethod\taction\texit_code\tresult_json\tmatrix_parent_log\n" > "$MANIFEST"

find_completed_json() {
  local model="$1"
  local dataset="$2"
  local split="$3"
  local method="$4"
  local target="$5"
  "/root/miniconda3/bin/python" - "$model" "$dataset" "$split" "$method" "$target" "$MINITEST_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

model, dataset, split, method, target, log_dir = sys.argv[1:]
target = int(target)
root = Path(log_dir)
candidates = sorted(root.glob(f"{method}_{dataset}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
for path in candidates:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('model') != model:
        continue
    if payload.get('dataset') != dataset:
        continue
    if payload.get('method') != method:
        continue
    if dataset == 'pope' and payload.get('pope_split') != split:
        continue
    sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)
    n_errors = int(payload.get('n_errors', 0) or 0)
    status = payload.get('status')
    if sample_count != target or n_errors != 0:
        continue
    if status not in (None, 'final'):
        continue
    print(str(path))
    break
PYEOF
}

echo "[CHECK] llava-v1.5-7b__pope__random__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "random" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__random__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "random" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "random" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__random__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "random" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "random" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__random__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "random" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "random" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__random__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "random" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "random" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__random__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "random" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "random" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__random__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "random" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "base" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "base" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "dola" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method dola --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "dola" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "opera" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method opera --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "opera" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "base" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "base" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "dola" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method dola --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "dola" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "opera" "40504")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method opera --mini_test 40504 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "opera" "40504")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "base" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method base --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "base" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "dola" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method dola --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "dola" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "opera" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method opera --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "opera" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "base" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method base --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "base" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "dola" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method dola --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "dola" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "opera" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method opera --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "opera" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[MATRIX] baseline_continue_gpu0 finished $(date -Iseconds)"
