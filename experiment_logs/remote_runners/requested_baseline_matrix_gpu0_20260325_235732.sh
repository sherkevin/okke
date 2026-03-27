#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/requested_baseline_matrix/requested_baseline_matrix_gpu0_20260325_235732.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/requested_baseline_matrix/requested_baseline_matrix_gpu0_20260325_235732.manifest.tsv"
MINITEST_DIR="/root/autodl-tmp/BRA_Project/logs/minitest"
echo "[MATRIX] requested_baseline_matrix_gpu0 start $(date -Iseconds)"
printf "iso_time\tmodel\tdataset\tsplit\tmethod\ttarget_samples\taction\texit_code\tresult_json\tmatrix_parent_log\n" > "$MANIFEST"

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
candidates = sorted(root.glob('*.json'), key=lambda p: p.name)
matched = []
for path in candidates:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('model') != model or payload.get('dataset') != dataset or payload.get('method') != method:
        continue
    if dataset == 'pope' and payload.get('pope_split') != split:
        continue
    sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)
    n_errors = int(payload.get('n_errors', 0) or 0)
    status = payload.get('status')
    if sample_count == target and n_errors == 0 and status in (None, 'final'):
        matched.append(str(path))
if matched:
    print(matched[-1])
PYEOF
}

echo "[CHECK] qwen3-vl-8b__pope__random__vcd $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "random" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__random__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__random__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "random" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__random__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__popular__opera $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "popular" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "opera" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "popular" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "opera" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__popular__vcd $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "popular" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "popular" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__base $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "base" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "base" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__opera $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "opera" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "opera" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__vcd $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__dola $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "dola" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "pope" "adversarial" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "dola" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__base $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "base" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "base" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method base --mini_test 4377
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "base" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "base" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__opera $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "opera" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "opera" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method opera --mini_test 4377
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "opera" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "opera" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "vcd" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "vcd" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method vcd --mini_test 4377
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "vcd" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "vcd" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__dola $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "dola" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "dola" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method dola --mini_test 4377
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "mmbench" "default" "dola" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "dola" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__base $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "chair" "default" "base" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__chair__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "base" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "chair" "default" "base" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "base" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__opera $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "chair" "default" "opera" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__chair__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "opera" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method opera --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "chair" "default" "opera" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "opera" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "chair" "default" "vcd" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__chair__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "vcd" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "chair" "default" "vcd" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "vcd" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__dola $(date -Iseconds)"
existing=$(find_completed_json "qwen3-vl-8b" "chair" "default" "dola" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] qwen3-vl-8b__chair__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "dola" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "qwen3-vl-8b" "chair" "default" "dola" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "dola" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__vcd $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "random" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__random__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split random
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "random" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "popular" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__popular__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "popular" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "base" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "base" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "opera" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "opera" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "pope" "adversarial" "dola" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__pope__adversarial__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "instructblip-7b" "pope" "adversarial" "dola" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "base" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method base --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "base" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "opera" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method opera --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "opera" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "vcd" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "vcd" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method vcd --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "vcd" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "vcd" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "mmbench" "default" "dola" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__mmbench__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method dola --mini_test 4377
  rc=$?
  result=$(find_completed_json "instructblip-7b" "mmbench" "default" "dola" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__base $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "base" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method base --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "base" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__opera $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "opera" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method opera --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "opera" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "vcd" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "vcd" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method vcd --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "vcd" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "vcd" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__dola $(date -Iseconds)"
existing=$(find_completed_json "instructblip-7b" "chair" "default" "dola" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] instructblip-7b__chair__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method dola --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "instructblip-7b" "chair" "default" "dola" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --pope_split popular
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "popular" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "vcd" "3000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "3000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --pope_split adversarial
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "pope" "adversarial" "vcd" "3000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "3000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "base" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method base --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "base" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "opera" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method opera --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "opera" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "vcd" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "vcd" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method vcd --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "vcd" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "vcd" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "dola" "4377")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "4377" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method dola --mini_test 4377
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "mmbench" "default" "dola" "4377")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "4377" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "base" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__base existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "base" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__base rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "opera" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__opera existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method opera --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "opera" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__opera rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "vcd" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__vcd existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method vcd --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "vcd" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__vcd rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
existing=$(find_completed_json "llava-v1.5-7b" "chair" "default" "dola" "5000")
if [ -n "$existing" ]; then
  echo "[SKIP] llava-v1.5-7b__chair__default__dola existing=$existing $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "5000" "skip_existing" "0" "$existing" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method dola --mini_test 5000 --chair_max_new_tokens 384
  rc=$?
  result=$(find_completed_json "llava-v1.5-7b" "chair" "default" "dola" "5000")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "5000" "executed" "$rc" "$result" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__dola rc=$rc result=$result $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[MATRIX] requested_baseline_matrix_gpu0 finished $(date -Iseconds)"
