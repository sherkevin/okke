#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix_resume/baseline_full_matrix_resume_gpu0_20260326_102441.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix_resume/baseline_full_matrix_resume_gpu0_20260326_102441.manifest.tsv"
echo "[MATRIX] baseline_full_matrix_resume_gpu0 start $(date -Iseconds)"
printf "iso_time\tmodel\tdataset\tsplit\tmethod\taction\texit_code\tresult_json\tmatrix_parent_log\n" > "$MANIFEST"

is_final_json() {
  local path="$1"
  "/root/miniconda3/bin/python" - "$path" <<'PYEOF'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(2)
status = payload.get('status')
sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)
n_errors = int(payload.get('n_errors', 0) or 0)
target = int(payload.get('target_samples', sample_count) or sample_count)
if status == 'final' and n_errors == 0 and sample_count == target:
    print('FINAL_OK')
    raise SystemExit(0)
raise SystemExit(3)
PYEOF
}

echo "[CHECK] qwen3-vl-8b__pope__random__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__random__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__random__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__random__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__random__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__random__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__random__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__random__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__random__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__random__dola $(date -Iseconds)"
echo "[SKIP] qwen3-vl-8b__pope__random__dola reason=skip_known_issue $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "dola" "skip_known_issue" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"

echo "[CHECK] qwen3-vl-8b__pope__popular__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__popular__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__popular__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__popular__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__popular__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__popular__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__pope__adversarial__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__pope__adversarial__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__pope__adversarial__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__mmbench__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__mmbench__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__mmbench__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__chair__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__chair__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__chair__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen3-vl-8b__chair__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen3-vl-8b__chair__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen3-vl-8b__chair__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen3-vl-8b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__random__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__random__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__random__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__random__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__random__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__random__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__random__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__random__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__random__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__random__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__random__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__random__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__popular__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__popular__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__popular__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__popular__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__popular__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__popular__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__popular__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__popular__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__popular__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__popular__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__popular__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__popular__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__adversarial__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__adversarial__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__adversarial__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__adversarial__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__adversarial__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__adversarial__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__adversarial__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__pope__adversarial__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__pope__adversarial__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__pope__adversarial__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__mmbench__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__mmbench__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__mmbench__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__mmbench__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__mmbench__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__mmbench__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__mmbench__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__mmbench__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__mmbench__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__mmbench__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__mmbench__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__mmbench__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__chair__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__chair__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__chair__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__chair__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__chair__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__chair__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__chair__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__chair__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__chair__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2-vl-7b__chair__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2-vl-7b__chair__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2-vl-7b__chair__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2-vl-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__random__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__random__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__random__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__random__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__random__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__random__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__random__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__random__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__random__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__random__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__random__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__random__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__popular__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__popular__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__popular__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__popular__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__popular__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__popular__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__popular__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__popular__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__popular__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__popular__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__popular__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__popular__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__adversarial__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__adversarial__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__adversarial__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__adversarial__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__adversarial__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__adversarial__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__adversarial__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__pope__adversarial__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__pope__adversarial__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__pope__adversarial__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__mmbench__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__mmbench__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__mmbench__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__mmbench__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__mmbench__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__mmbench__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__mmbench__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__mmbench__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__mmbench__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__mmbench__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__mmbench__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__mmbench__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__chair__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__chair__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__chair__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__chair__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__chair__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__chair__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__chair__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__chair__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__chair__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] qwen2.5-vl-7b__chair__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] qwen2.5-vl-7b__chair__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] qwen2.5-vl-7b__chair__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] qwen2.5-vl-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__random__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__random__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__random__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__random__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__random__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__random__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__popular__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__popular__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__popular__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__popular__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__popular__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__popular__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__adversarial__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__adversarial__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__adversarial__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__pope__adversarial__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__mmbench__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__mmbench__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__mmbench__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__mmbench__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__chair__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__chair__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__chair__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] instructblip-7b__chair__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] instructblip-7b__chair__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] instructblip-7b__chair__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] instructblip-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__random__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__random__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__random__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__random__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__random__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__random__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json --pope_split random
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__popular__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__popular__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__popular__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json --pope_split popular
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__pope__adversarial__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__pope__adversarial__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__pope__adversarial__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json --pope_split adversarial
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__mmbench__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__chair__default__base reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__chair__default__opera reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__chair__default__vcd reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[CHECK] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
if is_final_json "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json" >/dev/null 2>&1; then
  echo "[SKIP] llava-v1.5-7b__chair__default__dola reason=already_final $(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "skip_existing_final" "0" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
else
  echo "[START] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
  mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
  /root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json --chair_max_new_tokens 384
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "executed" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
  echo "[DONE] llava-v1.5-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json $(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
fi

echo "[MATRIX] baseline_full_matrix_resume_gpu0 finished $(date -Iseconds)"
