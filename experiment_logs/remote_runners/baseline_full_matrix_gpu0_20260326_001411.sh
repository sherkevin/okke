#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/baseline_full_matrix_gpu0_20260326_001411.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/baseline_full_matrix_gpu0_20260326_001411.manifest.tsv"
RESULT_ROOT="/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results"
echo "[MATRIX] baseline_full_matrix_gpu0 start $(date -Iseconds)"
printf "iso_time\tmodel\tdataset\tsplit\tmethod\texit_code\tresult_json\tmatrix_parent_log\n" > "$MANIFEST"
mkdir -p "$RESULT_ROOT"

echo "[START] qwen3-vl-8b__pope__random__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__random__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__random__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__random__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__dola.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "random" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/random__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__popular__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__popular__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__popular__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__popular__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "popular" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/popular__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__adversarial__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__adversarial__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__adversarial__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__pope__adversarial__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "pope" "adversarial" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/pope/adversarial__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__mmbench__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__mmbench__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__mmbench__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__mmbench__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "mmbench" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/mmbench/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__chair__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__chair__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__chair__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen3-vl-8b__chair__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "chair" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen3-vl-8b/chair/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__random__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__random__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__random__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__random__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "random" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/random__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__popular__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__popular__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__popular__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__popular__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "popular" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/popular__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__adversarial__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__adversarial__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__pope__adversarial__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "pope" "adversarial" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/pope/adversarial__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__mmbench__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__mmbench__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__mmbench__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__mmbench__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "mmbench" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/mmbench/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__chair__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__chair__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__chair__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2-vl-7b__chair__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2-vl-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2-vl-7b" "chair" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2-vl-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2-vl-7b/chair/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__random__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__random__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__random__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__random__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "random" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/random__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__popular__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__popular__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__popular__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__popular__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "popular" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/popular__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__adversarial__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__adversarial__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__adversarial__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__pope__adversarial__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "pope" "adversarial" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/pope/adversarial__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__mmbench__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__mmbench__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__mmbench__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__mmbench__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "mmbench" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/mmbench/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__chair__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__chair__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__chair__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] qwen2.5-vl-7b__chair__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen2.5-vl-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen2.5-vl-7b" "chair" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen2.5-vl-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/qwen2.5-vl-7b/chair/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__random__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__random__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__random__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__random__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/random__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__popular__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__popular__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__popular__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/popular__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__adversarial__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__pope__adversarial__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/pope/adversarial__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__mmbench__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__mmbench__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__mmbench__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__mmbench__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "mmbench" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/mmbench/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__chair__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__chair__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__chair__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] instructblip-7b__chair__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "chair" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/instructblip-7b/chair/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__random__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__random__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__random__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__random__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json --pope_split random
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/random__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__popular__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__popular__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__popular__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json --pope_split popular
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/popular__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__adversarial__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__adversarial__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__pope__adversarial__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method dola --mini_test 3000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json --pope_split adversarial
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/pope/adversarial__dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__mmbench__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method base --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__mmbench__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__mmbench__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method opera --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__mmbench__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__mmbench__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method vcd --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__mmbench__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__mmbench__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset mmbench --method dola --mini_test 4377 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "mmbench" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__mmbench__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/mmbench/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__base rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/base.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__chair__default__opera $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method opera --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "opera" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__opera rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/opera.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method vcd --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__vcd rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/vcd.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[START] llava-v1.5-7b__chair__default__dola $(date -Iseconds)"
mkdir -p "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method dola --mini_test 5000 --output_json /root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json --chair_max_new_tokens 384
rc=$?
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "dola" "$rc" "/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__dola rc=$rc output=/root/autodl-tmp/BRA_Project/logs/baseline_full_matrix/results/llava-v1.5-7b/chair/dola.json $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi

echo "[MATRIX] baseline_full_matrix_gpu0 finished $(date -Iseconds)"
