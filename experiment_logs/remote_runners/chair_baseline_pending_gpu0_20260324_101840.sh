#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/chair_full_matrix/chair_baseline_pending_gpu0_20260324_101840.manifest.tsv"
MINITEST_DIR="/root/autodl-tmp/BRA_Project/logs/minitest"
echo "[MATRIX] chair_baseline_pending_gpu0 start $(date -Iseconds)"
echo "[MATRIX] parent_log=${MATRIX_LOG}"
printf "iso_time\tmodel\tmethod\texit_code\tlatest_minitest_json_guess\tmatrix_parent_log\n" > "$MANIFEST"
echo "[START] llava-v1.5-7b__chair__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method beam_search --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method dola --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method opera --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__chair__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__chair__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method beam_search --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__chair__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__chair__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method opera --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__chair__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__chair__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__chair__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method beam_search --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__chair__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method dola --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__chair__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset chair --method opera --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__chair__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[MATRIX] chair_baseline_pending_gpu0 finished $(date -Iseconds)"
