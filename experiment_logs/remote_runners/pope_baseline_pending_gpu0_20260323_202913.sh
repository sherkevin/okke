#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/pope_baseline_pending_gpu0_20260323_202913.manifest.tsv"
MINITEST_DIR="/root/autodl-tmp/BRA_Project/logs/minitest"
echo "[MATRIX] pope_baseline_pending_gpu0 start $(date -Iseconds)"
echo "[MATRIX] parent_log=${MATRIX_LOG}"
printf "iso_time\tmodel\tpope_split\tmethod\texit_code\tlatest_minitest_json_guess\tmatrix_parent_log\n" > "$MANIFEST"
echo "[START] qwen3-vl-8b__random__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method beam_search --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "random" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__random__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__random__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "random" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__random__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__random__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "random" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__random__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__popular__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "popular" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__popular__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__popular__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method beam_search --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "popular" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__popular__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__popular__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "popular" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__popular__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__popular__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "popular" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__popular__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__adversarial__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method base --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "adversarial" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__adversarial__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__adversarial__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method beam_search --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "adversarial" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__adversarial__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__adversarial__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "adversarial" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__adversarial__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-8b__adversarial__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-8b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-8b" "adversarial" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-8b__adversarial__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__random__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "random" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__random__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__random__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method beam_search --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "random" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__random__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__random__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "random" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__random__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__random__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "random" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__random__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__popular__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "popular" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__popular__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__popular__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method beam_search --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "popular" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__popular__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__popular__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "popular" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__popular__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__popular__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "popular" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__popular__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__adversarial__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "adversarial" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__adversarial__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__adversarial__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method beam_search --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "adversarial" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__adversarial__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__adversarial__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "adversarial" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__adversarial__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__adversarial__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "adversarial" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__adversarial__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__random__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method base --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "random" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__random__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__random__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method beam_search --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "random" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__random__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__random__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method dola --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "random" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__random__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__random__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method opera --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "random" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__random__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__popular__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method base --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "popular" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__popular__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__popular__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method beam_search --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "popular" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__popular__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__popular__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method dola --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "popular" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__popular__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__popular__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method opera --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "popular" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__popular__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__adversarial__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method base --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "adversarial" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__adversarial__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__adversarial__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method beam_search --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "adversarial" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__adversarial__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__adversarial__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "adversarial" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__adversarial__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-4b__adversarial__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-4b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-4b" "adversarial" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-4b__adversarial__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__random__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method base --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "random" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__random__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__random__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method beam_search --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "random" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__random__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__random__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method dola --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "random" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__random__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__random__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method opera --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "random" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__random__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__popular__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method base --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "popular" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__popular__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__popular__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method beam_search --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "popular" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__popular__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__popular__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method dola --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "popular" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__popular__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__popular__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method opera --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "popular" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__popular__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__adversarial__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method base --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "adversarial" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__adversarial__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__adversarial__beam_search $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method beam_search --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/beam_search_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "adversarial" "beam_search" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__adversarial__beam_search rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__adversarial__dola $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method dola --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/dola_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "adversarial" "dola" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__adversarial__dola rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] qwen3-vl-2b__adversarial__opera $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model qwen3-vl-2b --dataset pope --method opera --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/opera_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "qwen3-vl-2b" "adversarial" "opera" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] qwen3-vl-2b__adversarial__opera rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[MATRIX] pope_baseline_pending_gpu0 finished $(date -Iseconds)"
