#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/baseline_pre325/baseline_pre325_gpu0_20260324_123920.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/baseline_pre325/baseline_pre325_gpu0_20260324_123920.manifest.tsv"
MINITEST_DIR="/root/autodl-tmp/BRA_Project/logs/minitest"
echo "[MATRIX] baseline_pre325_gpu0 start $(date -Iseconds)"
printf "iso_time\tmodel\tdataset\tsplit\tmethod\texit_code\tlatest_minitest_json_guess\tmatrix_parent_log\n" > "$MANIFEST"
echo "[START] llava-v1.5-7b__pope__random__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__pope__random__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method damo --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "random" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__random__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__pope__popular__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__pope__popular__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method damo --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "popular" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__popular__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__pope__adversarial__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method vcd --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__pope__adversarial__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method damo --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "pope" "adversarial" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__pope__adversarial__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__default__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method base --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__default__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method vcd --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__chair__default__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset chair --method damo --mini_test 40504 --chair_max_new_tokens 384
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_chair_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "llava-v1.5-7b" "chair" "default" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__chair__default__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__random__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__random__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__random__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method damo --mini_test 3000 --pope_split random
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "random" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__random__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__popular__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__popular__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__popular__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method damo --mini_test 3000 --pope_split popular
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "popular" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__popular__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__adversarial__base $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method base --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/base_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "base" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__base rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__adversarial__vcd $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method vcd --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/vcd_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "vcd" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__vcd rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] instructblip-7b__pope__adversarial__damo $(date -Iseconds)"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model instructblip-7b --dataset pope --method damo --mini_test 3000 --pope_split adversarial
rc=$?
guess=$(ls -t "$MINITEST_DIR"/damo_pope_*.json 2>/dev/null | head -1 || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "instructblip-7b" "pope" "adversarial" "damo" "$rc" "$guess" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] instructblip-7b__pope__adversarial__damo rc=$rc guess=$guess $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[MATRIX] baseline_pre325_gpu0 finished $(date -Iseconds)"
