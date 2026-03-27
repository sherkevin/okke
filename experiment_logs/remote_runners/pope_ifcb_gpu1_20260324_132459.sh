#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/BRA_Project
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
MATRIX_LOG="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/pope_ifcb_gpu1_20260324_132459.log"
MANIFEST="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/pope_ifcb_gpu1_20260324_132459.manifest.tsv"
RUN_ID="llava_ifcb_full_20260324_1324"
RESULT_DIR="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results"
mkdir -p "$RESULT_DIR"
echo "[MATRIX] pope_ifcb_gpu1 start $(date -Iseconds)"
echo "[MATRIX] parent_log=${MATRIX_LOG}"
printf "iso_time\tpope_split\tmethod\texit_code\toutput_json\tjson_exists\tmatrix_parent_log\n" > "$MANIFEST"
echo "[START] llava-v1.5-7b__random__ifcb $(date -Iseconds) -> /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_random_llava_ifcb_full_20260324_1324.json"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method ifcb --mini_test 3000 --pope_split random --run_id llava_ifcb_full_20260324_1324 --output_json /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_random_llava_ifcb_full_20260324_1324.json --checkpoint_every 50
rc=$?
json_path="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_random_llava_ifcb_full_20260324_1324.json"
if [ -f "$json_path" ]; then json_exists=1; else json_exists=0; fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "random" "ifcb" "$rc" "$json_path" "$json_exists" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__random__ifcb rc=$rc json=$json_path exists=$json_exists $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__popular__ifcb $(date -Iseconds) -> /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_popular_llava_ifcb_full_20260324_1324.json"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method ifcb --mini_test 3000 --pope_split popular --run_id llava_ifcb_full_20260324_1324 --output_json /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_popular_llava_ifcb_full_20260324_1324.json --checkpoint_every 50
rc=$?
json_path="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_popular_llava_ifcb_full_20260324_1324.json"
if [ -f "$json_path" ]; then json_exists=1; else json_exists=0; fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "popular" "ifcb" "$rc" "$json_path" "$json_exists" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__popular__ifcb rc=$rc json=$json_path exists=$json_exists $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[START] llava-v1.5-7b__adversarial__ifcb $(date -Iseconds) -> /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_adversarial_llava_ifcb_full_20260324_1324.json"
/root/miniconda3/bin/python /root/autodl-tmp/BRA_Project/run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method ifcb --mini_test 3000 --pope_split adversarial --run_id llava_ifcb_full_20260324_1324 --output_json /root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_adversarial_llava_ifcb_full_20260324_1324.json --checkpoint_every 50
rc=$?
json_path="/root/autodl-tmp/BRA_Project/logs/pope_ifcb_gpu1/results/ifcb_pope_adversarial_llava_ifcb_full_20260324_1324.json"
if [ -f "$json_path" ]; then json_exists=1; else json_exists=0; fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "adversarial" "ifcb" "$rc" "$json_path" "$json_exists" "$MATRIX_LOG" >> "$MANIFEST"
echo "[DONE] llava-v1.5-7b__adversarial__ifcb rc=$rc json=$json_path exists=$json_exists $(date -Iseconds)"
if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi
echo "[MATRIX] pope_ifcb_gpu1 finished $(date -Iseconds)"
