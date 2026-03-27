#!/bin/bash
# Run 2B scaling + TextVQA OCR shield sequentially (model loaded fresh per task)
set -euo pipefail
cd /root/autodl-tmp/A-OSP_Project
LOG="logs/eval_results/2b_pipeline_live.log"

echo "[$(date)] === 2B PIPELINE START ===" | tee "$LOG"
echo "[$(date)] Task 1: Scaling Matrix (MMHal + POPE)" | tee -a "$LOG"

PYTHONUNBUFFERED=1 python3 -u code/eval/run_2b_scaling.py \
    --n_extract 100 --n_mmhal 48 --n_pope 100 --alpha 1.0 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] Task 1 DONE. Sleeping 10s for GPU cooldown..." | tee -a "$LOG"
sleep 10

echo "[$(date)] Task 2: TextVQA OCR Shield" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python3 -u code/eval/run_textvqa_ocr_shield.py \
    --n_samples 100 --n_ocr_prompts 50 --alpha 1.0 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] === 2B PIPELINE COMPLETE ===" | tee -a "$LOG"
