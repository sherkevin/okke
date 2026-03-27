#!/bin/bash
# =============================================================================
# A-OSP Evaluation Pipeline — Run ALL Base Baselines
# =============================================================================
# This script sequentially runs POPE, MMMU, and Throughput benchmarks
# for the native Qwen2-VL-7B-Instruct (no intervention).
#
# Usage:
#   chmod +x run_all_base.sh
#   ./run_all_base.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/autodl-tmp/A-OSP_Project"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2-VL-7B-Instruct"
OUTPUT_DIR="${PROJECT_ROOT}/logs/eval_results"

echo "============================================"
echo "  A-OSP — Full Base Baseline Evaluation"
echo "  Model: Qwen2-VL-7B-Instruct (native)"
echo "============================================"

# ── Step 1: POPE ──
echo ""
echo "[1/3] Running POPE evaluation ..."
python "${SCRIPT_DIR}/run_base_eval.py" \
    --model_path "${MODEL_PATH}" \
    --pope_file "${PROJECT_ROOT}/data/pope/pope_coco_popular_mini.jsonl" \
    --image_dir "${PROJECT_ROOT}/data/coco_val2014" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b_pope_popular"

# ── Step 2: MMMU ──
echo ""
echo "[2/3] Running MMMU evaluation (first 200 samples) ..."
python "${SCRIPT_DIR}/run_base_mmmu.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b_mmmu_val" \
    --max_samples 200

# ── Step 3: Throughput ──
echo ""
echo "[3/3] Running Throughput benchmark (512 tokens) ..."
echo "  >>> REMINDER: Lock GPU frequency first!"
echo "  >>> sudo nvidia-smi -lgc 2000,2000"
python "${SCRIPT_DIR}/benchmark_throughput.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b" \
    --method_tag "base"

echo ""
echo "============================================"
echo "  ALL DONE — Results in: ${OUTPUT_DIR}/"
echo "============================================"
ls -lh "${OUTPUT_DIR}/"
