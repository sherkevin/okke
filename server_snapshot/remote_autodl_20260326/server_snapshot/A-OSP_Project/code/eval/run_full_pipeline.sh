#!/bin/bash
# =============================================================================
# A-OSP Full Evaluation Pipeline — Base + A-OSP Side-by-Side
# =============================================================================
# Runs all benchmarks for both methods sequentially, producing the complete
# data needed for Table 1, the Pareto bubble chart, and cross-method comparison.
#
# BEFORE RUNNING:
#   1. Ensure COCO val2014 images are in data/coco_val2014/
#   2. Lock GPU frequency: sudo nvidia-smi -lgc 2000,2000
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh [--pope_file path] [--image_dir path]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/autodl-tmp/A-OSP_Project"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2-VL-7B-Instruct"
V_MATRIX="${PROJECT_ROOT}/models/V_matrix.pt"
OUTPUT_DIR="${PROJECT_ROOT}/logs/eval_results"

POPE_FILE="${1:-${PROJECT_ROOT}/data/pope/pope_coco_popular_mini.jsonl}"
IMAGE_DIR="${2:-${PROJECT_ROOT}/data/coco_val2014}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   A-OSP Full Evaluation Pipeline                            ║"
echo "║   Model: Qwen2-VL-7B-Instruct                              ║"
echo "║   Methods: Base (native) vs A-OSP (intervention)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "POPE file : ${POPE_FILE}"
echo "Image dir : ${IMAGE_DIR}"
echo "V_matrix  : ${V_MATRIX}"
echo "Output    : ${OUTPUT_DIR}/"
echo ""

# ============================================================
# PHASE 1: BASE BASELINE
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: BASE BASELINE (no intervention)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "[1/6] Base POPE ..."
python "${SCRIPT_DIR}/run_base_eval.py" \
    --model_path "${MODEL_PATH}" \
    --pope_file "${POPE_FILE}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b_pope"

echo ""
echo "[2/6] Base Throughput (512 tokens) ..."
python "${SCRIPT_DIR}/benchmark_throughput.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b" \
    --method_tag "base"

# ============================================================
# PHASE 2: A-OSP INTERVENTION
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: A-OSP INTERVENTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "[3/6] A-OSP POPE ..."
python "${SCRIPT_DIR}/run_aosp_eval.py" \
    --model_path "${MODEL_PATH}" \
    --v_matrix "${V_MATRIX}" \
    --pope_file "${POPE_FILE}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "aosp_qwen2vl7b_pope"

echo ""
echo "[4/6] A-OSP Throughput (512 tokens) ..."
python "${SCRIPT_DIR}/benchmark_throughput.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "aosp_qwen2vl7b" \
    --method_tag "aosp" \
    --enable_aosp \
    --v_matrix "${V_MATRIX}"

# ============================================================
# PHASE 3: MMMU COMMONSENSE STRESS TEST (optional, slower)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3: MMMU COMMONSENSE STRESS TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "[5/6] Base MMMU (200 samples) ..."
python "${SCRIPT_DIR}/run_base_mmmu.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "base_qwen2vl7b_mmmu" \
    --max_samples 200

echo ""
echo "[6/6] A-OSP MMMU (200 samples) ..."
python "${SCRIPT_DIR}/run_aosp_mmmu.py" \
    --model_path "${MODEL_PATH}" \
    --v_matrix "${V_MATRIX}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "aosp_qwen2vl7b_mmmu" \
    --max_samples 200

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ALL DONE — Full Pipeline Complete                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory:"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "Key files for Agent 3 (visualization):"
echo "  POPE Base    : ${OUTPUT_DIR}/base_qwen2vl7b_pope_results.jsonl"
echo "  POPE A-OSP   : ${OUTPUT_DIR}/aosp_qwen2vl7b_pope_results.jsonl"
echo "  Throughput   : ${OUTPUT_DIR}/*_throughput.json"
echo "  MMMU Base    : ${OUTPUT_DIR}/base_qwen2vl7b_mmmu_results.jsonl"
echo "  MMMU A-OSP   : ${OUTPUT_DIR}/aosp_qwen2vl7b_mmmu_results.jsonl"
