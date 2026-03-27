#!/bin/bash
# Wait for Qwen3-VL-8B download to complete, then auto-run extraction pipeline.
# Usage: bash wait_and_extract.sh [--skip_extract]
#
# Steps:
#   1. Poll until all 4 safetensors shards are present and no .incomplete files exist
#   2. Run V_matrix extraction (10-sample mini-test)
#   3. Run CHAIR 50-sample smoke run (base)

PROJECT="/root/autodl-tmp/A-OSP_Project"
MODEL="$PROJECT/models/Qwen3-VL-8B-Instruct"
LOG="$PROJECT/logs/wait_and_extract.log"
mkdir -p "$PROJECT/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== wait_and_extract.sh started ==="

# ---------------------------------------------------------
# Step 1: Wait for model completeness
# ---------------------------------------------------------
while true; do
    N_SHARDS=$(ls "$MODEL"/model-*.safetensors 2>/dev/null | wc -l)
    N_INCOMPLETE=$(find "$MODEL/.cache" -name "*.incomplete" -size +0c 2>/dev/null | wc -l)
    TOTAL_GB=$(du -sh "$MODEL" 2>/dev/null | awk '{print $1}')
    log "Shards complete: $N_SHARDS/4 | Incomplete files: $N_INCOMPLETE | Total: $TOTAL_GB"

    if [ "$N_SHARDS" -eq 4 ] && [ "$N_INCOMPLETE" -eq 0 ]; then
        log "✓ Model download complete!"
        break
    fi
    sleep 120  # poll every 2 minutes
done

# ---------------------------------------------------------
# Step 1.5: Verify model loads correctly
# ---------------------------------------------------------
log "Verifying model load..."
cd "$PROJECT"
python3 code/scripts/verify_q3vl_load.py --skip_inference 2>&1 | tee -a "$LOG"
if [ $? -ne 0 ]; then
    log "✗ Model verification failed! Aborting."
    exit 1
fi
log "✓ Model verified"

# ---------------------------------------------------------
# Step 2: Extract V_matrix_q3.pt (10 images mini-test)
# ---------------------------------------------------------
if [ ! -f "$PROJECT/models/V_matrix_q3.pt" ]; then
    log "Extracting V_matrix_q3.pt (10-sample mini-test)..."
    cd "$PROJECT"
    python3 code/scripts/extract_vmatrix_q3.py \
        --n_images 10 \
        --layer 32 \
        --K 20 \
        --output models/V_matrix_q3_mini.pt \
        2>&1 | tee -a "$LOG"
    if [ $? -eq 0 ]; then
        log "✓ V_matrix_q3_mini.pt extracted"
        # Copy mini version as main V_matrix_q3.pt for downstream use
        cp "$PROJECT/models/V_matrix_q3_mini.pt" "$PROJECT/models/V_matrix_q3.pt"
        log "✓ Copied to V_matrix_q3.pt"
    else
        log "✗ V_matrix extraction FAILED. Check log."
        exit 1
    fi
else
    log "V_matrix_q3.pt already exists, skipping extraction."
fi

# ---------------------------------------------------------
# Step 3: POPE 50-sample Base mini-batch
# ---------------------------------------------------------
log "Starting POPE 50-sample BASE run (Qwen3-VL)..."
cd "$PROJECT"
python3 code/eval/run_base_eval.py \
    --model_path "$MODEL" \
    --pope_file  data/pope/pope_coco_popular.jsonl \
    --image_dir  data/coco_val2014 \
    --output_dir logs/eval_results \
    --tag        base_qwen3vl8b_pope_popular_n50 \
    --limit      50 \
    2>&1 | tee -a "$LOG"
if [ $? -eq 0 ]; then
    log "✓ POPE base 50-sample done"
else
    log "✗ POPE base run FAILED"
fi

# ---------------------------------------------------------
# Step 4: POPE 50-sample A-OSP mini-batch
# ---------------------------------------------------------
log "Starting POPE 50-sample A-OSP run (Qwen3-VL)..."
python3 code/eval/run_aosp_eval.py \
    --model_path "$MODEL" \
    --v_matrix   models/V_matrix_q3.pt \
    --pope_file  data/pope/pope_coco_popular.jsonl \
    --image_dir  data/coco_val2014 \
    --output_dir logs/eval_results \
    --tag        aosp_qwen3vl8b_pope_popular_n50 \
    --limit      50 \
    2>&1 | tee -a "$LOG"
if [ $? -eq 0 ]; then
    log "✓ POPE A-OSP 50-sample done"
else
    log "✗ POPE A-OSP run FAILED"
fi

# ---------------------------------------------------------
# Step 5: CHAIR 50-sample base smoke run
# ---------------------------------------------------------
log "Starting CHAIR 50-sample base run..."
python3 code/eval/run_chair_eval.py \
    --mode base \
    --limit 50 \
    --model_path "$MODEL" \
    2>&1 | tee -a "$LOG"
if [ $? -eq 0 ]; then
    log "✓ CHAIR base 50-sample done"
else
    log "✗ CHAIR base run FAILED"
fi

log "=== wait_and_extract.sh finished ==="
log "=== All mini-batches complete. Awaiting approval for full runs. ==="
