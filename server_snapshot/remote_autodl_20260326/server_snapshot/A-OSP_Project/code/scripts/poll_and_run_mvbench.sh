#!/bin/bash
# ============================================================
# Poll for Charades video upload, then auto-run MVBench eval
# ============================================================
# Usage: bash code/scripts/poll_and_run_mvbench.sh
# Logs:  logs/eval_results/mvbench_poll.log
#        logs/eval_results/mvbench_actseq_base_n10.jsonl
#        logs/eval_results/mvbench_actseq_aosp_n10.jsonl

PROJECT_ROOT="/root/autodl-tmp/A-OSP_Project"
EVAL_SCRIPT="$PROJECT_ROOT/code/eval/run_mvbench_action_sequence.py"
V_TEXT_ONLY="$PROJECT_ROOT/models/V_text_only_q3.pt"
V_MATRIX_Q3="$PROJECT_ROOT/models/V_matrix_q3.pt"
LOG="$PROJECT_ROOT/logs/eval_results/mvbench_poll.log"
INTERVAL=30   # seconds between polls

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$(dirname "$LOG")"
log "=== MVBench Poll-and-Run Started ==="
log "Polling for: /root/autodl-tmp/A-OSP_Project/data/MVBench/video/Charades_v1_480/"
log "Eval script: $EVAL_SCRIPT"
log "Poll interval: ${INTERVAL}s"

# ---- Poll loop ----
POLL_COUNT=0
while true; do
    POLL_COUNT=$((POLL_COUNT + 1))
    log "Poll #$POLL_COUNT — checking video paths..."

    python3 "$EVAL_SCRIPT" --check_video_paths --n_samples 10 >> "$LOG" 2>&1
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        log "✅ ALL 10 VIDEOS READY — launching evaluation pipeline!"
        break
    fi

    log "Videos not yet ready (exit=$EXIT_CODE). Sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
done

# ---- Check for official V_text_only_q3.pt ----
if [ -f "$V_TEXT_ONLY" ]; then
    log "✅ V_text_only_q3.pt found — using official S_text-only subspace"
    V_ARG=""
else
    log "⚠  V_text_only_q3.pt not found — using V_matrix_q3.pt as fallback"
    V_ARG="--v_matrix $V_MATRIX_Q3"
fi

# ---- Run BASE mode ----
log ""
log "=== [1/2] Running BASE mode (10 samples) ==="
PYTHONUNBUFFERED=1 python3 "$EVAL_SCRIPT" \
    --mode base \
    --n_samples 10 \
    --fps 1.0 \
    2>&1 | tee -a "$LOG"

log "BASE eval complete."

# ---- Run AOSP mode ----
log ""
log "=== [2/2] Running AOSP mode (10 samples) ==="
PYTHONUNBUFFERED=1 python3 "$EVAL_SCRIPT" \
    --mode aosp \
    --n_samples 10 \
    --fps 1.0 \
    --alpha 0.5 \
    $V_ARG \
    2>&1 | tee -a "$LOG"

log "AOSP eval complete."
log ""
log "=== Pipeline Finished ==="
log "Results in: $PROJECT_ROOT/logs/eval_results/mvbench_actseq_*.jsonl"
