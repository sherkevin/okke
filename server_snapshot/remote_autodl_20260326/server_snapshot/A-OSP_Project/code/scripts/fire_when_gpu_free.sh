#!/bin/bash
# ============================================================
# fire_when_gpu_free.sh
# Polls until GPU is free (Agent 2's POPE-3000 finishes),
# then fires Task 3.5b (MMMU Dual-Gain) and Task 2.3-50 (MVBench 50-sample).
# Run this script in background: bash code/scripts/fire_when_gpu_free.sh &
# ============================================================
set -euo pipefail
cd /root/autodl-tmp/A-OSP_Project

GPU_FREE_THRESHOLD_MIB=8000   # MiB — fire when GPU memory < this
POLL_INTERVAL=60               # seconds between polls
LOG="logs/eval_results/gpu_fire_daemon.log"
mkdir -p logs/eval_results

echo "[$(date)] GPU fire daemon started. Threshold < ${GPU_FREE_THRESHOLD_MIB} MiB" | tee -a "$LOG"

while true; do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[$(date)] GPU memory used: ${GPU_MEM} MiB" | tee -a "$LOG"

    if [ "${GPU_MEM:-99999}" -lt "$GPU_FREE_THRESHOLD_MIB" ]; then
        echo "[$(date)] ✅ GPU FREE — launching tasks" | tee -a "$LOG"
        break
    fi
    sleep "$POLL_INTERVAL"
done

# ---- Task 3.5b: MMMU Hard Dual-Gain (30 samples, all modes) ----
echo "[$(date)] Launching Task 3.5b: MMMU Dual-Gain..." | tee -a "$LOG"
PYTHONUNBUFFERED=1 python3 code/eval/run_mmmu_dual_gain.py \
    --mode all --n_samples 30 --alpha 1.0 \
    2>&1 | tee logs/eval_results/mmmu_dual_gain_live.log
echo "[$(date)] Task 3.5b COMPLETE" | tee -a "$LOG"

# ---- Task 2.3-50: MVBench 50-sample (requires Agent 4's Charades unzip) ----
CHARADES_DIR="data/mvbench/video/extracted/data0613/star/Charades_v1_480"
N_MP4=$(find "${CHARADES_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
if [ "$N_MP4" -ge 50 ]; then
    echo "[$(date)] Charades has ${N_MP4} videos — launching MVBench 50-sample..." | tee -a "$LOG"
    PYTHONUNBUFFERED=1 python3 code/eval/run_mvbench_action_sequence.py \
        --mode both --n_samples 50 --alpha 1.0 --max_pixels 151200 \
        2>&1 | tee logs/eval_results/mvbench_50sample_live.log
    echo "[$(date)] MVBench 50-sample COMPLETE" | tee -a "$LOG"
else
    echo "[$(date)] ⚠️  Only ${N_MP4} Charades videos (need ≥50). Skipping MVBench full run." | tee -a "$LOG"
    echo "[$(date)] Run manually when Agent 4 unzips: python3 code/eval/run_mvbench_action_sequence.py --mode both --n_samples 50" | tee -a "$LOG"
fi

echo "[$(date)] All queued tasks complete." | tee -a "$LOG"
