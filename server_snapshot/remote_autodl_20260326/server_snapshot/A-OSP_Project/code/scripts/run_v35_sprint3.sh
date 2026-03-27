#!/bin/bash
# V3.5 Sprint 3: Full Pipeline Script
# Step 1: Extract V_text_only_q3.pt (zero-vision S_text-only for Qwen3-VL-8B)
# Step 2: Run MVBench Base (10 samples)  
# Step 3: Run MVBench A-OSP (10 samples, uses V_text_only_q3.pt)
# Step 4: Compute Principal Angles (S_text-only vs S_blur)

set -e
cd /root/autodl-tmp/A-OSP_Project

LOGFILE="logs/eval_results/v35_sprint3_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/eval_results

echo "=== V3.5 Sprint 3 Pipeline ===" | tee "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# --- Step 1: Extract V_text_only_q3.pt ---
echo "[Step 1/4] Extracting V_text_only_q3.pt (Zero-Vision S_text-only)..." | tee -a "$LOGFILE"
if [ -f "models/V_text_only_q3.pt" ]; then
    echo "  Already exists. Skipping extraction." | tee -a "$LOGFILE"
else
    python3 code/scripts/extract_vmatrix_text_only_q3.py 2>&1 | tee -a "$LOGFILE"
    echo "  Done." | tee -a "$LOGFILE"
fi

# --- Step 2: MVBench Base ---
echo "" | tee -a "$LOGFILE"
echo "[Step 2/4] MVBench Base eval (10 samples)..." | tee -a "$LOGFILE"
python3 code/eval/run_mvbench_eval.py --mode base --n_samples 10 2>&1 | tee -a "$LOGFILE"

# --- Step 3: MVBench A-OSP ---
echo "" | tee -a "$LOGFILE"
echo "[Step 3/4] MVBench A-OSP eval (10 samples, S_text-only)..." | tee -a "$LOGFILE"
python3 code/eval/run_mvbench_eval.py --mode aosp --n_samples 10 2>&1 | tee -a "$LOGFILE"

# --- Step 4: Principal Angles ---
echo "" | tee -a "$LOGFILE"
echo "[Step 4/4] Computing Principal Angles (S_text-only vs S_blur)..." | tee -a "$LOGFILE"
python3 code/scripts/calc_principal_angles.py 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== PIPELINE COMPLETE ===" | tee -a "$LOGFILE"
echo "End: $(date)" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"
