#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600

M="/root/autodl-tmp/BRA_Project/models"
DS="/root/autodl-tmp/BRA_Project/datasets"
VID="$DS/video"
LOGDIR="/root/autodl-tmp/BRA_Project/logs/fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "========================================"
echo "[$(date -Iseconds)] fix_all.sh START"
echo "HF_ENDPOINT=$HF_ENDPOINT  LOGDIR=$LOGDIR"
echo "========================================"

# 0. Kill stale sessions
screen -S bra_monitor -X quit 2>/dev/null || true
pkill -f monitor_until_done.sh 2>/dev/null || true

# 1. Install hf_xet
echo "[step1] Installing hf_xet..."
pip install -q hf_xet 2>&1 | tail -2
python3 -c "import hf_xet; print(hf_xet.__version__)" 2>/dev/null && echo "[step1] hf_xet OK" || echo "[step1] hf_xet not available"

# 2. InstructBLIP - clean incomplete cache, full re-download
echo "[step2] InstructBLIP-Vicuna-7B full download..."
rm -rf "$M/instructblip-vicuna-7b/.cache"
(
  python3 -u <<'PY' >> "$LOGDIR/instructblip.log" 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = "Salesforce/instructblip-vicuna-7b", "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id=repo, repo_type="model", local_dir=dest, max_workers=8)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 3: time.sleep(30 * attempt)
sys.exit(1)
PY
) &
P2=$!

# 3. FREAK - missing shard
echo "[step3] FREAK supplement missing shard..."
rm -rf "$DS/FREAK_hf/.cache"
(
  python3 -u <<'PY' >> "$LOGDIR/freak.log" 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = "hansQAQ/FREAK", "/root/autodl-tmp/BRA_Project/datasets/FREAK_hf"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=8)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 3: time.sleep(30 * attempt)
sys.exit(1)
PY
) &
P3=$!

# 4. MVBench - incomplete videos
echo "[step4] MVBench resume missing videos..."
rm -rf "$VID/OpenGVLab_MVBench/.cache"
(
  python3 -u <<'PY' >> "$LOGDIR/mvbench.log" 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = "OpenGVLab/MVBench", "/root/autodl-tmp/BRA_Project/datasets/video/OpenGVLab_MVBench"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=8)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 3: time.sleep(30 * attempt)
sys.exit(1)
PY
) &
P4=$!

# 5. VidHalluc - xet-team/VidHalluc
echo "[step5] VidHalluc xet-team/VidHalluc..."
rm -rf "$VID/chaoyuli_VidHalluc/.cache"
mkdir -p "$VID/xet_team_VidHalluc"
(
  python3 -u <<'PY' >> "$LOGDIR/vidhalluc.log" 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = "xet-team/VidHalluc", "/root/autodl-tmp/BRA_Project/datasets/video/xet_team_VidHalluc"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=4)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 3: time.sleep(60 * attempt)
sys.exit(1)
PY
) &
P5=$!

# Wait all
echo "[wait] All jobs launched, waiting..."
FAIL=0
for pid in $P2 $P3 $P4 $P5; do
  wait $pid || FAIL=1
done

echo ""
echo "========================================"
echo "[$(date -Iseconds)] fix_all.sh END  FAIL=$FAIL"
du -sh "$M/instructblip-vicuna-7b" "$DS/FREAK_hf" "$VID/OpenGVLab_MVBench" "$VID/xet_team_VidHalluc" 2>/dev/null
echo "Logs in: $LOGDIR"
echo "========================================"
