#!/usr/bin/env bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true
unset HF_ENDPOINT
export HF_TOKEN="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"
export HUGGING_FACE_HUB_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=3600
DEST="/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc"
LOG="/root/autodl-tmp/BRA_Project/logs/vidhalluc_token.log"
mkdir -p "$DEST"
echo "[$(date -Iseconds)] chaoyuli/VidHalluc START with explicit token" | tee "$LOG"
python3 -u <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, time
from huggingface_hub import snapshot_download
TOKEN = "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"
repo = "chaoyuli/VidHalluc"
dest = "/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 6):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                          max_workers=4, token=TOKEN)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 5: time.sleep(60 * attempt)
sys.exit(1)
PY
echo "[$(date -Iseconds)] chaoyuli/VidHalluc END" | tee -a "$LOG"
