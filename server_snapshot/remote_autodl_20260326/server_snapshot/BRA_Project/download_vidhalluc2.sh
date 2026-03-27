#!/usr/bin/env bash
export PATH="/root/miniconda3/bin:$PATH"
# Enable AutoDL academic proxy for access to HuggingFace official
source /etc/network_turbo 2>/dev/null || true
unset HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=3600
DEST="/root/autodl-tmp/BRA_Project/datasets/video/xet_team_VidHalluc"
LOG="/root/autodl-tmp/BRA_Project/logs/vidhalluc_official.log"
mkdir -p "$DEST"
echo "[$(date -Iseconds)] xet-team/VidHalluc START (official HF + proxy + hf_xet)" | tee -a "$LOG"
python3 -u <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, time
from huggingface_hub import snapshot_download
repo = "xet-team/VidHalluc"
dest = "/root/autodl-tmp/BRA_Project/datasets/video/xet_team_VidHalluc"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 6):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=4)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 5: time.sleep(60 * attempt)
sys.exit(1)
PY
echo "[$(date -Iseconds)] xet-team/VidHalluc END" | tee -a "$LOG"
