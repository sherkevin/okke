#!/usr/bin/env bash
export PATH="/root/miniconda3/bin:$PATH"
unset HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
DEST="/root/autodl-tmp/BRA_Project/datasets/video/xet_team_VidHalluc"
mkdir -p "$DEST"
echo "[$(date -Iseconds)] xet-team/VidHalluc download start (official HF)"
python3 -u <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = "xet-team/VidHalluc", "/root/autodl-tmp/BRA_Project/datasets/video/xet_team_VidHalluc"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 5):
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=4)
        print(f"OK {repo} -> {dest}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 4: time.sleep(60 * attempt)
sys.exit(1)
PY
echo "[$(date -Iseconds)] xet-team/VidHalluc DONE"
