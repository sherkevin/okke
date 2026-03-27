#!/usr/bin/env bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true
unset HF_ENDPOINT
export HF_TOKEN="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"
export HF_HUB_DOWNLOAD_TIMEOUT=3600
LOG="/root/autodl-tmp/BRA_Project/logs/instructblip_bin.log"
DEST="/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
echo "[$(date -Iseconds)] InstructBLIP .bin download (skip safetensors/XET)" | tee "$LOG"
/root/miniconda3/bin/python3 -u -c '
import os, sys, time
from huggingface_hub import snapshot_download
dest = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
os.makedirs(dest, exist_ok=True)
for attempt in range(1, 4):
    try:
        snapshot_download(
            repo_id="Salesforce/instructblip-vicuna-7b",
            repo_type="model", local_dir=dest, max_workers=4,
            ignore_patterns=["*.safetensors", "*.safetensors.index.json"])
        print("OK InstructBLIP .bin -> " + dest, flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 3: time.sleep(60)
sys.exit(1)
' 2>&1 | tee -a "$LOG"
echo "[$(date -Iseconds)] InstructBLIP END" | tee -a "$LOG"
