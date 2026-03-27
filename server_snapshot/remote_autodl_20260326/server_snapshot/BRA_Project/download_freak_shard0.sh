#!/usr/bin/env bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true
unset HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=3600
LOG="/root/autodl-tmp/BRA_Project/logs/freak_shard0.log"
echo "[$(date -Iseconds)] FREAK test-00000 download start" | tee "$LOG"
python3 -u <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, time
from huggingface_hub import hf_hub_download
dest = "/root/autodl-tmp/BRA_Project/datasets/FREAK_hf"
os.makedirs(dest + "/data", exist_ok=True)
for attempt in range(1, 5):
    try:
        path = hf_hub_download(
            repo_id="hansQAQ/FREAK",
            filename="data/test-00000-of-00005.parquet",
            repo_type="dataset",
            local_dir=dest,
        )
        print(f"OK -> {path}", flush=True); sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt}: {e}", flush=True)
        if attempt < 4: time.sleep(60 * attempt)
sys.exit(1)
PY
echo "[$(date -Iseconds)] FREAK test-00000 END" | tee -a "$LOG"
