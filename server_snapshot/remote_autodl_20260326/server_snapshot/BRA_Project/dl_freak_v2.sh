#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true
unset HF_ENDPOINT
export HF_TOKEN="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"
export HUGGING_FACE_HUB_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=7200
LOG="/root/autodl-tmp/BRA_Project/logs/freak_shard0_v2.log"
echo "[$(date)] FREAK shard-0 restart via hf_hub_download + proxy" | tee $LOG
python3 -u << 'PY' 2>&1 | tee -a $LOG
import os, sys, time
from huggingface_hub import hf_hub_download
dest = "/root/autodl-tmp/BRA_Project/datasets/FREAK_hf"
for attempt in range(1, 6):
    try:
        path = hf_hub_download(
            repo_id="hansQAQ/FREAK",
            filename="data/test-00000-of-00005.parquet",
            repo_type="dataset",
            local_dir=dest,
            force_download=True,
        )
        print(f"OK -> {path}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"FAIL attempt {attempt}: {e}", flush=True)
        if attempt < 5:
            time.sleep(30 * attempt)
sys.exit(1)
PY
echo "[$(date)] FREAK shard-0 done" | tee -a $LOG
ls -lh /root/autodl-tmp/BRA_Project/datasets/FREAK_hf/data/ | tee -a $LOG
