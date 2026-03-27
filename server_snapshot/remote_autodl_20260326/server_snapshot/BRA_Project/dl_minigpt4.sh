#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true

MODELS="/root/autodl-tmp/BRA_Project/models"
LOG="/root/autodl-tmp/BRA_Project/logs"
MIRROR="https://hf-mirror.com"

echo "[$(date)] === MiniGPT-4 Component Download ==="

# Step 1: Download pretrained_minigpt4.pth (45 MB projection checkpoint)
echo "[$(date)] Step 1: Downloading pretrained_minigpt4.pth..."
if [ ! -f "$MODELS/minigpt4_proj/pretrained_minigpt4.pth" ]; then
    aria2c --split=4 --max-connection-per-server=4 --continue=true \
           --dir="$MODELS/minigpt4_proj" \
           "$MIRROR/Vision-CAIR/MiniGPT-4/resolve/main/pretrained_minigpt4.pth" \
        >> $LOG/minigpt4_dl.log 2>&1
    echo "[$(date)] pretrained_minigpt4.pth done"
else
    echo "pretrained_minigpt4.pth already exists, skipping"
fi

# Step 2: Download Salesforce/blip2-opt-2.7b via HuggingFace snapshot
echo "[$(date)] Step 2: Downloading blip2-opt-2.7b (visual encoder + Q-Former)..."
python3 -u << 'PY'
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_TOKEN'] = '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'
sys.path.insert(0, '/root/miniconda3/lib/python3.12/site-packages')
from huggingface_hub import snapshot_download

print("Downloading Salesforce/blip2-opt-2.7b...")
try:
    path = snapshot_download(
        repo_id='Salesforce/blip2-opt-2.7b',
        repo_type='model',
        local_dir='/root/autodl-tmp/BRA_Project/models/blip2-opt-2.7b',
        ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', 'rust_model*'],
    )
    print(f"Done: {path}")
except Exception as e:
    print(f"Error: {e}")
PY

echo "[$(date)] === MiniGPT-4 downloads complete ==="
ls -lh $MODELS/minigpt4_proj/
du -sh $MODELS/blip2-opt-2.7b/ 2>/dev/null
