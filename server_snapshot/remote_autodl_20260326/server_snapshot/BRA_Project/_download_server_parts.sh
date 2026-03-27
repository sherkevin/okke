#!/bin/bash
# Download things that server CAN handle: blip2 weights + FREAK shard0
# Clean up stale files too

BASE=/root/autodl-tmp/BRA_Project
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0

echo "=== Step 1: Clean stale aria2 files ==="
rm -f $BASE/models/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors.aria2
echo "  Removed Qwen3-VL-4B stale .aria2"

echo ""
echo "=== Step 2: FREAK shard 0 (~436MB) ==="
FREAK_DIR=$BASE/datasets/FREAK_hf/data
rm -f $FREAK_DIR/test-00000-of-00005.parquet.aria2
TARGET=$FREAK_DIR/test-00000-of-00005.parquet

if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 400000000 ]; then
    echo "  SKIP: already complete ($(du -sh $TARGET | cut -f1))"
else
    echo "  Downloading via wget..."
    URL="https://hf-mirror.com/datasets/Hiyouga/FREAK/resolve/main/data/test-00000-of-00005.parquet"
    wget -c -O "$TARGET" "$URL" 2>&1 | tail -3
    if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 400000000 ]; then
        echo "  OK: $(du -sh $TARGET | cut -f1)"
    else
        echo "  FAIL or incomplete: $(du -sh $TARGET 2>/dev/null | cut -f1)"
    fi
fi

echo ""
echo "=== Step 3: blip2-opt-2.7b weights ==="
# Clean stale incomplete cache first
BLIP2_DIR=$BASE/models/blip2-opt-2.7b
rm -rf $BLIP2_DIR/.cache
echo "  Cleaned cache. Starting fresh snapshot_download..."

python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
print('  Downloading blip2-opt-2.7b...')
try:
    r = snapshot_download(
        'Salesforce/blip2-opt-2.7b',
        local_dir='$BLIP2_DIR',
        ignore_patterns=['*.msgpack', '*.h5', 'flax_*'],
    )
    import os
    weights = [f for f in os.listdir('$BLIP2_DIR') if f.endswith('.safetensors') or (f.endswith('.bin') and 'model' in f)]
    print(f'  OK: {len(weights)} weight files')
    for w in sorted(weights):
        sz = os.path.getsize(os.path.join('$BLIP2_DIR', w))
        print(f'    {w}: {sz/1e9:.2f} GB')
except Exception as e:
    print(f'  FAIL: {e}')
"

echo ""
echo "=== Done ==="
echo "Disk: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free"
