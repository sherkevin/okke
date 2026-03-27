#!/bin/bash
BASE=/root/autodl-tmp/BRA_Project
MODELS=$BASE/models

echo "=== Qwen3-VL-4B (verify complete) ==="
ls -lh $MODELS/Qwen3-VL-4B-Instruct/model-*.safetensors 2>/dev/null
echo "aria2 stale files:"
ls -lh $MODELS/Qwen3-VL-4B-Instruct/model-*.aria2 2>/dev/null
echo "Total size: $(du -sh $MODELS/Qwen3-VL-4B-Instruct/ 2>/dev/null | cut -f1)"

echo ""
echo "=== blip2-opt-2.7b actual weight files ==="
find $MODELS/blip2-opt-2.7b -not -path '*/.*' -type f | sort | head -30
echo "Total size: $(du -sh $MODELS/blip2-opt-2.7b/ 2>/dev/null | cut -f1)"

echo ""
echo "=== InstructBLIP - all shards present? ==="
ls -lh $MODELS/instructblip-vicuna-7b/model-*.safetensors 2>/dev/null
echo "Expected: shards 1,2,3,4"

echo ""
echo "=== List InstructBLIP safetensors on HF mirror ==="
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import list_repo_files
try:
    files = [f for f in list_repo_files('Salesforce/instructblip-vicuna-7b') if 'safetensors' in f and 'model-' in f]
    for f in sorted(files):
        print(' ', f)
except Exception as e:
    print('Error:', e)
"

echo ""
echo "=== MiniGPT4 checkpoint location ==="
find /root/autodl-tmp/BRA_Project -name "pretrained_minigpt4*.pth" 2>/dev/null
