#!/bin/bash
# Download all models from hf-mirror on new server
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
BASE=/root/autodl-tmp/BRA_Project/models

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "$(date): Start downloading models" | tee /root/autodl-tmp/BRA_Project/dl_models_new.log

MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct:Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct:Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct:Qwen3-VL-8B-Instruct"
    "Qwen/Qwen2-VL-7B-Instruct:Qwen2-VL-7B-Instruct"
    "llava-hf/llava-1.5-7b-hf:llava-1.5-7b-hf"
    "Salesforce/blip2-opt-2.7b:blip2-opt-2.7b"
)

for entry in "${MODELS[@]}"; do
    repo="${entry%%:*}"
    local="${entry##*:}"
    dest="$BASE/$local"
    echo "" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
    echo "=== Downloading $repo -> $dest ===" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
    echo "$(date)" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log

    # Check if already complete (has weight files)
    n_weights=$(find "$dest" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)
    if [ "$n_weights" -gt 0 ]; then
        echo "  SKIP: already has $n_weights weight files" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
        continue
    fi

    python3 -c "
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
try:
    r = snapshot_download('$repo', local_dir='$dest',
                          ignore_patterns=['*.msgpack','*.h5','flax_*','*.ot'])
    import glob
    weights = glob.glob('$dest/*.safetensors') + glob.glob('$dest/*.bin')
    print(f'  OK: {len(weights)} weight files')
except Exception as e:
    print(f'  FAIL: {e}')
    sys.exit(1)
" 2>&1 | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log

    echo "$(date): Done $local" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
done

# MiniGPT4-LLaMA-7B (Vicuna)
echo "" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
echo "=== MiniGPT4-LLaMA-7B ===" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
dest="$BASE/MiniGPT-4-LLaMA-7B"
n_weights=$(find "$dest" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)
if [ "$n_weights" -gt 0 ]; then
    echo "  SKIP: already has $n_weights weight files" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
else
    python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
try:
    r = snapshot_download('lmsys/vicuna-7b-v1.3', local_dir='$dest',
                          ignore_patterns=['*.msgpack','*.h5','flax_*'])
    print('  OK')
except Exception as e:
    print(f'  FAIL: {e}')
" 2>&1 | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
fi

echo "" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
echo "$(date): ALL DONE" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
echo "Disk: $(df -h /root/autodl-tmp | tail -1)" | tee -a /root/autodl-tmp/BRA_Project/dl_models_new.log
