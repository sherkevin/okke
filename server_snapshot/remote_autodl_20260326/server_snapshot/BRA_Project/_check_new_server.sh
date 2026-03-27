#!/bin/bash
BASE=/root/autodl-tmp/BRA_Project/models

echo "=== MODEL WEIGHT STATUS ==="
for m in Qwen3-VL-2B-Instruct Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct Qwen2-VL-7B-Instruct llava-1.5-7b-hf MiniGPT-4-LLaMA-7B blip2-opt-2.7b instructblip-vicuna-7b; do
    dir="$BASE/$m"
    sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
    n_sf=$(find "$dir" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    n_bin=$(find "$dir" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
    echo "  $m: $sz  (safetensors=$n_sf, bin=$n_bin)"
done

echo ""
echo "=== INSTRUCTBLIP detail ==="
ls -lh $BASE/instructblip-vicuna-7b/ 2>/dev/null | grep -v '^total'

echo ""
echo "=== BLIP2 detail ==="
ls -lh $BASE/blip2-opt-2.7b/ 2>/dev/null | grep -v '^total'

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null

echo ""
echo "=== Disk ==="
df -h /root/autodl-tmp | tail -1
