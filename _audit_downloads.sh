#!/bin/bash
# Full audit of all incomplete/missing model and dataset files

BASE=/root/autodl-tmp/BRA_Project
MODELS=$BASE/models
DATASETS=$BASE/datasets

echo "================================================"
echo "INCOMPLETE FILES AUDIT"
echo "================================================"

echo ""
echo "--- Qwen3-VL-4B shards ---"
ls -lh $MODELS/Qwen3-VL-4B-Instruct/model-*.safetensors 2>/dev/null
ls -lh $MODELS/Qwen3-VL-4B-Instruct/model-*.safetensors.aria2 2>/dev/null

echo ""
echo "--- InstructBLIP shards ---"
ls -lh $MODELS/instructblip-vicuna-7b/model-*.safetensors 2>/dev/null
echo "  (expected: 4 shards total)"

echo ""
echo "--- blip2-opt-2.7b files ---"
ls -lh $MODELS/blip2-opt-2.7b/*.safetensors 2>/dev/null
ls -lh $MODELS/blip2-opt-2.7b/*.bin 2>/dev/null | head -15
echo "  --- incomplete cache ---"
for f in $MODELS/blip2-opt-2.7b/.cache/huggingface/download/*.incomplete; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null)
    hash=$(basename "$f" | sed 's/\.incomplete//')
    meta="${f%.incomplete}.metadata"
    filename=$(cat "$meta" 2>/dev/null | head -2 | tail -1)
    echo "  INCOMPLETE: $filename  (${sz} bytes partial)"
done

echo ""
echo "--- FREAK shard 0 ---"
ls -lh $DATASETS/FREAK_hf/data/ 2>/dev/null
echo "  aria2 file size: $(stat -c%s $DATASETS/FREAK_hf/data/test-00000-of-00005.parquet.aria2 2>/dev/null) bytes"
echo "  (other shards ~436-446MB each)"

echo ""
echo "--- MiniGPT4 checkpoint ---"
ls -lh /root/autodl-tmp/BRA_Project/models/minigpt4_proj/ 2>/dev/null
ls -lh /root/autodl-tmp/BRA_Project/MiniGPT-4/pretrained_minigpt4.pth 2>/dev/null || echo "  pth not in repo root"

echo ""
echo "================================================"
echo "SUMMARY: What to download locally with VPN"
echo "================================================"
echo ""

# Check Qwen3-VL-4B shard 2
if ls $MODELS/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors.aria2 2>/dev/null; then
    actual=$(ls $MODELS/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors 2>/dev/null && stat -c%s $MODELS/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors 2>/dev/null || echo 0)
    echo "[NEED] Qwen3-VL-4B shard2: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/model-00002-of-00002.safetensors"
    echo "       (~4GB, server has .aria2 placeholder only)"
fi

# InstructBLIP
echo "[NEED] InstructBLIP shard1: https://huggingface.co/Salesforce/instructblip-vicuna-7b/resolve/main/model-00001-of-00004.safetensors"
echo "[NEED] InstructBLIP shard2: https://huggingface.co/Salesforce/instructblip-vicuna-7b/resolve/main/model-00002-of-00004.safetensors"
echo "[NEED] InstructBLIP shard3: https://huggingface.co/Salesforce/instructblip-vicuna-7b/resolve/main/model-00003-of-00004.safetensors"
echo "       (each ~4-5GB)"

# FREAK shard 0
echo "[NEED] FREAK shard0: https://huggingface.co/datasets/Hiyouga/FREAK/resolve/main/data/test-00000-of-00005.parquet"
echo "       (~436MB)"

echo ""
echo "Disk remaining: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
