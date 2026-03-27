#!/bin/bash
echo "=== RUNNING PROCESSES ==="
ps aux | grep "python3 -u" | grep -v grep | awk '{print $1, $2, $11, $12}'

echo ""
echo "=== InstructBLIP shard 2 download ==="
INCOMPLETE=$(ls /root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b/.cache/huggingface/download/*.incomplete 2>/dev/null)
if [ -n "$INCOMPLETE" ]; then
    ls -lh $INCOMPLETE
    echo "  (target: ~4.7 GB, speed ~1MB/min = will take hours)"
else
    SHARD2=/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b/model-00002-of-00004.safetensors
    if [ -f "$SHARD2" ]; then
        echo "  DONE: $(du -sh $SHARD2 | cut -f1)"
    else
        echo "  Not found"
    fi
fi

echo ""
echo "=== MODELS ==="
for d in /root/autodl-tmp/BRA_Project/models/*/; do
    name=$(basename "$d")
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    n=$(find "$d" -maxdepth 1 -name "*.safetensors" -o -name "*.bin" | wc -l)
    echo "  $sz  $name  ($n weight files)"
done

echo ""
echo "=== DATASETS ==="
for d in /root/autodl-tmp/BRA_Project/datasets/*/; do
    name=$(basename "$d")
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    n=$(find "$d" -not -path '*/.*' -type f | wc -l)
    echo "  $sz  $name  ($n files)"
done

echo ""
echo "=== VIDEO DATASETS ==="
for d in /root/autodl-tmp/BRA_Project/datasets/video/*/; do
    name=$(basename "$d")
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    n=$(find "$d" -not -path '*/.*' -type f | wc -l)
    echo "  $sz  $name  ($n files)"
done

echo ""
echo "=== FREAK parquets ==="
ls /root/autodl-tmp/BRA_Project/datasets/FREAK_hf/data/ 2>/dev/null || echo "no data/ dir"

echo ""
echo "=== DISK ==="
df -h /root/autodl-tmp

echo ""
echo "=== MiniGPT4 setup ==="
ls /root/autodl-tmp/BRA_Project/MiniGPT-4/ 2>/dev/null | head -5
conda env list | grep minigpt4 || echo "no minigpt4 env"
