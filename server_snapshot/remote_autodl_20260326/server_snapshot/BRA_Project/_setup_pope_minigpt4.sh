#!/bin/bash
set -e

# Download POPE question files
echo "=== Downloading POPE ==="
cd /root/autodl-tmp/BRA_Project/datasets
rm -rf POPE
mkdir -p POPE/output/coco
cd POPE/output/coco

PROXY="https://ghfast.top"
BASE="${PROXY}/https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco"

for f in coco_pope_random.json coco_pope_popular.json coco_pope_adversarial.json; do
    echo "  Downloading ${f}..."
    wget -q "${BASE}/${f}" -O "${f}" && echo "  OK: ${f}" || echo "  FAILED: ${f}"
done
echo "POPE files:"
wc -l *.json 2>/dev/null || echo "  No files downloaded"

# Clone MiniGPT-4 via proxy
echo ""
echo "=== Cloning MiniGPT-4 ==="
cd /root/autodl-tmp/BRA_Project
rm -rf MiniGPT-4
git clone "${PROXY}/https://github.com/Vision-CAIR/MiniGPT-4.git" MiniGPT-4 && echo "MiniGPT-4 cloned OK" || echo "MiniGPT-4 clone FAILED, trying alternative..."

if [ ! -d "MiniGPT-4" ]; then
    # Try bgithub.xyz mirror
    git clone "https://bgithub.xyz/Vision-CAIR/MiniGPT-4.git" MiniGPT-4 && echo "MiniGPT-4 cloned via bgithub" || echo "MiniGPT-4 clone FAILED on both mirrors"
fi

if [ -d "MiniGPT-4" ]; then
    echo "MiniGPT-4 contents:"
    ls MiniGPT-4/
fi

echo ""
echo "=== DONE ==="
