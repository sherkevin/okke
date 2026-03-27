#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true

cd /root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc/data

echo "[$(date)] Starting VidHalluc direct download..."

for FILE in ACH_videos.zip TSH_videos.zip STH_videos.zip; do
    if [ -f "$FILE" ]; then
        echo "$FILE already exists, skipping"
        continue
    fi
    echo "[$(date)] Downloading $FILE ..."
    aria2c --split=8 --max-connection-per-server=8 --min-split-size=64M \
           --continue=true --max-tries=5 --retry-wait=10 \
           --dir="/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc/data" \
           "https://hf-mirror.com/datasets/chaoyuli/VidHalluc/resolve/main/data/$FILE" \
        >> /root/autodl-tmp/BRA_Project/logs/vidhalluc_direct.log 2>&1
    echo "[$(date)] Done: $FILE"
done

echo "[$(date)] All VidHalluc files downloaded!"
ls -lh /root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc/data/
