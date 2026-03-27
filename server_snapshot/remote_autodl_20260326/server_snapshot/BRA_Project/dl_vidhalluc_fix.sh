#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true

DATA="/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc/data"
MIRROR="https://hf-mirror.com/datasets/chaoyuli/VidHalluc/resolve/main/data"
LOG="/root/autodl-tmp/BRA_Project/logs/vidhalluc_direct.log"

echo "[$(date)] Restarting VidHalluc download (ACH complete check + STH/TSH)" >> $LOG

for FILE in ACH_videos.zip STH_videos.zip TSH_videos.zip; do
    if [ -f "$DATA/$FILE" ]; then
        SIZE=$(stat -c '%s' "$DATA/$FILE")
        echo "[$(date)] $FILE already exists ($SIZE bytes), skipping" >> $LOG
        continue
    fi
    echo "[$(date)] Downloading $FILE ..." >> $LOG
    aria2c --split=8 --max-connection-per-server=8 --min-split-size=64M \
           --continue=true --max-tries=10 --retry-wait=30 \
           --auto-file-renaming=false \
           --out="$FILE" \
           --dir="$DATA" \
           "$MIRROR/$FILE" \
        >> $LOG 2>&1
    echo "[$(date)] Done: $FILE" >> $LOG
done

echo "[$(date)] All VidHalluc files done!" >> $LOG
ls -lh $DATA/ >> $LOG 2>&1
