#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true

DATA="/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc/data"
MIRROR="https://hf-mirror.com/datasets/chaoyuli/VidHalluc/resolve/main/data"
LOG="/root/autodl-tmp/BRA_Project/logs/vidhalluc_sth_tsh.log"

echo "[$(date)] ACH complete. Downloading STH and TSH..." | tee $LOG

for FILE in STH_videos.zip TSH_videos.zip; do
    if [ -f "$DATA/$FILE" ]; then
        echo "[$(date)] $FILE already exists, skipping" | tee -a $LOG
        continue
    fi
    echo "[$(date)] Downloading $FILE ..." | tee -a $LOG
    aria2c --split=8 --max-connection-per-server=8 --min-split-size=32M \
           --continue=true --max-tries=10 --retry-wait=30 \
           --auto-file-renaming=false --out="$FILE" \
           --dir="$DATA" "$MIRROR/$FILE" >> $LOG 2>&1
    echo "[$(date)] Done: $FILE" | tee -a $LOG
done
echo "[$(date)] VidHalluc ALL done!" | tee -a $LOG
ls -lh $DATA/ | tee -a $LOG
