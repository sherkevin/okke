#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true

DEST="/root/autodl-tmp/BRA_Project/datasets/FREAK_hf/data"
MIRROR="https://hf-mirror.com/datasets/Harvard-NLP/FREAK/resolve/main/data"
LOG="/root/autodl-tmp/BRA_Project/logs/freak_aria2.log"

echo "[$(date)] Downloading FREAK test-00000-of-00005.parquet via aria2c..." | tee $LOG

aria2c --split=8 --max-connection-per-server=8 --min-split-size=16M \
       --continue=true --max-tries=10 --retry-wait=30 \
       --auto-file-renaming=false \
       --out="test-00000-of-00005.parquet" \
       --dir="$DEST" \
       "$MIRROR/test-00000-of-00005.parquet" \
    >> $LOG 2>&1

echo "[$(date)] FREAK shard-0 done:" | tee -a $LOG
ls -lh $DEST/ | tee -a $LOG
