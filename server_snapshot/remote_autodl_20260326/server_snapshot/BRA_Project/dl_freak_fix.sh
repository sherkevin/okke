#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
source /etc/network_turbo 2>/dev/null || true
echo "[$(date)] Downloading FREAK shard-0 (hansQAQ/FREAK)..." | tee /root/autodl-tmp/BRA_Project/logs/freak_aria2c2.log
aria2c --split=8 --max-connection-per-server=8 --min-split-size=16M \
       --continue=true --max-tries=5 --retry-wait=15 \
       --auto-file-renaming=false --out=test-00000-of-00005.parquet \
       --dir=/root/autodl-tmp/BRA_Project/datasets/FREAK_hf/data "https://hf-mirror.com/datasets/hansQAQ/FREAK/resolve/main/data/test-00000-of-00005.parquet" >> /root/autodl-tmp/BRA_Project/logs/freak_aria2c2.log 2>&1
echo "[$(date)] Done:" | tee -a /root/autodl-tmp/BRA_Project/logs/freak_aria2c2.log
ls -lh /root/autodl-tmp/BRA_Project/datasets/FREAK_hf/data/ | tee -a /root/autodl-tmp/BRA_Project/logs/freak_aria2c2.log
