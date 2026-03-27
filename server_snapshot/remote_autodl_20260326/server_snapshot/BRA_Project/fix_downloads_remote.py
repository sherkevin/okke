#!/usr/bin/env python3
import os, sys, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PATH"] = "/root/miniconda3/bin:" + os.environ.get("PATH", "")
from huggingface_hub import snapshot_download, hf_hub_download

LOG = "/root/autodl-tmp/BRA_Project/fix_downloads.log"
def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log("=== Fix downloads started ===")

cache_dir = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b/.cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    log("[InstructBLIP] Cleaned cache")

log("[InstructBLIP] Downloading from hf-mirror.com...")
try:
    snapshot_download(
        "Salesforce/instructblip-vicuna-7b",
        local_dir="/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b",
        max_workers=4,
        resume_download=True,
    )
    log("[InstructBLIP] DONE!")
except Exception as e:
    log(f"[InstructBLIP] FAILED: {e}")

log("[FREAK] Downloading missing shard test-00000-of-00005...")
try:
    hf_hub_download(
        "hansQAQ/FREAK",
        filename="data/test-00000-of-00005.parquet",
        repo_type="dataset",
        local_dir="/root/autodl-tmp/BRA_Project/datasets/FREAK_hf",
        resume_download=True,
    )
    log("[FREAK] DONE!")
except Exception as e:
    log(f"[FREAK] FAILED: {e}")

log("=== Fix downloads completed ===")
