"""
Download missing models and datasets for BRA evaluation.
Uses HF mirror (hf-mirror.com) for China server.
"""
import os
import sys
import subprocess
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download, hf_hub_download

BASE = "/root/autodl-tmp/BRA_Project"
MODELS = f"{BASE}/models"
DATASETS = f"{BASE}/datasets"

def download_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
    print(f"  FAILED after {max_retries} attempts")
    return None


# ====================================================================
# 1. InstructBLIP-Vicuna-7B: missing shard 2 of 4
# ====================================================================
print("=" * 60)
print("1. InstructBLIP-Vicuna-7B (missing shard)")
print("=" * 60)

ib_path = f"{MODELS}/instructblip-vicuna-7b"
missing_shard = f"{ib_path}/model-00002-of-00004.safetensors"
if os.path.exists(missing_shard):
    print(f"  Shard already exists: {os.path.getsize(missing_shard)/1e9:.2f} GB")
else:
    print("  Downloading missing shard...")
    result = download_with_retry(
        hf_hub_download,
        "Salesforce/instructblip-vicuna-7b",
        "model-00002-of-00004.safetensors",
        local_dir=ib_path,
        resume_download=True,
    )
    if result:
        print(f"  Downloaded: {os.path.getsize(result)/1e9:.2f} GB")


# ====================================================================
# 2. FREAK dataset (incomplete download)
# ====================================================================
print("\n" + "=" * 60)
print("2. FREAK dataset (resume incomplete)")
print("=" * 60)

freak_path = f"{DATASETS}/FREAK_hf"
# Clear incomplete cache and re-download
cache_dir = f"{freak_path}/.cache"
if os.path.isdir(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("  Cleared stale cache")

print("  Downloading FREAK...")
result = download_with_retry(
    snapshot_download,
    "Hiyouga/FREAK",
    local_dir=freak_path,
    resume_download=True,
)
if result:
    parquets = [f for f in os.listdir(freak_path) if f.endswith(".parquet")]
    print(f"  Downloaded. Parquet files: {len(parquets)}")


# ====================================================================
# 3. VidHalluc (video hallucination benchmark, incomplete)
# ====================================================================
print("\n" + "=" * 60)
print("3. VidHalluc (resume incomplete)")
print("=" * 60)

vid_path = f"{DATASETS}/video"
# Try downloading VidHalluc specifically
vidhalluc_path = f"{vid_path}/VidHalluc_VidHalluc"
cache_dir = f"{vidhalluc_path}/.cache"
if os.path.isdir(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("  Cleared stale VidHalluc cache")

print("  Downloading VidHalluc...")
try:
    result = download_with_retry(
        snapshot_download,
        "VidHalluc/VidHalluc",
        local_dir=f"{vid_path}/VidHalluc_VidHalluc",
        resume_download=True,
    )
    if result:
        print(f"  VidHalluc downloaded to {result}")
except Exception as e:
    print(f"  VidHalluc download error: {e}")
    print("  Trying alternative repo...")
    try:
        result = download_with_retry(
            snapshot_download,
            "chaoyuli/VidHalluc",
            local_dir=f"{vid_path}/chaoyuli_VidHalluc",
            resume_download=True,
        )
        if result:
            print(f"  VidHalluc (alt) downloaded to {result}")
    except Exception as e2:
        print(f"  VidHalluc alt also failed: {e2}")


# ====================================================================
# 4. MVBench (video understanding benchmark, NOT yet downloaded)
# ====================================================================
print("\n" + "=" * 60)
print("4. MVBench (new download)")
print("=" * 60)

mvbench_path = f"{DATASETS}/MVBench_hf"
if os.path.isdir(mvbench_path) and len(os.listdir(mvbench_path)) > 2:
    print(f"  Already exists with {len(os.listdir(mvbench_path))} items")
else:
    print("  Downloading MVBench...")
    try:
        result = download_with_retry(
            snapshot_download,
            "OpenGVLab/MVBench",
            local_dir=mvbench_path,
            resume_download=True,
        )
        if result:
            print(f"  MVBench downloaded to {result}")
    except Exception as e:
        print(f"  MVBench download error: {e}")


# ====================================================================
# Summary
# ====================================================================
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

# Check InstructBLIP
if os.path.exists(missing_shard):
    print(f"  InstructBLIP shard 2: OK ({os.path.getsize(missing_shard)/1e9:.2f} GB)")
else:
    print("  InstructBLIP shard 2: MISSING")

# Check FREAK
freak_parquets = [f for f in os.listdir(freak_path) if f.endswith(".parquet")] if os.path.isdir(freak_path) else []
print(f"  FREAK: {len(freak_parquets)} parquet files")

# Check VidHalluc
vid_items = []
for dp, dn, fns in os.walk(vid_path):
    vid_items.extend(fns)
print(f"  VidHalluc: {len(vid_items)} total files in video/")

# Check MVBench
if os.path.isdir(mvbench_path):
    mvb_items = os.listdir(mvbench_path)
    print(f"  MVBench: {len(mvb_items)} items")
else:
    print("  MVBench: NOT FOUND")

# Disk
import shutil
stat = shutil.disk_usage("/root/autodl-tmp")
print(f"\n  Disk free: {stat.free/1e9:.1f} GB / {stat.total/1e9:.1f} GB")
print("\nDONE")
