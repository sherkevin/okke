"""
Fast parallel download of missing models/datasets using hf_transfer + aria2.
"""
import os
import sys
import time
import subprocess
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download, hf_hub_download

BASE = "/root/autodl-tmp/BRA_Project"
MODELS = f"{BASE}/models"
DATASETS = f"{BASE}/datasets"

results = {}

def report(name, status, detail=""):
    results[name] = {"status": status, "detail": detail}
    sym = "OK" if status == "ok" else ("SKIP" if status == "skip" else "FAIL")
    print(f"  [{sym}] {name}: {detail}")


# ====================================================================
# 1. InstructBLIP-Vicuna-7B: missing shard 2
# ====================================================================
print("=" * 60)
print("1. InstructBLIP-Vicuna-7B")
print("=" * 60)

ib_path = f"{MODELS}/instructblip-vicuna-7b"
shard2 = f"{ib_path}/model-00002-of-00004.safetensors"

if os.path.exists(shard2) and os.path.getsize(shard2) > 1e9:
    report("instructblip-shard2", "skip",
           f"Already exists ({os.path.getsize(shard2)/1e9:.2f} GB)")
else:
    # Clean stale partial downloads
    cache = f"{ib_path}/.cache"
    if os.path.isdir(cache):
        shutil.rmtree(cache)

    print("  Downloading via hf_transfer...")
    try:
        f = hf_hub_download(
            "Salesforce/instructblip-vicuna-7b",
            "model-00002-of-00004.safetensors",
            local_dir=ib_path,
        )
        if f and os.path.exists(f):
            report("instructblip-shard2", "ok",
                   f"Downloaded ({os.path.getsize(f)/1e9:.2f} GB)")
        else:
            report("instructblip-shard2", "fail", "Download returned None")
    except Exception as e:
        report("instructblip-shard2", "fail", str(e)[:200])


# ====================================================================
# 2. FREAK dataset
# ====================================================================
print("\n" + "=" * 60)
print("2. FREAK dataset")
print("=" * 60)

freak_path = f"{DATASETS}/FREAK_hf"
freak_cache = f"{freak_path}/.cache"
if os.path.isdir(freak_cache):
    shutil.rmtree(freak_cache)

# Check how many data parquets we have
existing_parquets = []
data_dir = f"{freak_path}/data"
if os.path.isdir(data_dir):
    existing_parquets = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]

if len(existing_parquets) >= 5:
    report("freak", "skip", f"Already have {len(existing_parquets)} parquets in data/")
else:
    print(f"  Have {len(existing_parquets)} parquets, downloading...")
    try:
        r = snapshot_download("Hiyouga/FREAK", local_dir=freak_path)
        new_parquets = [f for f in os.listdir(data_dir) if f.endswith(".parquet")] if os.path.isdir(data_dir) else []
        report("freak", "ok", f"{len(new_parquets)} parquets")
    except Exception as e:
        report("freak", "fail", str(e)[:200])


# ====================================================================
# 3. VidHalluc
# ====================================================================
print("\n" + "=" * 60)
print("3. VidHalluc")
print("=" * 60)

vid_base = f"{DATASETS}/video"
os.makedirs(vid_base, exist_ok=True)

# Clean stale caches
for subdir in os.listdir(vid_base):
    cache = os.path.join(vid_base, subdir, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(f"  Cleaned cache: {subdir}")

vidhalluc_path = f"{vid_base}/VidHalluc"
# Check if we already have usable data
has_data = False
for dp, dn, fns in os.walk(vid_base):
    for f in fns:
        if f.endswith(".parquet") or f.endswith(".mp4"):
            has_data = True
            break

print(f"  Existing video data: {has_data}")
print("  Downloading VidHalluc...")
try:
    r = snapshot_download("VidHalluc/VidHalluc", local_dir=vidhalluc_path)
    report("vidhalluc", "ok", f"Downloaded to {r}")
except Exception as e:
    print(f"  Primary repo failed: {e}")
    try:
        alt_path = f"{vid_base}/chaoyuli_VidHalluc"
        r = snapshot_download("chaoyuli/VidHalluc", local_dir=alt_path)
        report("vidhalluc", "ok", f"Alt downloaded to {r}")
    except Exception as e2:
        report("vidhalluc", "fail", str(e2)[:200])


# ====================================================================
# 4. MVBench
# ====================================================================
print("\n" + "=" * 60)
print("4. MVBench")
print("=" * 60)

mvbench_path = f"{DATASETS}/MVBench_hf"
if os.path.isdir(mvbench_path) and len(os.listdir(mvbench_path)) > 3:
    report("mvbench", "skip", f"Already exists ({len(os.listdir(mvbench_path))} items)")
else:
    print("  Downloading MVBench...")
    try:
        r = snapshot_download("OpenGVLab/MVBench", local_dir=mvbench_path)
        report("mvbench", "ok", f"Downloaded to {r}")
    except Exception as e:
        report("mvbench", "fail", str(e)[:200])


# ====================================================================
# Final Summary
# ====================================================================
print("\n" + "=" * 60)
print("FINAL DOWNLOAD SUMMARY")
print("=" * 60)
for name, info in results.items():
    sym = "+" if info["status"] == "ok" else ("-" if info["status"] == "skip" else "X")
    print(f"  [{sym}] {name}: {info['detail']}")

stat = shutil.disk_usage("/root/autodl-tmp")
print(f"\n  Disk: {stat.used/1e9:.1f} GB used / {stat.total/1e9:.1f} GB total ({stat.free/1e9:.1f} GB free)")
print("\nALL DONE")
