"""Fix remaining downloads: InstructBLIP shard, MVBench, verify VidHalluc/FREAK."""
import os
import sys
import shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Disable hf_transfer to avoid SSL issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

BASE = "/root/autodl-tmp/BRA_Project"
MODELS = f"{BASE}/models"
DATASETS = f"{BASE}/datasets"

# ====================================================================
# 1. InstructBLIP-Vicuna-7B missing shard
# ====================================================================
print("=" * 60)
print("1. InstructBLIP-Vicuna-7B shard 2")
print("=" * 60)

ib_path = f"{MODELS}/instructblip-vicuna-7b"
shard2 = f"{ib_path}/model-00002-of-00004.safetensors"

if os.path.exists(shard2) and os.path.getsize(shard2) > 1e9:
    print(f"  SKIP: already exists ({os.path.getsize(shard2)/1e9:.2f} GB)")
else:
    cache = f"{ib_path}/.cache"
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print(f"  Cleaned stale cache")

    print("  Downloading shard 2 (regular mode)...")
    try:
        f = hf_hub_download(
            "Salesforce/instructblip-vicuna-7b",
            "model-00002-of-00004.safetensors",
            local_dir=ib_path,
        )
        print(f"  OK: {os.path.getsize(f)/1e9:.2f} GB")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Trying wget fallback...")
        url = "https://hf-mirror.com/Salesforce/instructblip-vicuna-7b/resolve/main/model-00002-of-00004.safetensors"
        os.system(f'wget -c -O "{shard2}" "{url}" 2>&1 | tail -3')
        if os.path.exists(shard2):
            print(f"  wget: {os.path.getsize(shard2)/1e9:.2f} GB")

# ====================================================================
# 2. MVBench (as dataset repo)
# ====================================================================
print("\n" + "=" * 60)
print("2. MVBench")
print("=" * 60)

mvbench_path = f"{DATASETS}/MVBench_hf"
try:
    files = list_repo_files("OpenGVLab/MVBench", repo_type="dataset")
    print(f"  Found {len(files)} files in dataset repo")
    r = snapshot_download("OpenGVLab/MVBench", repo_type="dataset",
                          local_dir=mvbench_path)
    print(f"  OK: Downloaded to {r}")
except Exception as e:
    print(f"  Dataset repo failed: {e}")
    try:
        files = list_repo_files("OpenGVLab/MVBench")
        print(f"  Model repo has {len(files)} files")
        r = snapshot_download("OpenGVLab/MVBench", local_dir=mvbench_path)
        print(f"  OK: Model repo downloaded to {r}")
    except Exception as e2:
        print(f"  FAIL: {e2}")

# ====================================================================
# 3. Verify FREAK
# ====================================================================
print("\n" + "=" * 60)
print("3. FREAK verification")
print("=" * 60)

freak_path = f"{DATASETS}/FREAK_hf"
data_dir = f"{freak_path}/data"
if os.path.isdir(data_dir):
    parquets = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    total_sz = sum(os.path.getsize(os.path.join(data_dir, f)) for f in parquets)
    print(f"  Parquets: {len(parquets)}, Total: {total_sz/1e6:.1f} MB")
    for p in sorted(parquets):
        sz = os.path.getsize(os.path.join(data_dir, p))
        print(f"    {p}: {sz/1e6:.1f} MB")
else:
    print(f"  No data/ directory found in {freak_path}")
    print(f"  Contents: {os.listdir(freak_path) if os.path.isdir(freak_path) else 'N/A'}")

# ====================================================================
# 4. Verify VidHalluc
# ====================================================================
print("\n" + "=" * 60)
print("4. VidHalluc verification")
print("=" * 60)

vid_base = f"{DATASETS}/video"
for d in os.listdir(vid_base) if os.path.isdir(vid_base) else []:
    p = os.path.join(vid_base, d)
    if os.path.isdir(p):
        n_files = sum(1 for _, _, fns in os.walk(p) for f in fns)
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(p) for f in fns)
        print(f"  {d}: {n_files} files, {sz/1e6:.1f} MB")

# ====================================================================
# 5. Full inventory
# ====================================================================
print("\n" + "=" * 60)
print("5. FULL INVENTORY")
print("=" * 60)

print("\nMODELS:")
for m in sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else []:
    mp = os.path.join(MODELS, m)
    if os.path.isdir(mp):
        n = sum(1 for _, _, fns in os.walk(mp) for f in fns if not f.startswith('.'))
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(mp) for f in fns
                 if not f.startswith('.'))
        shard_ok = True
        safetensors = [f for _, _, fns in os.walk(mp) for f in fns if f.endswith('.safetensors')]
        bins = [f for _, _, fns in os.walk(mp) for f in fns if f.endswith('.bin') and 'model' in f]
        weights = safetensors or bins
        print(f"  {m}: {n} files, {sz/1e9:.2f} GB, weights={len(weights)}")

print("\nDATASETS:")
for d in sorted(os.listdir(DATASETS)) if os.path.isdir(DATASETS) else []:
    dp = os.path.join(DATASETS, d)
    if os.path.isdir(dp):
        n = sum(1 for _, _, fns in os.walk(dp) for f in fns if not f.startswith('.'))
        sz = sum(os.path.getsize(os.path.join(ddp, f))
                 for ddp, _, fns in os.walk(dp) for f in fns
                 if not f.startswith('.'))
        print(f"  {d}: {n} files, {sz/1e6:.1f} MB")

stat = shutil.disk_usage("/root/autodl-tmp")
print(f"\nDisk: {stat.used/1e9:.1f} GB / {stat.total/1e9:.1f} GB ({stat.free/1e9:.1f} GB free)")
print("\nDONE")
