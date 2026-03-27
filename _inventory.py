"""Full inventory of models and datasets on server."""
import os
import json
import glob

BASE = "/root/autodl-tmp/BRA_Project"
MODELS = f"{BASE}/models"
DATASETS = f"{BASE}/datasets"

def dir_size_gb(path):
    total = 0
    for dp, dn, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9

def check_model_completeness(path):
    """Check if a HF model directory looks complete."""
    files = os.listdir(path) if os.path.isdir(path) else []
    has_config = any("config.json" in f for f in files)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    bin_files = [f for f in files if f.endswith(".bin")]
    has_weights = len(safetensors) > 0 or len(bin_files) > 0
    has_tokenizer = any("tokenizer" in f.lower() for f in files)

    # Check for incomplete downloads (*.incomplete, *.tmp)
    incomplete = [f for f in files if ".incomplete" in f or ".tmp" in f]

    # Check model-xxxxx-of-xxxxx pattern for missing shards
    import re
    shard_pattern = re.compile(r'model-(\d+)-of-(\d+)')
    shards = {}
    for f in files:
        m = shard_pattern.search(f)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            shards[current] = total
    missing_shards = []
    if shards:
        total = list(shards.values())[0]
        for i in range(1, total + 1):
            if i not in shards:
                missing_shards.append(i)

    return {
        "has_config": has_config,
        "has_weights": has_weights,
        "has_tokenizer": has_tokenizer,
        "safetensors": len(safetensors),
        "bin_files": len(bin_files),
        "incomplete": incomplete,
        "missing_shards": missing_shards,
        "total_files": len(files),
        "complete": has_config and has_weights and not incomplete and not missing_shards,
    }

print("=" * 70)
print("MODEL INVENTORY")
print("=" * 70)

if os.path.isdir(MODELS):
    for name in sorted(os.listdir(MODELS)):
        path = os.path.join(MODELS, name)
        if not os.path.isdir(path):
            continue
        size = dir_size_gb(path)
        info = check_model_completeness(path)
        status = "COMPLETE" if info["complete"] else "INCOMPLETE"
        print(f"\n  {name}")
        print(f"    Size: {size:.2f} GB | Files: {info['total_files']} | Status: {status}")
        print(f"    config: {info['has_config']} | weights: {info['safetensors']}st+{info['bin_files']}bin | tokenizer: {info['has_tokenizer']}")
        if info["missing_shards"]:
            print(f"    MISSING SHARDS: {info['missing_shards']}")
        if info["incomplete"]:
            print(f"    INCOMPLETE FILES: {info['incomplete']}")

print("\n" + "=" * 70)
print("DATASET INVENTORY")
print("=" * 70)

if os.path.isdir(DATASETS):
    for name in sorted(os.listdir(DATASETS)):
        path = os.path.join(DATASETS, name)
        if not os.path.isdir(path):
            continue
        size = dir_size_gb(path)

        # Count key files
        parquets = []
        jsons = []
        images = []
        for dp, dn, fns in os.walk(path):
            for f in fns:
                if f.endswith(".parquet"):
                    parquets.append(f)
                elif f.endswith(".json"):
                    jsons.append(f)
                elif f.endswith((".jpg", ".png", ".jpeg")):
                    images.append(f)

        print(f"\n  {name}")
        print(f"    Size: {size:.2f} GB | Parquet: {len(parquets)} | JSON: {len(jsons)} | Images: {len(images)}")

        # Special checks
        if "coco" in name.lower():
            val_dir = os.path.join(path, "val2014")
            if os.path.isdir(val_dir):
                n_imgs = len([f for f in os.listdir(val_dir) if f.endswith(".jpg")])
                print(f"    val2014 images: {n_imgs}")
            ann_dir = os.path.join(path, "annotations")
            if os.path.isdir(ann_dir):
                print(f"    annotations: {os.listdir(ann_dir)}")

        # Check for incomplete parquet downloads
        incomplete = [f for f in parquets if ".incomplete" in f or ".tmp" in f]
        for dp, dn, fns in os.walk(path):
            for f in fns:
                if f.endswith(".incomplete") or f.endswith(".tmp"):
                    incomplete.append(os.path.join(dp, f))
        if incomplete:
            print(f"    INCOMPLETE: {incomplete}")

# Check other locations
print("\n" + "=" * 70)
print("OTHER RESOURCES")
print("=" * 70)

# MiniGPT-4
mgpt4 = f"{BASE}/MiniGPT-4"
if os.path.isdir(mgpt4):
    print(f"\n  MiniGPT-4 repo: EXISTS ({len(os.listdir(mgpt4))} items)")

# Checkpoints
ckpt_dir = f"{BASE}/checkpoints"
if os.path.isdir(ckpt_dir):
    for f in os.listdir(ckpt_dir):
        fp = os.path.join(ckpt_dir, f)
        print(f"  Checkpoint: {f} ({os.path.getsize(fp)/1e6:.1f} MB)")

# Disk usage
print("\n" + "=" * 70)
print("DISK USAGE")
print("=" * 70)
total_models = dir_size_gb(MODELS) if os.path.isdir(MODELS) else 0
total_datasets = dir_size_gb(DATASETS) if os.path.isdir(DATASETS) else 0
print(f"  Models:   {total_models:.2f} GB")
print(f"  Datasets: {total_datasets:.2f} GB")
print(f"  Total:    {total_models + total_datasets:.2f} GB")

# Check disk space
import shutil
stat = shutil.disk_usage("/root/autodl-tmp")
print(f"\n  Disk free: {stat.free/1e9:.1f} GB / {stat.total/1e9:.1f} GB")
