#!/bin/bash
MODEL=/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b

echo "=== All files in model dir ==="
ls -lh $MODEL/ | grep -v "^total"

echo ""
echo "=== config.json (first 40 lines) ==="
python3 -m json.tool $MODEL/config.json 2>/dev/null | head -40
[ $? -ne 0 ] && echo "  FAIL: config.json not valid JSON or missing"

echo ""
echo "=== dtype check (shard 1) ==="
python3 << 'PYEOF'
from safetensors import safe_open
import os

MODEL = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
shard1 = f"{MODEL}/model-00001-of-00004.safetensors"

with safe_open(shard1, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"  Total keys in shard1: {len(keys)}")
    for k in keys[:3]:
        t = f.get_tensor(k)
        print(f"  {k}: dtype={t.dtype}, shape={t.shape}")
    # Check specific important weights
    for k in keys:
        if "embed_tokens.weight" in k:
            t = f.get_tensor(k)
            print(f"\n  EMBED: {k}: dtype={t.dtype}, shape={t.shape}")
        if "layers.31.self_attn" in k:
            print(f"  Found layer 31 -> 7B (32 layers)")
            break
        if "layers.39.self_attn" in k:
            print(f"  Found layer 39 -> 13B (40 layers)")
            break
PYEOF

echo ""
echo "=== Check if this matches HF original (by checking index) ==="
python3 << 'PYEOF'
import os, json, subprocess

MODEL = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
idx_path = f"{MODEL}/model.safetensors.index.json"
if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
    total = idx.get("metadata", {}).get("total_size", 0)
    print(f"  Index exists. total_size: {total/1e9:.2f} GB")
else:
    print("  No index file - need to download from HF")
    print("  Attempting to fetch just the index from HF mirror...")
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import hf_hub_download
    try:
        f = hf_hub_download(
            "Salesforce/instructblip-vicuna-7b",
            "model.safetensors.index.json",
            local_dir=MODEL,
        )
        idx = json.load(open(f))
        total = idx.get("metadata", {}).get("total_size", 0)
        shards = sorted(set(idx["weight_map"].values()))
        print(f"  Downloaded index. total_size: {total/1e9:.2f} GB")
        print(f"  Shards: {shards}")
    except Exception as e:
        print(f"  FAIL: {e}")
PYEOF

echo ""
echo "=== Check if config.json missing and download it ==="
python3 << 'PYEOF'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"

missing = []
for fname in ["config.json", "generation_config.json", "preprocessor_config.json", 
              "processor_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.model"]:
    p = os.path.join(MODEL, fname)
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        missing.append(fname)

if missing:
    print(f"  Missing config files: {missing}")
    from huggingface_hub import hf_hub_download
    for fname in missing:
        try:
            f = hf_hub_download("Salesforce/instructblip-vicuna-7b", fname, local_dir=MODEL)
            print(f"  Downloaded: {fname}")
        except Exception as e:
            print(f"  FAIL {fname}: {e}")
else:
    print("  All config files present")
PYEOF
