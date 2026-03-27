import json, os
import pandas as pd

BASE = "/root/autodl-tmp/BRA_Project/datasets"

# POPE
print("=== POPE ===")
pope_path = f"{BASE}/POPE/output/coco/coco_pope_adversarial.json"
with open(pope_path) as f:
    pope = [json.loads(l) for l in f]
print(f"Count: {len(pope)}")
print(f"Sample: {pope[0]}")
print(f"Keys: {list(pope[0].keys())}")

# COCO annotations
print("\n=== COCO Annotations ===")
ann_dir = f"{BASE}/coco2014/annotations"
if os.path.isdir(ann_dir):
    print(f"Files: {os.listdir(ann_dir)}")
    inst_path = f"{ann_dir}/instances_val2014.json"
    if os.path.exists(inst_path):
        with open(inst_path) as f:
            inst = json.load(f)
        cats = inst.get("categories", [])
        print(f"Categories: {len(cats)}, first: {cats[0] if cats else 'N/A'}")
else:
    print("NOT FOUND")

# COCO images count
coco_dir = f"{BASE}/coco2014/val2014"
if os.path.isdir(coco_dir):
    imgs = [f for f in os.listdir(coco_dir) if f.endswith(".jpg")]
    print(f"COCO val2014 images: {len(imgs)}")

# MMBench
print("\n=== MMBench ===")
mb = pd.read_parquet(f"{BASE}/MMBench_EN_hf/data/dev-00000-of-00001-75b6649fb044d38b.parquet")
print(f"Cols: {list(mb.columns)}")
print(f"Rows: {len(mb)}")
print(f"Sample answer: {mb.iloc[0].get('answer', 'N/A')}")

# MME
print("\n=== MME ===")
mme = pd.read_parquet(f"{BASE}/MME_hf/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet")
print(f"Cols: {list(mme.columns)}")
print(f"Rows: {len(mme)}")
print(f"Sample: question={mme.iloc[0].get('question', 'N/A')}, answer={mme.iloc[0].get('answer', 'N/A')}")

# HallusionBench
print("\n=== HallusionBench ===")
hb = pd.read_parquet(f"{BASE}/HallusionBench_hf/data/image-00000-of-00001.parquet")
print(f"Cols: {list(hb.columns)}")
print(f"Rows: {len(hb)}")

# MMMU
print("\n=== MMMU ===")
mmmu_dir = f"{BASE}/MMMU_hf/Art"
if os.path.isdir(mmmu_dir):
    files = [f for f in os.listdir(mmmu_dir) if f.endswith(".parquet")]
    print(f"Parquet files: {files}")
    if files:
        mm = pd.read_parquet(f"{mmmu_dir}/{files[0]}")
        print(f"Cols: {list(mm.columns)}")
        print(f"Rows: {len(mm)}")
