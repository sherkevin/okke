import pandas as pd
hb = pd.read_parquet("/root/autodl-tmp/BRA_Project/datasets/HallusionBench_hf/data/image-00000-of-00001.parquet")
print("gt_answer unique values:", list(hb["gt_answer"].unique()[:10]))
print("Sample gt:", repr(hb.iloc[0]["gt_answer"]))
print("Sample question:", hb.iloc[0]["question"][:150])
print()
# Also check the JSON results
import json
with open("/root/autodl-tmp/BRA_Project/bra_eval_qwen3vl.json") as f:
    results = json.load(f)
for r in results.get("qwen3vl2b", []):
    print(f"{r['dataset']}: baseline={r['baseline']}, bra={r['bra']}")
