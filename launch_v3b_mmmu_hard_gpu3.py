import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/v3_engineer_b datasets/MMMU_hf

cat > /root/autodl-tmp/BRA_Project/_mmmu_hard_manifest_builder.py <<'PY'
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Build a controllable MMMU hard manifest.")
    parser.add_argument("--root", default="/root/autodl-tmp/BRA_Project/datasets/MMMU_hf")
    parser.add_argument("--output", default="/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json")
    parser.add_argument("--split", default="validation", choices=["validation", "dev", "test"])
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit-per-subject", type=int, default=6)
    parser.add_argument("--split-tag", default="mmmu_hard_v3b_frozen")
    parser.add_argument("--difficulty", default="hard")
    return parser.parse_args()

def difficulty_matches(value, required: str) -> bool:
    if required == "*":
        return True
    return str(value or "").strip().lower() == required.lower()

def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    allowed_subjects = set(args.subjects) if args.subjects else None
    entries = []
    counts = defaultdict(int)
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue
        if allowed_subjects and subject_dir.name not in allowed_subjects:
            continue
        parquet_files = sorted(subject_dir.glob(f"{args.split}-*.parquet"))
        for parquet_path in parquet_files:
            df = pd.read_parquet(parquet_path)
            for row_idx, row in df.iterrows():
                if args.limit_per_subject is not None and counts[subject_dir.name] >= args.limit_per_subject:
                    break
                if not difficulty_matches(row.get("topic_difficulty"), args.difficulty):
                    continue
                sample_id = str(row.get("id", "")).strip()
                if not sample_id:
                    continue
                entries.append({
                    "subject": subject_dir.name,
                    "parquet_path": str(parquet_path),
                    "sample_id": sample_id,
                    "split_tag": args.split_tag,
                    "row_index": int(row_idx),
                    "topic_difficulty": str(row.get("topic_difficulty", "")),
                    "question_type": str(row.get("question_type", "")),
                })
                counts[subject_dir.name] += 1
    payload = {
        "root": str(root),
        "split": args.split,
        "split_tag": args.split_tag,
        "difficulty": args.difficulty,
        "subjects": sorted(counts.keys()),
        "sample_count": len(entries),
        "sample_count_by_subject": dict(sorted(counts.items())),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved MMMU manifest to {output}")
    print(f"Total samples: {len(entries)}")
    for subject, count in sorted(counts.items()):
        print(f"  {subject}: {count}")

if __name__ == "__main__":
    main()
PY

/root/miniconda3/bin/python _mmmu_hard_manifest_builder.py \
  --root /root/autodl-tmp/BRA_Project/datasets/MMMU_hf \
  --output /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json \
  --split validation \
  --difficulty hard \
  --limit-per-subject 6 > logs/v3_engineer_b/mmmu_hard_manifest_build.log 2>&1

CUDA_VISIBLE_DEVICES=3 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset mmmu \
  --n_samples 200 \
  --mmmu_manifest /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json \
  --bra_method tlra_full \
  --output logs/v3_engineer_b/mmmu_hard_tlra_full.json ;
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset mmmu \
  --n_samples 200 \
  --mmmu_manifest /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json \
  --bra_method tlra_no_vasm \
  --output logs/v3_engineer_b/mmmu_hard_tlra_no_vasm.json
' > logs/v3_engineer_b/mmmu_hard_main.log 2>&1 &

echo "launched: mmmu_hard_gpu3"
echo "manifest: /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json"
sed -n '1,80p' logs/v3_engineer_b/mmmu_hard_manifest_build.log || true
sleep 2
ps -eo pid,etimes,cmd | grep -E 'bra_eval_matrix.py --model qwen3vl2b --dataset mmmu' | grep -v grep || true
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
