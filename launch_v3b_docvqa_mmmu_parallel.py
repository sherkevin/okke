from __future__ import annotations

import posixpath
from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

LOCAL_ROOT = Path(r"d:\Shervin\OneDrive\Desktop\breaking")
REMOTE_ROOT = "/root/autodl-tmp/BRA_Project"

UPLOADS = [
    ("bra_eval_matrix.py", "bra_eval_matrix.py"),
    ("bra_logits_processor.py", "bra_logits_processor.py"),
]

REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/v3_engineer_b

CUDA_VISIBLE_DEVICES=3 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset docvqa \
  --n_samples 200 \
  --bra_method tlra_full \
  --output logs/v3_engineer_b/docvqa_tlra_full_rerun.json ;
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset docvqa \
  --n_samples 200 \
  --bra_method tlra_no_vasm \
  --output logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json
' > logs/v3_engineer_b/docvqa_contract_ready.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset mmmu \
  --n_samples 200 \
  --mmmu_manifest /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json \
  --bra_method tlra_full \
  --output logs/v3_engineer_b/mmmu_hard_tlra_full_rerun.json ;
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset mmmu \
  --n_samples 200 \
  --mmmu_manifest /root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json \
  --bra_method tlra_no_vasm \
  --output logs/v3_engineer_b/mmmu_hard_tlra_no_vasm_rerun.json
' > logs/v3_engineer_b/mmmu_hard_review.log 2>&1 &

echo "launched parallel reruns"
ps -eo pid,etimes,cmd | grep -E 'bra_eval_matrix.py --model qwen3vl2b --dataset (docvqa|mmmu)' | grep -v grep || true
"""


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()

    for local_name, remote_name in UPLOADS:
        local_path = LOCAL_ROOT / local_name
        remote_path = posixpath.join(REMOTE_ROOT, remote_name)
        print(f"Uploading {local_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)

    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
