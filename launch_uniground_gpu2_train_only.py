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
    "bra_universal_plugin.py",
    "uniground_runtime.py",
    "train_universal_plugin.py",
    "export_universal_coco_payload.py",
    "validate_uniground_universality.py",
]


REMOTE_BASH = r"""
set -e
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
mkdir -p logs/uniground_v6 train_data/uniground_v6 models/uniground_v6
rm -f /root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu2_export_train.log

CUDA_VISIBLE_DEVICES=2 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
/root/miniconda3/bin/python export_universal_coco_payload.py \
  --image-dir /root/autodl-tmp/BRA_Project/datasets/coco2014/val2014 \
  --instances-json /root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/instances_val2014.json \
  --encoder-path /root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14 \
  --output /root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt \
  --config-dump /root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.config.json \
  --device cuda:0 \
  --max-images 4096 \
  --image-batch-size 32 \
  --text-batch-size 256 && \
/root/miniconda3/bin/python train_universal_plugin.py \
  --features /root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.pt \
  --output /root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt \
  --config-dump /root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json \
  --epochs 6 \
  --batch-size 256 \
  --lr 1e-3 \
  --hidden-dim 512 && \
/root/miniconda3/bin/python validate_uniground_universality.py \
  --checkpoint /root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt \
  > /root/autodl-tmp/BRA_Project/logs/uniground_v6/psi_univ_checkpoint_validation.json
' > /root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu2_export_train.log 2>&1 &

ps -eo pid,etimes,cmd | grep -E "export_universal_coco_payload|train_universal_plugin" | grep -v grep
"""


def upload_file(sftp: paramiko.SFTPClient, local_name: str) -> None:
    local_path = LOCAL_ROOT / local_name
    remote_path = posixpath.join(REMOTE_ROOT, local_name)
    sftp.put(str(local_path), remote_path)
    print(f"uploaded {local_name}")


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()
    for name in UPLOADS:
        upload_file(sftp, name)
    sftp.close()

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
