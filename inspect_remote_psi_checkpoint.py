import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
import json
from pathlib import Path

import torch

from uniground_runtime import load_universal_scorer
from validate_uniground_universality import validate_checkpoint

ckpt = Path("/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt")
cfg = Path("/root/autodl-tmp/BRA_Project/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.config.json")
payload_cfg = Path("/root/autodl-tmp/BRA_Project/train_data/uniground_v6/psi_univ_coco_val2014_payload.config.json")
validator_out = Path("/root/autodl-tmp/BRA_Project/logs/uniground_v6/psi_univ_checkpoint_validation.json")

print("=" * 80)
print("LOAD_SMOKE")
print("=" * 80)
scorer = load_universal_scorer(ckpt, device="cpu")
print({"loaded_class": scorer.__class__.__name__})

print("=" * 80)
print("VALIDATOR")
print("=" * 80)
ok, checks, info = validate_checkpoint(ckpt)
print(json.dumps({"ok": ok, "checks": checks, "info": info}, ensure_ascii=False, indent=2))

for path in [cfg, payload_cfg, validator_out]:
    print("=" * 80)
    print(path)
    print("=" * 80)
    if not path.exists():
        print("MISSING")
        continue
    print(path.read_text(encoding="utf-8", errors="replace"))
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    cmd = f'export PATH="/root/miniconda3/bin:$PATH"; cd /root/autodl-tmp/BRA_Project; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_PY}\nPY'
    stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
