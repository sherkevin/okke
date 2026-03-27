import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path
import os

paths = [
    Path("/root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu2_export_train.log"),
    Path("/root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu3_qwen4b_smoke.log"),
]

for path in paths:
    print("=" * 80)
    print(path)
    print("=" * 80)
    if not path.exists():
        print("MISSING")
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for line in lines[-80:]:
        print(line)

print("=" * 80)
print("UNIGROUND OUTPUTS")
print("=" * 80)
root = Path("/root/autodl-tmp/BRA_Project")
for rel in [
    "train_data/uniground_v6",
    "models/uniground_v6",
    "logs/uniground_v6/second_host_smoke",
]:
    path = root / rel
    print(f"[{path}]")
    if not path.exists():
        print("MISSING")
        continue
    for child in sorted(path.iterdir()):
        if child.is_file():
            print(f"{child.name}\t{child.stat().st_size}")
        else:
            print(f"{child.name}/")
"""


REMOTE_CMD = (
    'export PATH="/root/miniconda3/bin:$PATH"; '
    'nvidia-smi && '
    f"/root/miniconda3/bin/python - <<'PY'\n{REMOTE_PY}\nPY"
)


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_CMD, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
