import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
import json
from pathlib import Path

paths = [
    Path("/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full.json"),
    Path("/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm.json"),
]

for path in paths:
    print("=" * 80)
    print(path)
    print("=" * 80)
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data["qwen3vl2b"][0]
    print("top_keys:", sorted(payload.keys()))
    print("baseline_keys:", sorted(payload["baseline"].keys()))
    print("bra_keys:", sorted(payload["bra"].keys()))
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    cmd = f'export PATH="/root/miniconda3/bin:$PATH"; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_PY}\nPY'
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
