import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

root = Path("/root/autodl-tmp/BRA_Project")
candidates = [
    root / "configs",
    root / "smoke_data",
    root / "data",
    root / "datasets",
    root / "logs" / "v3_contract",
]

for base in candidates:
    print("=" * 80)
    print(base)
    print("=" * 80)
    if not base.exists():
        print("MISSING")
        continue
    shown = 0
    for path in sorted(base.rglob("*")):
        low = path.name.lower()
        if any(key in low for key in ["lora", "matched", "phi", "jsonl", "train", "config"]):
            print(path)
            shown += 1
            if shown >= 80:
                break
    if shown == 0:
        print("NO_MATCHES")
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
