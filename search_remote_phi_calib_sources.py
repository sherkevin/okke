import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

roots = [
    Path("/root/autodl-tmp/BRA_Project"),
    Path("/root/autodl-tmp"),
]
keywords = [
    "phi",
    "calib",
    "matched",
    "train_data",
    "visual",
    "genome",
    "vg",
    "caption",
    "jsonl",
]

seen = set()
for root in roots:
    print("=" * 80)
    print(f"ROOT {root}")
    print("=" * 80)
    if not root.exists():
        print("MISSING")
        continue
    shown = 0
    for path in root.rglob("*"):
        low = path.name.lower()
        if any(key in low for key in keywords):
            resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            print(resolved)
            shown += 1
            if shown >= 250:
                print("...TRUNCATED...")
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
