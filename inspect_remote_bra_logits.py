import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

path = Path("/root/autodl-tmp/BRA_Project/bra_logits_processor.py")
text = path.read_text(encoding="utf-8", errors="replace").splitlines()
keys = ["def make_bra_config", "tlra_full", "tlra_no_vasm", "tlra_adaptivetopk", "bra_zero"]

for key in keys:
    print("=" * 80)
    print(f"SEARCH: {key}")
    print("=" * 80)
    found = False
    for idx, line in enumerate(text, start=1):
        if key in line:
            found = True
            lo = max(1, idx - 3)
            hi = min(len(text), idx + 6)
            for j in range(lo, hi + 1):
                print(f"{j}:{text[j-1]}")
            break
    if not found:
        print("NOT FOUND")
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
