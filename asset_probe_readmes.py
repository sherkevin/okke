import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_SCRIPT = r"""
from pathlib import Path

paths = [
    Path("/root/autodl-tmp/BRA_Project/datasets/MMBench_EN_hf/README.md"),
    Path("/root/autodl-tmp/BRA_Project/datasets/MME_hf/README.md"),
    Path("/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/README.md"),
    Path("/root/autodl-tmp/BRA_Project/datasets/FREAK_hf/README.md"),
]

for path in paths:
    print("=" * 80)
    print(path)
    print("=" * 80)
    if not path.exists():
        print("MISSING")
        continue
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 40:
                    break
                print(line.rstrip())
    except Exception as e:
        print(f"ERROR: {e}")
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    stdin, stdout, stderr = client.exec_command(
        f'export PATH="/root/miniconda3/bin:$PATH"; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_SCRIPT}\nPY',
        timeout=120,
    )
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    if err.strip():
        print("STDERR:")
        print(err)
    print(out)


if __name__ == "__main__":
    main()
