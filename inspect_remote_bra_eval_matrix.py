import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

path = Path("/root/autodl-tmp/BRA_Project/bra_eval_matrix.py")
text = path.read_text(encoding="utf-8", errors="replace").splitlines()

for start, end in [(1126, 1136), (945, 983)]:
    print("=" * 80)
    print(f"{path}:{start}-{end}")
    print("=" * 80)
    for idx in range(start - 1, min(end, len(text))):
        print(f"{idx+1}:{text[idx]}")
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
