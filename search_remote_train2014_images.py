import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

roots = [Path("/root/autodl-tmp")]
keywords = ["train2014", "instances_train2014.json", "captions_train2014.json"]

for root in roots:
    print("=" * 80)
    print(root)
    print("=" * 80)
    shown = 0
    for path in root.rglob("*"):
        low = path.name.lower()
        if any(key in low for key in keywords):
            print(path)
            shown += 1
            if shown >= 200:
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
