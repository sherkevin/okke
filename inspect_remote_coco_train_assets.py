import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

targets = [
    Path("/root/autodl-tmp/BRA_Project/datasets/coco2014/train2014"),
    Path("/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/captions_train2014.json"),
    Path("/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/instances_train2014.json"),
]

for path in targets:
    print("=" * 80)
    print(path)
    print("=" * 80)
    print("exists=", path.exists())
    if path.is_dir():
        count = 0
        for child in path.iterdir():
            count += 1
            if count <= 5:
                print(child.name)
        print("sample_count=", count)
    if path.is_file():
        print("size=", path.stat().st_size)
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
