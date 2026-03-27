import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

paths = [
    Path("/root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu2_qwen4b_pope_then_chair_full_matrix.log"),
    Path("/root/autodl-tmp/BRA_Project/logs/uniground_v6/gpu0_qwen2b_pope_then_mmbench_mme.log"),
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
    if not lines:
        print("EMPTY")
        continue
    for line in lines[-60:]:
        print(line)

print("=" * 80)
print("OUTPUT DIRECTORIES")
print("=" * 80)
for rel in [
    "logs/uniground_v6/second_host_qwen4b",
    "logs/uniground_v6/second_host_qwen2b",
    "logs/uniground_v6/qwen2b_table2",
]:
    path = Path("/root/autodl-tmp/BRA_Project") / rel
    print(f"[{path}]")
    if not path.exists():
        print("MISSING")
        continue
    children = sorted(path.iterdir())
    if not children:
        print("EMPTY")
        continue
    for child in children[-20:]:
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
