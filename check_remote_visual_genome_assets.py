import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

targets = [
    Path("/root/autodl-tmp/BRA_Project/train_data"),
    Path("/root/autodl-tmp/BRA_Project/train_data/phi_calib_matched_budget_50k.jsonl"),
    Path("/root/autodl-tmp/BRA_Project/datasets/VisualGenome"),
    Path("/root/autodl-tmp/BRA_Project/datasets/visual_genome"),
    Path("/root/autodl-tmp/VisualGenome"),
    Path("/root/autodl-tmp/visual_genome"),
    Path("/root/autodl-tmp/VG_100K"),
    Path("/root/autodl-tmp/VG_100K_2"),
]

for path in targets:
    print("=" * 80)
    print(path)
    print("=" * 80)
    print("exists=", path.exists())
    if path.is_file():
        print("size=", path.stat().st_size)
    if path.is_dir():
        try:
            entries = sorted(p.name for p in path.iterdir())[:40]
            for entry in entries:
                print(entry)
        except Exception as exc:
            print("dir_read_error=", repr(exc))

print("=" * 80)
print("COMMON VG FILE SEARCH")
print("=" * 80)
roots = [Path("/root/autodl-tmp"), Path("/root/autodl-tmp/BRA_Project")]
want = {
    "region_descriptions.json",
    "image_data.json",
    "objects.json",
    "attributes.json",
    "question_answers.json",
    "relationships.json",
}
seen = set()
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path.name in want or path.name.startswith("VG_100K"):
            s = str(path)
            if s not in seen:
                seen.add(s)
                print(s)
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
