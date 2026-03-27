from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/downloads datasets/video

cat > /root/autodl-tmp/BRA_Project/_download_docvqa_videomme.py <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

root = Path("/root/autodl-tmp/BRA_Project")
report = []

def record(name, path):
    p = Path(path)
    file_count = 0
    total_bytes = 0
    if p.exists():
        for cur, _, files in os.walk(p):
            for f in files:
                fp = Path(cur) / f
                try:
                    total_bytes += fp.stat().st_size
                    file_count += 1
                except Exception:
                    pass
    report.append({
        "name": name,
        "path": str(p),
        "exists": p.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    })

docvqa_dir = root / "datasets" / "DocVQA_hf"
videomme_dir = root / "datasets" / "video" / "Video-MME_hf"

print("[1/2] Downloading DocVQA ...", flush=True)
snapshot_download(
    repo_id="HuggingFaceM4/DocumentVQA",
    repo_type="dataset",
    local_dir=str(docvqa_dir),
    max_workers=8,
)
alias = root / "datasets" / "DocVQA"
if alias.exists() or alias.is_symlink():
    try:
        alias.unlink()
    except IsADirectoryError:
        pass
if not alias.exists():
    os.symlink(docvqa_dir, alias, target_is_directory=True)
record("DocVQA", docvqa_dir)

print("[2/2] Downloading Video-MME ...", flush=True)
snapshot_download(
    repo_id="lmms-lab/Video-MME",
    repo_type="dataset",
    local_dir=str(videomme_dir),
    max_workers=8,
)
alias2 = root / "datasets" / "video" / "Video-MME"
if alias2.exists() or alias2.is_symlink():
    try:
        alias2.unlink()
    except IsADirectoryError:
        pass
if not alias2.exists():
    os.symlink(videomme_dir, alias2, target_is_directory=True)
record("Video-MME", videomme_dir)

summary_path = root / "logs" / "downloads" / "docvqa_videomme_summary.json"
summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
PY

screen -dmS docvqa_videomme_dl bash -lc 'source /etc/network_turbo >/dev/null 2>&1 || true; export PATH="/root/miniconda3/bin:$PATH"; export HF_ENDPOINT="https://hf-mirror.com"; cd /root/autodl-tmp/BRA_Project; /root/miniconda3/bin/python _download_docvqa_videomme.py 2>&1 | tee logs/downloads/docvqa_videomme_download.log'

sleep 2
screen -ls | grep docvqa_videomme_dl || true
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
