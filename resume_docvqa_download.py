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
mkdir -p logs/downloads

cat > /root/autodl-tmp/BRA_Project/_resume_docvqa.py <<'PY'
import json
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
root = Path("/root/autodl-tmp/BRA_Project")
docvqa_dir = root / "datasets" / "DocVQA_hf"
alias = root / "datasets" / "DocVQA"

last_error = None
for attempt in range(1, 6):
    print(f"[DocVQA] attempt {attempt}/5", flush=True)
    try:
        snapshot_download(
            repo_id="HuggingFaceM4/DocumentVQA",
            repo_type="dataset",
            local_dir=str(docvqa_dir),
            max_workers=2,
        )
        break
    except Exception as e:
        last_error = str(e)
        print(f"[DocVQA] attempt {attempt} failed: {e}", flush=True)
        if attempt == 5:
            raise
        time.sleep(15)

if alias.exists() or alias.is_symlink():
    try:
        alias.unlink()
    except IsADirectoryError:
        pass
if not alias.exists():
    os.symlink(docvqa_dir, alias, target_is_directory=True)

file_count = 0
total_bytes = 0
for cur, _, files in os.walk(docvqa_dir):
    for f in files:
        fp = Path(cur) / f
        try:
            total_bytes += fp.stat().st_size
            file_count += 1
        except Exception:
            pass

summary = {
    "name": "DocVQA",
    "path": str(docvqa_dir),
    "alias": str(alias),
    "exists": docvqa_dir.exists(),
    "alias_exists": alias.exists(),
    "file_count": file_count,
    "total_bytes": total_bytes,
    "status": "completed",
    "last_error": last_error,
}

out = root / "logs" / "downloads" / "docvqa_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY

screen -dmS docvqa_resume bash -lc 'source /etc/network_turbo >/dev/null 2>&1 || true; export PATH="/root/miniconda3/bin:$PATH"; export HF_ENDPOINT="https://hf-mirror.com"; cd /root/autodl-tmp/BRA_Project; /root/miniconda3/bin/python _resume_docvqa.py 2>&1 | tee logs/downloads/docvqa_resume.log'
sleep 2
screen -ls | grep docvqa_resume || true
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
