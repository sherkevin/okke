import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")

targets = {
    "docvqa": "HuggingFaceM4/DocumentVQA",
    "videomme": "lmms-lab/Video-MME",
}

report = {}
for key, repo_id in targets.items():
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
        siblings = []
        total = 0
        for s in info.siblings:
            entry = {"rfilename": s.rfilename}
            size = getattr(s, "size", None)
            if size is not None:
                entry["size"] = size
                total += size
            siblings.append(entry)
        siblings = sorted(siblings, key=lambda x: x.get("size", 0), reverse=True)
        report[key] = {
            "repo_id": repo_id,
            "total_size_bytes_known": total,
            "largest_files": siblings[:20],
        }
    except Exception as e:
        report[key] = {"error": str(e)}

print(json.dumps(report, ensure_ascii=False, indent=2))
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    cmd = f'export PATH="/root/miniconda3/bin:$PATH"; source /etc/network_turbo >/dev/null 2>&1 || true; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_PY}\nPY'
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
