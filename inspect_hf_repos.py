import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

report = {}
try:
    from huggingface_hub import HfApi
    api = HfApi(endpoint="https://hf-mirror.com")
    report["huggingface_hub"] = "ok"
except Exception as e:
    print(json.dumps({"error": f"huggingface_hub import failed: {e}"}, ensure_ascii=False))
    raise SystemExit(0)

targets = {
    "docvqa": "HuggingFaceM4/DocumentVQA",
    "videomme": "lmms-lab/Video-MME",
}

for key, repo_id in targets.items():
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        report[key] = {
            "repo_id": repo_id,
            "file_count": len(files),
            "sample_files": files[:50],
        }
    except Exception as e:
        report[key] = {"repo_id": repo_id, "error": str(e)}

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
