import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
set -e
source /etc/network_turbo
export PATH="/root/miniconda3/bin:$PATH"
mkdir -p /root/autodl-tmp/BRA_Project/models
rm -rf /root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14
/root/miniconda3/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="/root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14",
    local_dir_use_symlinks=False,
)
print(path)
PY
ls -lah /root/autodl-tmp/BRA_Project/models/clip-vit-large-patch14
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=1800)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
