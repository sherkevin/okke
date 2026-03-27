import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
echo "=== WHOAMI ==="
whoami
echo "=== PWD ==="
pwd
echo "=== DISK ==="
df -h /root /root/autodl-tmp 2>/dev/null || true
echo "=== HF CLI ==="
which huggingface-cli || true
huggingface-cli --help >/dev/null 2>&1 && echo "huggingface-cli: OK" || echo "huggingface-cli: MISSING"
echo "=== DATASET ROOT ==="
ls -la /root/autodl-tmp/BRA_Project/datasets
echo "=== DOCVQA SEARCH ==="
ls -la /root/autodl-tmp/BRA_Project/datasets/DocVQA 2>/dev/null || echo "DocVQA path missing"
ls -la /root/autodl-tmp/BRA_Project/datasets | grep -i doc || true
echo "=== VIDEO SEARCH ==="
ls -la /root/autodl-tmp/BRA_Project/datasets/video 2>/dev/null || echo "video dir missing"
ls -la /root/autodl-tmp/BRA_Project/datasets | grep -i video || true
echo "=== VIDEO-MME SEARCH ==="
for p in \
  /root/autodl-tmp/BRA_Project/datasets/Video-MME \
  /root/autodl-tmp/BRA_Project/datasets/video/Video-MME \
  /root/autodl-tmp/BRA_Project/datasets/video/videomme \
  /root/autodl-tmp/BRA_Project/datasets/video/VideoMME
do
  if [ -e "$p" ]; then
    echo "FOUND: $p"
    ls -la "$p" | sed -n '1,20p'
  fi
done
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
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
