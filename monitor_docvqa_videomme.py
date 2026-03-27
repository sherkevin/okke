import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
echo "=== screen ==="
screen -ls | grep docvqa_videomme_dl || true
echo "=== process ==="
ps -eo pid,etimes,pcpu,pmem,cmd | grep _download_docvqa_videomme.py | grep -v grep || true
echo "=== tail log ==="
tail -n 40 /root/autodl-tmp/BRA_Project/logs/downloads/docvqa_videomme_download.log 2>/dev/null || true
echo "=== docvqa dir ==="
du -sh /root/autodl-tmp/BRA_Project/datasets/DocVQA_hf 2>/dev/null || echo "DocVQA_hf missing"
find /root/autodl-tmp/BRA_Project/datasets/DocVQA_hf -type f 2>/dev/null | wc -l || true
ls -la /root/autodl-tmp/BRA_Project/datasets/DocVQA 2>/dev/null || echo "DocVQA alias missing"
echo "=== videomme dir ==="
du -sh /root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf 2>/dev/null || echo "Video-MME_hf missing"
find /root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf -type f 2>/dev/null | wc -l || true
echo "=== disk ==="
df -h /root/autodl-tmp | sed -n '1,2p'
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
