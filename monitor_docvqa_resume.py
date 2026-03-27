import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
export PATH="/root/miniconda3/bin:$PATH"
echo "=== screen ==="
screen -ls | grep docvqa_resume || true
echo "=== tail log ==="
tail -n 60 /root/autodl-tmp/BRA_Project/logs/downloads/docvqa_resume.log 2>/dev/null || true
echo "=== summary ==="
cat /root/autodl-tmp/BRA_Project/logs/downloads/docvqa_summary.json 2>/dev/null || echo "docvqa_summary.json missing"
echo "=== path check ==="
du -sh /root/autodl-tmp/BRA_Project/datasets/DocVQA_hf 2>/dev/null || true
find /root/autodl-tmp/BRA_Project/datasets/DocVQA_hf -type f 2>/dev/null | wc -l || true
ls -la /root/autodl-tmp/BRA_Project/datasets/DocVQA 2>/dev/null || echo "DocVQA alias missing"
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
