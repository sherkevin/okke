import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
export PATH="/root/miniconda3/bin:$PATH"
echo "=== screen ==="
screen -ls | grep videomme_resume || true
echo "=== process ==="
ps -eo pid,etimes,pcpu,pmem,cmd | grep _resume_videomme.py | grep -v grep || true
echo "=== tail log ==="
tail -n 60 /root/autodl-tmp/BRA_Project/logs/downloads/videomme_resume.log 2>/dev/null || true
echo "=== summary ==="
cat /root/autodl-tmp/BRA_Project/logs/downloads/videomme_summary.json 2>/dev/null || echo "videomme_summary.json missing"
echo "=== path check ==="
du -sh /root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf 2>/dev/null || true
find /root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf -type f 2>/dev/null | wc -l || true
ls -la /root/autodl-tmp/BRA_Project/datasets/video/Video-MME 2>/dev/null || echo "Video-MME alias missing"
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
