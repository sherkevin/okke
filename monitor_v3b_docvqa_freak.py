import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
echo "=== gpu ==="
nvidia-smi
echo "=== docvqa main tail ==="
tail -n 80 logs/v3_engineer_b/docvqa_main.log 2>/dev/null || true
echo "=== freak parity tail ==="
tail -n 120 logs/v3_engineer_b/freak_parity.log 2>/dev/null || true
echo "=== current outputs ==="
ls -lah logs/v3_engineer_b 2>/dev/null || true
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
