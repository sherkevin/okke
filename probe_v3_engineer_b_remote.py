import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
echo "=== GPU ==="
nvidia-smi
echo "=== screen ==="
screen -ls 2>&1 || true
echo "=== logs/v3_engineer_b ==="
mkdir -p logs/v3_engineer_b
ls -lah logs/v3_engineer_b || true
echo "=== existing python jobs ==="
ps -eo pid,etimes,cmd | grep -E 'bra_eval_matrix.py|run_eval_pipeline.py' | grep -v grep || true
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
