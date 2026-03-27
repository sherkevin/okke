import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
cd /root/autodl-tmp/BRA_Project
echo "=== GPU ==="
nvidia-smi
echo "=== DOCVQA LOG ==="
tail -n 80 logs/v3_engineer_b/docvqa_contract_ready.log 2>/dev/null || true
echo "=== MMMU LOG ==="
tail -n 80 logs/v3_engineer_b/mmmu_hard_review.log 2>/dev/null || true
echo "=== OUTPUTS ==="
ls -lah logs/v3_engineer_b/*rerun* logs/v3_engineer_b/docvqa_contract_ready.log logs/v3_engineer_b/mmmu_hard_review.log 2>/dev/null || true
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
