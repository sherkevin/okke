import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
cd /root/autodl-tmp/BRA_Project
echo "=== DOCVQA_FULL ==="
sed -n '1,240p' logs/v3_engineer_b/docvqa_tlra_full.json 2>/dev/null || true
echo "=== DOCVQA_NO_VASM ==="
sed -n '1,240p' logs/v3_engineer_b/docvqa_tlra_no_vasm.json 2>/dev/null || true
echo "=== VIDHALLUC_LOG ==="
sed -n '1,220p' logs/v3_engineer_b/vidhalluc_main.log 2>/dev/null || true
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
