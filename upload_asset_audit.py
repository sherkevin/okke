from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


LOCAL = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\asset_audit_标杆V1_20260321.md")
REMOTE = "/root/autodl-tmp/BRA_Project/logs/asset_audit_标杆V1_20260321.md"


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    sftp = client.open_sftp()
    sftp.put(str(LOCAL), REMOTE)
    sftp.close()
    stdin, stdout, stderr = client.exec_command(
        f'export PATH="/root/miniconda3/bin:$PATH"; /root/miniconda3/bin/python - <<\'PY\'\nfrom pathlib import Path\np = Path("{REMOTE}")\nprint(p.exists(), p.stat().st_size)\nPY'
    )
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err)
    client.close()


if __name__ == "__main__":
    main()
