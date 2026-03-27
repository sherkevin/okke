from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

LOCAL = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\download_docvqa_videomme_20260321.md")
REMOTE = "/root/autodl-tmp/BRA_Project/logs/download_docvqa_videomme_20260321.md"


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()
    sftp.put(str(LOCAL), REMOTE)
    sftp.close()
    client.close()
    print("uploaded")


if __name__ == "__main__":
    main()
