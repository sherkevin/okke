from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

LOCAL_PATH = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\uniground_v6\checkpoint_release_feed_20260322.md")
REMOTE_PATH = "/root/autodl-tmp/BRA_Project/experiment_logs/uniground_v6/checkpoint_release_feed_20260322.md"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()
    try:
        sftp.mkdir("/root/autodl-tmp/BRA_Project/experiment_logs")
    except IOError:
        pass
    try:
        sftp.mkdir("/root/autodl-tmp/BRA_Project/experiment_logs/uniground_v6")
    except IOError:
        pass
    sftp.put(str(LOCAL_PATH), REMOTE_PATH)
    sftp.close()
    client.close()
    print(f"uploaded {LOCAL_PATH} -> {REMOTE_PATH}")


if __name__ == "__main__":
    main()
