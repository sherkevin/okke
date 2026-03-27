"""One-off: stream Charades zip from URL to local D: and remote SFTP; remove local after."""
import os
import sys
import urllib.request

import paramiko

URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
HOST = "connect.westd.seetacloud.com"
PORT = 23427
USER = "root"
REMOTE_DIR = "/root/autodl-tmp/A-OSP_Project/data/charades"
LOCAL_PATH = r"D:\Charades_v1_480.zip"
CHUNK = 4 * 1024 * 1024


def main() -> int:
    password = os.environ.get("SEETACLOUD_SSH_PASS")
    if not password:
        print("Set SEETACLOUD_SSH_PASS", file=sys.stderr)
        return 1

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=password, timeout=60)

    _, stdout, stderr = ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
    if stdout.channel.recv_exit_status() != 0:
        err = stderr.read().decode()
        print(err, file=sys.stderr)
        ssh.close()
        return 1

    sftp = ssh.open_sftp()
    remote_path = f"{REMOTE_DIR}/Charades_v1_480.zip"

    print("Opening HTTP stream…")
    resp = urllib.request.urlopen(URL, timeout=120)

    print(f"Writing: {LOCAL_PATH} + {remote_path}")
    total = 0
    last_log_mib = 0
    with open(LOCAL_PATH, "wb") as flocal, sftp.open(remote_path, "wb") as fremote:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            flocal.write(chunk)
            fremote.write(chunk)
            total += len(chunk)
            mib = total // (1024 * 1024)
            if mib >= last_log_mib + 200:
                print(f"  … {mib} MiB")
                last_log_mib = mib

    sftp.close()
    ssh.close()
    resp.close()

    print(f"Done. {total // (1024 * 1024)} MiB transferred. Removing local copy…")
    os.remove(LOCAL_PATH)
    print("Local file removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
