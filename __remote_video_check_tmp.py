import json
import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


def run(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return {"out": out, "err": err}


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    cmds = {
        "video_root": "bash -lc 'ls -1 /root/autodl-tmp/BRA_Project/datasets/video'",
        "vidhalluc": "bash -lc 'ls -1 /root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc'",
        "video_mme_dash": "bash -lc 'ls -1 /root/autodl-tmp/BRA_Project/datasets/video/Video-MME'",
        "video_mme_underscore": "bash -lc 'ls -1 /root/autodl-tmp/BRA_Project/datasets/video/Video_MME'",
    }
    results = {k: run(client, v) for k, v in cmds.items()}
    client.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
