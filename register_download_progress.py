from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

REMOTE_REGISTRY = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"
LOCAL_SNAPSHOT = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\SHARED_DATA_REGISTRY_snapshot_20260321.md")

SECTION = """

### Download Update: DocVQA + Video-MME (2026-03-22)

- Local log: `experiment_logs/download_docvqa_videomme_20260321.md`
- Remote log: `/root/autodl-tmp/BRA_Project/logs/download_docvqa_videomme_20260321.md`

#### Status

- `DocVQA`: completed
  - path: `/root/autodl-tmp/BRA_Project/datasets/DocVQA_hf`
  - alias: `/root/autodl-tmp/BRA_Project/datasets/DocVQA`
  - file_count: `223`
  - size: `9,591,618,321 bytes`
  - summary: `/root/autodl-tmp/BRA_Project/logs/downloads/docvqa_summary.json`

- `Video-MME`: completed
  - path: `/root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf`
  - alias: `/root/autodl-tmp/BRA_Project/datasets/video/Video-MME`
  - file_count: `73`
  - size: `101,002,238,065 bytes`
  - log: `/root/autodl-tmp/BRA_Project/logs/downloads/videomme_resume.log`
  - summary: `/root/autodl-tmp/BRA_Project/logs/downloads/videomme_summary.json`
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()
    with sftp.open(REMOTE_REGISTRY, "r") as f:
        content = f.read().decode("utf-8", errors="replace")

    marker = "### Download Update: DocVQA + Video-MME (2026-03-22)"
    if marker not in content:
        content = content.rstrip() + "\n\n" + SECTION.strip() + "\n"
        with sftp.open(REMOTE_REGISTRY, "w") as f:
            f.write(content)

    LOCAL_SNAPSHOT.write_text(content, encoding="utf-8")
    sftp.close()
    client.close()
    print("registered")


if __name__ == "__main__":
    main()
