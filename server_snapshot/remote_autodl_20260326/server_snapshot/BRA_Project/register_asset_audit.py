from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

REMOTE_REGISTRY = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"
LOCAL_SNAPSHOT = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\SHARED_DATA_REGISTRY_snapshot_20260321.md")

SECTION = """

---

## 6. Resource Audit

### Resource Audit: 标杆V1 download readiness (2026-03-21)

- Local log: `experiment_logs/asset_audit_标杆V1_20260321.md`
- Remote log: `/root/autodl-tmp/BRA_Project/logs/asset_audit_标杆V1_20260321.md`

#### Audit summary

- Ready now:
  - `Qwen3-VL-8B-Instruct`
  - `Qwen3-VL-2B-Instruct`
  - `COCO val2014 + annotations`
  - `POPE`
  - `CHAIR`
  - `MMBench_EN_hf` (version still worth confirming against V1.1)
  - `MME_hf`
  - `MMMU_hf` (full set only, not explicit Hard subset)
  - `FREAK_hf`

- Missing or not yet confirmed:
  - `DocVQA`
  - `Base + 5k LoRA`
  - `MMMU Hard` explicit split
  - explicit registration of `TLRA_calib` weights

#### Notes

- `Stage 0` is data-ready because `COCO val2014` and `instances_val2014.json` are present; only the held-out subset needs to be constructed at run time.
- `V_matrix.pt`, `V_matrix_q3.pt`, and `V_matrix_q3_mini.pt` exist under `models/`, but they are not clearly registered as the official `TLRA_calib` / `Phi_calib` assets yet.
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    sftp = client.open_sftp()

    with sftp.open(REMOTE_REGISTRY, "r") as f:
        content = f.read().decode("utf-8", errors="replace")

    marker = "### Resource Audit: 标杆V1 download readiness (2026-03-21)"
    if marker not in content:
        content = content.rstrip() + SECTION + "\n"
        with sftp.open(REMOTE_REGISTRY, "w") as f:
            f.write(content)

    LOCAL_SNAPSHOT.write_text(content, encoding="utf-8")

    sftp.close()
    client.close()
    print("registered")


if __name__ == "__main__":
    main()
