from __future__ import annotations

from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

REMOTE_FILES = [
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_smoke_tlra_full.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_smoke_tlra_full.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full_rerun.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_main.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_meanpool.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_adaptivetopk.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_parity.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_full.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_adaptivetopk.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_main.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/video_mme_tlra_full_smoke.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/video_mme_smoke.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full_rerun.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm_rerun.json",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_main.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_review.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_manifest_build.log",
    "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/closeout_20260322.md",
    "/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json",
    "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md",
]


def main():
    local_root = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\remote_mirror\v3_engineer_b")
    local_root.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()

    for remote in REMOTE_FILES:
        remote_name = Path(remote).name
        local_path = local_root / remote_name
        try:
            print(f"Downloading {remote} -> {local_path}")
            sftp.get(remote, str(local_path))
        except FileNotFoundError:
            print(f"Skipping missing remote file: {remote}")

    sftp.close()
    client.close()
    print(f"Mirrored {len(REMOTE_FILES)} files to {local_root}")


if __name__ == "__main__":
    main()
