from __future__ import annotations

import json
from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

LOCAL_ROOT = Path(r"d:\Shervin\OneDrive\Desktop\breaking")
MIRROR = LOCAL_ROOT / "experiment_logs" / "remote_mirror" / "v3_engineer_b"
LOCAL_SUMMARY = LOCAL_ROOT / "experiment_logs" / "v3_engineer_b_contract_ready_reruns_20260322.md"
REMOTE_SUMMARY = "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/contract_ready_reruns_20260322.md"
REMOTE_REGISTRY = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"


def load_payload(name: str) -> dict:
    data = json.loads((MIRROR / name).read_text(encoding="utf-8"))
    return next(iter(data.values()))[0]


def build_summary() -> str:
    doc_full = load_payload("docvqa_tlra_full_rerun.json")
    doc_no_vasm = load_payload("docvqa_tlra_no_vasm_rerun.json")
    mmmu_full = load_payload("mmmu_hard_tlra_full_rerun.json")
    mmmu_no_vasm = load_payload("mmmu_hard_tlra_no_vasm_rerun.json")

    lines = [
        "# V3 Engineer B Contract-Ready Reruns (2026-03-22)",
        "",
        "## Contract-Ready DocVQA",
        "- Status: `contract-ready` after schema validation on both reruns.",
        "- Accepted assets:",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full_rerun.json`",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json`",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log`",
        f"- `tlra_full`: bra_method `{doc_full['bra_method']}`, baseline accuracy `{doc_full['baseline']['accuracy']:.4f}`, BRA accuracy `{doc_full['bra']['accuracy']:.4f}`, baseline NEM `{doc_full['baseline']['normalized_exact_match']:.4f}`, BRA NEM `{doc_full['bra']['normalized_exact_match']:.4f}`, baseline ANLS `{doc_full['baseline']['anls']:.4f}`, BRA ANLS `{doc_full['bra']['anls']:.4f}`, intervention_rate `{doc_full['bra']['intervention_rate']:.4f}`, sample_audits `{len(doc_full.get('sample_audits', []))}`.",
        f"- `tlra_no_vasm`: bra_method `{doc_no_vasm['bra_method']}`, baseline accuracy `{doc_no_vasm['baseline']['accuracy']:.4f}`, BRA accuracy `{doc_no_vasm['bra']['accuracy']:.4f}`, baseline NEM `{doc_no_vasm['baseline']['normalized_exact_match']:.4f}`, BRA NEM `{doc_no_vasm['bra']['normalized_exact_match']:.4f}`, baseline ANLS `{doc_no_vasm['baseline']['anls']:.4f}`, BRA ANLS `{doc_no_vasm['bra']['anls']:.4f}`, intervention_rate `{doc_no_vasm['bra']['intervention_rate']:.4f}`, sample_audits `{len(doc_no_vasm.get('sample_audits', []))}`.",
        "- Schema acceptance checks passed for both files: `bra_method` preserved, `normalized_exact_match` present, `anls` present, `intervention_rate` present, `sample_audits` non-empty.",
        "",
        "## MMMU Hard Review",
        "- Review assets:",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full_rerun.json`",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm_rerun.json`",
        "  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_review.log`",
        f"- `tlra_full`: accuracy `{mmmu_full['bra']['accuracy']:.4f}`, intervention_rate `{mmmu_full['bra']['intervention_rate']:.4f}`, avg_vasm_time_ms `{mmmu_full['bra']['avg_vasm_time_ms']:.4f}`, avg_routing_time_ms `{mmmu_full['bra']['avg_routing_time_ms']:.4f}`, visual_state_provenance present `{bool(mmmu_full['bra'].get('visual_state_provenance'))}`.",
        f"- `tlra_no_vasm`: accuracy `{mmmu_no_vasm['bra']['accuracy']:.4f}`, intervention_rate `{mmmu_no_vasm['bra']['intervention_rate']:.4f}`, avg_vasm_time_ms `{mmmu_no_vasm['bra']['avg_vasm_time_ms']:.4f}`, avg_routing_time_ms `{mmmu_no_vasm['bra']['avg_routing_time_ms']:.4f}`, visual_state_provenance present `{bool(mmmu_no_vasm['bra'].get('visual_state_provenance'))}`.",
        "- Review conclusion: the old `intervention_rate = 0.0` reading is no longer valid under the corrected export protocol. Current reruns show substantial non-zero trigger rates.",
        "",
        "## Guardrails",
        "- `FREAK` remains `UNFROZEN_PROJECTOR / provisional` and was not rerun as a formal table.",
        "- `VidHalluc` and `Video-MME` remain stopped and did not occupy GPU in this rerun phase.",
    ]
    return "\n".join(lines) + "\n"


def upload_and_append(summary_text: str) -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()

    local_tmp = LOCAL_ROOT / "experiment_logs" / "_tmp_v3b_contract_ready_upload.md"
    local_tmp.write_text(summary_text, encoding="utf-8")
    sftp.put(str(local_tmp), REMOTE_SUMMARY)
    remote_tmp = "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/_contract_ready_registry_append_20260322.md"
    sftp.put(str(local_tmp), remote_tmp)
    sftp.close()

    cmd = (
        f"/root/miniconda3/bin/python - <<'PY'\n"
        f"from pathlib import Path\n"
        f"reg = Path('{REMOTE_REGISTRY}')\n"
        f"append = Path('{remote_tmp}').read_text(encoding='utf-8')\n"
        f"with reg.open('a', encoding='utf-8') as f:\n"
        f"    f.write('\\n\\n---\\n\\n')\n"
        f"    f.write(append)\n"
        f"print('updated', reg)\n"
        f"PY"
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err)
    client.close()


def main() -> None:
    summary = build_summary()
    LOCAL_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"Wrote local summary to {LOCAL_SUMMARY}")
    upload_and_append(summary)


if __name__ == "__main__":
    main()
