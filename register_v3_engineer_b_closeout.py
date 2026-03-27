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
LOCAL_SUMMARY = LOCAL_ROOT / "experiment_logs" / "v3_engineer_b_closeout_20260322.md"
REMOTE_SUMMARY = "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/closeout_20260322.md"
REMOTE_REGISTRY = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"


def load_payload(name: str) -> dict:
    path = MIRROR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return next(iter(data.values()))[0]


def metric_line(label: str, payload: dict, metric: str = "accuracy") -> str:
    base = payload["baseline"].get(metric)
    bra = payload["bra"].get(metric)
    return f"- `{label}`: baseline `{base:.4f}` -> BRA `{bra:.4f}`"


def docvqa_line(label: str, payload: dict) -> str:
    base = payload["baseline"]
    bra = payload["bra"]
    return (
        f"- `{label}`: accuracy `{base.get('accuracy', 0):.4f}` -> `{bra.get('accuracy', 0):.4f}`, "
        f"AGL `{base.get('agl', 0):.2f}` -> `{bra.get('agl', 0):.2f}`, "
        f"ITL `{base.get('itl_ms_per_token', 0):.2f}` -> `{bra.get('itl_ms_per_token', 0):.2f}` ms/token, "
        f"trigger rate `{bra.get('intervention_rate', 0):.3f}`"
    )


def build_summary() -> str:
    docvqa_full = load_payload("docvqa_tlra_full.json")
    docvqa_no_vasm = load_payload("docvqa_tlra_no_vasm.json")
    freak_meanpool = load_payload("freak_tlra_meanpool.json")
    freak_adapt = load_payload("freak_tlra_adaptivetopk.json")
    mmmu_full = load_payload("mmmu_hard_tlra_full.json")
    mmmu_no_vasm = load_payload("mmmu_hard_tlra_no_vasm.json")

    docvqa_missing = {
        "normalized_exact_match": "normalized_exact_match" not in docvqa_full["baseline"],
        "anls": "anls" not in docvqa_full["baseline"],
    }

    lines = [
        "# V3 Engineer B Closeout (2026-03-22)",
        "",
        "## Remote Assets",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_smoke_tlra_full.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_meanpool.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_adaptivetopk.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_full.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_adaptivetopk.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/video_mme_tlra_full_smoke.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full.json`",
        "- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm.json`",
        "- `/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json`",
        "",
        "## Local Mirror",
        f"- `{(MIRROR / 'docvqa_tlra_full.json').as_posix()}`",
        f"- `{(MIRROR / 'docvqa_tlra_no_vasm.json').as_posix()}`",
        f"- `{(MIRROR / 'freak_tlra_meanpool.json').as_posix()}`",
        f"- `{(MIRROR / 'freak_tlra_adaptivetopk.json').as_posix()}`",
        f"- `{(MIRROR / 'mmmu_hard_tlra_full.json').as_posix()}`",
        f"- `{(MIRROR / 'mmmu_hard_tlra_no_vasm.json').as_posix()}`",
        "",
        "## DocVQA Negative Control",
        docvqa_line("tlra_full", docvqa_full),
        docvqa_line("tlra_no_vasm", docvqa_no_vasm),
        f"- Required field check: `normalized_exact_match` missing = `{docvqa_missing['normalized_exact_match']}`, `anls` missing = `{docvqa_missing['anls']}`.",
        f"- Trigger/audit fields are preserved: `intervention_rate={docvqa_full['bra'].get('intervention_rate')}`, sample audits present = `{bool(docvqa_full.get('sample_audits'))}`.",
        f"- Trace anomaly: output `bra_method` values are `{docvqa_full.get('bra_method')}` and `{docvqa_no_vasm.get('bra_method')}` rather than the requested `tlra_full` / `tlra_no_vasm` labels.",
        "",
        "## FREAK Parity Review",
        metric_line("meanpool accuracy", freak_meanpool),
        metric_line("adaptivetopk accuracy", freak_adapt),
        f"- `UNFROZEN_PROJECTOR / provisional`: projector identity is not frozen, so this pair is retained only as provisional evidence.",
        f"- On this run, `MeanPool` accuracy `{freak_meanpool['bra']['accuracy']:.4f}` vs `AdaptiveTopK` `{freak_adapt['bra']['accuracy']:.4f}`; this does not support a strong local-evidence-superiority claim.",
        "",
        "## MMMU Hard Formal Row",
        metric_line("tlra_full accuracy", mmmu_full),
        metric_line("tlra_no_vasm accuracy", mmmu_no_vasm),
        f"- Frozen manifest: `{mmmu_full.get('manifest_path')}`",
        f"- Loaded sample count: `{mmmu_full.get('sample_count')}` across `{len(mmmu_full.get('mmmu_subjects', []))}` subjects.",
        f"- Current evidence favors `tlra_no_vasm` over `tlra_full` on this frozen manifest (`{mmmu_no_vasm['bra']['accuracy']:.4f}` vs `{mmmu_full['bra']['accuracy']:.4f}`), while both runs show `intervention_rate=0.0`.",
        "",
        "## Video Status",
        "- `VidHalluc`: both current runs loaded `0 samples`; keep as appendix-only exploratory pilot and stop here.",
        "- `Video-MME`: loader/index is auditable, but the current blocking point is the lower-level video decoding stack rather than data ingress; therefore it does not enter the main-paper benchmark and remains appendix-only diagnosis.",
        "",
        "## Interpretation Guardrails",
        "- `DocVQA` is useful as an OCR-concession probe, but the current JSONs are not fully compliant because `normalized_exact_match` and `anls` were not preserved in output.",
        "- `FREAK` remains provisional because `Phi_calib` / projector identity is unfrozen.",
        "- `MMMU Hard` is now materially upgraded from pilot to a frozen-manifest run, but its current result does not support the stronger `tlra_full > tlra_no_vasm` thesis.",
    ]
    return "\n".join(lines) + "\n"


def upload_and_append(summary_text: str) -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    sftp = client.open_sftp()

    local_tmp = LOCAL_ROOT / "experiment_logs" / "_tmp_v3b_closeout_upload.md"
    local_tmp.write_text(summary_text, encoding="utf-8")
    sftp.put(str(local_tmp), REMOTE_SUMMARY)

    registry_append = "\n\n---\n\n" + summary_text
    remote_tmp = "/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/_registry_append_20260322.md"
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


def main():
    summary = build_summary()
    LOCAL_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"Wrote local summary to {LOCAL_SUMMARY}")
    upload_and_append(summary)


if __name__ == "__main__":
    main()
