import json
from pathlib import Path


def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = next(iter(data.values()))[0]
    issues: list[str] = []

    if payload.get("bra_method") == "bra_zero":
        issues.append("bra_method collapsed to bra_zero")
    if "normalized_exact_match" not in payload.get("baseline", {}):
        issues.append("baseline missing normalized_exact_match")
    if "normalized_exact_match" not in payload.get("bra", {}):
        issues.append("bra missing normalized_exact_match")
    if "anls" not in payload.get("baseline", {}):
        issues.append("baseline missing anls")
    if "anls" not in payload.get("bra", {}):
        issues.append("bra missing anls")
    if "intervention_rate" not in payload.get("bra", {}):
        issues.append("bra missing intervention_rate")
    if not payload.get("sample_audits"):
        issues.append("sample_audits missing or empty")
    return issues


def main() -> None:
    root = Path(r"d:\Shervin\OneDrive\Desktop\breaking\experiment_logs\remote_mirror\v3_engineer_b")
    for name in ["docvqa_tlra_full_rerun.json", "docvqa_tlra_no_vasm_rerun.json"]:
        path = root / name
        if not path.exists():
            print(f"{name}: missing")
            continue
        issues = validate(path)
        if issues:
            print(f"{name}: FAIL")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"{name}: PASS")


if __name__ == "__main__":
    main()
