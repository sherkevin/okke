# V3 Engineer B Closeout (2026-03-22)

## Remote Assets
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_smoke_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_meanpool.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/freak_tlra_adaptivetopk.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/vidhalluc_tlra_adaptivetopk.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/video_mme_tlra_full_smoke.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full.json`
- `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm.json`
- `/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json`

## Local Mirror
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/docvqa_tlra_full.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/docvqa_tlra_no_vasm.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/freak_tlra_meanpool.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/freak_tlra_adaptivetopk.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/mmmu_hard_tlra_full.json`
- `d:/Shervin/OneDrive/Desktop/breaking/experiment_logs/remote_mirror/v3_engineer_b/mmmu_hard_tlra_no_vasm.json`

## DocVQA Negative Control
- `tlra_full`: accuracy `0.0000` -> `0.0000`, AGL `6.37` -> `6.38`, ITL `88.33` -> `87.64` ms/token, trigger rate `0.820`
- `tlra_no_vasm`: accuracy `0.0000` -> `0.0000`, AGL `6.45` -> `6.37`, ITL `87.12` -> `87.76` ms/token, trigger rate `0.815`
- Required field check: `normalized_exact_match` missing = `True`, `anls` missing = `True`.
- Trigger/audit fields are preserved: `intervention_rate=0.82`, sample audits present = `True`.
- Trace anomaly: output `bra_method` values are `bra_zero` and `bra_zero` rather than the requested `tlra_full` / `tlra_no_vasm` labels.

## FREAK Parity Review
- `meanpool accuracy`: baseline `0.2233` -> BRA `0.2100`
- `adaptivetopk accuracy`: baseline `0.2200` -> BRA `0.2200`
- `UNFROZEN_PROJECTOR / provisional`: projector identity is not frozen, so this pair is retained only as provisional evidence.
- On this run, `MeanPool` accuracy `0.2100` vs `AdaptiveTopK` `0.2200`; this does not support a strong local-evidence-superiority claim.

## MMMU Hard Formal Row
- `tlra_full accuracy`: baseline `0.1930` -> BRA `0.1930`
- `tlra_no_vasm accuracy`: baseline `0.1579` -> BRA `0.2193`
- Frozen manifest: `/root/autodl-tmp/BRA_Project/datasets/MMMU_hf/mmmu_hard_manifest_v3b.json`
- Loaded sample count: `114` across `26` subjects.
- Current evidence favors `tlra_no_vasm` over `tlra_full` on this frozen manifest (`0.2193` vs `0.1930`), while both runs show `intervention_rate=0.0`.

## Video Status
- `VidHalluc`: both current runs loaded `0 samples`; keep as appendix-only exploratory pilot and stop here.
- `Video-MME`: loader/index is auditable, but the current blocking point is the lower-level video decoding stack rather than data ingress; therefore it does not enter the main-paper benchmark and remains appendix-only diagnosis.

## Interpretation Guardrails
- `DocVQA` is useful as an OCR-concession probe, but the current JSONs are not fully compliant because `normalized_exact_match` and `anls` were not preserved in output.
- `FREAK` remains provisional because `Phi_calib` / projector identity is unfrozen.
- `MMMU Hard` is now materially upgraded from pilot to a frozen-manifest run, but its current result does not support the stronger `tlra_full > tlra_no_vasm` thesis.
