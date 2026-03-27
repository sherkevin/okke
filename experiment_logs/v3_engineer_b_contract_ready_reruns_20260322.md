# V3 Engineer B Contract-Ready Reruns (2026-03-22)

## Contract-Ready DocVQA
- Status: `contract-ready` after schema validation on both reruns.
- Accepted assets:
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_full_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log`
- `tlra_full`: bra_method `tlra_full`, baseline accuracy `0.6550`, BRA accuracy `0.6650`, baseline NEM `0.6550`, BRA NEM `0.6650`, baseline ANLS `0.7978`, BRA ANLS `0.8235`, intervention_rate `0.9000`, sample_audits `5`.
- `tlra_no_vasm`: bra_method `tlra_no_vasm`, baseline accuracy `0.6650`, BRA accuracy `0.6550`, baseline NEM `0.6650`, BRA NEM `0.6550`, baseline ANLS `0.8168`, BRA ANLS `0.8118`, intervention_rate `0.8950`, sample_audits `5`.
- Schema acceptance checks passed for both files: `bra_method` preserved, `normalized_exact_match` present, `anls` present, `intervention_rate` present, `sample_audits` non-empty.

## MMMU Hard Review
- Review assets:
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_full_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_tlra_no_vasm_rerun.json`
  - `/root/autodl-tmp/BRA_Project/logs/v3_engineer_b/mmmu_hard_review.log`
- `tlra_full`: accuracy `0.1754`, intervention_rate `0.3684`, avg_vasm_time_ms `0.0252`, avg_routing_time_ms `0.1831`, visual_state_provenance present `True`.
- `tlra_no_vasm`: accuracy `0.1667`, intervention_rate `0.3772`, avg_vasm_time_ms `0.0073`, avg_routing_time_ms `0.3074`, visual_state_provenance present `True`.
- Review conclusion: the old `intervention_rate = 0.0` reading is no longer valid under the corrected export protocol. Current reruns show substantial non-zero trigger rates.

## Guardrails
- `FREAK` remains `UNFROZEN_PROJECTOR / provisional` and was not rerun as a formal table.
- `VidHalluc` and `Video-MME` remain stopped and did not occupy GPU in this rerun phase.
