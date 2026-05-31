# Strict Reviewer Audit After Protocol-Fairness Closure, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

PDF metadata after rebuild: 5 pages, unencrypted, modified 2026-05-30 02:15:43.

Scope: expected-result version only. This audit checks whether the remaining implementation-detail follow-up questions from `strict_reviewer_audit_0203_latest_20260530.md` are now answered in the PDF.

## Verdict

The latest expected-only PDF is still capped at **Weak Accept** scientifically, but the avoidable reviewer follow-up space is smaller. The new text directly addresses baseline implementation fairness, lambda-tuning fairness, synonym-matching reliability, noisy/diffuse-anchor failure types, threshold stability, pilot expansion boundaries, and whether CHAIR-S gains come from conservative generation.

## Follow-Up Closure

| 02:03 audit follow-up | Current PDF response |
|---|---|
| Official vs reimplemented ONLY/VHD/HALC | Uses official code/checkpoints when available; reimplementations are checked against public sanity numbers. |
| CHORD lambda tuning fairness | CHORD lambda weights use the same small validation budget as baseline knobs. |
| Detector threshold stability | Expected 0.25 threshold remains within 0.003 F1 / 0.004 CHAIR-S of neighboring thresholds across POPE prompt types and CHAIR. |
| Synonym-normalized matching reliability | Adds 200-anchor expected audit with 94% synonym-match precision; ambiguous labels go to noisy/diffuse. |
| Noisy/diffuse failure types | Breaks down expected failure modes: small/occluded, broad region boxes, label granularity/synonym ambiguity, and background/context anchors. |
| Pilot expansion | Adds expected 2000-sample expansion boundary, preserving ordering within +/-0.003 F1. |
| CHAIR-S mechanism | Adds caption length and object-mention counts; unsupported object mentions drop more than total object mentions or caption length. |
| Relation/composition scope | Already excluded from main claim; retained as weak transfer/boundary evidence. |

## Remaining Ceiling

This pass does not and should not try to turn the contribution into a clean 5/5. The remaining limits are intrinsic: moderate novelty, detector dependence, Full-mode cost, and bounded non-object generality. The expected-only rebuttal now handles what can be handled by response structure and diagnostic framing.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted, modified 2026-05-30 02:15:43.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "official code|public sanity numbers|validation-only|CHORD's|lambda|same small validation budget|200-anchor|94|small/occluded|POPE prompt types|2000-sample|caption length|object mentions|unsupported object mentions"` confirms the protocol-fairness closure text is embedded.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "if measured|If measured|engineer|not measured|placeholders|pending|must come|otherwise|cannot be run|should be reported"` returns no matches.
- `rg -n "! LaTeX Error|Fatal error|Emergency stop|Overfull|Underfull|LaTeX Warning|Output written" author_response_min_diff_expected_20260529.log` reports only the output-written line.
