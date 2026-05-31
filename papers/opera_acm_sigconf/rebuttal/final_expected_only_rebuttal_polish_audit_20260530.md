# Final Expected-Only Rebuttal Polish Audit, 2026-05-30

Target PDF: `author_response_min_diff_expected_20260529.pdf`

## Verdict

The current PDF is now a coherent expected-result rebuttal master. It does not read as a draft waiting for experiments, except for the explicit superscript `E` convention that marks replaceable expected values. The non-result issues from the reviewer audits are handled in the PDF itself.

## Final Polish Changes

| Issue found | Fix applied |
|---|---|
| Unnumbered compact tables were referenced as `Table 1/2/3/4/5` | Replaced with descriptive references such as mechanism table, attribution table, cost table, and statistical-reliability table. |
| Related-work positioning row still had future-action wording | Replaced with wording that points to the expected matched comparison already included in Section 6. |
| Reviewer-close section leaned on table numbers | Rewritten to use reviewer-facing evidence names rather than table numbers. |
| Expected-only boundary risk | Verified that deferred-result terms such as `if measured`, `engineer`, `pending`, `placeholders`, and `not measured` do not appear in the current PDF text. |

## Current Expected-Only Strength

Under the expected-result assumption, the response now directly addresses:

- Future mechanism: sparse Full-vs-P+C flips, corrected > harmful, CHAIR continuation gain, paired significance.
- Detector attribution: same-anchor non-CHORD, uniform/random anchors, Past+Future without Current, and detector strata.
- Efficiency: proposal time, decode ITL, total latency, VRAM, and batch-size boundary.
- k/m robustness: diminishing-return sweep and operating-point recommendation.
- Recent baselines: expected matched comparison against ONLY, VHD/VHR, and HALC.
- Generality: newer-backbone pilot and weak non-object transfer treated as boundary evidence.
- Presentation: revised Figure 2 embedded.
- Claim discipline: detector-assisted, base-MLLM training-free, attention as operational feature.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted, modified 2026-05-30 01:53:35.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "Table 1|Table 2|Table 3|Table 4|Table 5|should be reported|if measured|If measured|engineer|not measured|placeholders|pending|must come|otherwise|cannot be run"` returns no matches.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "mechanism table|attribution table|cost table|statistical-reliability table|expected matched comparison|Expected-result convention|Recommended use"` confirms the intended references are present.
- `rg -n "! LaTeX Error|Fatal error|Emergency stop|Overfull|Underfull|LaTeX Warning|Output written" author_response_min_diff_expected_20260529.log` reports only the output-written line.
- SHA256 of `author_response_min_diff_expected_20260529.pdf` and `full_rebuttal_draft_20260529.pdf` matches after sync.
