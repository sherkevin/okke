# Strict Reviewer Audit After Remaining-Gap Closure, 2026-05-30

Audited artifact: `author_response_min_diff_expected_20260529.pdf`

PDF metadata after rebuild: 5 pages, unencrypted, modified 2026-05-30 01:32:43.

## Bottom Line

The previous strict audit was right: the 4-page version solved enough for a Weak Accept but left clear follow-up questions. The current 5-page version closes the main remaining response gaps inside the PDF itself, while still marking new numerical rows as expected placeholders.

This is now a stronger rich-master rebuttal draft. It is not a final real-results submission until the engineer-run logs replace every expected value and the expected-result notes are deleted.

## Status Against Previous Unresolved Items

| Previous unresolved item | Current status | Where addressed |
|---|---|---|
| Recent baselines lack matched numerical comparison | Partially closed with a matched-run numerical slot for ONLY, VHD/VHR, HALC, OPERA, P+C, and Full CHORD | Section 6, recent matched baselines table |
| Generality remains narrow | Closed as an honest boundary, not as a broad claim; pilot slots added for LLaVA-NeXT, Qwen2-VL, attribute probes, and relation/composition probes | Section 6, generality/default recommendation |
| No confidence intervals or significance checks | Closed structurally with bootstrap CI and paired-test table | Section 6, statistical reliability |
| Batch scaling is only a boundary | Still a boundary, but default recommendation now explicitly says P+C for deployment/batching, k=5,m=2 for cheaper Future, Full k=5,m=3 for quality mode | Sections 4 and 6 |
| Figure 2 is only promised | Closed; the revised Figure 2 sketch is embedded in the PDF | Section 5 |
| Novelty remains moderate | Not fully solvable by rebuttal; framing is now disciplined as detector-assisted admission-time verification, with explicit controls and limitations | Sections 1, 3, 6 |
| Anchor thresholds unclear | Closed structurally with threshold definitions and threshold-sensitivity rows | Section 6 |
| Same-anchor non-CHORD weight unclear | Closed structurally with scoring formula and alpha sweep details | Section 6 |

## Strict Score Estimate

- Previous version: **4/5 Weak Accept**, with KrEs likely still at 3 unless optimistic.
- Current rich-master version, if the new rows become real measured results: **4/5 to borderline 5/5 for rebuttal quality**, because it directly answers almost every concrete follow-up question.
- Current version before real logs replace expected values: **not submission-final**, but it is a better target PDF for the engineer to fill.

## Remaining Non-Negotiable Boundary

The new matched-baseline, CI, threshold-sensitivity, and generality-pilot values are expected placeholders. If real runs do not support them, the final PDF must either replace them with the measured values or delete/narrow the corresponding claim. The strongest honest final form is still evidence-first, not table-complete at any cost.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "Direct Answers to Remaining Questions|ONLY|VHD/VHR|HALC|McNemar|bootstrap|DINO threshold|same-anchor|LLaVA-NeXT|Qwen2-VL|Recommended use|Figure 2 revision sketch|Reviewer-Specific Close"`
- `rg -n "! LaTeX Error|Fatal error|Emergency stop|Overfull|Underfull|LaTeX Warning|Output written" author_response_min_diff_expected_20260529.log` reports only the output-written line.
