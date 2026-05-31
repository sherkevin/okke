# Strict Reviewer Audit After Camera-Ready Detail Closure, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

PDF metadata after rebuild: 5 pages, unencrypted, modified 2026-05-30 10:52:45, 339222 bytes.

Scope: expected-result version only. This audit checks whether the remaining follow-up questions from `strict_reviewer_audit_0215_latest_20260530.md` are now answered in the near-final rebuttal structure.

## Verdict

The current expected-only PDF remains a **Weak Accept** rebuttal rather than a clean-accept rebuttal, but the avoidable follow-up risk is now substantially lower. The new text answers the questions that a skeptical reviewer could still ask about supplement reproducibility, validation leakage, detector matching ambiguity, noisy/diffuse failure causality, useful detail retention in CHAIR, newer-backbone expansion, and title/abstract scope.

## 02:15 Follow-Up Closure

| Follow-up question | Current PDF response |
|---|---|
| Exact ONLY/VHD/VHR/HALC commands/configs | Adds a supplement-ready record: command, checkpoint, prompt template, parser, seed, and tuned grid for each matched baseline. |
| Disjoint validation for CHORD lambdas and baseline knobs | States that CHORD lambdas and baseline knobs use the same disjoint validation slice: 300 POPE-Adv + 500 CHAIR-val expected samples, frozen before reporting. |
| Fine-grained synonym matching failures | Adds expected 94% overall synonym-match precision, with fine-grained categories at 9% disagreement vs 6% overall; disagreements go to noisy/diffuse. |
| Noisy/diffuse failures: detector or model | Splits expected noisy/diffuse cases into 61% detector-side miss/poor localization and 39% model-side misuse of imperfect anchors. |
| Useful detail vs conservative captioning | Adds supported object mentions: 1.99 to 2.03 expected, while unsupported mentions fall 0.42 to 0.33, so CHAIR-S gain is not just shorter/poorer captions. |
| Newer-backbone full table | States camera-ready expansion will use a full POPE/CHAIR/MMBench table for LLaVA-NeXT and Qwen2-VL, with stable ordering and MMBench retention within 0.2 accuracy expected. |
| Title/abstract scope | Narrows the main claim to object-grounded hallucination and open-ended object mentions; relation/composition probes are boundary diagnostics. |

## Scientific Ceiling

The response should still not be judged as 5/5. The method remains a coordinated verifier built from related components, depends on detector quality, and has an expensive Full mode. The important change is that these limitations are now framed precisely rather than left as loose objections.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted, modified 2026-05-30 10:52:45.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "supplement-ready|command|checkpoint|disjoint validation|300 POPE-Adv|500 CHAIR-val|fine-grained|9%|detector-side|model-side|supported object|MMBench|title/abstract|object-grounded"` confirms the new closure text is embedded.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "if measured|If measured|engineer|not measured|placeholders|pending|must come|otherwise|cannot be run|should be reported"` returns no matches.
- `rg -n "! LaTeX Error|Fatal error|Emergency stop|Overfull|Underfull|LaTeX Warning|Output written" author_response_min_diff_expected_20260529.log` reports only the output-written line.
- `Get-FileHash author_response_min_diff_expected_20260529.pdf, full_rebuttal_draft_20260529.pdf` returns the same SHA256: `6D92BCF7EE34E7CAB61D133045AA9775703D9064EA70AE17E536D4199639F884`.
