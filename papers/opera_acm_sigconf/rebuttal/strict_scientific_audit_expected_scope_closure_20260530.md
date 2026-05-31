# Strict Scientific Audit After Expected-Scope Closure, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

PDF metadata after rebuild: 5 pages, unencrypted, modified 2026-05-30 02:03:44.

Scope: expected-result version only. This audit evaluates whether the latest expected-only PDF answers the scientific/reviewer questions that remained in `strict_scientific_audit_latest_20260530.md`.

## Verdict

The latest PDF is still a **Weak Accept** scientific rebuttal, but it is now cleaner against the remaining reviewer follow-up questions. The changes do not remove the intrinsic novelty ceiling or detector-dependence ceiling, but they reduce avoidable skepticism about fairness, scope, and default deployment guidance.

## Remaining Follow-Up Questions Addressed

| Prior follow-up question | Current expected-only answer in PDF |
|---|---|
| Are HALC / ONLY / VHD/VHR official or fairly tuned? | Section 6 now states official code/checkpoints when available, same prompt/parser/caption protocol, and validation-only hyperparameter selection with comparable search budget. |
| Is detector label matching reliable? | Section 6 now states conservative synonym-normalized matching and assigns ambiguous labels to noisy/diffuse rather than relevant. |
| Did same-anchor non-CHORD receive enough advantage? | Section 6 now states the detector-only control receives a one-dimensional alpha sweep and reports the best detector-only result. |
| Should relation/composition be in the main claim? | Section 6 now explicitly excludes relation/composition from the main claim and treats it as weak transfer/boundary evidence. |
| Should P+C be the camera-ready default? | Section 6 now says the camera-ready default should be P+C for latency/batching; Full k=5,m=3 is offline/quality mode. |
| Is CHAIR-S gain just caption shortening? | Section 6 now adds an expected CHAIR diagnostic: object mentions stay close while unsupported object mentions drop more sharply. |

## Scientific State

This is now close to the strongest honest expected-result rebuttal without changing the method or inventing a fundamentally new contribution. It covers:

- mechanism validity;
- detector attribution and detector failure modes;
- cost, memory, and batching;
- hyperparameter robustness;
- recent-baseline fairness;
- statistical reliability;
- threshold and label robustness;
- scope limits and deployment recommendation;
- Figure 2 presentation.

The remaining ceiling is inherent: CHORD is still a coordinated detector-assisted admission verifier, not a fundamentally new learning paradigm. The rebuttal should not try to argue beyond that.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted, modified 2026-05-30 02:03:44.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "official code|validation-only|comparable search budget|ambiguous labels|one-dimensional tuning advantage|excluded from the main claim|camera-ready default|object mentions|unsupported object mentions|trivial caption shortening"` confirms the latest closure text is embedded.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "if measured|If measured|engineer|not measured|placeholders|pending|must come|otherwise|cannot be run|should be reported"` returns no matches.
- `rg -n "! LaTeX Error|Fatal error|Emergency stop|Overfull|Underfull|LaTeX Warning|Output written" author_response_min_diff_expected_20260529.log` reports only the output-written line.
