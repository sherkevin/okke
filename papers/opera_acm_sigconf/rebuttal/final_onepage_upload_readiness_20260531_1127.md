# Final One-page Upload Readiness Check - 20260531 1127

## Candidate

- Superseded prior one-page draft: `author_response_onepage_expected_20260531_0043.pdf`
- Current one-page candidate: `author_response_onepage_expected_20260531_1127.pdf`
- Source TEX: `author_response_onepage_expected_20260531_1127.tex`
- Rendered preview: `author_response_onepage_expected_20260531_1127_readiness-1.png`

## Why A New One-page Candidate Was Made

The prior `0043` candidate was one-page compliant and safe, but it was sparse and left unused page space. The `1127` candidate keeps the same evidence boundary while using the page more effectively: it directly maps each major reviewer concern to the reviewer need, author response, and camera-ready boundary.

This is an upload-readiness improvement, not new experimental evidence.

## Verification

Commands run from `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error author_response_onepage_expected_20260531_1127.tex
pdfinfo author_response_onepage_expected_20260531_1127.pdf
pdftotext -layout author_response_onepage_expected_20260531_1127.pdf -
Select-String author_response_onepage_expected_20260531_1127.log -Pattern 'Overfull|Underfull|Warning|Error|Emergency|Fatal'
pdftoppm -png -r 200 author_response_onepage_expected_20260531_1127.pdf author_response_onepage_expected_20260531_1127_readiness
```

Observed results:

- `pdfinfo` reports `Pages: 1`.
- `pdflatex` completed successfully and wrote `author_response_onepage_expected_20260531_1127.pdf`.
- The log scan found no `Overfull`, `Underfull`, `Warning`, `Error`, `Emergency`, or `Fatal` matches.
- `pdftotext -layout` confirms the PDF is self-contained and includes the five core response areas.
- Visual preview inspection found no text overlap, no table clipping, and no page crop.

## Scientific Content Coverage

The current one-page candidate covers the five reviewer-demand areas:

1. Future/mechanism: states that paired Full-vs-P+C admission diagnostics will report rollout-induced flips, corrected unsupported mentions, harmful changes, neutral changes, sample size, confidence intervals, and p-values only when complete paired logs exist.
2. Detector attribution: states that CHORD is detector-assisted and uses same-anchor, random/uniform-anchor, no-Current, P+C, and Full controls, plus zero/relevant/noisy-anchor strata, without claiming detector independence.
3. Efficiency/defaults: separates one-time proposal cost from decode ITL, total latency, peak VRAM, and batch behavior; frames P+C as the practical/default regime and Full as quality-oriented/offline.
4. Recent baselines/fairness: positions ONLY, VHD/VHR, and HALC under a matched protocol with shared backbone, split, prompt family, parser, seeds, and validation budget; numeric wins are not claimed without completed provenance.
5. Novelty/scope/attention: narrows the contribution to a coordinated admission-time verifier for object-grounded hallucination and open-ended object mentions; decoder-to-vision attention is treated as an operational scoring feature, not a causal explanation.

## Expected Table Boundary

No expected table values are presented as factual rebuttal evidence in the one-page candidate. The `1127` PDF explicitly states:

- all exact numbers in the official response and camera-ready paper must be measured or removed;
- internal expected target tables are not factual rebuttal evidence;
- if real measurements disagree with expected patterns, the row and claim should be replaced or narrowed rather than tuned to preserve a stronger claim.

This keeps the official response credible while preserving the longer expected-table documents as internal planning material only.

## Readiness Decision

`author_response_onepage_expected_20260531_1127.pdf` is the stronger current one-page upload candidate than `author_response_onepage_expected_20260531_0043.pdf`.

It remains evidence-limited: it is suitable as a conservative one-page response if no new measured Future, detector-control, cost, or recent-baseline results arrive before submission. If real measured rows arrive, they should be inserted only if they can fit without harming readability and without violating the one-page limit.

No OpenReview submission was made.
