# One-page Numeric Rebuttal Readiness 20260531 2208

## Purpose

Fix the 21:54 strict-audit issue: the official one-page rebuttal candidate was readable but undersold the strongest scientific evidence from `review_v11`. This revision compresses the score-moving numeric rows into the official one-page PDF/TEX instead of creating another long author-response markdown.

## Source And Output

- Source TEX: `author_response_onepage_expected_20260531_1127.tex`
- Source PDF: `author_response_onepage_expected_20260531_1127.pdf`
- New TEX: `author_response_onepage_expected_20260531_2208.tex`
- New PDF: `author_response_onepage_expected_20260531_2208.pdf`
- Rendered check: `author_response_onepage_expected_20260531_2208_readiness-1.png`

## Evidence Compressed Into The Page

- Future mechanism: Full--P+C flip rates `4.6%/4.9%`, corrected/harmful counts `96/42` and `102/44`, Adv.F1 `+0.013`, CIs and p-values, and CHAIR-S `-0.020`.
- Detector attribution: compact control ladder `0.820/0.823/0.821/0.824/0.825/0.832/0.845`, plus P+C/Full CHAIR-S `0.175/0.155`.
- Cost/default: OPERA `434 ms, 16.2 GB`; P+C `663 ms, 17.5 GB`; Full `864 ms, 20.3 GB`; total-latency arithmetic via proposal plus ITL times 20 tokens.
- k/m practical-quality boundary: P+C `0.832/0.175/663 ms`, `k=5,m=2` `0.843/0.159/807 ms`, Full `k=5,m=3` `0.845/0.155/864 ms`.
- Recent baselines: ONLY `0.826`, VHD/VHR `0.829`, HALC `0.831`, CHORD P+C `0.832`, Full `0.845`; CHAIR-S `0.190/0.184/0.178/0.175/0.155`.
- Claim boundary preserved: detector-assisted, object-grounded, operational-attention feature, P+C practical/default, Full quality/offline, no detector-independent or broad relation/composition claim.

## Verification

Commands run from `papers/opera_acm_sigconf/rebuttal`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error author_response_onepage_expected_20260531_2208.tex
pdfinfo author_response_onepage_expected_20260531_2208.pdf
Select-String author_response_onepage_expected_20260531_2208.log -Pattern 'Output written|Overfull|Underfull|Warning|Error|Fatal|Emergency'
pdftotext -layout author_response_onepage_expected_20260531_2208.pdf -
pdftoppm -png -r 220 author_response_onepage_expected_20260531_2208.pdf author_response_onepage_expected_20260531_2208_readiness
```

Results:

- PDF compiles successfully.
- `pdfinfo` reports exactly `Pages: 1`.
- LaTeX log check reports output written and no matched overfull/underfull/warning/error/fatal/emergency lines.
- Text extraction confirms Future, detector, cost, baseline, and evidence-discipline rows are present.
- Rendered PNG was visually inspected: the page is readable, academic, and has no obvious text overlap or cropping.

## Submission Boundary

No OpenReview upload was made. Official ACM MM submission remains a single strict one-page PDF; Official Comment must not be used as overflow.
