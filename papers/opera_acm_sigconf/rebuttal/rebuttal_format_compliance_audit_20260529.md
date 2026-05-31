# Rebuttal Format Compliance Audit, 2026-05-29

Target artifact: `author_response_min_diff_expected_20260529.pdf`.

## Public Requirements Checked

| Source | Confirmed requirement | Implication |
|---|---|---|
| ACM MM 2026 Call for Technical Papers | Authors may submit rebuttal in OpenReview after reviews. Rebuttal must maintain anonymity and cannot include external links to code/videos/etc. | The response must be anonymous and self-contained. |
| ACM MM 2026 Important Dates | Rebuttal date is listed as 04-June for Main Track and 28-May--04-June for Brave New Ideas. | Deadline must be checked against the submission track. |
| ACM MM 2026 public author instructions | Public page gives paper PDF format, but does not publish a rebuttal PDF template, one-page rule, or word count. | No public evidence that a 3-page rebuttal PDF is accepted. |
| OpenReview Rebuttal Stage docs | Venue chairs configure rebuttal form, number of rebuttals, and additional form fields. | The logged-in invitation controls the final field type/limit. |
| OpenReview Default Rebuttal Form | Default form is a Markdown textarea with `maxLength: 2500`. | If ACM MM uses the default, the current 3-page PDF is too long and wrong format. |

## Current PDF Measurements

| Property | Measured value |
|---|---:|
| Pages | 3 |
| Page size | Letter, 612 x 792 pt |
| File size | 188799 bytes |
| Encrypted | No |
| Approx. words from `pdftotext` | 1163 |
| Approx. characters with whitespace | 8243 |
| Approx. characters without whitespace | 6969 |
| External links in extracted text | None found |
| Author-identifying text in extracted text | Only `Anonymous Authors` found |

## Compliance Verdict

| Possible final requirement | Does current PDF satisfy it? | Reason |
|---|---|---|
| Anonymous, self-contained rebuttal content | Mostly yes | It uses `Anonymous Authors` and no external links were found. |
| PDF upload with no public page/word limit | Unknown | Public ACM MM pages do not confirm PDF upload for rebuttal. |
| One-page PDF rebuttal | No | Current PDF is 3 pages. |
| OpenReview default Markdown field, max 2500 characters | No | Current text is about 8243 characters with whitespace. |
| Internal minimum-difference master for later replacement | Yes | It has final-like structure and removable expected-result notes. |

## Required Next Check

Before submission, open the logged-in ACM MM 2026 OpenReview rebuttal invitation and record:

1. Is the field a PDF upload, Markdown textbox, or both?
2. Is the limit per paper or per review?
3. What is the exact max length/page limit?
4. Does it allow tables/LaTeX?
5. Does it permit new experiment results in the response text?

## Operational Recommendation

Keep `author_response_min_diff_expected_20260529.pdf` as the rich master/reference. Prepare a separate compressed OpenReview text version at <=2500 characters unless the logged-in form explicitly allows a longer PDF upload.
