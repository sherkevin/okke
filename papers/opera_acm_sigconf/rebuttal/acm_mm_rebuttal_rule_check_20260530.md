# ACM MM 2026 Rebuttal Rule Check, 2026-05-30

Scope: verify the real rebuttal / author-response format for Submission 8826.

## Result

The current working assumption must be corrected:

- There is no verified ACM MM 2026 rule saying the rebuttal may be five pages.
- The public ACM MM 2026 website only says rebuttals are submitted in OpenReview, must preserve anonymity, and cannot include external links.
- The actual OpenReview Author Console task for Submission 8826 is `Rebuttal PDF`, not a plain text rebuttal field.
- The latest PC Chairs email explicitly says each paper may include only a single one-page PDF containing all responses; the one-page limit is strict; over-page rebuttals will not be considered.
- The latest PC Chairs email also says not to use `Official Comment` to reply individually to reviewers because comments submitted there are not visible to reviewers and will not be considered.
- The authenticated OpenReview invitation `acmmm.org/ACMMM/2026/Conference/Submission8826/-/Rebuttal_PDF` states:
  - content field: `pdf`
  - file extension: `pdf`
  - max size: `50`
  - description: `Upload a single page PDF file that ends with .pdf`

Therefore, the uploadable rebuttal must be a **single-page PDF**, not the current 5-page planning/master PDF, and all substantive response content must be inside that single page.

## Source Evidence

Public ACM MM 2026 Call for Technical Papers:

- Rebuttal is optional and submitted in OpenReview.
- Rebuttal must maintain anonymity.
- Rebuttal cannot include links to external material such as code or videos.

Public ACM MM 2026 Important Dates:

- Main Track rebuttal date: 04-June.
- Brave New Ideas rebuttal window: 28-May to 04-June.

Authenticated OpenReview Author Console / invitation API:

- Author task visible for Submission 8826: `Submission8826 Rebuttal PDF`.
- Due date visible in Author Console: 05 Jun 2026, 17:59 China Standard Time.
- Invitation content requires PDF upload and describes it as a single-page PDF.

Latest PC Chairs email:

- Single one-page PDF only.
- No strict template, but reasonable legible layout is required.
- The single page includes all rebuttal content, references, and supporting material.
- Do not use Official Comment to reply individually to reviewers.
- Official Comments are not visible to reviewers and will not be considered.
- One-page PDF is visible to reviewers, AC, SAC, and PCs.

OpenReview default rebuttal form:

- The default OpenReview rebuttal form has a `maxLength: 2500` string field, but this is not the active ACM MM Submission 8826 task.
- Because ACM MM uses a `Rebuttal_PDF` invitation for this paper, the default text-field limit should not be used as the controlling rule.

## Current Artifact Implication

The latest timestamped `author_response_min_diff_expected_YYYYMMDD_HHMM.pdf` file is useful as an internal expected-result master, but it is not upload-compliant because it has 5 pages.

Next required deliverable should be a new single-page rebuttal PDF distilled from the current timestamped master, preserving only the highest-impact evidence:

1. One-sentence contribution clarification and claim calibration.
2. Mechanism evidence: Future flip / corrected-harmful / CHAIR continuation.
3. Detector attribution controls: same-anchor, random/uniform, Past+Future without Current, detector strata.
4. Cost transparency: end-to-end latency, P+C practical vs Full quality.
5. k/m and recent-baseline response, likely in compressed text or one compact table.
6. Reviewer-specific close, preferably one compact sentence covering each main reviewer cluster.

Do not create per-reviewer Official Comments as overflow. If content cannot fit on the single page, it must be cut or compressed.

## Current Count Snapshot

Current master PDF:

- Historical master `texcount` snapshot: about 2512 text words plus table/header tokens.
- PDF text extraction: about 2631 word-like tokens.
- Page count: 5.

This is close to 2500 words, but that no longer matters for the actual ACM MM rebuttal upload task. The controlling constraint is single-page PDF.
