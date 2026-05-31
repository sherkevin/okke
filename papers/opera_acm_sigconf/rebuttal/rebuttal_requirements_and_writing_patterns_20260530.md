# ACM MM 2026 Rebuttal Requirements and Writing Patterns, 2026-05-30

Submission: `8826`

Scope: current operational guidance for preparing the CHORD rebuttal deliverables. No comment, mediation request, or PDF was submitted during this check.

## Evidence Priority

1. Highest priority: latest PC Chairs email recorded in `pc_chairs_rebuttal_format_email_20260530.md`.
2. Controlling upload form evidence: authenticated OpenReview invitations saved in `openreview_invitation_snapshot_submission8826_20260530.json`.
3. Current public ACM MM 2026 guidance: ACM MM 2026 CFP, important dates, and Author Advocate pages.
4. General OpenReview documentation: useful to interpret `maxLength` and venue-level configuration, but not controlling when the venue-specific instruction differs.
5. Historical ACM MM / other-venue examples: writing and workflow reference only, not binding for ACM MM 2026 Submission8826.

## PC Chairs Email Rule

The latest PC Chairs email clarifies the rebuttal format and overrides earlier tactical interpretations:

- Each paper may include only a single one-page PDF containing all responses.
- The one-page limit is strict; over-page rebuttals will not be considered.
- There is no strict formatting template, but the layout must be reasonable, legible, and clear.
- The single page must include all rebuttal content, including references and supporting material.
- Do not use `Official Comment` to reply individually to reviewers.
- Official Comments are not visible to reviewers and will not be considered.
- The one-page PDF will be visible to reviewers, AC, SAC, and PCs.
- Rebuttal is optional, anonymous, and may focus only on comments that require clarification.

## Current Requirements for Submission 8826

### Rebuttal PDF

OpenReview invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Rebuttal_PDF`

Observed fields:

- `minReplies`: 1
- `maxReplies`: 1
- `pdf.value.param.type`: `file`
- `pdf.value.param.extensions`: [`pdf`]
- `pdf.value.param.maxSize`: 50
- `pdf.description`: `Upload a single page PDF file that ends with .pdf`
- due: `2026-06-05 17:59:00` China Standard Time
- expiration: `2026-06-05 18:29:00` China Standard Time
- observed readers: Program Chairs, Submission8826 Senior Area Chairs, Submission8826 Area Chairs, Submission8826 Authors

Interpretation:

- Prepare one global PDF response for the paper, not one PDF per reviewer.
- The machine-readable fields enforce one submitted reply, PDF type/extension, and size.
- The one-page requirement is strict under the PC Chairs email. Prepare a single-page PDF regardless of whether OpenReview visibly enforces page count.
- There is no observed 2500-word PDF limit. The old `maxLength: 2500` reference is OpenReview's default text-field form, not this PDF upload invitation.

### Official Comment

OpenReview invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Official_Comment`

Observed fields:

- `title`: optional string, `maxLength`: 500
- `comment`: required string textarea, `maxLength`: 5000, markdown enabled
- `replyto`: can target a note in the submission forum, so it can be attached at paper level or under a review-level note
- no `duedate` observed
- expiration: `2026-06-05 19:59:00` China Standard Time
- observed mandatory readers: Program Chairs and Submission8826 Senior Area Chairs
- observed optional readers: Submission8826 Area Chairs and Submission8826 Authors
- no reviewer reader option observed

Interpretation:

- Official Comment is a text-comment channel, not the main rebuttal PDF.
- Do not submit one comment per reviewer.
- Do not use Official Comment for reviewer responses, overflow evidence, pointers to content outside the PDF, or scientific rebuttal material.
- The PC Chairs email says comments submitted there are not visible to reviewers and will not be considered.

### Author Advocate Mediation

OpenReview invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Author_Advocate_Mediation`

Observed fields:

- `minReplies`: 1
- `maxReplies`: 1
- due: `2026-06-05 17:59:00` China Standard Time
- expiration: `2026-06-05 18:29:00` China Standard Time
- `target_reviewer_ids`: string, `maxLength`: 50
- checkbox grounds are factual/process issues only

Interpretation:

- This is optional and separate from the rebuttal.
- Use only for qualifying factual/process cases. General scientific disagreements, misunderstandings, novelty disputes, or "reviewer lacks expertise" should go to AC/SAC through the normal response/comment route, not AA mediation.

## Public ACM MM 2026 Guidance

ACM MM 2026 CFP says authors may optionally submit a rebuttal in OpenReview, must maintain anonymity, and cannot include external links or materials such as code/videos.

The public Important Dates page lists Main Track `Rebuttal | 04-June`.

The public Author Advocate page limits AA mediation to factual/process issues and says other scientific concerns should be addressed to the Area Chair.

## Historical/Related Rules Checked

ACM MM 2025 public author instructions said to click `Rebuttal` to add one rebuttal per review, keep responses self-contained, avoid external links, and maintain anonymity. This shows ACM MM workflow can vary by year and OpenReview configuration; it does not override the 2026 Submission8826 invitation.

ACM MM 2024 had a downloadable rebuttal template whose `rebuttal.tex` says the author response was limited to a one-page PDF file and was intended for factual errors or requested information, not new unrequested contributions. This supports the one-page-PDF convention historically, but the controlling 2026 evidence remains the current OpenReview invitation.

OpenReview's rebuttal-stage documentation says venues can configure one rebuttal per paper, one rebuttal per review, or multiple rebuttals per paper, and can add/overwrite default form fields. Therefore, year-specific and submission-specific invitations matter more than generic defaults.

## Writing Patterns From Strong Public Rebuttals/Comments

Observed pattern across public OpenReview author responses and rebuttals:

- Start with a one-sentence scope statement: what changed, what evidence was added, and which main concerns are addressed.
- Group duplicated concerns across reviewers instead of repeating the same paragraph five times.
- Use reviewer IDs only where needed for traceability; use concern labels for shared issues, e.g., `Novelty`, `Detector attribution`, `Efficiency`, `Generality`.
- Lead with quantitative answers when the critique is empirical. A small table with matched protocols is better than a long promise.
- Define diagnostics exactly before using them, especially for new quantities such as corrected/harmful flips, anchor strata, or unsupported mentions.
- Separate "we fixed a misunderstanding" from "we acknowledge a limitation". Defensive prose tends to be weaker than bounded claim calibration.
- Avoid broad future-work promises unless paired with a concrete camera-ready edit or a current result.
- Mention paper locations or camera-ready edits when possible, but do not overuse "we will add" as a substitute for answering the concern.
- Do not introduce external links, code links, or anonymous project pages when the venue prohibits external materials.
- Keep the tone factual. Politeness helps; flattery and emotional language waste space.

Local reference PDFs are saved in:

`D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530`

Use them as layout/presentation references only. They are public examples and historical references, not binding ACM MM 2026 instructions.

## Recommended CHORD Response Strategy

Primary PDF:

- Use one-page PDF as the only rebuttal deliverable.
- Compress around five reviewer concerns: mechanism, detector attribution, recent baselines/fairness, efficiency/default setting, and claim/generality boundary.
- Put the strongest expected-result table(s) in the PDF only if readable at one page. Prefer one compact integrated table over multiple cramped tables.
- Include exact definitions for the most important diagnostics; avoid introducing table values whose definitions are not clear.
- State that CHORD is base-MLLM training-free but detector-assisted.
- State practical default clearly: P+C or smaller rollout for deployment; Full CHORD as quality-oriented/offline setting.
- Keep relation/composition claims out of the main claim if expected gains remain weak.

Final pre-submit check:

- Re-open Author Console immediately before upload.
- Confirm the `Rebuttal_PDF` invitation still says single-page PDF and due time has not changed.
- Do not post Official Comments as reviewer replies.
- Verify PDF page count locally with `pdfinfo`.
- Verify no external links and no author-identifying text.
