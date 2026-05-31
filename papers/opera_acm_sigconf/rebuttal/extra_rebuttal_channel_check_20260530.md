# Extra Rebuttal Channel Check, 2026-05-30

Submission: `8826`

Purpose: verify whether the authors can submit any rebuttal material beyond the one-page `Rebuttal_PDF`, such as extra attachments, supplementary files, external links, or additional comments.

No comment, mediation request, or rebuttal PDF was submitted during this check.

## Latest PC Chairs Email

The latest PC Chairs email supersedes earlier tactical interpretations of `Official Comment`:

- Each paper may include only one strict one-page PDF containing all responses.
- The one-page document must include all rebuttal content, including references and supporting material.
- Do not use `Official Comment` to reply individually to reviewers.
- Official Comments are not visible to reviewers and will not be considered.
- The one-page PDF will be visible to reviewers, AC, SAC, and PCs.

## Public Official Evidence

ACM MM 2026 Call for Technical Papers:

- Rebuttal: authors may optionally submit a rebuttal in OpenReview.
- Rebuttal must maintain anonymity.
- Rebuttal cannot include links to external material such as code, videos, etc.
- Supplementary materials were a separate submission item with a 50 MB limit during the submission phase.

ACM MM 2026 Important Dates:

- Main Track `Supplementary Submission`: 08-April.
- Main Track `Rebuttal`: 04-June.

Interpretation:

- The public site distinguishes supplementary submission from rebuttal.
- It does not advertise a new rebuttal-stage supplementary-material upload.
- External links/materials are explicitly disallowed inside the rebuttal.

OpenReview documentation:

- Rebuttal stages are configurable and can be one rebuttal per paper, one per review, or multiple rebuttals per paper.
- Additional rebuttal form options are venue-configured.
- Supplementary material upload is a separate form field/stage that must be configured by the venue.

Interpretation:

- Extra files are only available if the active invitation/form includes such a field.
- For Submission8826, the active rebuttal invitation must control.

## Authenticated Submission8826 OpenReview Evidence

Saved schema:

`D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_invitation_snapshot_submission8826_20260530.json`

Active invitations and content fields:

- `Rebuttal_PDF`
  - content fields: `pdf` only.
  - `pdf.type`: `file`.
  - `pdf.extensions`: [`pdf`].
  - `pdf.maxSize`: 50.
  - description: `Upload a single page PDF file that ends with .pdf`.
  - `minReplies`: 1, `maxReplies`: 1.
- `Official_Comment`
  - content fields: `title`, `comment`.
  - `title.maxLength`: 500.
  - `comment.maxLength`: 5000.
  - no file upload field.
- `Author_Advocate_Mediation`
  - content fields are mediation instructions, reviewer id, and factual/process checkboxes.
  - no general rebuttal attachment field.

OpenReview UI check:

- The submission page shows original `Supplementary Material: zip`, but this is the already-submitted supplementary file from the original submission phase.
- The active `New Rebuttal PDF` form shows `PDF`, `Readers`, `Signatures`, and `Edit History`; no separate attachment/supplement field was observed.
- The Add buttons visible are `Withdrawal`, `Official Comment`, `Author Advocate Mediation`, and `Rebuttal PDF`.

## Operational Conclusion

Allowed / available:

1. One `Rebuttal_PDF` upload, with a single `pdf` file field.
2. Optional `Author_Advocate_Mediation` only for qualifying factual/process issues.

Not available / not allowed:

1. No rebuttal-stage supplementary upload field was observed.
2. No extra attachment field was observed in the rebuttal PDF invitation.
3. External links/materials such as code/videos/etc. are disallowed by ACM MM 2026 public guidance.
4. The original supplementary zip exists on the submission page, but there is no evidence that it can be replaced or augmented during rebuttal.
5. `Official Comment` must not be used as per-reviewer rebuttal, overflow content, or extra scientific evidence; PC Chairs say it is not visible to reviewers and will not be considered.

Practical implication:

- The rebuttal must be fully self-contained in one PDF.
- The scientific conversion argument cannot depend on comments, extra files, external links, old supplementary material, or any off-page references carrying substantive rebuttal content.
- If extra detail cannot fit, prioritize inside the PDF: mechanism, detector attribution, recent baselines, end-to-end efficiency, and claim calibration.
