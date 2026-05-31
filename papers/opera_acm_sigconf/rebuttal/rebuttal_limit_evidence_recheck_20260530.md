# Rebuttal Limit Evidence Recheck, 2026-05-30

Submission: `8826`

Forum: `0f8ta8WBat`

Purpose: high-stakes re-verification of ACM MM 2026 rebuttal/comment limits.

No comment, mediation request, or rebuttal PDF was submitted during this check.

## Confidence Summary

I am confident about the current rules for Submission 8826 because the highest-priority evidence is now the PC Chairs' latest rebuttal-format email, consistent with the authenticated OpenReview `Rebuttal_PDF` invitation.

The public ACM MM 2026 website gives general rebuttal guidance. The authenticated OpenReview invitation confirms a single-page PDF upload. The PC Chairs email further clarifies that Official Comments must not be used for individual reviewer replies and will not be considered for rebuttal.

Residual risk: organizers can technically change an OpenReview invitation before the deadline. Re-check the Author Console once immediately before final submission.

## Latest PC Chairs Email, Highest-Priority Rule

Local memo:

`D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\pc_chairs_rebuttal_format_email_20260530.md`

Binding implications:

- Each paper may include only one single-page PDF containing all responses.
- The one-page limit is strict; over-page rebuttals will not be considered.
- There is no strict template, but layout must be legible with reasonable font size and line spacing.
- The page must include all rebuttal content, including references and supporting material.
- Do not use `Official Comment` to reply individually to reviewers.
- Official Comments are not visible to reviewers and will not be considered.
- The one-page PDF is visible to reviewers, AC, SAC, and PCs.
- Rebuttal is optional, anonymous, and may focus only on comments requiring clarification.

## Public Official Evidence

ACM MM 2026 Call for Technical Papers:

Source: https://2026.acmmm.org/site/cfp-guidelines.html

- Authors may optionally submit a rebuttal in OpenReview.
- Rebuttal must maintain anonymity.
- Rebuttal cannot include external links/material such as code or videos.

ACM MM 2026 Important Dates:

Source: https://2026.acmmm.org/site/important-dates.html

- Main Track public date says `Rebuttal | 04-June`.

ACM MM 2026 Author Advocate Mediation:

Source: https://2026.acmmm.org/site/author-advocate-mediation.html

- AA mediation is restricted to factual/process issues, not general scientific disagreements.

OpenReview documentation:

Sources:

- https://docs.openreview.net/reference/stages/rebuttal-stage
- https://docs.openreview.net/reference/default-forms/default-rebuttal-form

- Rebuttal stages can be configured as one per paper, one per review, or multiple per paper.
- Default OpenReview rebuttal form uses a string field with `maxLength: 2500`.
- Therefore, a `maxLength` value is a text-field character limit, not a PDF page/word limit.
- Venue-specific invitations can overwrite the default form.

## Authenticated OpenReview Evidence For Submission 8826

Saved evidence file:

`D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_invitation_snapshot_submission8826_20260530.json`

UI screenshot evidence:

`D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_official_comment_form_submission8826_20260530.png`

### Rebuttal PDF

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Rebuttal_PDF`

Fields:

- `minReplies`: 1
- `maxReplies`: 1
- `contentFields.pdf.value.param.type`: `file`
- `contentFields.pdf.value.param.extensions`: [`pdf`]
- `contentFields.pdf.value.param.maxSize`: 50
- `contentFields.pdf.description`: `Upload a single page PDF file that ends with .pdf`
- Enforcement nuance: the machine-readable validator observed here enforces file type/extension/size and one submitted reply. No numeric `maxPages` validator was observed. The one-page requirement is a textual instruction in the active upload field description, so it should be treated as a submission rule even if page count may not be automatically checked by OpenReview.

Deadline:

- UTC due: 2026-06-05 09:59:00
- China Standard Time due: 2026-06-05 17:59:00
- Expiration: 2026-06-05 18:29:00 China Standard Time

Readers observed in the saved OpenReview schema:

- Program Chairs
- Submission8826 Senior Area Chairs
- Submission8826 Area Chairs
- Submission8826 Authors

No reviewer reader is listed in the saved schema for the submitted PDF note. The later PC Chairs email explicitly clarifies that the one-page PDF will be visible to reviewers as well as AC, SAC, and PCs; use the PC email as the controlling visibility statement.

Conclusion:

- This is one global rebuttal PDF for the submission.
- It should be prepared as a single-page PDF because the active upload field explicitly says `Upload a single page PDF file that ends with .pdf`.
- It is not governed by a 2500-word or 2500-character PDF text limit in the schema.

### Official Comment

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Official_Comment`

Fields:

- `title`: optional string, `maxLength`: 500.
- `comment`: required string textarea, `maxLength`: 5000.
- Comment description explicitly says `max 5000 characters`.
- `replyto` can target a note in the forum, so comments can be attached at paper-level or under a review-level note.

Deadline / expiration:

- No `duedate` field observed.
- Expiration: 2026-06-05 19:59:00 China Standard Time.

Readers observed:

- Program Chairs: mandatory.
- Submission8826 Senior Area Chairs: mandatory.
- Submission8826 Area Chairs: optional.
- Submission8826 Authors: optional.

UI note:

- Program Chairs will not be notified.
- Senior Area Chairs are notified for comments visible only to mandatory readers.
- All other selected readers are notified.

No reviewer reader option was observed in the form or saved schema.

Conclusion:

- Official Comment is a text-comment channel, max 5000 characters per comment.
- It is not the rebuttal PDF and not a PDF word/page limit.
- Under the PC Chairs email, it must **not** be used to reply individually to reviewers.
- Official Comments are not visible to reviewers and will not be considered for rebuttal, so do not use them for scientific response content.

### Author Advocate Mediation

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Author_Advocate_Mediation`

Fields:

- `minReplies`: 1
- `maxReplies`: 1
- `target_reviewer_ids`: string, `maxLength`: 50
- checkboxes for limited factual/process reasons only.

Deadline:

- China Standard Time due: 2026-06-05 17:59:00
- Expiration: 2026-06-05 18:29:00 China Standard Time

Allowed grounds:

- fewer than 3 reviews,
- data entry error,
- requested comparison to unpublished/post-deadline paper,
- factual errors directly impacting score,
- review summary copied from abstract,
- unprofessional/hostile/dismissive language.

The form explicitly says general scientific issues, misunderstandings, lack of expertise, contradictory statements, or extreme ratings without justification should be addressed to the Area Chair, not the Author Advocate.

Conclusion:

- Optional. Use only if there is a qualifying factual/process case.

## Current Operational Rule

For Submission 8826, prepare:

1. **One single-page rebuttal PDF** as the only reviewer-facing rebuttal deliverable.
2. **No Official Comments for reviewer response.** Do not split rebuttal content into per-reviewer comments; they are not visible to reviewers and will not be considered.
3. Optional Author Advocate Mediation only for qualifying factual/process problems.

All substantive response content, references, supporting material, definitions, and evidence must fit inside the one-page PDF.
