# OpenReview Rebuttal Workflow Check, 2026-05-30

Submission: `8826`

Forum: `0f8ta8WBat`

Scope: verify the actual ACM MM 2026 OpenReview workflow without submitting any comment or PDF.

## Short Answer

The active workflow is not a 2500-word PDF limit and not a per-reviewer comment workflow.

The latest PC Chairs email is now the highest-priority rule source:

1. **Rebuttal PDF**: one global strict single-page PDF upload for Submission 8826; all response content, references, and supporting material must fit on this page.
2. **Official Comment**: do not use this to reply individually to reviewers; PC Chairs state that Official Comments are not visible to reviewers and will not be considered.
3. **Author Advocate Mediation**: optional, one request, only for tightly defined review-process/factual-error cases.

No comment or PDF was submitted during this inspection.

## What `maxLength` Means

`maxLength` is an OpenReview schema parameter for a **string/text field**. It counts characters, not PDF words.

Examples found in the active invitations:

- `Official_Comment.comment`: `type=string`, `input=textarea`, `maxLength=5000`.
- `Official_Comment.title`: `type=string`, `maxLength=500`.
- `Rebuttal_PDF.pdf`: `type=file`, `extensions=[pdf]`, `maxSize=50`, description says `Upload a single page PDF file that ends with .pdf`.

Therefore, the 2500/default OpenReview text-form idea is not the controlling rule for this submission's PDF upload.

## Rebuttal PDF

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Rebuttal_PDF`

Observed schema:

- `type`: note
- `minReplies`: 1
- `maxReplies`: 1
- field: `pdf`
- field type: `file`
- extension: `pdf`
- max size: `50`
- description: `Upload a single page PDF file that ends with .pdf`
- readers: Program Chairs, Submission8826 Senior Area Chairs, Submission8826 Area Chairs, Submission8826 Authors
- due date: 2026-06-05 17:59 China Standard Time
- expiration: 2026-06-05 18:29 China Standard Time

Interpretation:

- This is the main global rebuttal deliverable.
- It is not per reviewer.
- The one-page limit is strict. Over-page rebuttals will not be considered.
- There is no strict template, but font size, line spacing, and layout must be reasonable and legible.
- The current 5-page timestamped `author_response_min_diff_expected_YYYYMMDD_HHMM.pdf` is an internal master and must be distilled into a one-page upload PDF.

## Official Comment

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Official_Comment`

Observed schema:

- field `title`: optional, max 500 characters.
- field `comment`: required, max 5000 characters.
- comment supports Markdown and LaTeX.
- `replyto` can target a note in the forum, so a comment can be attached to the paper-level thread or to an individual review.
- no `minReplies` / `maxReplies` observed for this invitation, so it is not a required one-shot global task in the same way as `Rebuttal_PDF`.
- expiration: 2026-06-05 19:59 China Standard Time.

UI text when adding a comment under a review:

- It asks authors to select readers.
- Program Chairs and Senior Area Chairs are mandatory readers.
- Area Chairs and Authors are selectable and were checked in the UI.
- The form states Program Chairs will not be notified.
- It states Senior Area Chairs will be notified for comments visible only to mandatory readers.
- It states all other readers will be notified of all comments.

Important reader finding:

- The observed Official Comment reader options did **not** include the individual reviewer or reviewers group.
- Thus, under the current schema, Official Comments appear to be AC/SAC/PC-facing comments, not direct reviewer-facing rebuttal messages.

Interpretation:

- Do not submit one Official Comment per reviewer.
- Do not use Official Comment for scientific rebuttal content.
- The PC Chairs email says Official Comments are not visible to reviewers and will not be considered, so the single-page Rebuttal PDF must carry the entire rebuttal.

## Author Advocate Mediation

Invitation:

`acmmm.org/ACMMM/2026/Conference/Submission8826/-/Author_Advocate_Mediation`

Observed schema:

- `minReplies`: 1
- `maxReplies`: 1
- due date: 2026-06-05 17:59 China Standard Time
- expiration: 2026-06-05 18:29 China Standard Time
- requires `target_reviewer_ids`, max 50 characters.
- checkboxes cover limited grounds:
  - fewer than 3 reviews,
  - data entry error,
  - requested comparison to unpublished/post-deadline paper,
  - factual errors regarding paper contents that directly impacted the score,
  - review summary copied directly from abstract,
  - unprofessional / hostile / dismissive language.

The form explicitly says AA mediation is exclusively for these factual/process reasons and not for general scientific misunderstandings, lack of expertise, contradictory statements, or extreme ratings without justification.

Interpretation:

- This is optional and should not be used unless we have a clearly qualifying process/factual-error case.
- It is not a substitute for the rebuttal PDF.

## Public Conference Sources

ACM MM 2026 Call for Technical Papers:

- Rebuttal is submitted in OpenReview.
- Rebuttal must maintain anonymity.
- Rebuttal cannot include links to external material such as code or videos.

ACM MM 2026 Important Dates:

- Main Track lists `Rebuttal | 04-June`.

Authenticated OpenReview task is more specific for this submission:

- Author Console displays `Submission8826 Rebuttal PDF`.
- Due: 05 Jun 2026, 17:59 China Standard Time.

## Practical Recommendation

Use the single-page `Rebuttal PDF` as the only rebuttal artifact.

Do not use `Official Comment` for reviewer replies, AC pointers, extra supporting material, or overflow content. The PC Chairs email says comments submitted there are not visible to reviewers and will not be considered.

Do not use Author Advocate Mediation unless we can identify a qualifying factual/process issue.
