# Rebuttal / Author-Response Reference Pattern Summary, 2026-05-29

Scope: formatting and writing-pattern guidance for the CHORD minimum-difference expected-results rebuttal PDF. These references are not evidence for CHORD results.

## Downloaded Local References

Directory: `papers/opera_acm_sigconf/rebuttal/reference_rebuttals_20260529/`

| Local file | Source type | Main use |
|---|---|---|
| `neurips2019_paper692_author_feedback.pdf` | NeurIPS one-page author feedback | Compact reviewer-by-reviewer answer style. |
| `neurips2019_adaptive_nn_author_feedback.pdf` | NeurIPS one-page author feedback | Title and section style: "Author feedback" plus reviewer subsections. |
| `neurips2020_author_feedback.pdf` | NeurIPS one-page author feedback | Concern-clustered response style under severe space limits. |
| `openreview_to_reviewers_E1JWvjfuIM.pdf` | OpenReview response/revision PDF | "To Reviewers" title and point-by-point handling. |
| `esurf_response_to_reviewers_comments.pdf` | Journal-style response to reviewers | Detailed point-by-point response pattern; too long for conference rebuttal but useful for completeness. |
| `cvpr_author_response_template_rebuttal.pdf` | CVPR author-response template | Confirms "rebuttal" is a stage/action word, while the document is framed as author response. |
| `sigcomm2014_author_response_guidelines.pdf` | SIGCOMM author-response guidelines | Best-practice guidance: answer factual points, reviewer questions, and avoid unsupported new claims. |

## Title Pattern

Public examples use several acceptable labels:

- "Paper #... Author Feedback"
- "Author feedback: ..."
- "To Reviewers"
- "Response to Reviewers' Comments"
- "LATEX Guidelines for Author Response"

The word "rebuttal" is common in venue instructions and platform buttons, but polished response documents usually use the less adversarial label "Author Response", "Author Feedback", or "Response to Reviewers".

Recommendation for CHORD:

- Use final-looking title: `Author Response for Submission 8826: CHORD`.
- Avoid title text like `Rebuttal Draft` in the PDF itself.
- Keep "rebuttal" in filenames and internal planning documents only.

Rationale: "Rebuttal" is likely allowed if the venue calls the stage a rebuttal, but "Author Response" reads more professional and matches common conference examples. It also avoids sounding argumentative to reviewers.

## Structure Pattern

The strongest examples do not restate the whole paper. They:

1. Open with one short thank-you and a one-sentence diagnosis of the main issue.
2. Group by reviewer concern or by high-level issue, not necessarily by reviewer order.
3. Use direct evidence and precise commitments.
4. Acknowledge valid limitations instead of trying to win every point.
5. Mention exact revision actions when a concern is about related work, wording, or presentation.

For CHORD, this means:

- Lead with the shared concern: attribution and mechanism, not motivation.
- Put Future flip/correctness and detector controls before generic novelty discussion.
- Use tables for mechanism, detector attribution, cost, and k/m robustness.
- End with reviewer-specific closure so every reviewer can locate their concern.

## Table Pattern

Conference author responses often use very compact tables when the table directly resolves a concern. For this CHORD rebuttal, expected-result tables should keep the final table shape now:

- Same row labels as the final measured table.
- Same metrics and units as the final measured table.
- Numeric expected values marked with superscript `E`.
- A small note below the table: expected placeholders, to be replaced by measured SSH results.

When real results arrive, the final edit should only:

1. replace expected values;
2. remove superscript `E`;
3. delete the small expected-result note if all rows are measured;
4. downgrade claims if the measured values miss the pass criteria.

## Writing Rules For CHORD

| Rule | Reason |
|---|---|
| Say "detector-assisted, training-free for the base MLLM" | Addresses KrEs/yx8u without hiding Grounding DINO. |
| Treat attention as an operational scoring signal | Avoids the known "attention is not explanation" objection. |
| Present Past+Current and Full as two operating points | Preserves ve3y/yx8u trust on latency. |
| Use expected values only as removable placeholders | Prevents false measured-evidence claims. |
| Avoid "reviewers misunderstood" wording | Public author-response guidance and examples reward factual clarification, not confrontation. |

## Sources Checked

- ACM MM 2026 Call for Technical Papers: OpenReview rebuttal is optional and used to address reviewer comments.
- OpenReview Rebuttal Stage / Default Rebuttal Form: rebuttal can be venue-configured; default form is text/Markdown with length constraints.
- CVPR author-response template: title and body frame the document as an author response, while the stage is called a rebuttal.
- SIGCOMM author-response guidance: focus on factual corrections/questions; PC can ignore unsupported new results.
