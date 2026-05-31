# CHORD Rebuttal PDF Plan, 2026-05-29

Status: execution plan for an internal rebuttal PDF draft. The PDF is not a final submission until expected values are replaced by measured remote results.

Canonical review source: `reviews_20260528.md` only.

Format boundary checked on 2026-05-29: ACM MM 2026 confirms OpenReview rebuttal, anonymity, and no external links. OpenReview's default rebuttal form is Markdown text with a 2500-character limit, but the logged-in ACMMM 2026 form controls the final field type and length.

Format sources:

- ACM MM 2026 Call for Technical Papers: `https://2026.acmmm.org/site/cfp-guidelines.html`
- ACM MM 2026 Important Dates: `https://2026.acmmm.org/site/important-dates.html`
- OpenReview Default Rebuttal Form: `https://docs.openreview.net/reference/default-forms/default-rebuttal-form`
- OpenReview Rebuttal Stage: `https://docs.openreview.net/reference/stages/rebuttal-stage`

| Step | Owner | Artifact | Acceptance check |
|---|---|---|---|
| 1. Freeze reviewer concern map | Codex | `reviewer_audited_expected_result_tables_20260529.md` | Every real reviewer concern maps to one evidence table and a stop rule. |
| 2. Build expected table package | Codex | same file plus PDF tables | Values are labeled as expected/target ranges, not measurements. |
| 3. Draft rebuttal PDF | Codex | `rebuttal_expected_tables_draft_20260529.tex` | PDF has an opening response, compact expected tables, and OpenReview text skeleton. |
| 4. Compile and inspect | Codex | `rebuttal_expected_tables_draft_20260529.pdf` | LaTeX compiles without fatal errors; output exists and is readable. |
| 5. Remote experiment replacement | Engineer | measured JSON/JSONL and actual result tables | Replace every `TBD` / `expected` field with measured values or downgrade the claim. |
| 6. Final OpenReview response | Authors | OpenReview text field | Fits the logged-in ACMMM 2026 form, stays anonymous, and includes no external links. |

Reviewer-audit rule: if any P0 table fails its stop rule, the final rebuttal must narrow the corresponding claim rather than argue around the result.
