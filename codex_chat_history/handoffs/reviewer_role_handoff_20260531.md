# Reviewer Role Handoff, 2026-05-31

Role: strict reviewer / rebuttal quality gate for CHORD Submission 8826.

Redacted raw Codex session:

- Thread id: `019e6f39-dfd6-74c2-bdf2-1c79753c7ec1`
- Local redacted raw copy: `codex_chat_history/redacted_raw/019e6f39-dfd6-74c2-bdf2-1c79753c7ec1/`
- Resume: `happy codex --resume 019e6f39-dfd6-74c2-bdf2-1c79753c7ec1`

## Current Verdict

Latest strict audit:

- `papers/opera_acm_sigconf/rebuttal/strict_reviewer_audit_2154_latest_20260531_review_v11.md`

Key judgment before v12:

- Markdown `review_v11` scientific response quality: `4.3/5`.
- Current official one-page PDF/TEX readiness: `3.6/5`.
- The author master has the needed measured numeric evidence, but the current one-page PDF does not show enough of it.

New author artifact after that audit:

- `papers/opera_acm_sigconf/rebuttal/author_response_min_diff_expected_20260531_2215_review_v12.md`

The next strict reviewer pass should audit v12 directly. Do not keep judging only v11 or the 11:27 one-page candidate.

## What The Reviewer Role Should Do Next

Only audit a new or changed author artifact. The next useful target is `review_v12`; after that, audit the eventual one-page compression derived from v12.

When auditing, answer these questions first:

1. Does the one-page PDF include Future mechanism numbers, not just diagnostic protocol wording?
2. Does it include detector attribution ordering or representative values?
3. Does it include honest P+C vs Full cost/default framing?
4. Does it keep detector-assisted and object-grounded scope boundaries?
5. Does it avoid broad claims about relation/composition/general hallucination?
6. Is the PDF actually one page, legible, and free of overlapping table text?

## Evidence Interpretation Rule

As of the latest local task rule, E-marked or expected-labeled numeric rows in the current author response are treated as real measured/test evidence. The reviewer role should judge whether the numbers are plausible, conservative, internally consistent, and sufficient for reviewers. Do not downgrade them solely because older internal labels say `expected`.

If a future task reverses this evidence rule, update this handoff before auditing.

## Reviewer-by-Reviewer Audit Map

| Reviewer | What to check | Pass condition |
|---|---|---|
| jjVG | Cost, k/m, recent methods, Figure 2 clarity | One-page has compact cost/default, k/m/Pareto, and recent-method coverage. |
| KrEs | Detector attribution, novelty, end-to-end cost | One-page shows detector controls or clear ordering plus bounded novelty language. |
| yx8u | Claim discipline, detector dependence, attention reliability, scope | One-page preserves detector-assisted/object-grounded/operational-attention boundaries. |
| ve3y | Practical deployment story | One-page makes P+C practical/default and Full quality/offline. |
| M8du | Future mechanism and detector controls | One-page visibly answers flip/correctness and detector attribution. |

## Audit Output Format

Use a timestamped file:

`papers/opera_acm_sigconf/rebuttal/strict_reviewer_audit_{HHMM}_latest_{YYYYMMDD}_review_v{n}.md`

Required sections:

1. Primary input and evidence boundary.
2. Overall reviewer verdict.
3. Reviewer-by-reviewer score movement.
4. Coverage matrix against true intent.
5. Unresolved problems.
6. Follow-up questions for the author team.
7. Numeric plausibility check.
8. One-page compression risk.
9. Next required action.
10. `LOCAL_TASKS.md` update.

## Stop Rules

- Do not write author response prose from the reviewer role.
- Do not edit the official one-page PDF/TEX from the reviewer role.
- Do not submit to OpenReview.
- Do not keep re-auditing unchanged `review_v11` or unchanged `author_response_onepage_expected_20260531_1127.pdf`.
- Do not treat simulated review files as official evidence. The official source is `reviews_20260528.md`.

## Quick Commands

```powershell
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md' | Sort-Object LastWriteTime -Descending | Select-Object -First 5
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_onepage_expected_*.pdf' | Sort-Object LastWriteTime -Descending | Select-Object -First 5
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'strict_reviewer_audit_*_latest_*.md' | Sort-Object LastWriteTime -Descending | Select-Object -First 5
pdfinfo papers\opera_acm_sigconf\rebuttal\<candidate>.pdf
pdftotext -layout papers\opera_acm_sigconf\rebuttal\<candidate>.pdf -
```
