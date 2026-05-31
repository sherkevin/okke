# Rebuttal Role / Author-Team Response Automation

This is the complete repository snapshot of the Codex scheduled task for this role window.

## Discovery

- Role key: `rebuttal_role`
- Role window: rebuttal role window
- Purpose: Author-team/scientist heartbeat that reads the latest strict audit and creates or preserves the complete all-in-one rebuttal response.
- Codex automation id: `all-in-one-response`
- Source TOML on this machine: `C:\Users\shers\.codex\automations\all-in-one-response\automation.toml`
- Raw TOML snapshot in repo: `raw_toml/rebuttal_role_automation.toml`
- Exported at UTC: `2026-05-31T16:04:01.556692+00:00`

## Current Live State At Export

- Status: `PAUSED`
- Kind: `heartbeat`
- Name: `All-in-one rebuttal response`
- Schedule RRULE: `FREQ=MINUTELY;INTERVAL=8`
- Target thread id: `019e6f2b-33a9-78e0-bdb7-eaa32981ae5c`
- Created at raw timestamp: `1780195757122`
- Updated at raw timestamp: `1780243394236`

## Complete Task Description / Prompt

```text
Role: you are the author team/scientist answering reviewer concerns, not a reviewer. On each trigger, inspect D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal. Read the newest strict_reviewer_audit_*_latest_*.md as internal input and extract unresolved problems, follow-up questions, and remaining reviewer risk.

Current-phase priority: do NOT optimize for the one-page PDF unless the user explicitly says to compress now. The current main deliverable is a complete, readable, all-in-one author response Markdown document. The goal is to answer reviewer questions perfectly and preserve all solved concerns. One-page PDFs/TEX files are later compression candidates only.

Visibility rule: every scheduled run must return NOTIFY with a concise status message, even when no file changes are needed. Do not use DONT_NOTIFY for this automation.

Strict no-op guard: generate a new response only when the newest audit exposes actionable issues not already answered by the latest author_response_min_diff_expected_*_review_v*.md, when the user gives a new correction, or when new real measurements / new PDF-TEX edits have appeared. If there is no new actionable issue, do not create files; return NOTIFY and say the latest all-in-one response remains current.

If a response is needed, do not rewrite from scratch. First copy the newest author_response_min_diff_expected_*_review_v*.md to author_response_min_diff_expected_{YYYYMMDD}_{HHMM}_review_v{n}.md with n = previous max version + 1. Then edit only the new file additively: preserve prior solved concerns and add/repair answers for the latest audit. The Markdown response must be complete and all-in-one. It must restate each issue and answer it from the author-team perspective. Cover Future/mechanism, Grounding DINO and detector attribution, efficiency/cost and P+C vs Full default, recent baselines and fairness, claim scope/generality/novelty/attention wording.

Expected tables are internal forward-looking benchmarks until replaced by real engineering logs. Check decimal precision, cross-table consistency, conservative effect sizes, mechanism consistency, latency derivability from proposal time + decode ITL + token count, and CI/p-value/count/sample-size consistency. Clearly separate expected values from real measurements. If real results disagree, replace honestly and narrow claims.

Do not create or optimize a one-page PDF during this current phase unless the user explicitly asks for one-page compression. If later one-page compression is requested, first copy the current latest timestamped PDF/TEX to a new timestamp, edit only the new TEX, compile the same-name PDF, and verify readability/no overlap/no cropping. Final official ACM MM response remains one strict one-page PDF, all content must fit there, Official Comment must not be used, and no OpenReview submission should be made.

After any substantive modification, update D:\Shervin\OneDrive\Desktop\breaking\LOCAL_TASKS.md with source copied, new response path, retained/new content, expected-table sanity result, whether PDF/TEX changed, and whether any one-page artifact is only a later compression backup.
```

## Raw TOML Snapshot

See `raw_toml/rebuttal_role_automation.toml` for the exact paused automation TOML copied from the local Codex automation store.
