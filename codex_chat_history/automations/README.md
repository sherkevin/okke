# Codex Role Automations

This directory is the durable repository entry point for the review-role and rebuttal-role scheduled tasks.

- Exported at UTC: `2026-05-31T16:04:01.556692+00:00`
- Live action taken: both role automations were paused after their complete definitions were captured.
- Full prompts are in the per-role Markdown files and exact TOML copies are under `raw_toml/`.

| Role | Automation id | Status at export | Schedule | Target thread | Complete definition | Raw TOML |
|---|---|---|---|---|---|---|
| Review Role / Strict Reviewer Audit | `reviewer-strict-audit-current-thread-8min` | `PAUSED` | `FREQ=MINUTELY;INTERVAL=8` | `019e6f39-dfd6-74c2-bdf2-1c79753c7ec1` | `review_role_automation.md` | `raw_toml/review_role_automation.toml` |
| Rebuttal Role / Author-Team Response | `all-in-one-response` | `PAUSED` | `FREQ=MINUTELY;INTERVAL=8` | `019e6f2b-33a9-78e0-bdb7-eaa32981ae5c` | `rebuttal_role_automation.md` | `raw_toml/rebuttal_role_automation.toml` |

## Operational Note

These snapshots are documentation only. To resume either task, update the live Codex automation from the app or `automation_update`, using the corresponding Markdown/TOML here as the source of truth for the prompt and schedule.
