# Codex Chat History And Role Handoffs

This directory packages the Codex sessions that matter for the CHORD rebuttal handoff.

Redacted raw logs are kept because the user explicitly requested that chat records be uploaded with the project. They are not the best entry point for a collaborator. Start from the handoff files and the cleaned transcripts below, then open the redacted jsonl only when exact prior messages or full tool outputs are needed.

## Start Here

| Role / purpose | Clean handoff | Raw session id | Raw log |
|---|---|---|---|
| Rebuttal author role | `handoffs/rebuttal_role_handoff_20260531.md` | `019e6f2b-33a9-78e0-bdb7-eaa32981ae5c` | `redacted_raw/019e6f2b-33a9-78e0-bdb7-eaa32981ae5c/` |
| Strict reviewer role | `handoffs/reviewer_role_handoff_20260531.md` | `019e6f39-dfd6-74c2-bdf2-1c79753c7ec1` | `redacted_raw/019e6f39-dfd6-74c2-bdf2-1c79753c7ec1/` |
| Current sync/upload work | this README and `LOCAL_TASKS.md` | `019e7e49-9136-7c02-b3f4-7f17404d2932` | `redacted_raw/019e7e49-9136-7c02-b3f4-7f17404d2932/` |

## Cleaned Reading Layer

Use `cleaned/README.md` first for a compact index. Each session under `cleaned/<session-id>/` contains:

- `transcript.md`: chronological human-readable transcript with long tool outputs summarized.
- `compact_events.jsonl`: deduplicated structured event stream for programmatic inspection.
- `summary.json`: source SHA256, source byte count, original/compact event counts, duplicate count, and omitted encrypted-reasoning count.

The cleaned layer removes exact duplicate compact events, omits encrypted reasoning payloads, and summarizes long tool outputs by preview plus SHA256. The complete redacted raw logs remain in `redacted_raw/` for exact reconstruction.

## Resume Commands

```powershell
happy codex --resume 019e6f2b-33a9-78e0-bdb7-eaa32981ae5c
happy codex --resume 019e6f39-dfd6-74c2-bdf2-1c79753c7ec1
happy codex --resume 019e7e49-9136-7c02-b3f4-7f17404d2932
```

The local helper can also find sessions by name:

```powershell
codex --query "rebuttal role" --exact
codex --query "reviewer role" --exact
```

## Canonical Project Files

Use these repo files before reading redacted raw chat logs:

| Need | File |
|---|---|
| Official reviewer text only | `papers/opera_acm_sigconf/rebuttal/reviews_20260528.md` |
| Rebuttal strategy map | `papers/opera_acm_sigconf/rebuttal/official_review_response_plan_20260528.md` |
| Reviewer true-intent map | `papers/opera_acm_sigconf/rebuttal/reviewer_true_intent_analysis_20260529.md` |
| Latest author-team master | `papers/opera_acm_sigconf/rebuttal/author_response_min_diff_expected_20260531_2215_review_v12.md` |
| One-page compression backups | `papers/opera_acm_sigconf/rebuttal/author_response_onepage_expected_20260531_1127.pdf` / `author_response_onepage_expected_20260531_2208.pdf` and matching `.tex` files |
| Latest strict reviewer audit | `papers/opera_acm_sigconf/rebuttal/strict_reviewer_audit_2154_latest_20260531_review_v11.md` |
| Live task ledger | `LOCAL_TASKS.md` |

## Evidence Boundary

The latest author-team master is `review_v12`, which corrects the immediate priority: answer the science completely first, then compress later. Older files may still say to freeze or optimize the 11:27 one-page candidate. When in conflict, follow the newest `LOCAL_TASKS.md` entries and `author_response_min_diff_expected_20260531_2215_review_v12.md`.

OpenAI-style `sk-*` tokens and GitHub-style `gh*_...` tokens were mechanically replaced with redacted placeholders before GitHub upload. The local unsanitized Codex source remains outside this tracked directory.

Do not submit anything to OpenReview from these handoff files. The final upload decision remains a human step.
