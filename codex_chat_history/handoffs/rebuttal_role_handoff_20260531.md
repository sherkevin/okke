# Rebuttal Role Handoff, 2026-05-31

Role: author-team / rebuttal-builder for CHORD Submission 8826.

Redacted raw Codex session:

- Thread id: `019e6f2b-33a9-78e0-bdb7-eaa32981ae5c`
- Local redacted raw copy: `codex_chat_history/redacted_raw/019e6f2b-33a9-78e0-bdb7-eaa32981ae5c/`
- Resume: `happy codex --resume 019e6f2b-33a9-78e0-bdb7-eaa32981ae5c`

## Current State

The rebuttal work is still in convergence, but `review_v12` changed the immediate priority. The current author-team decision is to make the full scientific response complete and readable first, then compress later. The 11:27 and 22:08 one-page PDFs are backups for later compression, not the current scientific master.

Latest high-signal files:

| Purpose | File |
|---|---|
| Official review source | `papers/opera_acm_sigconf/rebuttal/reviews_20260528.md` |
| Review response plan | `papers/opera_acm_sigconf/rebuttal/official_review_response_plan_20260528.md` |
| Reviewer intent | `papers/opera_acm_sigconf/rebuttal/reviewer_true_intent_analysis_20260529.md` |
| Latest author master | `papers/opera_acm_sigconf/rebuttal/author_response_min_diff_expected_20260531_2215_review_v12.md` |
| Previous author master | `papers/opera_acm_sigconf/rebuttal/author_response_min_diff_expected_20260531_1133_review_v11.md` |
| One-page backup 1 | `papers/opera_acm_sigconf/rebuttal/author_response_onepage_expected_20260531_1127.tex` / `.pdf` |
| One-page backup 2 | `papers/opera_acm_sigconf/rebuttal/author_response_onepage_expected_20260531_2208.tex` / `.pdf` |
| Latest reviewer audit | `papers/opera_acm_sigconf/rebuttal/strict_reviewer_audit_2154_latest_20260531_review_v11.md` |

## Mission

Build from the current full scientific master, then compress only after the master is stable. Do not let one-page constraints remove the score-moving scientific answers too early.

The next useful author action is:

1. Review `author_response_min_diff_expected_20260531_2215_review_v12.md` as the current scientific master.
2. Make only targeted edits that improve scientific completeness, reviewer-specific clarity, or evidence boundaries.
3. After the master is accepted, create a new timestamped one-page `.tex` derived from the v12 content, using the 11:27/22:08 PDFs only as compression references.
4. Rebuild and visually audit the one-page PDF.
5. Ask for human review before any OpenReview upload.

## Highest-Value Content To Preserve

| Priority | Must appear if space permits | Why |
|---|---|---|
| P0 | Future mechanism row: Full vs P+C flip rate, corrected/harmful counts, Adv. F1 delta, CHAIR-S delta | Directly answers M8du's score-moving question. |
| P0 | Detector attribution ordering: same-anchor/non-CHORD, random/uniform/no-Current controls below P+C and Full | Directly addresses KrEs's main Weak Reject reason. |
| P0 | Cost/default row: P+C practical/default, Full quality/offline, total latency/VRAM if compact | Addresses jjVG, KrEs, and ve3y without pretending Full is cheap. |
| P1 | Recent baseline compact comparison: ONLY, VHD/VHR, HALC, P+C, Full under matched protocol | Closes jjVG/KrEs completeness concerns. |
| P1 | Claim boundary: detector-assisted, base-MLLM training-free, object-grounded scope, attention as operational feature | Preserves yx8u and avoids overclaiming. |

## Content To Cut First

Cut these before cutting numeric Future or detector evidence:

1. Figure 2 wording.
2. Repeated camera-ready promises.
3. Long explanations of evidence policy.
4. Detector threshold details.
5. Broad generality discussion beyond object-grounded hallucination.

## Reviewer Targets

| Reviewer | Current concern | Rebuttal role action |
|---|---|---|
| jjVG | Cost, k/m, recent methods, Figure 2 | Include compact cost/default, k/m/Pareto, and recent-baseline line. |
| KrEs | Novelty, Grounding DINO attribution, end-to-end cost | Include detector attribution ordering and bounded novelty language. |
| yx8u | Incremental novelty, detector dependence, attention reliability, scope | Keep claim discipline; do not overclaim detector independence or attention causality. |
| ve3y | Practical value under overhead | Frame P+C as practical/default and Full as quality/offline. |
| M8du | Future mechanism and detector controls | Make Future row first and detector attribution second. |

## Stop Rules

- Do not submit to OpenReview automatically.
- Do not use any simulated review files as official evidence.
- Do not erase old expected-only boundary text without checking the newest `LOCAL_TASKS.md` rule and latest strict audit.
- Do not broaden the paper's claim to relation/composition/general hallucination unless measured evidence exists.
- Do not compress again until the full v12 scientific master is accepted as complete.

## Verification Checklist

Run these before declaring the rebuttal candidate ready:

```powershell
pdfinfo papers\opera_acm_sigconf\rebuttal\<new-onepage>.pdf
pdftotext -layout papers\opera_acm_sigconf\rebuttal\<new-onepage>.pdf -
Select-String papers\opera_acm_sigconf\rebuttal\<new-onepage>.log -Pattern 'Output written|Overfull|Underfull|Warning|Error|Fatal|Emergency'
```

Then have the strict reviewer role audit the exact new `.tex` and `.pdf`.
