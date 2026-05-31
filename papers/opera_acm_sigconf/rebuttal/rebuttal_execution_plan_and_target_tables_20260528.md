# CHORD Rebuttal Execution Plan and Target Tables, 2026-05-28

Scope: execution plan plus ideal target tables. Target tables below are not measured results. They define the evidence pattern we want to verify before writing the final rebuttal.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record.

## ACM MM / OpenReview Format Boundary

Verified public rules:

- ACM MM 2026 uses OpenReview for submission, peer review, and author rebuttal.
- Authors may optionally submit a rebuttal after receiving reviews.
- The rebuttal must maintain anonymity.
- The rebuttal cannot include links to external material such as code, videos, or other externally hosted evidence.
- Official public pages do not publish a character limit or a rebuttal PDF template.
- OpenReview's generic rebuttal stage can be configured as one rebuttal per paper, one rebuttal per review, or multiple rebuttals per paper. The actual ACMMM 2026 OpenReview form must therefore be checked before final drafting.
- OpenReview's default rebuttal form is a Markdown text field with `maxLength: 2500`, but venues can override it.
- Public ACM MM 2026 sources do not confirm a one-page PDF rebuttal limit. Do not draft or package the response as a one-page PDF unless the logged-in ACMMM 2026 OpenReview form or official email explicitly requires it.
- Consolidated demand note: `rebuttal_demand_20260529.md`.

Operational drafting rule until the OpenReview form is checked:

- assume short OpenReview text, target <=2500 characters;
- no external links;
- no author-identifying language;
- no claim that depends on uploaded revised paper/PDF;
- use compact inline tables only if OpenReview field supports Markdown or readable monospace text.

Sources:

- ACM MM 2026 Call for Technical Papers: https://2026.acmmm.org/site/cfp-guidelines.html
- ACM MM 2026 Author Instructions: https://2026.acmmm.org/site/author-instructions.html
- ACM MM 2026 Important Dates: https://2026.acmmm.org/site/important-dates.html
- OpenReview Rebuttal Stage documentation: https://docs.openreview.net/reference/stages/rebuttal-stage
- OpenReview Default Rebuttal Form documentation: https://docs.openreview.net/reference/default-forms/default-rebuttal-form

## Execution Plan

| Phase | Objective | Concrete action | Output artifact | Stop condition |
|---:|---|---|---|---|
| 1 | Confirm format | Inspect official pages and, when available, the actual OpenReview response field. | Format note in this file and final rebuttal header constraints. | Do not draft final prose until char limit/per-review vs per-paper mode is known. |
| 2 | Build target tables | Define what convincing evidence should look like before running. | This plan file. | Targets must be labeled as targets, not results. |
| 3 | Code/result inventory | Find local evaluation scripts, logs, and result JSONs for CHORD variants. | `actual_evidence_inventory_20260528.md`. | If no runnable path exists locally, mark remote/GPU requirement. |
| 4 | Mechanism diagnostic | Compare Full CHORD vs Past+Current at answer/token-decision level. | Table 1 actual results. | Do not claim future helps unless flip/correctness evidence supports it. |
| 5 | Detector attribution | Compare real anchors, no/uniform anchors, random anchors, and no-current-anchor variants. | Table 2 actual results. | Do not claim detector-independent gains unless controls support it. |
| 6 | Cost accounting | Measure/report detector time, decode ITL, total latency, VRAM, and batch note. | Table 3 actual results. | Do not hide one-time proposer cost. |
| 7 | Secondary checks | k/m sweep and related-work/baseline positioning. | Compact appendix notes or final rebuttal bullets. | If not run, state as camera-ready update, not measured evidence. |
| 8 | Draft rebuttal | Compose a short evidence-first author response. | `final_rebuttal_draft_20260528.md`. | Every numeric claim links to actual local result/log. |

## Target Table 1: Future Rollout Mechanism

Purpose: answer M8du's main question: does Future actually change admitted decisions, and are those changes correct?

Ideal pattern: future should not flip too often, but when it flips, most flips should correct hallucination or preserve correctness, especially in open-ended CHAIR-style generation.

| Dataset / setting | Unit | Target Full vs P+C difference | Target correct-flip pattern | Target harmful-flip ceiling | Desired rebuttal interpretation if achieved |
|---|---|---:|---:|---:|---|
| POPE adversarial | answer / first decisive yes-no token | Nonzero but modest | Correct flips exceed harmful flips | Low | Future is not the main POPE driver; it helps ambiguous hard negatives without collapsing recall. |
| CHAIR / captioning | object mention or sentence-level hallucination event | Higher than POPE | Correct flips clearly dominate | Low to moderate | Future mainly helps longer open-ended outputs where early admissions compound. |
| MMBench retention | answer-level correctness | Very low | Neutral or mildly positive | Very low | Future does not materially damage general multimodal ability. |

Decision rule:

- If POPE flip rate is low but CHAIR benefit is clear, the rebuttal should narrow the claim: Past+Current is the efficient POPE regime; Full CHORD is quality-oriented for open-ended generation.
- If harmful flips are high, do not foreground Full CHORD as universally better.

## Target Table 2: Detector Attribution

Purpose: answer KrEs and M8du: are gains from CHORD's decoding rule or just Grounding DINO?

Ideal pattern: real anchors should help, but CHORD should retain value relative to no/random-anchor controls; same-anchor non-CHORD control should not match Full CHORD.

| Condition | What it isolates | Target outcome | Desired rebuttal interpretation if achieved |
|---|---|---|---|
| Full CHORD + real anchors | Complete method | Best or near-best quality | Query-conditioned anchors help the admission verifier. |
| Full CHORD + no/uniform anchors | Remove detector localization | Drops but remains above rollback-only where possible | CHORD is not solely detector lookup; rollout/admission control still contributes. |
| Full CHORD + random anchors | Controls for extra visual weighting | Worse than real anchors | Gains depend on semantically meaningful anchors, not arbitrary added weights. |
| Past+Future without current anchor term | Future without detector-backed current grounding | Below Full but informative | Current grounding and future rollout are complementary. |
| Same-anchor non-CHORD control | Detector available without CHORD admission rule | Does not match Full CHORD | The detector alone is insufficient; the admission rule matters. |
| Missed/noisy-anchor cases | Robustness boundary | Mixed but explainable | The method is detector-assisted and fails gracefully or is bounded by proposer quality. |

Decision rule:

- If same-anchor non-CHORD matches Full CHORD, the novelty claim must be weakened.
- If no/random anchors perform similarly to real anchors, the current grounding mechanism needs re-interpretation.

## Target Table 3: End-to-End Cost Accounting

Purpose: answer efficiency criticism honestly and prevent accusations of hidden Grounding DINO cost.

Ideal pattern: Full CHORD is slower but transparent; Past+Current is the recommended latency-oriented operating point.

| Method / regime | Detector proposal time | Decode ITL | Total answer latency | Peak VRAM | Desired rebuttal interpretation |
|---|---:|---:|---:|---:|---|
| Greedy | 0 | measured | measured | measured | Baseline cost floor. |
| OPERA | 0 | measured | measured | measured | Closest rollback-only comparison. |
| CHORD Past+Current | one-time cached proposer | measured | measured | measured | Practical latency-oriented CHORD regime. |
| Full CHORD | one-time cached proposer + rollout cost | measured | measured | measured | Quality-oriented regime; overhead is real and now fully accounted for. |

Decision rule:

- If Full CHORD total latency is too high, lead with Past+Current as practical default.
- If proposal time is small relative to rollout, emphasize cached one-time proposal but still report it.
- If VRAM overhead is large, state it plainly and avoid deployment-cheap language.

## Secondary Evidence Needed

| Evidence | Reviewer covered | Minimum acceptable output |
|---|---|---|
| k/m sweep | jjVG, M8du | small grid with quality and ITL; if too expensive, run POPE adversarial subset and label scope. |
| ONLY/VHD/HALC related-work patch | jjVG, KrEs, M8du | citation/axis table; direct numbers only if verifiable. |
| Figure 2 cleanup plan | jjVG | one-sentence camera-ready commitment. |
| Attention-window boundary | yx8u | clarify operational signal, not causal explanation; add last-4 evidence if available. |

## Final Rebuttal Drafting Rules

- Start from actual evidence, not from prose.
- Avoid unsupported adjectives such as "negligible", "universal", or "fully training-free".
- Explicitly separate measured new evidence from camera-ready promises.
- If a result is weak, narrow the claim rather than hiding it.
- Prioritize M8du and KrEs because they are the decision-critical skeptics.
