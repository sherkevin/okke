# Strict Reviewer Audit, 2026-05-30 17:39, review_v2

## Primary Input And Evidence Boundary

Primary input audited: `author_response_min_diff_expected_20260530_1733_review_v2.md`.

Reviewer-demand contract: `reviewer_true_intent_analysis_20260529.md`.

Current scientific master context: `author_response_min_diff_expected_20260530_1541.pdf` and `author_response_min_diff_expected_20260530_1541.tex`.

Previous audit context: `strict_reviewer_audit_1717_latest_20260530.md`.

Evidence commands used:

```powershell
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*.pdf'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'strict_reviewer_audit_*_latest_*.md'
pdfinfo papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1541.pdf
pdftotext -layout papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1541.pdf -
rg -n "Future|Full|P\+C|Grounding|DINO|same-anchor|HALC|ONLY|VHD|VHR|latency|VRAM|CHAIR|CI|McNemar|bootstrap|expected" papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1541.tex
rg -n "Expected-Table Reasonability Check|4\.6%|4\.9%|Same-anchor non-CHORD|one strict page|Full--P\+C|Recent Baseline|Statistical Reliability|second-proposer" papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1733_review_v2.md
```

Boundary: `review_v2` is an author-response planning artifact, not the current PDF/TEX. The newest PDF/TEX remains the 15:41 master. The PDF/TEX still contains at least two known issues that `review_v2` identifies but has not yet fixed in the official artifact: the Full-vs-P+C flip-rate/count mismatch (`4.1%` vs `96+42`, `4.3%` vs `102+44`) and the same-anchor/random-control note contradiction. Therefore I score the Markdown response and the current official artifact separately.

## Overall Reviewer Verdict

Current author-response Markdown quality: **4/5 Weak Accept**, conditional.

Current official PDF/TEX state: **3/5 Borderline to weak 4**, not yet secure.

As a strict reviewer, I would not treat `review_v2` as final evidence. It is a much better response plan than `review_v1`: it directly catches numeric contradictions, gives a coherent detector-attribution story, states P+C as the practical default, and narrows claims to detector-assisted object-grounded admission. However, much of the persuasive content is still framed as expected targets rather than measured results, and the latest PDF/TEX still predates the fixes. If the final one-page rebuttal contains real or explicitly bounded values from `review_v2`, the likely score is Weak Accept. If the one-page PDF remains the 15:41 content or compresses expected tables without evidence boundaries, the score stays Borderline.

## Reviewer-by-Reviewer Score Movement

| Reviewer | Original real concern | What `review_v2` satisfies | Still unresolved | Likely score after reading `review_v2` | Most useful action |
|---|---|---|---|---:|---|
| jjVG | Cost, k/m, recent related work, figure clarity. | k/m table is now coherent; P+C vs Full cost story is clear; ONLY/VHD/VHR/HALC are explicitly included; Figure 2 is deemphasized in favor of evidence. | They still need the final one-page to visibly include k/m or at least a Pareto sentence plus recent-baseline fairness. | 4 if integrated; 3 if PDF remains unchanged. | Put one compact k/m/P+C default sentence and recent-baseline fairness sentence into the one-page PDF. |
| KrEs | Novelty, Grounding DINO attribution, end-to-end cost, recent baselines/backbones. | Detector controls are much better specified; same-anchor non-CHORD is now favorable to the detector; cost arithmetic is explicit; claim is detector-assisted. | Evidence is still expected; no second real proposer; no exact baseline commits/checkpoints; official PDF still has a control-table contradiction. | 3, possibly 4 only with real measured controls and fixed PDF. | Replace expected detector/baseline rows with measured values or explicitly mark them as revision commitments; fix the same-anchor table in TEX. |
| yx8u | Incremental novelty, detector dependence, attention reliability, scope and overclaiming. | `review_v2` handles this well: base-MLLM training-free but detector-assisted; attention is operational, not causal; relation/composition is boundary evidence. | Final title/abstract/main claim still must be checked; long response is safer than one-page compression. | 4. | Preserve the exact claim-boundary sentence in the final one-page PDF. |
| ve3y | Practical value under nontrivial overhead. | P+C is now the practical/default regime and Full is quality/offline; total latency arithmetic is clear. | Proposal time, VRAM, and batch rows are still expected unless measured. | 4 if cost table is included honestly. | Keep the P+C default and end-to-end cost table in the final rebuttal; do not sell Full as deployment-cheap. |
| M8du | Whether Future actually changes decisions correctly; detector attribution; hyperparameters; robustness. | `review_v2` directly answers with flip counts, corrected/harmful counts, CHAIR continuation rows, detector controls, and k/m sweep. | The Future evidence remains expected; current PDF has inconsistent flip rates; no measured harmful-flip audit is shown. | 4 if real flip/correctness numbers exist; otherwise 3/4. | Fix the flip-rate/count mismatch and state whether the numbers are measured, expected, or pending. |

## Coverage Matrix Against True Intent

| Gate | Status | Evidence | Reviewer risk |
|---|---|---|---|
| Mechanism/Future effectiveness | **Partially resolved** | `review_v2` gives corrected `4.6%` and `4.9%` flip-rate targets from `96/42` and `102/44`, plus CHAIR continuation targets. | Still expected. Current PDF/TEX still says `4.1%` and `4.3%`, which is arithmetically inconsistent with the counts. M8du can still object if these are not real measurements. |
| Grounding DINO and detector attribution | **Partially resolved** | Same-anchor non-CHORD, uniform/random anchors, Past+Future without Current, detector strata, threshold sensitivity, and no detector-agnostic claim are all present in `review_v2`. | No second real proposer. The 15:41 TEX says same-anchor/random improve over rollback-only but table values put some below Past; `review_v2` fixes the target but not the PDF. KrEs can still call this planned rather than evidenced. |
| Efficiency/cost and P+C/Full default | **Mostly resolved in response, partially in artifact** | End-to-end formula is clear: proposal ms + ITL * tokens. P+C is practical; Full is quality/offline. Batch/VRAM limits are named. | Proposal time, VRAM, and batch rows are expected. If final page cannot fit these, ve3y/KrEs may still suspect hidden cost. |
| Recent baselines and fairness | **Partially resolved** | ONLY, VHD/VHR, HALC are included with a matched-protocol table and fairness wording. | Expected numbers only; exact official code/checkpoints/commit IDs and parser details are not in the one-page plan. If the final rebuttal only says "we compare" without implementation facts, KrEs/jjVG remain unsatisfied. |
| Claim scope, generality, novelty, attention wording | **Mostly resolved** | Strong claim boundary: detector-assisted, base-MLLM training-free, object-grounded token admission; attention is operational, not causal; relation/composition is a limitation. | The final one-page must preserve this discipline. Novelty remains moderate even with good wording, so this cannot become a high-score argument by itself. |

## Unresolved Problems

1. **The strongest new evidence is still expected, not measured.** Reviewers asked for empirical isolation, not a target table. Expected rows are useful internally but cannot carry the official rebuttal unless clearly labeled or replaced. This affects Future flips, recent baselines, detector strata, VRAM/batch cost, InstructBLIP CHAIR CI, and prompt robustness.

2. **The current PDF/TEX still contains numeric contradictions.** The 15:41 master has `96/42` and `102/44` corrected/harmful counts but `4.1%` and `4.3%` flip rates. The correct rates for total flips over 3000 are `4.6%` and `4.9%`. A reviewer noticing this would question the care behind the expected tables.

3. **Detector attribution is improved but not fully closed.** Same-anchor non-CHORD is the right control, but without real measured results and without a second real proposer, KrEs can still say the method is detector-assisted and detector-specific. That is acceptable only if the claim is narrowed.

4. **Recent-baseline fairness is still abstract.** The response says official/sanity-checked implementations, same parser, prompts, split, seeds, and tuning budget. It does not list concrete repositories, checkpoints, commits, or sanity-check criteria. In a one-page rebuttal this may be impossible to fully show, but at minimum one dense fairness sentence must survive.

5. **One-page compression may remove the evidence that makes `review_v2` credible.** `review_v2` is convincing because it is long and table-rich. The final page cannot carry every row. The team must choose three compact evidence blocks: Future mechanism, detector attribution, and cost/recent-baseline fairness.

6. **The score ceiling is still moderate novelty.** Even a well-executed rebuttal probably moves Borderline reviewers to Weak Accept and weakens KrEs's rejection, but it does not turn CHORD into a fundamentally new paradigm. The response should not try to oversell novelty.

## Follow-up Questions For The Author Team

1. Which values in `review_v2` are already measured today, and which are still expected targets? Produce a table with columns `measured`, `expected`, `pending`, and `remove from final if unavailable`.

2. Will the next TEX/PDF revision actually fix the flip-rate/count mismatch to `4.6%` and `4.9%`, or will the counts be changed to match `4.1%` and `4.3%`? Pick one and make the table arithmetically consistent.

3. Will the detector-attribution table in TEX be updated so same-anchor non-CHORD and random-anchor controls no longer contradict the note saying they improve over rollback-only?

4. Are the recent-baseline rows for ONLY, VHD/VHR, and HALC measured under a matched protocol, or are they expected placeholders? If measured, what official code/checkpoint or sanity-check source was used?

5. Can the one-page rebuttal include enough implementation fairness detail to be credible without Official Comment overflow? At minimum, can it state official/sanity-checked implementations, same parser, prompt family, split, seeds, and validation budget?

6. Are proposal time, VRAM, and batch-size behavior measured on the same hardware as decode ITL, or estimated? If estimated, do not present them as final measured cost.

7. Is the InstructBLIP CHAIR bootstrap row measured? If not, should it be removed from the final one-page or explicitly treated as an expected diagnostic for the scientific master only?

8. What is the exact final one-page priority order? I recommend: Future flip/correctness, detector attribution controls, end-to-end cost plus recent baselines, then claim boundary.

## Expected Table And Numeric Plausibility Check

`review_v2` does the right thing by treating expected tables as forward-looking targets, not casual placeholders. It also catches the two most important consistency bugs from the 15:41 master. That said, the current numeric package is still not final-reviewer safe.

Passes:

- Latency arithmetic is internally consistent: `118 + 27.24 * 20 = 663` for P+C and `118 + 37.31 * 20 = 864` for Full.
- P+C and Full values are consistent across the response: P+C `0.832` / `0.175` / `663`, Full `0.845` / `0.155` / `864`.
- Effect sizes are conservative enough to be believable: Full is better than P+C but not dramatically, while P+C is close to strong recent baselines.
- Detector strata sum correctly to 3000 and threshold rows sum to 100.0%.
- The same-anchor target in `review_v2` is more credible than the 15:41 master because it gives the detector-only control a small benefit while keeping it below real-anchor P+C.

Remaining numeric risks:

- The official TEX/PDF has not yet been corrected from `4.1%/4.3%` to `4.6%/4.9%`.
- The recent-baseline table could look invented if `ONLY`, `VHD/VHR`, and `HALC` rows are not clearly marked as measured, expected, or pending.
- Confidence intervals and p-values are plausible but unsupported unless the paired samples and bootstrap settings exist.
- VRAM and batch-size rows are plausible, but reviewers will treat them as deployment evidence; estimated values need explicit boundaries.

Required table-level fix before final PDF:

| Table | Required change |
|---|---|
| Future mechanism | Make flip rate and corrected/harmful counts mathematically consistent. |
| Detector attribution | Align same-anchor/random rows with the note, or rewrite the note. |
| Statistical reliability | Add InstructBLIP CHAIR row only if measured or clearly expected. |
| Recent baselines | Mark measured vs expected and include fairness constraints in the same page. |
| Cost | Keep proposal time separate from decode ITL and state hardware/batch boundary. |

## One-page Rebuttal Compression Risk

The final one-page PDF should not try to carry all of `review_v2`. The risk is that the response becomes broad but thin. A strict reviewer needs to see the following, in this order:

1. Future mechanism: Full-vs-P+C flip rate, corrected/harmful counts, and CHAIR-S delta.
2. Detector attribution: same-anchor non-CHORD, random/uniform anchors, P+C real anchors, Full real anchors.
3. Cost and practical regime: P+C default, Full quality/offline, proposal time included.
4. Recent baselines: ONLY, VHD/VHR, HALC under matched protocol, with fairness constraints.
5. Claim boundary: detector-assisted, object-grounded, attention as operational signal, no broad relation/composition claim.

Do not spend one-page space on a full Figure 2 redesign promise. One sentence is enough. Do not rely on Official Comment, supplementary material, or hidden command logs.

## Next Required Action

The author team should not keep expanding the response indefinitely. The next action should be a narrow execution step:

1. Produce a `review_v3` or direct master-edit plan that labels every `review_v2` number as measured, expected, pending, or remove-if-unavailable.
2. Copy the latest timestamped TEX/PDF to a new timestamp and fix the two confirmed table inconsistencies.
3. Replace expected values with real experiment outputs as they arrive.
4. Compress only the highest-evidence rows into the final strict one-page rebuttal.

If real measurements are not available, the final claim should be weaker: "we add planned diagnostics and will include them in camera-ready" is not enough to move KrEs; it only preserves honesty.

## LOCAL_TASKS Update

This audit should close the `Reviewer heartbeat audit of response v2 at 17:39` task in `LOCAL_TASKS.md`.

Output path: `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_1739_latest_20260530_review_v2.md`.

Primary input: `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1733_review_v2.md`.

Evidence boundary: this audit judges the latest author-response Markdown, the unchanged 15:41 PDF/TEX, the reviewer true-intent contract, and the previous 17:17 audit. It does not independently validate raw experiment logs and does not edit PDF/TEX.
