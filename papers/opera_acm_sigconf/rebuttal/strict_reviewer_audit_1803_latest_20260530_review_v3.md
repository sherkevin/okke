# Strict Reviewer Audit, 2026-05-30 18:03, review_v3

## Primary Input And Evidence Boundary

Primary input audited: `author_response_min_diff_expected_20260530_1744_review_v3.md`.

Reviewer-demand contract: `reviewer_true_intent_analysis_20260529.md`.

Current scientific master context: `author_response_min_diff_expected_20260530_1744.pdf` and `author_response_min_diff_expected_20260530_1744.tex`.

Previous reviewer audit context: `strict_reviewer_audit_1739_latest_20260530_review_v2.md`.

Evidence commands used:

```powershell
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*.pdf'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'strict_reviewer_audit_*_latest_*.md'
pdfinfo papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf
pdftotext -layout papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf -
rg -n "4\.6|4\.9|0\.823|0\.821|InstructBLIP CHAIR-S|measured|expected|pending|one-page|ONLY|VHD|HALC|P\+C|Full|second-proposer|detector-assisted|attention" papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744_review_v3.md papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.tex
rg -n "Shared True Needs|Mechanism validation|Detector/Grounding DINO attribution|End-to-end cost|Missing ONLY|Novelty is incremental|Reviewer KrEs|Reviewer M8du|Reviewer jjVG|Reviewer yx8u|Reviewer ve3y" papers\opera_acm_sigconf\rebuttal\reviewer_true_intent_analysis_20260529.md
```

Boundary: `review_v3` is a stronger author response than `review_v2`, and the 17:44 scientific master now incorporates the arithmetic and detector-control fixes that were missing from the 15:41 master. However, the 17:44 PDF is still a 5-page scientific master, not the final strict one-page ACM MM rebuttal. Most decisive numbers are still expected-result targets unless raw experiment logs replace them. This audit does not validate raw logs and does not edit PDF/TEX.

## Overall Reviewer Verdict

Current author-response Markdown quality: **4/5 Weak Accept, stronger than v2 but still conditional**.

Current 5-page scientific master quality: **4/5 as an internal scientific rebuttal master**.

Current official one-page readiness: **3/5 Borderline** until the response is compressed around measured or explicitly bounded evidence.

The main improvement since the last audit is real: `review_v3` answers the measured/expected/pending boundary, and the 17:44 master fixes the two concrete table inconsistencies that would have damaged reviewer trust. The reviewer-facing scientific story is now coherent: Future is sparse and beneficial if measured, detector controls are ordered correctly, P+C is the practical regime, and the claim is detector-assisted/object-grounded. The remaining problem is no longer structure; it is evidentiary eligibility. A reviewer will not raise their score because a target table is internally plausible. They will raise it only if the final one-page PDF contains measured diagnostics or unmistakably bounded revision commitments.

## Reviewer-by-Reviewer Score Movement

| Reviewer | Original real concern | What is now satisfied | What remains unresolved | Likely score after reading current state | Most useful action |
|---|---|---|---|---:|---|
| jjVG | Visible completeness gaps: cost, k/m, related work, figure clarity. | The 17:44 master contains k/m sensitivity, recent baseline positioning and numeric slots, end-to-end cost accounting, and Figure 2 cleanup intent. | Final one-page must still visibly mention k/m robustness and missing related baselines; otherwise the visible checklist concern returns. | 4 if one-page preserves a compact k/m/baseline/cost block. | Put one Pareto sentence for k/m and one fairness sentence for ONLY/VHD/VHR/HALC in the final page. |
| KrEs | Attribution prosecutor: novelty, Grounding DINO, full cost, strong baselines/backbones. | Same-anchor, uniform/random anchors, Past+Future without Current, detector strata, and corrected detector-control ordering are now coherent. Cost includes proposal/decode/VRAM/batch structure. | This remains target evidence unless measured. No second real proposer. Recent-baseline implementation details are still too abstract for a skeptical reproducibility read. | 3 to weak 4, depending on whether measured rows replace targets. | Provide measured detector-control and recent-baseline rows, or explicitly narrow to detector-assisted evidence without claiming detector-agnostic robustness. |
| yx8u | Supportive but claim-sensitive: detector dependence, attention reliability, scope. | Very well handled. The response repeatedly frames CHORD as detector-assisted, base-MLLM training-free, object-grounded, and uses attention only as an operational feature. | Compression risk: the final one-page must not drop the claim-boundary paragraph. | 4. | Preserve the exact claim-boundary sentence in the final page. |
| ve3y | Practical value despite overhead. | P+C is clearly the practical/default setting and Full is a quality/offline setting. Full is no longer presented as deployment-cheap. | Cost rows include expected proposal/VRAM/batch values; if not measured, one-page should avoid presenting them as final deployment evidence. | 4. | Include measured or explicitly bounded end-to-end cost, not decode-only timing. |
| M8du | Mechanism auditor: Future flip correctness, detector attribution, hyperparameters, robustness. | The mechanism table now has consistent flip rates/counts and a clear corrected-vs-harmful story; k/m table addresses hyperparameter concern; detector controls are coherent. | The mechanism evidence is still only persuasive if measured. Without raw flip/correctness counts, M8du can still say the mechanism remains planned rather than validated. | 4 if measured; 3/4 if expected-only. | Use a measured Full-vs-P+C flip/correctness row as the first evidence item in the final page. |

## Coverage Matrix Against True Intent

| Gate | Status | Evidence | Reviewer risk |
|---|---|---|---|
| Mechanism/Future effectiveness | **Partially resolved** | `review_v3` and the 17:44 master now use consistent `4.6%` and `4.9%` flip-rate targets, corrected/harmful counts, CHAIR continuation deltas, and statistical reliability rows. | Still expected-result target unless raw runs are available. M8du asked for actual flip frequency and correctness; plausible numbers do not fully answer that. |
| Grounding DINO and detector attribution | **Partially resolved** | Same-anchor non-CHORD is now `0.823`, random anchors `0.821`, both below real-anchor P+C and Full. The attribution story no longer contradicts itself. | No second real proposer and no raw measurement proof. KrEs can still treat this as a detector-assisted heuristic unless controls are measured. |
| Efficiency/cost and P+C/Full default | **Mostly resolved** | End-to-end cost decomposition is visible; P+C default and Full quality/offline are clear; batch/VRAM limits are acknowledged. | Proposal time, VRAM, and batch behavior need hardware-backed status. Estimated rows should not be used as final deployment evidence. |
| Recent baselines and fairness | **Partially resolved** | ONLY, VHD/VHR, HALC are named and given a matched protocol with conservative expected ordering. | The table still lacks concrete repo/checkpoint/commit/sanity-check evidence. If rows are not measured, final one-page should use them as fairness commitments, not numerical wins. |
| Claim scope, generality, novelty, attention wording | **Mostly resolved** | Claim boundary is strong and reviewer-aligned: detector-assisted, object-grounded, operational attention, no broad relation/composition claim. | Novelty remains moderate. This gate prevents downgrade more than it creates a high-score upside. |

## Unresolved Problems

1. **Evidence eligibility is now the central risk.** `review_v3` correctly distinguishes expected targets from measured context, but the final reviewer response cannot rely on internal target tables. The next author-side artifact must decide which numbers are measured enough for the final page and which must be omitted or phrased as planned camera-ready additions.

2. **The 17:44 PDF is not the final rebuttal artifact.** It is five pages and therefore cannot be the official ACM MM rebuttal. It is scientifically organized, but the final one-page version may lose the very details that make it convincing.

3. **Recent-baseline comparisons remain vulnerable.** The rows are conservative, but without exact implementation provenance they can still read as invented placeholders. KrEs and jjVG will not be fully satisfied by names and expected numbers alone.

4. **Detector attribution still lacks a second-proposer result.** The response wisely avoids detector-agnostic claims. That is enough to protect yx8u/ve3y, but KrEs can still retain a residual objection that the method is tied to Grounding DINO.

5. **Statistical rows are too precise unless backed by actual paired samples.** Confidence intervals and p-values are powerful, but if they are not measured they become a liability in the final one-page rebuttal.

6. **The likely score ceiling is still Weak Accept.** The response can plausibly convert M8du and jjVG and soften KrEs, but moderate novelty and detector dependence still cap the upside.

## Follow-up Questions For The Author Team

1. Which exact numeric rows in the 17:44 master are now backed by real experiment logs? Answer row-by-row for Future flips, detector controls, recent baselines, proposal time, VRAM/batch, CHAIR prompt robustness, and statistical reliability.

2. What is the final one-page evidence set? Choose at most three blocks: Future mechanism, detector attribution, and cost/recent-baseline fairness. Do not attempt to include every 5-page table.

3. For ONLY, VHD/VHR, and HALC, what implementation source or sanity-check criterion will be stated on-page? If none can be stated, remove numeric comparisons from the final page and keep only the fairness protocol.

4. Are the p-values and confidence intervals computed from real paired samples? If not, remove them from the final one-page rebuttal.

5. Is proposal time measured on the same hardware and preprocessing path as decode ITL? If not, phrase cost as "we will report proposal+decode accounting" rather than a final deployment number.

6. Will the final page explicitly say "detector-assisted" and "we do not claim detector independence"? This is necessary to keep yx8u from reading the response as overclaiming.

7. If no second real proposer can be run, will the authors explicitly present same-anchor/random/uniform controls as attribution checks rather than detector-robustness proof?

8. What exact one-sentence answer will M8du see first? It should be about Full-vs-P+C flip rate and corrected/harmful flips, not about aggregate POPE F1.

## Expected Table And Numeric Plausibility Check

The expected tables are now internally stronger than before.

Passes:

- Future flip-rate/count arithmetic is fixed in the 17:44 TEX/PDF: `96+42=138`, `138/3000=4.6%`; `102+44=146`, `146/3000=4.9%`.
- Detector attribution ordering is now coherent: rollback-only < same-anchor/random/uniform controls < real-anchor P+C < Full, with small and conservative gaps.
- End-to-end latency arithmetic remains consistent: P+C `118 + 27.24*20 = 663`, Full `118 + 37.31*20 = 864`.
- P+C and Full are consistently positioned across mechanism, cost, k/m, baselines, and claim boundary.
- Effect sizes are conservative enough to be credible: P+C is near strong recent baselines, and Full is better but slower.

Remaining plausibility risks:

- Several expected values are too specific for a final rebuttal unless real logs exist, especially p-values, confidence intervals, VRAM, batch behavior, and recent-baseline latency.
- The recent-baseline table is plausible but reviewer-sensitive. A single unsupported numeric row can damage trust more than a shorter fairness statement would.
- The newer-backbone and relation/attribute pilot rows should remain limitations or camera-ready plans unless measured; they should not compete for final one-page space.

Required final-page numeric policy:

| Evidence type | Safe for final one-page only if | Otherwise |
|---|---|---|
| Future flip/correctness | Measured with the same parser and sample set. | State it as planned diagnostic, not a result. |
| Detector controls | Measured under same backbone/split/prompt/parser. | Describe control design and claim boundary only. |
| Recent baselines | Implementation provenance and matched protocol are defensible. | Keep only fairness commitment; no numeric win claim. |
| Cost/VRAM/batch | Hardware and measurement path are known. | Report decode ITL plus promise full accounting. |
| Statistical tests | Paired samples and bootstrap/McNemar settings exist. | Remove p-values and confidence intervals. |

## One-page Rebuttal Compression Risk

This is now the largest practical risk. The 17:44 scientific master is useful because it is five pages; the official rebuttal is one page. The final page must be evidence-selective, not comprehensive.

Recommended one-page order:

1. One opening sentence: aggregate scores alone were insufficient, so the rebuttal adds targeted diagnostics and narrows claims.
2. A compact mechanism row: Full-vs-P+C flip rate, corrected/harmful counts, and CHAIR-S delta.
3. A compact attribution row: same-anchor, random/uniform, P+C real anchors, Full real anchors.
4. A compact cost/baseline row: P+C default, Full quality/offline, proposal included, recent baselines under matched protocol.
5. One boundary sentence: base-MLLM training-free but detector-assisted; attention is operational; no detector independence or broad relation/composition claim.

Do not spend final-page space on lengthy figure-redesign prose. Do not depend on Official Comment, hidden logs, or supplement-like overflow.

## Next Required Action

The author team should stop adding new broad diagnostics and produce one of two artifacts:

1. If enough real results exist: a final one-page rebuttal draft using only measured rows plus the claim-boundary sentence.
2. If real results are still pending: a `review_v4` that is not longer, but stricter. It should mark each candidate final-page row as `include measured`, `include wording only`, or `drop`.

The next response should not merely repeat the full expected tables. It should make the final one-page eligibility decision.

## LOCAL_TASKS Update

This audit should close the `Reviewer heartbeat audit of response v3 at 18:03` task in `LOCAL_TASKS.md`.

Output path: `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_1803_latest_20260530_review_v3.md`.

Primary input: `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744_review_v3.md`.

Context PDF/TEX: `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf` and `.tex`.

Evidence boundary: this audit verifies the current planning/summary documents and the 17:44 compiled scientific master. It does not independently validate raw experiment logs and does not edit PDF/TEX.
