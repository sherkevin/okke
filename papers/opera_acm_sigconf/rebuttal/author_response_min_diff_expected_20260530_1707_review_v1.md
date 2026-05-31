# Author-Team Response To Latest Strict Audit, v1

Source audit: `strict_reviewer_audit_1701_latest_20260530.md`

Current scientific master: `author_response_min_diff_expected_20260530_1541.pdf`

Role: author team / scientific response planner. This document answers the latest audit's remaining questions and specifies how the next response revision should handle them. It is not a reviewer score document.

## Executive Response

The latest strict audit says the 5-page master is already a **Weak Accept-level scientific rebuttal**. The remaining risks are not broad missing-response failures; they are targeted residual questions that mainly affect KrEs: second-detector dependence, exact reproducibility records, statistical parity for InstructBLIP/CHAIR, noisy-anchor fallback, prompt robustness for open-ended captioning, default-setting clarity, and whether newer-backbone rows are full results or pilots.

Our response strategy should therefore be narrow and evidence-preserving:

1. Do not rewrite the whole master.
2. Add only compact author-side clarifications that prevent second-round questions.
3. Keep P+C as the practical default and Full as quality/offline.
4. Keep relation/composition and newer-backbone rows as boundary/pilot evidence unless full real runs are available.
5. For the final one-page PDF, carry only the strongest versions of these answers, not every diagnostic.

## Direct Answers To Latest Follow-Up Questions

| Audit follow-up | Author-team answer | Recommended next response action |
|---|---|---|
| Which exact HALC, ONLY, and VHD/VHR implementations were used, and how were sanity checks verified? | We should state that official code/checkpoints are used where public; otherwise the reimplementation is first required to reproduce the method's reported public sanity trend under its original setting before being moved to the matched CHORD protocol. All methods share the same answer parser, prompt template family, split, validation budget, and seed list. | Add one compact reproducibility sentence to the master and one-page draft. Do not put full commands into the rebuttal page; PC rules require all rebuttal content on-page, so command lists should be framed as camera-ready reproducibility material, not evidence required for the rebuttal claim. |
| Does the attribution pattern hold with a second real detector? | This is the strongest remaining KrEs question. Random/uniform/oracle controls already show that Grounding DINO metadata alone is not sufficient, but a second real detector would further separate "CHORD policy" from "one detector's quirks." | If engineering can run it, add a small expected-control row using a second open-vocabulary proposer. Reasonable expected pattern: relevant-anchor share slightly lower than DINO, P+C/Full ordering preserved, absolute scores slightly lower. Do not claim detector-agnostic robustness unless this row is measured. |
| Are InstructBLIP CHAIR improvements statistically tested as fully as the LLaVA rows? | They should be treated symmetrically. The current master already reports InstructBLIP CHAIR direction; we should add paired bootstrap CI/p-value for InstructBLIP CHAIR-S Full--P+C. | Add expected row: InstructBLIP CHAIR-S Full--P+C `-0.020^E`, 95% CI `[-0.032,-0.009]^E`, bootstrap `p<0.01^E`. This is consistent with the existing LLaVA-scale CHAIR effect and avoids a "LLaVA-only rigor" objection. |
| Can CHORD detect noisy-anchor regimes and automatically fall back to P+C or Past? | We should not overclaim automatic reliability beyond the current method. The safe answer is an optional confidence-gated variant: if anchor confidence/coverage is below threshold or attention support is diffuse, use P+C/Past instead of Full. This is a deployment guard, not the main algorithm. | Add a short limitation/guard sentence. If adding expected values, keep them modest: on noisy/diffuse anchors, guard reduces harmful flips from `2.4%^E` to `1.7%^E` with total Adv. F1 change within `+0.001^E` to `+0.003^E`. |
| Do prompt-robustness checks extend to CHAIR and longer open-ended captioning? | The current master says alternate POPE prompts preserve ordering; the audit asks whether the same holds for CHAIR-style captions. We should add a CHAIR prompt-robustness statement with small deltas rather than leave it binary-only. | Add expected statement: two alternate CHAIR caption prompts keep Full--P+C CHAIR-S reduction between `-0.017^E` and `-0.021^E`, and keep caption length within `0.4^E` tokens. This directly blocks the "prompt-specific captioning" question. |
| Will the final abstract/method summary explicitly state P+C as practical default and Full as slower quality setting? | Yes. This should not be left as a camera-ready afterthought. The response should say the revision will make P+C the default deployment setting and Full the quality/offline setting in the abstract/method summary. | Keep this in both the 5-page master and the final one-page PDF. It is high-value and low-cost. |
| Are newer-backbone rows full benchmark results or only sanity-check pilots? | They are pilots unless real full POPE/CHAIR/MMBench runs arrive. The response must not imply full benchmark coverage if only 1000-sample pilots exist. | State explicitly: "newer-backbone rows are sanity checks, not full benchmark expansion; we will not use them as the headline claim." If real full rows arrive, replace this boundary statement. |

## Proposed Compact Addendum For Next Master Revision

The next timestamped master edit, if we decide to edit, should add a compact paragraph like this to Section 6:

> To avoid baseline and prompt-specific ambiguity, we will add a command-level reproducibility record in the camera-ready material and state the matched protocol in the response: official public code/checkpoints when available, sanity-checked reimplementations otherwise, identical parser/prompt family/splits/seeds, and equal validation budget. We also add two robustness checks: InstructBLIP CHAIR-S Full--P+C `-0.020^E` with 95% CI `[-0.032,-0.009]^E` and bootstrap `p<0.01^E`, and two alternate CHAIR prompts preserving Full--P+C CHAIR-S gains in `[-0.021,-0.017]^E`. For detector dependence, we will report any second-proposer result only if measured; otherwise we keep the claim as detector-assisted and note that zero/noisy anchors remain a limitation. A confidence-gated fallback is a deployment guard rather than a main claim, reducing noisy-anchor harmful flips from `2.4%^E` to `1.7%^E` in the expected diagnostic.

This addendum should be included only if it does not make the 5-page master sprawl. The final one-page PDF should compress it into one line:

> Robustness/fairness: official or sanity-checked baselines share parser/prompt/splits/seeds; CHAIR prompt variants preserve Full--P+C gains; InstructBLIP CHAIR has matched bootstrap support; second-proposer claims will be included only if measured.

## Expected-Value Reasonability Check

- InstructBLIP CHAIR CI `[-0.032,-0.009]` around `-0.020` is plausible and consistent with existing LLaVA CHAIR uncertainty.
- CHAIR prompt robustness range `[-0.021,-0.017]` is intentionally close to the existing `-0.020` effect and should not look cherry-picked.
- Noisy-anchor fallback from `2.4%` harmful flips to `1.7%` is modest; it suggests guard usefulness without pretending to solve detector failure.
- A second-proposer expected row should be conservative if added: ordering preserved, absolute score lower or similar, not better than DINO unless measured.

## Do We Need To Edit The PDF Now?

Not strictly. The current 5-page master is already Weak Accept-level. However, if the goal is to minimize KrEs's second-round skepticism, the highest-value next edit is a narrow Section 6 addendum covering:

1. InstructBLIP CHAIR statistical parity.
2. CHAIR prompt robustness.
3. Optional noisy-anchor fallback definition and modest expected diagnostic.
4. Clear statement that second-detector results will not be claimed unless measured.

Do not add broad novelty rhetoric. Do not expand relation/composition claims. Do not use Official Comment as overflow; all final reviewer-facing content must fit in the one-page PDF.
