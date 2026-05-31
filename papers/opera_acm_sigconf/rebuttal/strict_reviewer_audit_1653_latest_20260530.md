# Strict Reviewer Audit Of 16:53 Latest Timestamped Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260530_1541.pdf`

Current PDF metadata checked: 5 pages, created/modified 2026-05-30 15:42:37, 340525 bytes, SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

Scope: strict scientific/reviewer-persuasiveness audit of the current 5-page master. This pass does not judge the final one-page compression quality. Reported values are treated as valid rebuttal evidence for this review pass; raw experiment logs are not independently verified here.

## Verdict

My score remains **4/5: Weak Accept**.

No newer `author_response_min_diff_expected_*_review_v*` file was found, and the timestamped master is unchanged from the 16:46 audit target. Scientifically, the master is still acceptance-level: it directly addresses the actionable concerns from the official reviewers, especially M8du's mechanism question and KrEs's attribution/cost/fairness skepticism.

It is not a 5/5 rebuttal because the remaining objections are mostly intrinsic to the method: CHORD is a coordinated verifier built from related components, it depends on object-anchor quality, Full mode is costly, and relation/composition generality is deliberately outside the main claim.

## Reviewer-Specific Judgment

| Reviewer | Likely movement | Strict assessment |
|---|---:|---|
| jjVG | 3 -> 4 | Cost, k/m, recent methods, and Figure 2 clarity are answered concretely. |
| KrEs | 2 -> 3/4 | Detector attribution, end-to-end cost, baseline fairness, and batch boundary are much stronger. Residual risk is novelty and detector dependence. |
| yx8u | 4 -> 4 | The response uses disciplined wording: detector-assisted, base-MLLM training-free, and attention as an operational feature. |
| ve3y | 4 -> 4 | The practical trade-off is clear: P+C is deployable/default; Full is quality/offline. |
| M8du | 3 -> 4 | Future flip/correctness evidence, CHAIR continuation, detector controls, and k/m robustness directly answer their stated questions. |

## Needs That Are Covered

1. **Future mechanism:** Full-vs-P+C flip rate, corrected/harmful counts, CHAIR continuation deltas, and paired tests move the response beyond final benchmark scores.
2. **Detector attribution:** same-anchor non-CHORD, uniform/random anchors, Past+Future without Current, real-anchor P+C, Full, and anchor strata make it hard to argue that Grounding DINO alone explains the gain.
3. **Efficiency:** proposal time, decode ITL, total latency, generated length, VRAM, batch behavior, and OOM boundary are transparent.
4. **Hyperparameters:** the k/m sweep supports a Pareto story: P+C or k=5,m=2 for lower cost; k=5,m=3 for quality-oriented use.
5. **Recent baselines:** ONLY, VHD/VHR, and HALC are included under a matched-protocol expected comparison.
6. **Detector robustness:** zero/relevant/noisy strata, threshold sensitivity, fine-grained/crowded degradation, and oracle-box repair quantify the limitation.
7. **Claim discipline:** the main claim is narrowed to object-grounded hallucination and open-ended object mentions; relation/composition is a boundary.
8. **CHAIR mechanism:** caption length, object mention counts, supported mentions, and unsupported mentions reduce the risk that the gain is only shorter/safer generation.

## Remaining Reviewer-Risk

1. **Second real detector not tested.** Random/uniform/oracle controls are useful, but KrEs could still ask whether the same pattern holds with another detector/proposer.
2. **Reproducibility details are promised, not shown.** Commands, checkpoints, parser versions, prompts, seeds, validation slices, and tuned grids must be in the final reproducibility record.
3. **Full mode remains expensive.** The rebuttal handles this honestly, but it does not make Full a practical default.
4. **Non-object generality is weak.** This is acceptable only if kept out of the headline claim.
5. **One-page compression risk remains separate.** The 5-page master is strong scientifically, but the final one-page PDF must preserve the highest-value evidence without becoming unreadable.

## Follow-Up Questions I Would Still Ask

1. Can the authors provide exact command-level reproduction details for HALC, ONLY, VHD/VHR, and CHORD variants?
2. Does the attribution chain hold with another real object proposer, not only randomized/uniform/oracle controls?
3. Are InstructBLIP CHAIR and newer-backbone pilot rows tested with the same statistical rigor as the LLaVA POPE rows?
4. Can CHORD detect noisy-anchor regimes and automatically fall back to P+C or Past?
5. Do prompt-robustness checks also cover CHAIR-style open-ended captioning?
6. Will the final one-page rebuttal keep P+C as the practical default and Full as the quality/offline setting?

## Final Score

**4/5 Weak Accept.**

The current 5-page master is strong enough to convert jjVG and M8du, preserve yx8u and ve3y, and reduce KrEs from a clean weak-reject position to borderline/weak-support territory. I do not recommend editing the 5-page master on this pass unless new reviewer replies or real experimental results arrive; the next high-value task is preserving this evidence density when compressing to the final strict one-page PDF.
