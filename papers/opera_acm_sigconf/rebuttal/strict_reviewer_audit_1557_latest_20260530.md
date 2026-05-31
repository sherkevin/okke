# Strict Reviewer Audit Of 15:57 Latest Timestamped Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260530_1541.pdf`

Current PDF metadata checked: 5 pages, created/modified 2026-05-30 15:42:37, 340525 bytes, SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

Scope: strict scientific adequacy review against the five official-reviewer intent map. Reported table values are treated as valid rebuttal evidence for this review pass; raw experiment logs are not independently audited here.

## Verdict

As the composite reviewer, I would still assign **4/5: Weak Accept**.

The current timestamped master is scientifically adequate at rebuttal level. It answers the main hidden acceptance gates in the intent document: M8du's mechanism evidence, KrEs's attribution and cost isolation, jjVG's completeness checklist, yx8u's claim discipline, and ve3y's practical runtime framing.

It is not a 5/5 response. The response makes the paper acceptable, not unassailable. The remaining objections are now mostly about the paper's intrinsic ceiling: moderate novelty, detector-assisted dependence, Full-mode cost, and bounded generality beyond object-grounded hallucination.

## Reviewer-By-Reviewer Judgment

| Reviewer | Original stance | My likely post-rebuttal stance | Reason |
|---|---:|---:|---|
| jjVG | 3 Borderline | 4 Weak Accept | k/m, recent methods, cost, and Figure 2 are all explicitly addressed. For this reviewer, the visible incompleteness is largely gone. |
| KrEs | 2 Weak Reject | 3 Borderline or 4 Weak Accept | The response now gives real/uniform/random/same-anchor controls, Past+Future without Current, end-to-end latency, VRAM, batch behavior, and recent baseline comparisons. They can still object on novelty and detector dependence. |
| yx8u | 4 Weak Accept | 4 Weak Accept | The response is careful: detector-assisted, training-free only for the base MLLM, attention as operational scoring, and relation/composition outside the headline claim. |
| ve3y | 4 Weak Accept | 4 Weak Accept | The quality-latency framing is honest: P+C is practical, Full is quality/offline. This preserves trust. |
| M8du | 3 Borderline | 4 Weak Accept | Their explicit questions are answered by flip/correctness diagnostics, detector controls, latency decomposition, k/m sweep, and detector robustness strata. |

## What The Response Explains Well

1. **Future rollout mechanism.** It reports Full-vs-P+C flip rate, corrected and harmful flips, CHAIR continuation deltas, and paired statistics. This directly answers whether Future changes decisions and whether those changes help.

2. **Grounding DINO attribution.** The control ladder is strong enough for rebuttal: same-anchor non-CHORD, uniform anchors, random matched anchors, Past+Future without Current, real-anchor P+C, and Full real anchors. This makes it harder to claim the detector alone explains the gain.

3. **End-to-end cost.** Proposal time, decode ITL, total latency, tokens, VRAM, batch behavior, and OOM boundary are separated. This removes the most serious suspicion of hidden overhead.

4. **Hyperparameter robustness.** The k/m table shows a Pareto trend and makes the default a quality setting rather than an arbitrary universal choice.

5. **Recent baselines.** ONLY, VHD/VHR, and HALC are now included in a matched-protocol table. The response also commits to validation-only tuning and reproducibility records.

6. **Detector failure boundary.** Zero-anchor, relevant-anchor, noisy/diffuse-anchor strata, threshold sensitivity, fine-grained/crowded degradation, detector-side vs model-side split, and oracle-box repair turn detector robustness into a measured limitation.

7. **Scope discipline.** The claim is narrowed to object-grounded hallucination and open-ended object mentions. Relation/composition is correctly treated as weak-transfer boundary evidence, not a main contribution.

8. **CHAIR interpretation.** Caption length and object-mention diagnostics reduce the risk that CHAIR gains are just shorter or more conservative outputs.

## What Is Still Not Fully Solved

1. **Novelty is still moderate.** The rebuttal shows coordination matters, but the ingredients are still recognizable: rollback, detector anchors, attention scoring, and rollout reranking.

2. **Detector quality remains central.** The best gains concentrate in relevant-anchor cases; zero-anchor and noisy/diffuse cases remain weaker.

3. **No second real detector is tested.** The anchor controls are useful, but they do not fully answer the "replace the proposer" version of KrEs's concern.

4. **Full CHORD remains expensive.** The rebuttal handles this honestly, but the method's best-quality setting is not suitable as a lightweight default.

5. **Non-object hallucination remains weak.** Attribute transfer is small and relation/composition transfer is weaker. This is acceptable only if the final paper keeps those claims out of the headline contribution.

6. **Baseline fairness depends on final reproducibility details.** I would want the camera-ready to include exact commands, commits/checkpoints, parser versions, seeds, validation splits, and tuned grids for every baseline.

## Questions I Would Still Ask

1. Which exact HALC, ONLY, and VHD/VHR implementations were used, and what public sanity checks verified them?

2. Can the authors release or append the exact prompt templates, answer parser, caption parser, seeds, validation splits, and hyperparameter grids?

3. Does the conclusion still hold if Grounding DINO is replaced by another real proposer with different recall/precision behavior?

4. Are InstructBLIP CHAIR improvements statistically tested with the same rigor as the LLaVA rows?

5. Can CHORD detect noisy-anchor regimes and automatically back off to P+C or Past?

6. Do the prompt-template robustness checks extend to CHAIR and caption-style generation, or only to POPE binary QA?

7. Will the final paper explicitly make P+C the practical default and Full the quality/offline setting in the abstract/method summary?

8. Are newer-backbone results intended as complete benchmark evidence or only sanity checks?

## Final Score

**4/5 Weak Accept.**

This rebuttal is good enough to support acceptance if the reported results are credible. It probably converts M8du and jjVG, preserves yx8u and ve3y, and weakens KrEs's reject from "missing evidence" to "moderate novelty and bounded scope." That is a strong rebuttal outcome, but not a guarantee of high score or meta-review acceptance.
