# Strict Reviewer Audit Of 17:11 Latest Timestamped Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260530_1541.pdf`

Current PDF metadata checked: 5 pages, created/modified 2026-05-30 15:42:37, 340525 bytes, SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

Scope: manual execution of the current strict reviewer-audit automation. The judgment below evaluates the latest PDF itself against the five official reviewers' real concerns. Reported values are treated as valid rebuttal evidence for this pass; raw experiment logs are not independently audited here.

## Verdict

My score remains **4/5: Weak Accept**.

The latest PDF hash is unchanged from the recent strict-audit rounds. As a composite reviewer, I still view the rebuttal as scientifically adequate at acceptance level: it directly addresses mechanism evidence, detector attribution, end-to-end cost, k/m robustness, recent baselines, detector failure analysis, and claim calibration.

I would not raise it to 5/5. The rebuttal is strong, but it cannot fully remove the paper's intrinsic limits: moderate novelty, dependence on a frozen detector, expensive Full mode, and bounded generality beyond object-grounded hallucination.

## Reviewer-Specific Judgment

| Reviewer | Likely movement | Strict assessment |
|---|---:|---|
| jjVG | 3 -> 4 | Their concrete completeness concerns are addressed: k/m rationale, recent methods, end-to-end cost, and Figure 2 cleanup. |
| KrEs | 2 -> 3/4 | The response now gives meaningful detector attribution and cost accounting. Their remaining objections would be novelty, detector dependence, and limited generality. |
| yx8u | 4 -> 4 | The response preserves support by narrowing claims, stating detector assistance, and treating attention as an operational scoring feature. |
| ve3y | 4 -> 4 | The quality-latency trade-off is honestly framed: P+C is practical, Full is quality/offline. |
| M8du | 3 -> 4 | The response directly answers their mechanism concerns with Full-vs-P+C flip/correctness evidence, detector controls, latency decomposition, k/m sweep, and robustness strata. |

## What Is Satisfied

1. **Future mechanism.** The rebuttal gives admission-level evidence about when Future changes P+C decisions and whether those changes help.

2. **Detector attribution.** Same-anchor non-CHORD, uniform anchors, random matched anchors, Past+Future without Current, P+C real anchors, and Full real anchors form a credible attribution ladder.

3. **Cost transparency.** Proposal time, decode ITL, total latency, VRAM, batch behavior, and memory boundary are explicit enough to address hidden-overhead concerns.

4. **Hyperparameter choice.** The k/m sweep creates a defensible Pareto story rather than leaving k=5,m=3 as an arbitrary choice.

5. **Recent baselines.** ONLY, VHD/VHR, and HALC are included in the rebuttal comparison story, reducing the evaluation-completeness objection.

6. **Detector robustness.** Zero-anchor, relevant-anchor, noisy/diffuse-anchor, threshold sensitivity, fine-grained/crowded degradation, and oracle-box repair make detector dependence measurable.

7. **Claim discipline.** The method is framed as detector-assisted and object-grounded, with relation/composition treated as boundary evidence.

8. **CHAIR explanation.** Caption length plus supported/unsupported object-mention diagnostics make the CHAIR gain more credible than a simple conservative-generation effect.

## What Is Still Not Solved

1. **Novelty remains moderate.** CHORD is still an organized combination of rollback, object proposals, attention-derived scoring, and rollout reranking.

2. **Detector quality remains central.** The largest gains occur when useful anchors are present.

3. **No second real proposer is evaluated.** Random/uniform/oracle controls help, but a second real detector would be the cleaner answer to KrEs.

4. **Full mode remains costly.** The response is honest, but the best-quality setting is not deployment-light.

5. **Generality is bounded.** Attribute, relation, and composition results should not become headline claims.

6. **Reproducibility remains a camera-ready burden.** Baseline code versions, commands, checkpoints, prompts, parsers, seeds, validation splits, and tuned grids must be recorded completely.

## Follow-Up Questions I Would Still Ask

1. Which exact HALC, ONLY, and VHD/VHR implementations and checkpoints were used?

2. What sanity checks show those baseline implementations are faithful?

3. Does the attribution pattern hold with another real detector?

4. Are InstructBLIP CHAIR improvements statistically tested as fully as LLaVA?

5. Can CHORD identify noisy-anchor cases and fall back to P+C or Past?

6. Do prompt-robustness checks extend to CHAIR and longer open-ended captioning?

7. Will the final abstract and method summary explicitly state P+C as practical default and Full as slower quality setting?

## Final Score

**4/5 Weak Accept.**

This rebuttal is strong enough to satisfy the actionable reviewer concerns. The likely reviewer coalition remains favorable: jjVG and M8du move upward, yx8u and ve3y stay supportive, and KrEs remains the main risk because their residual objections are intrinsic rather than easy rebuttal omissions.
