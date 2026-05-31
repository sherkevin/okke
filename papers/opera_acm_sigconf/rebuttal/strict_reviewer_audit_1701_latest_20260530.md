# Strict Reviewer Audit Of 17:01 Latest Timestamped Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260530_1541.pdf`

Current PDF metadata checked: 5 pages, created/modified 2026-05-30 15:42:37, 340525 bytes, SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

Scope: strict reviewer judgment against the five official reviewers' real concerns. Reported values are treated as valid rebuttal evidence for this pass; raw experiment logs are not independently audited here.

## Verdict

My score remains **4/5: Weak Accept**.

The latest timestamped PDF is unchanged by hash from the prior strict-audit rounds. As a reviewer, I would still consider this rebuttal acceptance-level because it directly addresses the main evidence gaps identified in the intent map: Future mechanism, detector attribution, end-to-end cost, k/m robustness, recent baselines, detector failure analysis, and claim calibration.

This is not a 5/5 response. It is a strong credibility repair for a paper with real intrinsic constraints: moderate novelty, detector-assisted dependence, nontrivial Full-mode cost, and bounded transfer outside object-grounded hallucination.

## Reviewer-Specific Judgment

| Reviewer | Likely movement | Strict assessment |
|---|---:|---|
| jjVG | 3 -> 4 | Their concrete checklist is answered: k/m rationale, recent related methods, transparent cost, and cleaner Figure 2. |
| KrEs | 2 -> 3/4 | The response substantially weakens their attribution and hidden-cost objections. They may still resist on novelty, detector dependence, and limited non-object generality. |
| yx8u | 4 -> 4 | The response preserves support by narrowing claims, describing the method as detector-assisted, and treating attention as an operational feature. |
| ve3y | 4 -> 4 | The practical story is credible because P+C is presented as the practical setting and Full as a slower quality setting. |
| M8du | 3 -> 4 | Their explicit concerns are directly addressed by Full-vs-P+C flip/correctness diagnostics, detector controls, latency decomposition, k/m sweep, and robustness strata. |

## Needs Satisfied

1. **Mechanism validation.** The rebuttal answers whether Future actually changes admitted decisions and whether those changes are beneficial, rather than relying only on aggregate benchmark improvements.

2. **Detector attribution.** Same-anchor non-CHORD, uniform anchors, random matched anchors, Past+Future without Current, P+C real anchors, and Full real anchors give a reasonable attribution ladder.

3. **End-to-end efficiency.** Proposal time, decode ITL, total latency, VRAM, batch behavior, and memory boundary make the cost story transparent.

4. **Hyperparameter robustness.** The k/m sweep supports a Pareto interpretation and prevents k=5,m=3 from looking arbitrary.

5. **Recent baselines.** ONLY, VHD/VHR, and HALC are included in the rebuttal's comparison story, reducing the evaluation-completeness objection.

6. **Detector robustness.** Zero-anchor, relevant-anchor, noisy/diffuse-anchor, threshold sensitivity, fine-grained/crowded degradation, and oracle-box repair turn detector dependence into a measured limitation.

7. **Claim discipline.** The paper is framed around detector-assisted object-grounded hallucination and open-ended object mentions, not broad hallucination elimination.

8. **CHAIR interpretation.** Caption length and supported/unsupported object-mention diagnostics make the CHAIR improvement more credible.

## Remaining Problems

1. **Novelty remains moderate.** The method is still a coordinated combination of rollback, object proposals, attention-derived scoring, and rollout reranking.

2. **Detector quality remains central.** The strongest gains require useful anchors, while weak or noisy detector cases remain a real boundary.

3. **No second real proposer is evaluated.** The current controls are meaningful, but replacing Grounding DINO with another real proposer would more directly answer KrEs.

4. **Full mode remains expensive.** The response is honest, but quality-mode compute cost remains a practical limitation.

5. **Generality is bounded.** Attribute, relation, and composition results should remain boundary evidence rather than a headline claim.

6. **Reproducibility must be completed later.** The final paper needs exact baseline code versions, commands, checkpoints, prompts, parsers, seeds, validation splits, and tuned grids.

## Follow-Up Questions

1. Which exact HALC, ONLY, and VHD/VHR implementations were used, and what sanity checks verified them?

2. Does the attribution pattern hold with a second real detector?

3. Are InstructBLIP CHAIR improvements statistically tested as fully as the LLaVA rows?

4. Can CHORD detect noisy-anchor regimes and automatically fall back to P+C or Past?

5. Do prompt-robustness checks extend to CHAIR and longer open-ended captioning?

6. Will the final abstract/method summary explicitly state P+C as the practical default and Full as a slower quality setting?

7. Are the newer-backbone rows full benchmark results or only sanity-check pilots?

## Final Score

**4/5 Weak Accept.**

The rebuttal is strong enough to satisfy the actionable reviewer concerns. The likely reviewer coalition remains favorable: jjVG and M8du move upward, yx8u and ve3y stay supportive, and KrEs becomes the main residual risk because their remaining concerns are about intrinsic novelty and detector dependence rather than missing rebuttal evidence.
