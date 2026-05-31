# Strict Reviewer Audit Of Latest Author Response, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

PDF metadata checked: 4 pages, last modified 2026-05-30 00:01:30, size 197511 bytes.

Reference: `reviewer_true_intent_analysis_20260529.md`

Assumption carried from user clarification: table numbers are treated as real measured results; stale `E` marks and expected-note wording are not used as evidence-status objections, though they should be removed before submission.

## Verdict

As a strict composite reviewer, I would score the current latest rebuttal **4/5: Weak Accept**.

This is not a high-confidence accept. It does not satisfy every reviewer demand. But compared with the prior version, it now directly addresses the main missing gates from `reviewer_true_intent_analysis_20260529.md`: mechanism validation, detector attribution, end-to-end cost, hyperparameter robustness, detector-failure behavior, and claim discipline.

If I were only Reviewer KrEs, I would still be closer to **3/5 Borderline** because recent matched baselines and broader generality remain weak. As the combined five-reviewer judgment, however, the response is now strong enough for Weak Accept.

## Does It Satisfy All Needs?

No. It satisfies the decision-critical needs enough to justify a weak accept, but it does not close every substantive concern.

### Satisfied Or Mostly Satisfied

- **Future mechanism**: Table 1 gives Full-vs-P+C flip rates, corrected/harmful counts, and CHAIR-S deltas. The new definitions clarify corrected/harmful flips and the CHAIR protocol.
- **Detector attribution**: Table 2 includes same-anchor non-CHORD, uniform anchors, random anchors, Past+Future without Current, P+C real anchors, and Full real anchors. The new control definitions reduce ambiguity.
- **Detector robustness**: the new detector-stratum table gives zero-anchor, relevant-anchor, and noisy/diffuse-anchor counts and effects.
- **Efficiency honesty**: Table 3 separates proposal time, decode ITL, total latency, tokens, peak VRAM, and batch-1 setting.
- **Batch-size boundary**: the added batch table directly acknowledges scaling and OOM behavior for Full at batch 4.
- **k/m robustness**: Table 4 shows diminishing returns and now frames k=5,m=3 as quality-oriented rather than universal default.
- **Claim calibration**: the response says detector-assisted, training-free for base MLLM, attention as operational feature, scoped to evaluated hallucination settings.
- **Recent-method awareness**: HALC, ONLY, and VHD/VHR are now included in a positioning table.

### Still Not Fully Satisfied

- **No direct matched recent-baseline numbers** for HALC, ONLY, VHD/VHR.
- **No stronger/newer MLLM evidence** such as Qwen2-VL, InternVL, LLaVA-NeXT.
- **No relation/attribute/compositional hallucination evidence**.
- **No confidence intervals actually shown**, only "will report if space allows".
- **Figure 2 is still a promise**, not shown in the rebuttal.
- **Some protocol details remain compact**: same-anchor non-CHORD is defined, but exact scoring formula and implementation details are still not reproducible from the rebuttal alone.

## Reviewer-Specific Score Movement

| Reviewer | Original | Latest likely score | Strict rationale |
|---|---:|---:|---|
| jjVG | 3 | 4 | k/m, cost, related methods, and Figure 2 are all addressed at the level this low-confidence reviewer likely needs. |
| KrEs | 2 | 3, possible 4 | Detector attribution and end-to-end cost are much stronger now, but no matched recent baselines or broader MLLM tests remain a serious gap. |
| yx8u | 4 | 4 | Claim discipline is good. Remaining generality concerns prevent a higher score. |
| ve3y | 4 | 4 | Practical framing is honest; P+C vs Full is now clear. Full remains costly, so no upgrade beyond weak accept. |
| M8du | 3 | 4 | The response now directly answers Future flip/correctness, detector dependency, hyperparameters, and detector robustness. |

The likely coalition is now: jjVG weak accept, yx8u weak accept, ve3y weak accept, M8du weak accept, KrEs borderline or weak accept. That is enough for a weak-accept synthesis, not a clean accept.

## What I Would Still Criticize In The Review Discussion

1. **Recent methods are positioned but not experimentally compared.** A positioning table is better than omission, but KrEs asked for stronger baselines in evaluation, not only related-work taxonomy.

2. **Generality remains narrow.** The method is still validated around object-level anchors, POPE/CHAIR/MMBench, two 7B backbones, and object/open-ended hallucination. This does not prove transfer to relation, attribute, compositional, or reasoning-heavy hallucination.

3. **Batch scaling is acknowledged, not solved.** Full CHORD OOMs at batch 4 under the reported boundary. That is honest, but it weakens deployment claims.

4. **The detector remains a core dependency.** The stratum table is useful, but the method still relies on relevant anchors for most gains. Zero-anchor cases gain only marginally. The response should not overstate detector robustness.

5. **Statistical reliability is underdeveloped.** The response says bootstrap CIs will be reported if space allows. A strict reviewer would prefer at least one compact CI or significance statement for Table 1 and key Table 2 deltas.

6. **Novelty is still moderate.** The response reframes novelty as coordinated admission-time verification. That is acceptable, but it does not eliminate the concern that CHORD combines known components.

## Questions I Would Still Ask

1. Can you provide matched numerical comparisons to HALC, ONLY, and VHD/VHR on at least one shared backbone/split?
2. Are the Table 1 flip/correction improvements statistically significant?
3. What are the exact thresholds for classifying anchors as relevant versus noisy/diffuse?
4. How sensitive are detector-stratum results to Grounding DINO thresholds?
5. Does the same-anchor non-CHORD baseline use a fixed weighting coefficient, and how is it tuned?
6. Why should the method generalize to relation/attribute hallucinations when the mechanism is object-anchor-centric?
7. Can P+C or Full be validated on one newer MLLM, even as a small pilot?
8. For deployment, should the recommended default be P+C rather than Full?


## Final Score

**4/5 Weak Accept** as the combined reviewer simulation.

Not all demands are solved. The decisive improvement is that the latest PDF now answers enough of M8du and jjVG, preserves yx8u and ve3y, and weakens KrEs's rejection enough that the overall decision can plausibly cross the acceptance threshold.

