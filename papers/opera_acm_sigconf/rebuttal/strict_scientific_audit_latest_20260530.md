# Strict Scientific Audit Of Latest Author Response, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

Current PDF metadata: 5 pages, modified 2026-05-30 01:53:36, 334909 bytes.

Reference: `reviewer_true_intent_analysis_20260529.md`

Scope: this audit treats the reported table values as valid results and evaluates only scientific/reviewer-response adequacy.

## Verdict

As a strict composite reviewer, I would score the latest rebuttal **4/5: Weak Accept**.

This is not a 5/5 response. It does not eliminate the incremental-novelty concern, the detector-dependence concern, or the limited-generality concern. But it now satisfies enough decision-critical reviewer needs that I would no longer keep the paper at Borderline.

The rebuttal is strongest where it directly answers M8du and KrEs: it gives mechanism diagnostics, detector controls, latency accounting, statistical stability, detector-threshold sensitivity, and matched recent-baseline comparisons. It also preserves yx8u and ve3y by avoiding overclaiming and by presenting Full CHORD as a quality-oriented setting rather than a cheap default.

## Does It Satisfy All Reviewer Needs?

No. It satisfies the major acceptance gates, but not every concern.

Satisfied or mostly satisfied:

- Future mechanism: Full-vs-P+C flip rate, corrected/harmful counts, CHAIR continuation deltas, definitions, and statistical tests are present.
- Detector attribution: same-anchor non-CHORD, uniform/random anchors, Past+Future without Current, real-anchor P+C, Full, detector strata, and threshold sensitivity are present.
- Efficiency: proposal time, decode ITL, total latency, VRAM, and batch behavior are reported.
- Hyperparameters: k/m sweep now supports a quality-vs-cost operating-point story.
- Recent baselines: ONLY, VHD/VHR, HALC are included under matched protocol.
- Claim discipline: the paper is framed as detector-assisted base-MLLM training-free admission verification, not as detector-free or causally explained by attention.
- Generality: newer-backbone and attribute/relation/composition pilot rows are included with bounded claims.

Still not fully satisfied:

- Novelty remains moderate; the rebuttal reframes the contribution but does not make it fundamentally new.
- Generality evidence is still pilot-scale and boundary-style, especially for relation/composition cases.
- Full CHORD remains expensive and memory-limited; the practical recommendation relies on P+C or smaller rollout settings.
- The method still depends strongly on relevant object anchors; zero-anchor and noisy-anchor cases show limited gains.
- Recent-baseline comparison is only one matched protocol; it does not exhaust all settings or implementations.

## Reviewer-Specific Assessment

| Reviewer | Original | Likely after latest rebuttal | Strict assessment |
|---|---:|---:|---|
| jjVG | 3 | 4 | Their concrete completeness requests are now addressed: k/m, cost, recent methods, and Figure 2 clarity. |
| KrEs | 2 | 3 to 4 | The response now seriously weakens their rejection. I would expect at least Borderline; Weak Accept is possible if they accept the matched protocol and detector controls. |
| yx8u | 4 | 4 | The response preserves their support through claim discipline and bounded generality. |
| ve3y | 4 | 4 | Practical trade-off is honestly presented; no reason to downgrade. |
| M8du | 3 | 4 | Their main mechanism and detector-dependency questions are directly answered. |

Combined reviewer simulation: likely enough for acceptance, but not a clean or high-confidence accept.

## What Still Prevents A Higher Score

1. **Novelty ceiling**: even with better evidence, the method is still a coordinated combination of rollback, detector anchors, attention-derived scoring, and rollout. The rebuttal makes this acceptable, not exceptional.

2. **Detector dependence**: the response is honest that gains concentrate when relevant anchors exist. This is scientifically fair, but it limits robustness claims.

3. **Efficiency trade-off**: P+C is the practical setting; Full is costly. The response handles this honestly, but the cost remains a real limitation.

4. **Generality is not broad**: the newer-backbone and relation/composition rows are useful, but they look like boundary evidence rather than a broad validation suite.

5. **Recent baselines are limited to the chosen matched protocol**: this is acceptable for rebuttal, but a skeptical reviewer can still ask whether implementation details favor CHORD.

## Questions I Would Still Ask

1. Are the matched HALC/ONLY/VHD/VHR implementations official or reimplementations, and were their hyperparameters tuned comparably?
2. How sensitive are CHORD gains to the Grounding DINO text/box threshold beyond the three reported settings?
3. For relevant/noisy anchor strata, how reliable are synonym-normalized label matches?
4. Does the same-anchor non-CHORD baseline receive as much tuning as CHORD's scoring weights?
5. Can the newer-backbone pilot be expanded beyond 1000 samples?
6. Why are relation/composition gains so small, and should those settings be excluded from the method's claimed scope?
7. In deployment, should the camera-ready recommend P+C as the default and Full only for high-accuracy offline generation?
8. How much of the CHAIR-S improvement comes from fewer object mentions versus better grounded object mentions?

## Final Score

**4/5 Weak Accept.**

This rebuttal now satisfies the main acceptance-critical concerns from the five-reviewer intent map. It does not fully solve novelty, detector dependence, deployment cost, or broad generality, but it explains these limits honestly and adds enough targeted evidence that I would support acceptance rather than remain at Borderline.

