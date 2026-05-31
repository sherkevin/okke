# Final Rebuttal Reviewer Audit Under Real-Values Assumption, 2026-05-29

Audited artifact: `author_response_min_diff_expected_20260529.pdf`

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

User clarification: all table values in the PDF are real measured results; the `E` marks and expected-result notes are stale cleanup artifacts and should not be treated as evidence uncertainty.

## Revised Bottom-Line Verdict

Under this assumption, I would score the rebuttal **4/5, Weak Accept**.

This is a real improvement over the previous audit because the response now contains the exact type of measured evidence that the most important skeptical reviewers requested:

- Future-vs-P+C flip/correctness counts;
- detector attribution controls;
- end-to-end latency/VRAM accounting;
- k/m robustness sweep;
- claim calibration around detector-assisted, base-MLLM training-free decoding.

I still would not score it as 5/5 or "sure accept" because the rebuttal does not fully solve recent-baseline comparison, broader model/benchmark generality, batch-size behavior, or quantified detector-failure strata.

## Reviewer-By-Reviewer Revised Judgment

| Reviewer | Original score | Revised likely score | Why |
|---|---:|---:|---|
| jjVG | 3 Borderline | 4 Weak Accept | Their visible checklist is mostly closed: k/m sweep, cost breakdown, recent-method acknowledgement, and Figure 2 revision promise. Low confidence makes them movable. |
| KrEs | 2 Weak Reject | 3 Borderline, possible 4 if generous | Detector attribution is now meaningfully addressed by same-anchor, uniform/random anchor, Past+Future, P+C, and Full controls. End-to-end cost is also clearer. But they also asked for stronger recent baselines, batch behavior, and broader experiments, so I would not assume full conversion. |
| yx8u | 4 Weak Accept | 4 Weak Accept | The response is well calibrated: detector-assisted wording, attention-as-feature wording, scoped claims, and honest latency. It preserves this reviewer well. |
| ve3y | 4 Weak Accept | 4 Weak Accept | The practical-regime framing is credible: P+C is the lower-cost operating point, Full is a quality mode. They likely remain supportive. |
| M8du | 3 Borderline | 4 Weak Accept | Table 1 directly answers the central Future mechanism question: Full flips P+C on about 4.1--4.3% of POPE-Adv cases and corrected flips exceed harmful flips by roughly 2.3x. Together with Table 2 and Table 4, this is enough to raise their score. |

Likely coalition after rebuttal: yx8u + ve3y remain Weak Accept, M8du likely moves to Weak Accept, jjVG likely moves to Weak Accept, KrEs softens to Borderline or possibly Weak Accept. That is a plausible acceptance coalition.

## What Is Now Satisfied

### 1. Future mechanism is substantially answered

The most important M8du question is now answered. The rebuttal does not merely show aggregate gains; it reports admission-change behavior:

- LLaVA POPE-Adv: Full vs P+C flips 4.1% of decisions, with 96 corrected and 42 harmful flips.
- InstructBLIP POPE-Adv: flips 4.3%, with 102 corrected and 44 harmful flips.
- CHAIR-S improves by 0.020 on both backbones.

This supports the bounded claim that Future is not a broad perturbation mechanism; it acts on a small set of hard cases and its helpful flips outnumber harmful flips.

### 2. Detector attribution is meaningfully isolated

Table 2 is strong enough to weaken the "Grounding DINO did all the work" objection:

- same-anchor non-CHORD does not match P+C or Full;
- uniform/random anchors are worse than real query-conditioned anchors;
- Past+Future without Current improves over Past but remains below P+C real anchors and Full real anchors;
- Full real anchors are best across Adv. F1, CHAIR-S, and FP rate.

This does not prove CHORD is independent of the detector, and the response correctly avoids that claim. It shows that the detector alone is insufficient and that the admission rule adds measurable value.

### 3. Efficiency honesty is mostly solved

Table 3 is reviewer-facing in the right way:

- Greedy: 395 ms/sample;
- OPERA: 434 ms/sample;
- P+C: 663 ms/sample;
- Full: 864 ms/sample.

This makes the cost explicit. It also correctly positions P+C as the practical operating point and Full as the quality-oriented operating point. The rebuttal no longer tries to sell Full CHORD as cheap.

### 4. k/m robustness is largely solved

Table 4 shows a sensible Pareto pattern:

- P+C without future is lower cost but weaker.
- k=5,m=2 is close to k=5,m=3 at lower cost.
- k=5,m=4 and k=10,m=3 give small gains at much higher cost.

This supports robustness and diminishing returns. The only wording issue is that the paper should not over-defend k=5,m=3 as the universal default. It should call k=5,m=3 the quality-oriented default and k=5,m=2 or P+C the practical setting.

### 5. Claim discipline is strong

The response says the contribution is coordination at the token-admission decision, not novelty of every ingredient. It also says CHORD is training-free for the base MLLM but detector-assisted. That is exactly the tone needed to preserve yx8u and ve3y.

## Remaining Substantive Risks

### 1. KrEs may still want stronger recent baselines

The response says it will add ONLY, VHD/VHR, and HALC to related work, but it does not provide direct matched numerical comparisons. For jjVG this may be enough; for KrEs it may not be.

If space allows, add a compact positioning table directly in the rebuttal:

`Method | training-free | detector use | lookahead/rollout | intervention stage | matched-code availability | reason not directly compared`

If any matched run exists, include even one or two direct rows.

### 2. Batch-size behavior remains missing

KrEs explicitly asked for behavior under different batch sizes. Table 3 says batch 1. That is honest, but incomplete.

Minimum fix: add one sentence: "All rebuttal latency numbers are batch-1 single-GPU measurements; we will add batch-size scaling in the revision." Better fix: include batch 1/4 latency or say batch scaling is not optimized because rollout branches are sequential.

### 3. Detector-failure strata are promised but not quantified in the PDF

The PDF says zero-anchor cases, relevant-anchor cases, and noisy/missed-object proposals will be reported, but no numbers are shown. Since both M8du and yx8u worried about detector failure, this remains a real gap.

Even a tiny line would help:

`zero-anchor / relevant-anchor / noisy-anchor: count, delta vs Past, failure interpretation`.

### 4. Generality is bounded, not expanded

The rebuttal honestly limits claims to object/open-ended hallucination settings. That is good. But it does not add Qwen2-VL, InternVL, LLaVA-NeXT, relation hallucination, attribute hallucination, or compositional VQA. This leaves yx8u/KrEs/M8du with a residual scope concern.

This is probably acceptable if the response is space-limited, but it prevents a high score.

### 5. Some stale wording still weakens final polish

Even ignoring E marks as evidence status, the final submitted version should remove phrases like:

- `final table shape we will use`;
- `Table is designed to test`;
- `Expected-result note`;
- `engineer-run SSH outputs`.

Replace them with measured-result language:

- `Table 1 reports measured Full-vs-P+C decision changes`;
- `Table 2 isolates detector attribution`;
- `All values are measured on ...`.

This is not a scientific weakness if the values are real, but it is a presentation risk.

## Questions I Would Still Ask As A Reviewer

1. What are the exact definitions of `corrected` and `harmful` flips?
2. Are flip/correction counts statistically stable across seeds or prompt order?
3. How is same-anchor non-CHORD implemented?
4. What are the zero-anchor, missed-anchor, and noisy-anchor strata results?
5. Why is k=5,m=3 the submitted quality default if k=5,m=2 is close and cheaper?
6. Are Table 3 numbers synchronized GPU timings including detector proposal and preprocessing?
7. What happens for batch sizes above 1?
8. Why no direct matched comparison with ONLY, VHD/VHR, or HALC?
9. How should readers expect the method to behave on relation/attribute/compositional hallucinations?

## Final Score

As reviewer after accepting the numbers as real:

**4/5 Weak Accept.**

As area-chair-style synthesis:

The rebuttal likely creates an acceptance coalition, but it is not a guaranteed high-score conversion. The most likely final pattern is:

- jjVG: 3 -> 4
- KrEs: 2 -> 3, possibly 4
- yx8u: 4 -> 4
- ve3y: 4 -> 4
- M8du: 3 -> 4

The paper still has moderate novelty and scope limitations, but the rebuttal now answers the central mechanism and attribution doubts well enough that I would support acceptance.

