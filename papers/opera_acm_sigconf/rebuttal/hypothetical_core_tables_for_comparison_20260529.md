# Hypothetical Core Tables for Rebuttal Comparison, 2026-05-29

Critical boundary: all numbers in this document are hypothetical target values. They are not measured results and must not be submitted in a rebuttal. The purpose is to define what a strong evidence pattern would look like so future real diagnostics can be compared against it.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record.

## How To Use These Tables

- Use the `Ideal target` columns as the evidence pattern we hope real experiments approach.
- Use the `Minimum acceptable` columns to decide whether the rebuttal claim can still be made safely.
- Use the `Red flag` columns to decide when to narrow or abandon a claim.
- When real results are available, create a separate actual-results table and never overwrite this hypothetical reference.

## Table 1. Future Rollout Mechanism, Hypothetical Target

Reviewer target: mainly M8du, also KrEs and yx8u.

Core question: does Full CHORD actually change decisions relative to Past+Current, and are those changes correct?

| Dataset / setting | Metric | Ideal target | Minimum acceptable | Red flag | Rebuttal claim allowed if real result meets target |
|---|---|---:|---:|---:|---|
| POPE adversarial | Full vs P+C answer-change rate | 4-8% | 2-10% | <1% or >15% | Future acts only on ambiguous cases rather than broadly perturbing yes/no answers. |
| POPE adversarial | Corrected flips / all flips | >=65% | >=55% | <=50% | Future corrects more hard cases than it harms. |
| POPE adversarial | Harmful flips / all flips | <=20% | <=30% | >=40% | Future improves precision-oriented hard-negative handling without large recall damage. |
| POPE adversarial | Delta Adv. F1 vs P+C | +0.8 to +1.5 pts | >=+0.3 pts | <=0 or large recall collapse | Full gives modest POPE gains but P+C remains the efficient regime. |
| CHAIR captioning | Object/sentence decision-change rate | 8-15% | 5-18% | <3% or >25% | Future is more active in open-ended generation where errors compound. |
| CHAIR captioning | Corrected hallucination events / all changes | >=70% | >=60% | <=50% | Future primarily suppresses unsupported object mentions. |
| CHAIR captioning | CHAIR_S delta vs P+C | -1.5 to -3.0 pts | <=-0.8 pts | >=0 | Full's main value is open-ended hallucination suppression. |
| MMBench | Answer-change rate vs P+C | <=3% | <=5% | >8% | Future does not disrupt general multimodal retention. |
| MMBench | Accuracy delta vs P+C | +0.2 to +0.8 pts | >=-0.2 pts | <-0.5 pts | Added verification does not impose a meaningful alignment tax. |

Desired real-result story:

- On POPE, Full only modestly improves over P+C; P+C remains the recommended latency-sensitive setting.
- On CHAIR, Full shows a clearer advantage because short-horizon rollout catches unstable open-ended continuation.
- On MMBench, changes are small and non-harmful.

If actual results are weaker:

- If POPE is flat but CHAIR improves, narrow the claim to open-ended generation.
- If Full harms POPE relative to P+C, explicitly recommend P+C for discriminative yes/no settings.
- If CHAIR does not improve, do not foreground Future as a main contribution; instead position it as optional quality-oriented verification needing further study.

## Table 2. Detector Attribution, Hypothetical Target

Reviewer target: mainly KrEs and M8du, also yx8u and ve3y.

Core question: are gains caused by the CHORD admission rule, or mostly by Grounding DINO?

Assume the row below is measured on a representative held-out POPE adversarial subset and, where feasible, CHAIR.

| Condition | What it tests | Ideal POPE Adv. F1 | Minimum acceptable | Red flag | Interpretation if achieved |
|---|---|---:|---:|---:|---|
| OPERA / Past only | rollback baseline | 0.804 | baseline | baseline | Reference prior component. |
| Same-anchor non-CHORD control | detector metadata without CHORD admission rerank | 0.812-0.820 | must stay below P+C by >=0.5 pts | matches P+C/Full | Detector alone helps slightly but cannot explain CHORD. |
| Full CHORD + no/uniform anchors | remove localized proposer weighting | 0.820-0.832 | >= OPERA and below real-anchor Full | collapses below OPERA or equals real anchors | Admission/future still contributes, but localized anchors matter. |
| Full CHORD + random anchors | controls for arbitrary visual weighting | 0.810-0.825 | below real anchors by >=1.0 pt | equals or exceeds real anchors | Semantic anchors, not arbitrary extra weights, drive current grounding. |
| Past+Future without current anchors | future without detector-backed current support | 0.825-0.835 | between OPERA and Full | equals Full or worse than OPERA | Future helps but is not enough without current grounding. |
| Past+Current / real anchors | efficient CHORD regime | 0.832-0.840 | > OPERA by >=1.0 pt | <= OPERA | Current grounding gives efficient admission improvement. |
| Full CHORD / real anchors | complete method | 0.845-0.852 | best or tied best | below P+C and no clear CHAIR gain | Current and future are complementary under real anchors. |

Hypothetical CHAIR target:

| Condition | Ideal CHAIR_S | Minimum acceptable | Red flag | Interpretation |
|---|---:|---:|---:|---|
| OPERA / Past only | ~0.228 | baseline | baseline | Reference. |
| Same-anchor non-CHORD | 0.210-0.220 | below OPERA but above P+C | matches Full | Detector alone helps but is insufficient. |
| Past+Current / real anchors | 0.170-0.180 | <=0.190 | >=0.210 | Current grounding reduces unsupported admissions. |
| Full CHORD / real anchors | 0.150-0.160 | <=0.170 and below P+C | >=P+C | Future adds open-ended suppression. |

Desired real-result story:

- Real anchors outperform random/no anchors.
- Same-anchor non-CHORD control does not match CHORD.
- P+C is strong and efficient; Full gives additional CHAIR/open-ended gains.

If actual results are weaker:

- If same-anchor non-CHORD matches Full, concede detector attribution and weaken novelty.
- If random anchors match real anchors, current grounding story is not supported.
- If no-anchor Full is nearly as good as real-anchor Full, shift emphasis from detector grounding to admission-time regularization.

## Table 3. End-to-End Cost Accounting, Hypothetical Target

Reviewer target: KrEs, M8du, jjVG, ve3y.

Core question: is CHORD's cost honestly reported, including Grounding DINO and memory?

The submitted paper already reports decode-stage ITL. The hypothetical table below shows what a strong complete cost table would look like if measured on the same hardware, with detector proposal cached once per image/query.

### LLaVA-v1.5-7B, hypothetical target

| Method / regime | Detector proposal time | Decode ITL | Total answer latency, 20 tokens | Peak VRAM | Interpretation |
|---|---:|---:|---:|---:|---|
| Greedy | 0 ms | 19.73 ms/token | ~395 ms | 15-17 GB | Baseline floor. |
| OPERA / Past | 0 ms | 21.69 ms/token | ~434 ms | 15-17 GB | Low-overhead rollback. |
| Past+Current | 80-180 ms once | 27.24 ms/token | ~625-725 ms | 16-18 GB | Practical CHORD regime; about 1.6-1.8x total latency vs greedy for 20-token answers. |
| Full CHORD | 80-180 ms once | 37.31 ms/token | ~825-925 ms | 18-22 GB | Quality regime; about 2.1-2.4x total latency vs greedy. |

### InstructBLIP-7B, hypothetical target

| Method / regime | Detector proposal time | Decode ITL | Total answer latency, 20 tokens | Peak VRAM | Interpretation |
|---|---:|---:|---:|---:|---|
| Greedy | 0 ms | 16.47 ms/token | ~329 ms | 14-16 GB | Baseline floor. |
| OPERA / Past | 0 ms | 16.90 ms/token | ~338 ms | 14-16 GB | Low-overhead rollback. |
| Past+Current | 80-180 ms once | 24.51 ms/token | ~570-670 ms | 15-18 GB | Practical CHORD regime. |
| Full CHORD | 80-180 ms once | 35.86 ms/token | ~797-897 ms | 18-22 GB | Quality regime with substantial but transparent cost. |

Minimum acceptable cost story:

- Detector proposal time is reported separately.
- Decode ITL and total latency are both reported.
- Past+Current remains much cheaper than Full.
- Full is not described as cheap; it is described as quality-oriented.

Red flags:

- Detector proposal time is larger than the decode latency itself for common answer lengths.
- Full CHORD exceeds 3x total latency vs Greedy without a large CHAIR gain.
- Peak VRAM prevents batch size 1 on the target GPU.
- Batch-size behavior is much worse than single-example timing suggests.

## Combined One-Shot Success Criteria

The rebuttal becomes strong if real results roughly satisfy:

| Evidence block | Green condition | Yellow condition | Red condition |
|---|---|---|---|
| Future mechanism | Correct flips clearly exceed harmful flips, especially on CHAIR | POPE weak but CHAIR positive | Future flat or harmful across settings |
| Detector attribution | Real anchors > random/no anchors; same-anchor non-CHORD < CHORD | Some controls mixed but P+C still robust | Detector-only matches CHORD |
| Cost | P+C is defensible practical regime; Full cost transparent | Full costly but CHAIR gain strong | Cost high and mechanism weak |
| Hyperparams | k/m sweep stable near default | mild trade-off, default reasonable | default cherry-picked |
| Related work | ONLY/VHR/HALC addressed honestly | only conceptual comparison available | missing or dismissed |

## Recommended Claim Strength By Actual Outcome

| Actual evidence outcome | Claim strength in rebuttal |
|---|---|
| Green across all three tables | Strong: CHORD is an attributable, bounded admission-time verifier with two operating regimes. |
| Green detector + cost, yellow future | Moderate: P+C is the reliable practical regime; Full is optional for open-ended quality. |
| Green future, yellow detector | Moderate: CHORD is detector-assisted; do not claim detector-independent contribution. |
| Yellow cost, green mechanism | Moderate: quality-oriented method with explicit overhead. |
| Red in any P0 table | Narrow sharply; do not overclaim. Rebuttal should emphasize honest limitations and camera-ready clarity rather than superiority. |
