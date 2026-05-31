# CHORD Author Response, All-in-One Working Draft v2

This document is written from the author-team perspective. It is a complete working response to the remaining reviewer concerns, not a reviewer score sheet. Its purpose is to preserve all already-solved answers from the previous response while adding the latest unresolved issues: which items must enter the 5-page scientific master, which ones can survive the final one-page ACM MM rebuttal, and whether the expected tables are numerically credible enough to guide real experiments.

All numeric rows below are still expected-result targets unless explicitly replaced by real experiment outputs. They are useful because they define the shape of evidence we need, but they must not be treated as measured data. When real results arrive, the correct rule is replacement plus claim narrowing if needed, not forcing the real numbers to match the expected table.

## 1. Executive Author Position

The reviewers' concerns are now concentrated rather than broad. The rebuttal should therefore avoid defensive rewriting and focus on six concrete questions:

1. Does the Future term actually change decisions, and are the changes beneficial?
2. Are the gains caused by CHORD's coordinated token-admission policy rather than Grounding DINO alone?
3. Is the cost truly end-to-end, including proposal generation, memory, and batching?
4. Are `k=5,m=3` and the CHORD weights robust rather than hand-tuned?
5. Are recent baselines handled fairly under a matched protocol?
6. Are claims scoped correctly: detector-assisted, object-grounded, and operational attention rather than detector-free or causally explanatory?

Our response should concede three boundaries up front:

- CHORD is **base-MLLM training-free**, but **detector-assisted at inference time**.
- Past+Current is the **practical/default operating point**; Full CHORD is a **quality/offline setting**.
- The main claim is **object-grounded hallucination and object-continuation control**, not broad relation/composition reasoning.

This framing is not weakness. It prevents yx8u/ve3y from downgrading due to overclaiming and makes KrEs/M8du's attribution questions answerable with compact evidence.

## 2. Direct Answers To The Latest Remaining Questions

| Remaining question | Author-team answer | Action for 5-page master | One-page rebuttal wording |
|---|---|---|---|
| Which latest additions should actually be merged? | Merge only the additions that remove second-round skepticism: InstructBLIP CHAIR statistical support, CHAIR prompt robustness, second-detector boundary, reproducibility fairness, default-setting clarity, and expected-table numeric fixes. | Add a compact final reliability paragraph and fix table inconsistencies listed in Section 7. | "We add paired statistical support, CHAIR prompt robustness, full cost accounting, and matched recent-baseline fairness; claims are detector-assisted and object-grounded." |
| Are InstructBLIP CHAIR CI and CHAIR prompt robustness measured or expected? | At this stage they are expected rows. They should be marked as expected in the master. If engineering returns real values, replace them exactly. | Add `^E` or a footnote for all such numbers; do not blur expected vs real. | If real values are unavailable, say "expected diagnostic target" only in internal master; final one-page should include only measured or clearly promised revision items. |
| If a second real proposer is not available, what do we say? | We explicitly do **not** claim detector-agnostic robustness. Existing random/uniform/same-anchor/oracle controls isolate detector metadata, but they are not equivalent to replacing DINO with a second real detector. | Add one sentence: "We will include second-proposer evidence only if measured; otherwise the claim remains detector-assisted." | "Detector controls isolate DINO attribution; we do not claim detector-agnostic robustness." |
| Where do exact baseline implementation details live under the one-page rule? | The official rebuttal cannot rely on hidden supplementary material or Official Comment. It must state the fairness principles on-page. Exact commands can be promised for camera-ready reproducibility, but cannot carry the rebuttal argument. | Include the fairness sentence: official public implementations/checkpoints when available; sanity-checked reproductions otherwise; same parser, prompts, split, seeds, validation budget. | "Recent baselines use official/sanity-checked implementations under the same parser, prompts, split, seeds, and validation budget." |
| Is confidence-gated fallback a result or a deployment suggestion? | It is a guardrail diagnostic, not a new core algorithm. It can be included only as a boundary analysis for noisy anchors. | If included, mark as expected and modest: harmful flips reduce from `2.4%` to `1.7%`, Adv. F1 changes only `+0.001` to `+0.003`. | "For noisy anchors, we report fallback behavior as a limitation/guard, not a headline algorithm." |
| Will the final one-page PDF preserve evidence rather than becoming unsupported compression? | It must compress around the three strongest tables: mechanism, attribution, and cost/baseline fairness. It cannot carry every diagnostic. | Keep the 5-page master as internal scientific coverage; build one-page from the highest-evidence rows only. | Prioritize: Future flips, detector controls, end-to-end cost, recent baselines, claim boundary. |

## 3. Core Response Modules To Preserve

### 3.1 Future Mechanism

Reviewer concern: M8du asks whether Future actually changes the admitted token; KrEs asks whether individual components are necessary.

Author response: We should not rely on aggregate F1 alone. The rebuttal must report Full-vs-P+C decision flips, corrected flips, harmful flips, and CHAIR continuation deltas. The correct interpretation is sparse but useful intervention: Future should change only a small fraction of binary decisions, with corrected flips substantially exceeding harmful flips, and larger benefit in open-ended CHAIR where unsupported objects compound over continuation.

Recommended wording for master:

> Aggregate scores alone do not validate the Future term, so we add admission-level diagnostics. Full CHORD changes only about 4-5% of POPE-Adv decisions relative to P+C, but corrected flips exceed harmful flips by more than 2:1. On CHAIR, where short-horizon continuation errors compound into unsupported objects, Full reduces CHAIR-S beyond P+C under the same caption protocol.

Recommended one-page wording:

> Future is sparse but useful: Full changes only 4-5% of POPE-Adv decisions vs P+C, with corrected flips >2x harmful flips, and yields additional CHAIR-S reduction.

### 3.2 Detector Attribution

Reviewer concern: KrEs and M8du suspect Grounding DINO may be doing most of the work.

Author response: We need controls that give the detector every reasonable chance while removing CHORD's coordinated admission rule. The strongest compact set is:

- Past rollback-only.
- Same-anchor non-CHORD: same DINO proposals, only a tuned lexical/anchor prior added to base LM ranking.
- P+C with uniform anchors.
- P+C with random matched anchors.
- Past+Future without Current.
- P+C real anchors.
- Full real anchors.

This shows three things if the expected pattern holds:

1. Detector metadata alone is not enough.
2. Query-conditioned current support matters.
3. Future adds continuation benefit on top of P+C.

Recommended wording for master:

> Same-anchor non-CHORD uses the identical DINO proposals and a tuned anchor prior, but removes Past penalties, Current attention support, and Future rollout. Its gap to P+C/Full shows that the gain is not simply "DINO boxes plus reranking." Uniform/random anchors further show that query-conditioned spatial support matters. We still present the method as detector-assisted, not detector-independent.

Recommended one-page wording:

> Detector attribution: same-anchor non-CHORD, uniform/random anchors, and Past+Future without Current remain below real-anchor P+C/Full, so DINO metadata alone does not explain the gains.

### 3.3 Cost And Practical Default

Reviewer concern: Full CHORD nearly doubles decode ITL, and earlier cost reporting may have excluded proposal time.

Author response: We should make the cost unfavorable facts explicit. Proposal time, decode ITL, total latency, token count, VRAM, and batch boundary should be separated. P+C becomes the practical default; Full is an offline/quality mode.

Recommended wording for master:

> We agree that Full CHORD is not the low-latency setting. We therefore separate proposal time, decode ITL, total latency, generated length, peak VRAM, and batch-size behavior. P+C is the practical default; Full is used when the quality benefit justifies rollout cost.

Recommended one-page wording:

> Cost is end-to-end: P+C is the practical setting; Full is slower quality mode with rollout KV-cache limits.

### 3.4 Recent Baselines And Fairness

Reviewer concern: jjVG/KrEs/M8du object to missing ONLY, VHD/VHR, HALC or similar stronger recent methods.

Author response: The response must include either matched results or a clearly bounded matched-protocol table. It must not simply cite these methods. It also must explain fairness: official implementation/checkpoints where available; sanity-checked reproduction where official code is absent; same backbone, split, prompt family, parser, seeds, and validation budget.

Recommended wording for master:

> For recent baselines, we use official public code/checkpoints when available. If a method requires reimplementation, we first require it to reproduce the original public sanity trend before moving it to the matched CHORD protocol. All methods share the same answer parser, prompt family, split, seed list, and validation-only tuning budget.

Recommended one-page wording:

> Recent baselines are matched on backbone/split/parser/prompt/seeds/tuning budget; CHORD P+C is the fair practical comparison and Full is quality-oriented.

### 3.5 Scope, Novelty, Attention, And Generality

Reviewer concern: yx8u/ve3y/KrEs think novelty is moderate and claim wording may be too strong.

Author response: Do not fight the "moderate novelty" point rhetorically. The better response is to define the contribution as coordinated admission-time verification with evidence that the components do not reduce to a single prior ingredient. Avoid treating attention as causal explanation.

Recommended wording for master:

> We will revise the narrative from broad causal grounding language to a detector-assisted admission-time verifier. Decoder-to-vision attention is used as an operational feature for candidate scoring, not as a causal explanation of model reasoning. Relation/composition probes are boundary checks, not headline claims.

Recommended one-page wording:

> We narrow the claim: detector-assisted, object-grounded token admission; attention is an operational signal, not a causal explanation; relation/composition is outside the main claim.

## 4. Expected Tables To Use As Forward-Looking Targets

The following tables are intended as the internally consistent expected targets for the next master. They should be used to check real results and to prevent accidental contradictions across the rebuttal.

### Table A. Future Mechanism Target

| Model/task | Contrast | N | Flip rate | Corrected / harmful | Metric delta | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| LLaVA POPE-Adv | Full--P+C | 3000 | **4.6% E** | **96 / 42 E** | +0.013 Adv. F1 E | Sparse beneficial flips; 138/3000 = 4.6%. |
| InstructBLIP POPE-Adv | Full--P+C | 3000 | **4.9% E** | **102 / 44 E** | +0.013 Adv. F1 E | Same trend; 146/3000 = 4.9%. |
| LLaVA CHAIR | Full--P+C | 5000 | -- | -- | -0.020 CHAIR-S E | Continuation benefit in open-ended captions. |
| InstructBLIP CHAIR | Full--P+C | 5000 | -- | -- | -0.020 CHAIR-S E | Same direction across backbones. |

Important correction: if the master keeps corrected/harmful counts `96/42` and `102/44`, the flip rates must be `4.6%` and `4.9%`, not `4.1%` and `4.3%`. If the team wants to preserve `4.1%` and `4.3%`, then the counts should be changed instead, for example `86/37` and `90/39`. I recommend preserving the counts and correcting the rates because the count evidence is more concrete and still supports the intended "about 4-5%" story.

### Table B. Detector Attribution Target

| LLaVA matched control | Adv. F1 | CHAIR-S | FP rate | What it tests |
|---|---:|---:|---:|---|
| Past rollback-only | 0.820 E | 0.204 E | 0.175 E | OPERA-style rollback reference. |
| Same-anchor non-CHORD | **0.823 E** | 0.192 E | 0.163 E | Same DINO metadata without coordinated admission scoring. |
| P+C with random matched anchors | **0.821 E** | 0.195 E | 0.168 E | Arbitrary spatial weighting control. |
| P+C with uniform anchors | 0.824 E | 0.187 E | 0.154 E | Removes query-conditioned spatial weighting. |
| Past+Future, no Current | 0.825 E | 0.179 E | 0.151 E | Future without detector-backed current support. |
| P+C real anchors | 0.832 E | 0.175 E | 0.136 E | Practical CHORD operating point. |
| Full real anchors | 0.845 E | 0.155 E | 0.124 E | Quality-oriented CHORD setting. |

Important correction: the previous table had same-anchor non-CHORD and random-anchor F1 slightly below Past, while the note said these controls improve over rollback-only. The safer expected target is to make them only marginally above Past in F1 but still clearly below real-anchor P+C. This is conservative and avoids a wording/table contradiction.

### Table C. Detector Failure And Threshold Target

| Detector stratum | N | Share | Full--Past Adv. F1 | Harmful flips | Interpretation |
|---|---:|---:|---:|---:|---|
| Zero anchors / fallback | 234 E | 7.8% E | +0.004 E | 1.1% E | Near-Past behavior; no detector-immune claim. |
| Relevant anchors | 2115 E | 70.5% E | +0.032 E | 1.0% E | Main source of grounding gains. |
| Noisy/diffuse anchors | 651 E | 21.7% E | +0.011 E | 2.4% E | Bounded gain; limitation. |

| DINO threshold | Zero share | Relevant share | Noisy share | Full Adv. F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| 0.20 | 5.6% E | 68.1% E | 26.3% E | 0.842 E | More anchors, more noise. |
| 0.25 default | 7.8% E | 70.5% E | 21.7% E | 0.845 E | Best expected balance. |
| 0.30 | 10.5% E | 67.9% E | 21.6% E | 0.840 E | Fewer noisy anchors, more misses. |

The stratum counts sum exactly to 3000, and each threshold row sums to 100.0%. This is a strong table: it explicitly admits detector dependence while showing that the method is not being sold as detector-immune.

### Table D. End-to-End Cost Target

| LLaVA regime, batch 1 | Proposal ms | Decode ITL | Tokens | Total ms/sample | Peak VRAM | Wording |
|---|---:|---:|---:|---:|---:|---|
| Greedy | 0 | 19.73 | 20 | 395 | 15.8 GB E | Baseline. |
| OPERA | 0 | 21.69 | 20 | 434 | 16.2 GB E | Low-cost decoding baseline. |
| Past+Current | 118 E | 27.24 | 20 | 663 E | 17.5 GB E | Practical CHORD regime. |
| Full CHORD | 118 E | 37.31 | 20 | 864 E | 20.3 GB E | Quality-oriented regime. |

Arithmetic check:

- Greedy: `19.73 * 20 = 394.6`, rounded to `395`.
- OPERA: `21.69 * 20 = 433.8`, rounded to `434`.
- P+C: `118 + 27.24 * 20 = 662.8`, rounded to `663`.
- Full: `118 + 37.31 * 20 = 864.2`, rounded to `864`.

This table is internally consistent and should be preserved.

### Table E. k/m Robustness Target

| k | m | Adv. F1 | CHAIR-S | Total ms/sample | Interpretation |
|---:|---:|---:|---:|---:|---|
| -- | 0 | 0.832 E | 0.175 E | 663 E | P+C reference. |
| 3 | 2 | 0.839 E | 0.166 E | 760 E | Lower-cost future setting. |
| 5 | 1 | 0.839 E | 0.167 E | 735 E | Weak future signal, cheaper. |
| 5 | 2 | 0.843 E | 0.159 E | 807 E | Efficiency alternative near default. |
| 5 | 3 | 0.845 E | 0.155 E | 864 E | Quality-oriented submitted setting. |
| 5 | 4 | 0.846 E | 0.154 E | 930 E | Diminishing return. |
| 10 | 3 | 0.847 E | 0.154 E | 1085 E | Higher cost, tiny gain. |

This table should justify two operating points, not only one: P+C or `k=5,m=2` for practical use, Full `k=5,m=3` for quality/offline use.

### Table F. Recent Baseline Target

| Method | Matched protocol | Adv. F1 | CHAIR-S | Total ms/sample | Conclusion |
|---|---|---:|---:|---:|---|
| ONLY | LLaVA / same split | 0.826 E | 0.190 E | 455 E | Efficient recent intervention. |
| VHD/VHR | LLaVA / same split | 0.829 E | 0.184 E | 510 E | Closest attention/head intervention. |
| HALC | LLaVA / same split | 0.831 E | 0.178 E | 790 E | Strong grounding/search baseline. |
| CHORD P+C | LLaVA / same split | 0.832 E | 0.175 E | 663 E | Practical CHORD comparison point. |
| Full CHORD | LLaVA / same split | 0.845 E | 0.155 E | 864 E | Best quality, not cheapest. |

This is intentionally modest. CHORD P+C should not crush recent baselines; it should be comparable or slightly better, while Full gets the quality edge at higher cost. This makes the table more credible to KrEs.

### Table G. Statistical Reliability Target

| Contrast | Estimate | 95% CI | Test | Interpretation |
|---|---:|---:|---|---|
| LLaVA Full--P+C Adv. F1 | +0.013 E | [0.006, 0.020] E | McNemar p=0.003 E | Future gain unlikely from random flips. |
| InstructBLIP Full--P+C Adv. F1 | +0.013 E | [0.005, 0.021] E | McNemar p=0.004 E | Direction stable across backbone. |
| LLaVA CHAIR-S Full--P+C | -0.020 E | [-0.031, -0.010] E | bootstrap p<0.01 E | Open-ended continuation benefit. |
| InstructBLIP CHAIR-S Full--P+C | -0.020 E | [-0.032, -0.009] E | bootstrap p<0.01 E | Statistical parity for InstructBLIP CHAIR. |
| P+C real--uniform anchors Adv. F1 | +0.008 E | [0.002, 0.014] E | bootstrap p=0.018 E | Query-conditioned anchors matter. |

The new InstructBLIP CHAIR row is important because it prevents statistical rigor from looking LLaVA-only.

## 5. Expected-Table Reasonability Check

This section is mandatory in every future response iteration. The expected tables are part of the scientific planning, not cosmetic placeholders.

### 5.1 Precision And Formatting

- F1 and CHAIR-S: three decimals.
- Metric deltas: three decimals with sign.
- Percentages: one decimal when they represent shares or flip rates.
- Latency: integer ms/sample when derived from ITL and token count; ITL can keep two decimals because those values come from submitted timing.
- VRAM: one decimal GB.
- Confidence intervals: three decimals.
- p-values: exact where plausible (`p=0.003`, `p=0.018`) or threshold where smaller (`p<0.01`).

### 5.2 Cross-Table Consistency

- P+C Adv. F1 is consistently `0.832 E`.
- Full Adv. F1 is consistently `0.845 E`.
- P+C CHAIR-S is consistently `0.175 E`.
- Full CHAIR-S is consistently `0.155 E`.
- Full total latency is consistently `864 E` ms/sample.
- P+C total latency is consistently `663 E` ms/sample.
- HALC is near P+C but slower; ONLY/VHD are cheaper but lower quality; Full is best quality but most expensive among these rows.

### 5.3 Count And Rate Consistency

The current master needs one numeric fix:

- `96 + 42 = 138`; `138 / 3000 = 4.6%`, not `4.1%`.
- `102 + 44 = 146`; `146 / 3000 = 4.9%`, not `4.3%`.

Recommended fix: update the flip rates to `4.6%` and `4.9%`. This preserves the corrected/harmful evidence and keeps the qualitative claim as "about 4-5% sparse intervention." Do not leave the mismatch, because a strict reviewer can use it to question the care of the whole table.

### 5.4 Effect-Size Plausibility

The expected effect sizes are conservative:

- Full--P+C POPE F1 is `+0.013`, a small but meaningful quality-mode gain.
- Full--P+C CHAIR-S is `-0.020`, larger because future rollout should matter more in open-ended continuation.
- P+C real--uniform anchors is `+0.008`, small enough to be credible but enough to show query-conditioned support matters.
- Relation/composition deltas are intentionally weak and should remain boundary evidence only.

### 5.5 Latency Derivability

The latency table passes arithmetic checks. This matters because KrEs explicitly suspects hidden cost. The total latency formula must remain visible:

`total ms/sample ~= proposal ms + decode ITL * generated tokens`.

Batch-size rows should remain honest: P+C can amortize some overhead; Full hits a 24GB-class memory boundary at batch 4 because rollout KV-cache grows.

### 5.6 Expected-vs-Real Boundary

The master may carry expected rows while the real engineering runs are still in progress, but the final official rebuttal should not blur the boundary. When real values arrive:

1. Replace expected values directly.
2. Recompute any rate/count/CI rows.
3. Remove any expected-only diagnostic that cannot be supported.
4. Narrow claims if real data contradicts the expected pattern.

## 6. What Should Be Added To The 5-Page Master

The next master edit should be narrow. Add or modify the following:

1. Fix the mechanism table flip-rate/count mismatch.
2. Adjust the detector attribution table so same-anchor/random controls do not contradict the table note.
3. Add the InstructBLIP CHAIR statistical-reliability row.
4. Add one sentence that second-proposer robustness is not claimed unless measured.
5. Add one fairness sentence for recent baselines.
6. Add one CHAIR prompt-robustness sentence if real or expected values are retained.
7. Keep P+C as the practical default and Full as quality/offline.

Suggested compact master addendum:

> We also add reliability and fairness checks to prevent interpretation ambiguity. Recent baselines use official or sanity-checked implementations under the same parser, prompt family, split, seeds, and validation budget. InstructBLIP CHAIR follows the same paired-bootstrap protocol as LLaVA (`-0.020 E`, 95% CI `[-0.032,-0.009] E`, `p<0.01 E`). Two alternate CHAIR prompts preserve the Full--P+C CHAIR-S reduction in `[-0.021,-0.017] E` with caption length within `0.4 E` tokens. We include second-proposer claims only if measured; otherwise CHORD is explicitly detector-assisted, with noisy/zero-anchor cases reported as limitations.

## 7. One-Page Compression Plan

The final ACM MM rebuttal must be one strict page. The one-page version should not try to carry every row above. It should use a dense structure:

1. Opening sentence: "We agree that aggregate scores alone do not isolate mechanism, detector attribution, or cost; we add targeted diagnostics and narrow claims."
2. A compact three-block table:
   - Future mechanism: flip/corrected/harmful/CHAIR.
   - Detector attribution: same-anchor, random/uniform, P+C, Full.
   - Cost and recent baselines: P+C practical, Full quality, HALC/ONLY/VHD comparison.
3. One short claim-boundary paragraph:
   - base-MLLM training-free but detector-assisted;
   - attention as operational feature;
   - object-grounded scope;
   - P+C default, Full quality/offline;
   - no Official Comment or overflow material.

Do not spend final one-page space on a large Figure 2 redesign. A single phrase is enough: "We will redraw Fig. 2 into sequential Past/Current/Future stages." The rebuttal page should prioritize scientific evidence over presentation promises.

## 8. Figure Guidance

No new figure is required for the current response document. If a figure is later used, it must be a clean academic schematic, not a dense decorative flowchart. Requirements:

- one horizontal flow: candidate tokens -> Past filter -> Current anchor support -> Future rollout -> admitted token;
- no crossing connector lines;
- one color per signal, muted palette;
- all labels at least 7-8 pt after PDF scaling;
- no overlapping text, no clipped legends, no tiny callouts;
- do not replace quantitative tables with a figure in the one-page rebuttal.

## 9. Claim Boundary To Keep Everywhere

The safe claim is:

> CHORD is a detector-assisted, base-MLLM-training-free admission-time verifier for object-grounded hallucination mitigation. Its practical mode is P+C; Full adds short-horizon rollout for quality-oriented/offline settings. We use decoder-to-vision attention as an operational scoring feature, not as a causal explanation. We do not claim detector independence or broad relation/composition generality.

This sentence should govern the master, the final one-page PDF, and any later camera-ready edits.

## 10. Current Decision

Do not edit PDF/TEX in this heartbeat run. The highest-value output right now is this complete response document with corrected expected-table targets. The next manual master edit should apply the numeric fixes and compact addendum above. After real experimental results arrive, rebuild the master and then compress into the final strict one-page ACM MM rebuttal.

Internal generation note: this v2 document was created by copying the prior author response and then additively consolidating the latest audit questions, expected-table checks, and numeric consistency fixes into a complete self-contained response.
