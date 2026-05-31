# Final Rebuttal Reviewer Audit, 2026-05-29

Audited artifact: `author_response_min_diff_expected_20260529.pdf`

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

## Bottom-Line Verdict

If this PDF is submitted exactly as-is, I would rate the response as **Borderline / 3 out of 5**, not Weak Accept.

Reason: the document is strategically well aligned with the reviewers' real concerns, but it still contains explicit expected-result placeholders rather than measured evidence. As a reviewer, I would read the structure as promising and unusually honest, but I could not treat Tables 1--4 as evidence. The strongest skeptical reviewers, especially M8du and KrEs, asked for diagnostics, not proposed table shapes.

If all expected values are replaced by real run-backed values and the values remain close to the current expected pattern, the same rebuttal could plausibly support a **Weak Accept / 4 out of 5** outcome from the overall reviewer coalition.

## Evidence Seen In The PDF

- The PDF is 3 pages and titled `Author Response for Submission 8826: CHORD`.
- It explicitly focuses on five concerns: Future mechanism, detector attribution, end-to-end cost, k/m robustness, and related-work/claim calibration.
- It correctly softens the novelty claim: CHORD is framed as a detector-assisted admission-time verifier, not as a fully new detector-free decoding paradigm.
- It correctly states that CHORD is training-free for the base MLLM, but not detector-free.
- It correctly treats decoder-to-vision attention as an operational scoring feature rather than a causal explanation.
- It includes four core diagnostic tables, but each table contains `E` marked expected placeholders and notes saying that engineer-run SSH outputs should replace them.

## Reviewer-By-Reviewer Judgment

| Reviewer | Original score | Real acceptance gate | As-is coverage | Likely movement |
|---|---:|---|---|---|
| jjVG | 3 Borderline | Completeness: k/m, recent work, cost, Figure 2 | Mostly covered structurally, but k/m and cost values are expected; recent work is promised, not compared | 3 -> 3 or weak 4 only if they accept promises |
| KrEs | 2 Weak Reject | Attribution: prove gains are not just Grounding DINO plus known decoding, and report true cost | Correctly targets the issue, but no measured detector controls or end-to-end cost yet | 2 -> 2 or 3; unlikely 4 as-is |
| yx8u | 4 Weak Accept | Claim discipline: detector-assisted wording, attention caution, scope honesty | Strong coverage; the tone is well calibrated | 4 stays 4, unless expected placeholders reduce trust |
| ve3y | 4 Weak Accept | Practical value: honest quality-latency frontier | Good framing with P+C vs Full, but latency/VRAM are not measured | 4 probably stays 4 |
| M8du | 3 Borderline | Mechanism: how often Future flips P+C, whether flips are correct, and detector controls | The right table is present, but it is not evidence yet | 3 stays 3; this reviewer cannot upgrade without real numbers |

## What Is Actually Solved

1. The rebuttal has the right priority ordering. It leads with mechanism, attribution, cost, hyperparameters, and claim calibration instead of generic thanks or generic benchmark gains.
2. The novelty response is credible. It no longer pretends every component is novel in isolation, and it explains the contribution as coordination at token admission.
3. The training-free ambiguity is handled correctly. The response says training-free for the base MLLM, not detector-free.
4. The attention-explanation risk is handled correctly. This protects against yx8u's concern.
5. The cost story is more honest than the submitted paper. It separates P+C as practical and Full as quality-oriented.
6. The reviewer-specific close is useful. Each reviewer can see their concern acknowledged.

## What Is Not Solved Yet

### 1. Mechanism validation is still not evidence

M8du's main question was: how often does Full CHORD change the admitted token relative to Past+Current, and are those changes correct? Table 1 is exactly the right diagnostic shape, but because all key values are expected placeholders, the question is not actually answered.

As a reviewer, I would ask for raw counts, definitions, seeds, and logs before changing my score.

### 2. Detector attribution remains unproven

KrEs and M8du wanted to know whether the gains come from CHORD or from Grounding DINO. Table 2 includes the right controls: same-anchor non-CHORD, uniform anchors, random anchors, Past+Future without Current, P+C real anchors, and Full real anchors. But again, these are expected values.

The methodologically weakest row is `Same-anchor non-CHORD`: the rebuttal must define exactly how the anchor metadata is used without CHORD scoring. Otherwise a skeptical reviewer may call it an artificial baseline.

### 3. End-to-end efficiency is still incomplete

Table 3 adds proposal time, total latency, VRAM, and tokens, which is the right response. However, KrEs also asked about behavior under different batch sizes. The current table only shows batch 1. It also uses expected proposal time and VRAM.

This is a partial answer, not a full answer.

### 4. Hyperparameter robustness is not yet demonstrated

Table 4 has the right k/m sweep, but the values are expected. Also, the expected table itself may create a new issue: k=5,m=2 appears close to k=5,m=3 at lower cost. If measured values match this pattern, the final text should recommend P+C or k=5,m=2 as the practical/default setting, and present k=5,m=3 as quality-oriented.

### 5. Recent baselines are only positioned, not compared

The PDF says ONLY, VHD/VHR, and HALC will be added. That addresses jjVG's related-work complaint only partially. KrEs and M8du asked for stronger recent baselines in the experimental evaluation. If no matched comparison can be run, the rebuttal should explicitly say why direct numerical comparison is not protocol-compatible and include a compact positioning table in the response itself.

### 6. Generality remains mostly unresolved

yx8u and M8du asked about newer MLLMs and non-object hallucinations: Qwen2-VL, InternVL, LLaVA-NeXT, relation/attribute/compositional/reasoning-heavy settings. The current rebuttal mainly limits the claim to object/open-ended hallucination settings. That is honest, but it does not expand evidence.

This may be acceptable for yx8u and ve3y, but not enough for KrEs.

### 7. The wording is still too promise-heavy

The PDF repeatedly uses `we will revise`, `we will add`, `Table is designed to`, and expected-result notes. That is acceptable for an internal final template, but not for a final rebuttal response. The submitted version should sound like:

- `We ran this diagnostic...`
- `The measured results show...`
- `We will revise the camera-ready accordingly...`

not:

- `Table reports the final shape we will use...`
- `Engineer-run outputs should replace these values...`

## Questions I Would Still Ask As A Reviewer

1. Are the Table 1--4 values measured? If yes, where are the raw logs, seeds, prompts, splits, and parser definitions?
2. How exactly is `corrected / harmful` defined for Full-vs-P+C flips on POPE?
3. Are the CHAIR continuation gains statistically significant across samples?
4. How is `same-anchor non-CHORD` implemented, and why is it a fair detector-only control?
5. What fraction of samples have zero anchors, noisy anchors, or missed relevant objects, and what is performance in each stratum?
6. Does end-to-end latency include detector proposal, image preprocessing, model loading, and GPU synchronization?
7. What happens at batch sizes larger than 1?
8. Why not directly compare against ONLY, VHD/VHR, HALC, or other recent methods under the same backbone and prompt protocol?
9. If k=5,m=2 is nearly as good as k=5,m=3 with lower cost, why should k=5,m=3 remain the default?
10. Why should the result generalize beyond object-level anchors to relation, attribute, and compositional hallucinations?

## Required Last-Mile Fixes Before Submission

1. Replace all `E` expected values with measured values or remove the corresponding table rows.
2. Delete every expected-result note before final submission.
3. Add one short provenance sentence per table: dataset split, sample count, backbone, decoding budget, seeds, and where logs are archived.
4. Add batch-size evidence or explicitly say the current rebuttal only validates batch-1 latency.
5. Put the recent-method positioning table directly in the rebuttal if direct matched baselines cannot be run.
6. Change promise-heavy phrasing into measured-result phrasing wherever data are available.
7. If measured values do not pass the stated pattern, narrow the claim rather than submitting the expected story.

## Score Estimate

As-is final response: **3/5 Borderline**.

As an internal final template awaiting real values: **8/10**.

As a true final rebuttal after real values replace expected placeholders and pass the current pattern: **4/5 Weak Accept is plausible**, with the likely reviewer coalition being yx8u + ve3y + M8du or jjVG, and KrEs softened but not necessarily converted.

