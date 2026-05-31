# Strict Reviewer Audit, 2026-05-30 21:23, review_v4

## Primary Input And Evidence Boundary

Primary input audited: `author_response_min_diff_expected_20260530_2024_review_v4.md`.

Reviewer-demand contract: `reviewer_true_intent_analysis_20260529.md`.

Current scientific master context: `author_response_min_diff_expected_20260530_1744.pdf` and `author_response_min_diff_expected_20260530_1744.tex`.

Previous reviewer audit context: `strict_reviewer_audit_1803_latest_20260530_review_v3.md`.

Evidence commands used:

```powershell
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'strict_reviewer_audit_*_latest_*.md'
pdfinfo papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf
rg -n "Final One-Page Eligibility Matrix|include measured|include wording only|Drop from final|Final One-Page Payload Rule|Current Decision|measured|expected|pending|ONLY|VHD|HALC|Future|detector|P\+C|Full|Official Comment" papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2024_review_v4.md papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.tex
rg -n "Hard Answer|Can This Document Solve|Shared True Needs|Mechanism validation|Detector/Grounding DINO attribution|End-to-end cost|Missing ONLY|Novelty is incremental|Reviewer jjVG|Reviewer KrEs|Reviewer yx8u|Reviewer ve3y|Reviewer M8du" papers\opera_acm_sigconf\rebuttal\reviewer_true_intent_analysis_20260529.md
```

Boundary: `review_v4` is still an author-side planning/decision document, not a final one-page rebuttal. It does not add new measured experimental evidence. The 17:44 scientific master remains the latest compiled PDF and is a 5-page internal master, not the official one-page ACM MM rebuttal. This audit does not independently validate raw experiment logs and does not edit PDF/TEX.

## Overall Reviewer Verdict

Current `review_v4` Markdown quality: **4/5 Weak Accept as an internal response-control document**.

Current 5-page scientific master quality: **4/5 as an internal scientific master**.

Current official one-page readiness: **3/5 Borderline, possibly weak 4 only after an actual one-page evidence-eligible draft exists**.

The v4 response makes the right strategic move: it stops expanding expected tables and instead introduces a final one-page eligibility matrix. That directly addresses the previous audit's central risk, namely that precise expected values could leak into the final official rebuttal as if they were measured. However, from a strict reviewer standpoint, v4 still does not itself answer the empirical questions. It says which values should be included only if measured; it does not show that they are measured. Therefore the score does not move above Weak Accept for the internal response, and the actual official-rebuttal readiness remains Borderline until a one-page draft exists.

## Reviewer-by-Reviewer Score Movement

| Reviewer | Original real concern | What v4 satisfies | What remains unresolved | Likely score after current state | Most useful action |
|---|---|---|---|---:|---|
| jjVG | Completeness: k/m, cost, recent related work, figure clarity. | v4 preserves the k/m, cost, and recent-baseline answers while correctly saying expected numeric tables should not all enter the one-page rebuttal. | If the final page drops k/m and recent related-work signals completely, jjVG may still see visible omissions. | 4 if final page keeps a compact k/m plus baseline fairness sentence; 3 otherwise. | Put one Pareto/default sentence and one ONLY/VHD/HALC fairness sentence into the final page. |
| KrEs | Attribution, novelty, Grounding DINO, full cost, stronger baselines. | v4 is honest: measured detector-control rows are required for numeric claims; otherwise only control design and detector-assisted boundary should be included. | No measured detector-control proof is shown in v4; no second real proposer; no implementation provenance for recent baselines. | 3; weak 4 only with measured detector/baseline rows. | Give KrEs measured attribution rows or remove numeric wins and state detector-assisted limitation explicitly. |
| yx8u | Claim discipline, detector dependence, attention reliability, scope. | Strongly satisfied. v4 forces the final page to keep the detector-assisted/base-MLLM-training-free/object-grounded/operational-attention boundary. | The risk is accidental overcompression: if the final page drops this boundary, support weakens. | 4. | Preserve the exact boundary sentence in the one-page rebuttal. |
| ve3y | Practical value under overhead. | P+C practical vs Full quality/offline remains clear; v4 prevents unmeasured VRAM/batch estimates from being oversold. | The final page still needs either measured end-to-end cost or a transparent accounting commitment. | 4 if cost boundary is retained; 3/4 if not. | Include proposal+decode accounting and P+C default in the final page. |
| M8du | Future mechanism, corrected/harmful flips, detector controls, hyperparameter sensitivity. | v4 prioritizes Future mechanism as the first evidence block and states exact counts must be included only if measured. | This is still not the measured flip/correctness evidence M8du asked for. If no measured row appears, M8du may stay Borderline. | 3/4; 4 only with measured Full-vs-P+C flip/correctness. | Make the first final-page evidence row a measured Full-vs-P+C flip/correctness row, or explicitly admit it is pending. |

## Coverage Matrix Against True Intent

| Gate | Status | Evidence | Reviewer risk |
|---|---|---|---|
| Mechanism/Future effectiveness | **Partially resolved** | v4 says Future flip/correctness rows are final-page eligible only if measured, and otherwise should be phrased as diagnostic design. | Correct policy, but not empirical closure. M8du asked for actual flip frequency and correctness. |
| Grounding DINO and detector attribution | **Partially resolved** | v4 keeps same-anchor/uniform/random/no-Current controls as the right attribution design and refuses detector-independent claims without evidence. | KrEs can still say attribution is planned, not proven, unless measured rows are available. |
| Efficiency/cost and P+C/Full default | **Mostly resolved** | v4 preserves P+C as practical/default and Full as quality/offline; it blocks unmeasured VRAM/batch rows from becoming final claims. | If proposal time is not measured, final page must avoid exact total-latency claims beyond submitted decode ITL. |
| Recent baselines and fairness | **Partially resolved** | v4 says ONLY/VHD/HALC numeric wins must be omitted unless measured and provenance-backed; otherwise use fairness protocol wording. | This is honest but less persuasive. jjVG/KrEs may still want direct numbers. |
| Claim scope, generality, novelty, attention wording | **Mostly resolved** | v4 keeps the safest claim boundary and avoids broad relation/composition or causal-attention claims. | Novelty remains moderate and cannot be fully solved in rebuttal. |

## Unresolved Problems

1. **No final one-page rebuttal exists yet.** v4 is a correct decision document, but the official artifact still needs to be written and audited. This is now the main operational blocker.

2. **Measured-vs-expected eligibility is still unresolved at the row level.** v4 provides the rule, but not the row-by-row real status. The author team must now mark each candidate row as actually measured, wording-only, or dropped.

3. **KrEs remains the hardest reviewer.** Without measured detector-control rows and recent-baseline provenance, KrEs can maintain the view that CHORD is a detector-assisted engineering combination with insufficient attribution evidence.

4. **M8du still needs actual mechanism evidence.** A diagnostic design is not enough. The exact Full-vs-P+C flip/correctness row either needs real logs or should not be presented numerically.

5. **One-page compression could become too weak.** If all key numeric rows are unavailable, the final page may become mostly promises and boundaries. That is honest, but the likely score then stays Borderline to Weak Accept rather than improving.

6. **The author/reviewer automation loop can produce more planning documents instead of convergence.** At this stage, further expansion is less valuable than one-page eligibility and final-page drafting.

## Follow-up Questions For The Author Team

1. Which of the three evidence blocks can be backed by measured logs today: Future mechanism, detector attribution, and cost/recent-baseline fairness?

2. For each final-page candidate row, what is the status: `include measured`, `include wording only`, or `drop`?

3. If Future flip/correctness is not measured, what is the exact wording that avoids pretending the mechanism has been empirically validated?

4. If detector attribution rows are not measured, will the final page still explicitly say "detector-assisted" and "we do not claim detector independence"?

5. Are ONLY/VHD/HALC direct numbers measured with implementation provenance? If not, the final page should not report numeric superiority over them.

6. What is the final one-page draft file path and when will it be audited? This is now more important than producing v5.

7. If only one numeric block can fit, which one will be prioritized? My recommendation remains Future mechanism first, then detector attribution, then cost/baseline fairness.

8. Will the final page preserve P+C as the practical default and Full as quality/offline? Dropping this distinction would reopen ve3y/yx8u concerns.

## Expected Table And Numeric Plausibility Check

v4's numeric policy is correct and safer than v3. It does not introduce new numeric contradictions. It correctly prevents expected values from being used as final claims unless real logs support them.

What passes:

- Future target arithmetic remains coherent: `4.6%` for `96+42` over 3000 and `4.9%` for `102+44` over 3000.
- Detector-control ordering remains conservative and internally consistent.
- Cost arithmetic remains derivable from proposal time plus decode ITL times token count.
- Recent-baseline expected ordering is modest, with P+C near strong baselines and Full stronger but slower.
- The final one-page eligibility matrix is the right safeguard against overclaiming.

Remaining numeric risks:

- p-values, confidence intervals, VRAM/batch rows, recent-baseline latencies, and detector-control deltas are too precise for the official page unless measured.
- If the final page uses only wording and no measured numeric diagnostic, the response becomes safer but less score-moving.
- If measured results differ from expected tables, the expected tables must be replaced rather than reconciled rhetorically.

Required final numeric policy:

| Candidate evidence | Reviewer-safe final-page action |
|---|---|
| Measured Future flip/correctness | Include exact row if logs exist. |
| Expected Future flip/correctness | Mention diagnostic design only; no exact counts/p-values. |
| Measured detector controls | Include compact attribution row. |
| Expected detector controls | Include control design plus detector-assisted boundary only. |
| Measured cost | Include proposal+decode, P+C default, Full quality/offline. |
| Expected cost | State full accounting will be reported; avoid exact total/VRAM claims. |
| Measured recent baselines | Include compact fair-comparison row. |
| Expected recent baselines | Use fairness protocol wording; no numeric wins. |

## One-page Rebuttal Compression Risk

This remains the dominant risk. v4 correctly says the final page should not be a compressed five-page master. I agree with the payload rule, but I would make it even stricter:

1. **One sentence of concession and scope.** Aggregate scores alone did not isolate mechanism; we add targeted diagnostics and narrow claims.
2. **One compact evidence table.** No more than three blocks: mechanism, attribution, cost/baseline.
3. **One boundary sentence.** Detector-assisted, base-MLLM-training-free, P+C practical, Full quality/offline, attention operational.

If a block has no measured numbers, use wording only and do not use exact expected values. The final page should be less ambitious and more credible rather than dense with unsupported expected numbers.

## Next Required Action

Do not produce a broader `review_v5` unless it directly creates the final one-page payload decision. The next author-side task should be one of:

1. Create a final one-page rebuttal draft using only evidence-eligible content; or
2. Create a short final-page eligibility sheet with every row marked `include measured`, `include wording only`, or `drop`, then immediately draft the one-page PDF/TEX.

From a reviewer standpoint, the paper is now in a convergence phase. More internal expected-table refinement will not materially raise scores. Only measured mechanism/attribution evidence and a clean one-page final artifact can.

## LOCAL_TASKS Update

This audit should close the `Manual recovery reviewer audit of response v4 at 21:23` task in `LOCAL_TASKS.md`.

Output path: `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_2123_latest_20260530_review_v4.md`.

Primary input: `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2024_review_v4.md`.

Context PDF/TEX: `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf` and `.tex`.

Evidence boundary: this audit judges the latest author-response Markdown and compiled 17:44 scientific master against the five-reviewer intent map. It does not validate raw experiment logs, does not edit PDF/TEX, and does not submit anything to OpenReview.
