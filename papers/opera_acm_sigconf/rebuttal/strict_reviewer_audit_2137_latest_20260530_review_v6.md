# Strict Reviewer Audit 21:37 Latest 20260530 Review v6

## Primary Input And Evidence Boundary

Primary input audited:

- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2132_review_v6.md`

Context files used:

- `papers\opera_acm_sigconf\rebuttal\reviewer_true_intent_analysis_20260529.md`
- `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_2134_latest_20260530_review_v6.md`
- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf`
- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.tex`

Verification commands:

```powershell
Get-Date -Format 'yyyyMMdd_HHmm'; Get-Date -Format 'HHmm'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md' | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime,Length
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'strict_reviewer_audit_*_latest_*.md' | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime,Length
Select-String -Path papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2132_review_v6.md -Pattern '^#|^##|Working Draft v6|No New Audit|Immediate Next Step|Final One-Page|Evidence Eligibility|Pre-Submission Stop Rule|Current Decision|Future|Detector|detector|P\+C|Full|baseline|Expected-Table'
pdfinfo papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf
```

Evidence boundary:

- This is a fresh 21:37 reviewer heartbeat audit, not a skipped duplicate.
- No newer `author_response_min_diff_expected_*_review_v*.md` exists after `author_response_min_diff_expected_20260530_2132_review_v6.md`.
- No newer timestamped PDF/TEX exists after the 17:44 five-page scientific master.
- The input is materially unchanged since the 21:34 audit, so score movement is intentionally unchanged.
- I did not generate an author response, edit/copy/compile PDF/TEX, validate raw experiment logs, or submit anything to OpenReview.

## Overall Reviewer Verdict

`review_v6` Markdown response-control quality: **4/5**.

Current 17:44 scientific master quality: **4/5 as an internal five-page master**.

Current official one-page PDF/TEX readiness: **3/5 Borderline**.

As the strict reviewer, I do not raise the score in this heartbeat because the primary input has not changed since the 21:34 audit. The v6 document is useful because it correctly stops broad response expansion and says the next meaningful artifact is the strict one-page rebuttal PDF/TEX. But the official reviewer-facing risk remains: the actual one-page artifact is still absent, and the hardest empirical concerns from KrEs and M8du remain gated on measured Future and detector-attribution evidence. The response strategy is now good; the submittable artifact is still not proven.

## Reviewer-by-Reviewer Score Movement

| Reviewer | Original true concern | Already satisfied in v6 | Still unresolved | Likely score after current state | Most score-moving action |
|---|---|---|---|---:|---|
| jjVG | Cost, `k/m`, Figure 2 clarity, recent related work. | v6 preserves P+C/Full operating regimes, a compact `k/m` Pareto sentence, recent-baseline fairness wording, and deprioritizes Figure 2 relative to evidence. | The final one-page PDF has not shown these compressed answers. Recent-baseline numbers are not safe unless measured. | **3 to 4**. | Put the `k/m` Pareto sentence, cost/default sentence, and ONLY/VHD/HALC fairness sentence into the actual one-page. |
| KrEs | Novelty, Grounding DINO attribution, end-to-end efficiency, strong baselines/backbones. | v6 uses detector-assisted boundaries and requires measured attribution rows before numeric claims. | There is no measured detector attribution in the current official PDF/TEX; recent-baseline provenance is still not visible. | **3**. | Include a measured detector-control row or make the detector-assisted limitation explicit in final one-page prose. |
| yx8u | Incremental novelty, detector dependency, attention reliability, generality, cautious claims. | v6 is strong on claim discipline: base-MLLM-training-free but detector-assisted, attention operational not causal, object-grounded scope. | Compression could still drop the boundary sentence. | **4**. | Preserve the boundary sentence in the final one-page. |
| ve3y | Practical value despite runtime overhead. | v6 clearly says P+C is practical/default and Full is quality/offline. | Actual one-page needs measured or clearly bounded cost accounting. | **4 if preserved; 3/4 if omitted**. | Keep P+C/Full cost framing in the official one-page. |
| M8du | Future term mechanism, flip correctness, detector attribution, hyperparameter sensitivity, transfer beyond object-level. | v6 gives Future mechanism top priority and forbids expected flip counts from being treated as facts. | It still does not answer with measured flip frequency/correctness. | **3 to weak 4**. | The first numeric one-page row should be measured Full-vs-P+C flip/correctness, or absent with explicit wording-only diagnostic framing. |

## Coverage Matrix Against True Intent

| True-intent area | Status | Strict evidence |
|---|---|---|
| Mechanism / Future effectiveness | **Partially resolved** | v6 prioritizes Future flip/correctness and gives safe wording if unmeasured. It does not provide measured closure. |
| Grounding DINO / detector attribution | **Partially resolved** | v6 gives the right controls and boundary. It does not prove attribution with measured rows. |
| Efficiency / cost / P+C versus Full default | **Mostly resolved as wording, partially as evidence** | P+C practical/default and Full quality/offline are clear. Proposal/total/VRAM/batch numbers remain unsafe unless measured. |
| Recent baselines and fairness | **Partially resolved** | The matched-protocol wording is good. Numeric claims over ONLY/VHD/HALC remain unsafe unless real matched runs exist. |
| Claim scope / generality / novelty / attention | **Mostly resolved** | The response now uses the correct conservative scope. It cannot fully solve moderate novelty or limited generality. |

## Unresolved Problems

1. **No strict one-page PDF/TEX exists yet.**
   This is the central blocker. Reviewers will not see v6; they will see the official one-page response.

2. **Future mechanism remains unmeasured in the current artifact.**
   M8du's most concrete question remains open unless paired Full-vs-P+C flip/correctness logs are available and included.

3. **Detector attribution remains empirically open.**
   KrEs can still argue that the control design is not the same as measured attribution. Same-anchor, random/uniform anchors, no-Current, P+C, and Full need measured evidence or a strict limitation sentence.

4. **Expected numbers remain dangerous.**
   v6 correctly says they must not leak into the official page. The final PDF must prove this discipline by excluding exact expected CI, p-values, VRAM, batch, flip counts, and recent-baseline wins unless measured.

5. **More author-response Markdown expansion is now counterproductive.**
   v6 already states this. A v7 without new measured evidence or a new reviewer audit would not improve reviewer confidence.

6. **The current compiled PDF context is still five pages.**
   The 17:44 master is useful internally but does not satisfy the PC rule. Official readiness remains capped at 3/5.

## Follow-up Questions For The Author Team

1. What is the exact path of the first strict one-page rebuttal TEX/PDF draft?

2. Which rows in that one-page draft are measured, which are wording-only, and which are dropped?

3. If Future flip/correctness is not measured, will the official page remove all exact flip percentages, corrected/harmful counts, confidence intervals, and p-values?

4. If detector controls are not measured, will the official page explicitly state that CHORD is detector-assisted and does not claim detector independence?

5. Is proposal time measured, or is the only safe timing evidence still submitted decode ITL? The final page must not mix these.

6. Are ONLY/VHD/HALC results measured under official/sanity-checked implementations and matched protocol? If not, will the final page use fairness wording only?

7. Will the one-page PDF preserve P+C practical/default and Full quality/offline?

8. Should the reviewer heartbeat keep producing new audits before a new author artifact appears? From a reviewer standpoint, the next audit should ideally target the one-page draft, not another unchanged v6.

## Expected Table And Numeric Plausibility Check

The v6 expected-table policy remains plausible and conservative as internal planning. It is not final evidence.

Checks passed:

- Future flip rates are internally coherent at `4.6%` and `4.9%` for the stated counts.
- P+C and Full expected values remain consistent across tables.
- Detector attribution ordering is conservative: noisy/no-current controls below real-anchor P+C, Full strongest but more expensive.
- Recent-baseline expectations are modest enough not to look like implausible domination.
- Cost arithmetic remains derivable from proposal time plus decode ITL times token count.

Remaining numeric risks:

- CI/p-value/sample-size claims are not safe unless backed by real paired/bootstrap runs.
- Proposal time, VRAM, and batch-size behavior must be measured, not inferred.
- Recent-baseline wins require implementation provenance.
- The final one-page should carry fewer exact numbers than the master, not more.

Required final-page numeric rule:

| Candidate content | Final-page action |
|---|---|
| Measured Future flip/correctness | Include first. |
| Expected Future flip/correctness | Wording-only diagnostic; no exact numbers. |
| Measured detector controls | Include compact attribution row. |
| Expected detector controls | Control design plus detector-assisted boundary only. |
| Existing decode ITL | Include only if labeled as decode ITL. |
| Expected proposal/VRAM/batch | Drop unless measured. |
| Measured recent baselines | Include only with provenance and matched protocol. |
| Expected recent baselines | Use fairness-protocol wording only. |

## One-page Rebuttal Compression Risk

The one-page compression risk remains the main risk. v6 is persuasive because it is long and procedural; the official rebuttal must be short and evidence-selective.

Must be preserved in one page:

- A concession that aggregate scores alone did not isolate mechanism, attribution, or cost.
- One compact table with at most three blocks: Future mechanism, detector attribution, cost/baseline fairness.
- P+C as practical/default and Full as quality/offline.
- Detector-assisted, base-MLLM-training-free boundary.
- Attention as operational scoring feature, not causal explanation.
- No Official Comment or hidden overflow.

Should be minimized:

- Figure 2 redesign discussion.
- Long reviewer-by-reviewer prose.
- Extended expected-table explanations.
- Exact expected statistics.

Strict reviewer view: a narrow one-page with one measured diagnostic and honest boundaries is stronger than a dense page that mixes expected values with final claims.

## Next Required Action

Create the strict one-page rebuttal PDF/TEX draft. Do not generate another broad author response unless new measured evidence or a new strict reviewer audit changes the input state.

If the one-page cannot be generated immediately, create a one-page eligibility ledger with only three labels:

- `include measured`;
- `include wording only`;
- `drop unless measured`.

The next reviewer audit should ideally target that one-page draft. Re-auditing unchanged v6 can satisfy the heartbeat rule, but it does not improve the paper.

## LOCAL_TASKS Update

This audit closes `Reviewer heartbeat audit at 21:37` in `LOCAL_TASKS.md`.

Output path:

- `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_2137_latest_20260530_review_v6.md`

Primary input:

- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2132_review_v6.md`

Evidence boundary:

- Fresh audit completed despite unchanged v6 input.
- Latest PDF/TEX context remains the 17:44 five-page master.
- No PDF/TEX edit, no author-response generation, no experiment rerun, and no OpenReview action.
