# Strict Reviewer Audit 22:25 Latest 20260530 Review v8

## Primary Input And Evidence Boundary

Primary input audited:

- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2221_review_v8.md`

Official one-page draft audited:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf`

Context files used:

- `papers\opera_acm_sigconf\rebuttal\reviewer_true_intent_analysis_20260529.md`
- `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_2221_latest_20260530_review_v7.md`
- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.pdf`
- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1744.tex`

Verification commands and checks:

```powershell
Get-Date -Format 'yyyyMMdd_HHmm'; Get-Date -Format 'HHmm'
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_min_diff_expected_*_review_v*.md' | Sort-Object LastWriteTime -Descending
Get-ChildItem papers\opera_acm_sigconf\rebuttal -Filter 'author_response_onepage_expected_20260530_2221*' | Sort-Object LastWriteTime -Descending
pdfinfo papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf
pdftotext -layout papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf -
Select-String -Path papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.log -Pattern 'Overfull|Underfull|Warning|Error|Output written|Rerun'
pdftoppm -f 1 -l 1 -png -r 140 papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf $env:TEMP\chord_onepage_2221_preview
```

Visual/text boundary:

- `pdfinfo` confirms the one-page draft has **1 page**.
- The LaTeX log shows `Output written` and no captured overfull/underfull warning in the inspected warning pattern.
- The rendered preview is readable and not clipped. The table is dense but visually coherent.
- `pdftotext -layout` confirms the page contains the key blocks: mechanism, detector attribution, cost/baselines/hyperparameters, P+C/Full default, detector-assisted boundary, no hidden overflow.

Evidence boundary:

- This audit does not validate raw experimental logs.
- The one-page PDF intentionally reports no exact unverified mechanism counts, detector-control deltas, confidence intervals, p-values, VRAM, batch behavior, or recent-baseline wins.
- I did not edit/copy/compile PDF/TEX and did not submit anything to OpenReview.

## Overall Reviewer Verdict

`review_v8` Markdown response-control quality: **4/5**.

Official one-page PDF/TEX readiness: **4/5, weak-accept-level rebuttal artifact but evidence-limited**.

This is the first state where official readiness clearly improves from the prior **3/5 Borderline**. The reason is not that the scientific evidence became stronger; it is that the rebuttal is now a real one-page PDF, is self-contained, obeys the PC one-page constraint, and avoids presenting unverified target values as facts. It directly addresses all five reviewer categories at least at the level of claim discipline and diagnostic design. The cap remains that KrEs and M8du asked for measured attribution/mechanism evidence, while the current one-page mostly provides protocols and boundaries. Therefore I would not call this high-confidence acceptance material, but it is now a credible, safe official rebuttal draft.

## Reviewer-by-Reviewer Score Movement

| Reviewer | Original true concern | What the one-page now satisfies | Still unresolved | Likely score after current one-page | Most score-moving action |
|---|---|---|---|---:|---|
| jjVG | Cost, `k/m`, Figure 2 clarity, recent work. | The page includes cost/default framing, a compact `k/m` rationale, recent-baseline fairness wording, and a Figure 2 cleanup phrase. | No direct recent-baseline numbers; no full `k/m` table. | **4**. | If space allows, preserve the current `k/m` and baseline sentence exactly; do not drop them. |
| KrEs | Novelty, Grounding DINO attribution, end-to-end efficiency, stronger baselines/backbones. | The page explicitly states detector-assisted scope and includes the detector-control design. It does not overclaim detector independence. | No measured attribution row; no second proposer; no measured recent-baseline comparison. | **3, maybe weak 4 if they value the claim boundary**. | Add one measured attribution row if real logs exist; otherwise keep the current limitation. |
| yx8u | Incremental novelty, detector dependency, attention reliability, generality, cautious claims. | Strongly satisfied. The page is cautious, uses operational-attention wording, and avoids broad detector-independent/general claims. | Main risk is only if later edits reintroduce broad claims. | **4**. | Freeze the boundary language. |
| ve3y | Runtime/deployment practicality. | P+C is practical/default and Full is quality/offline; the page does not pretend Full is cheap. | Exact proposal/total cost is not measured in the page. | **4**. | Keep decode/proposal distinction and avoid unmeasured totals. |
| M8du | Future mechanism, flip correctness, detector attribution, hyperparameter sensitivity, transfer beyond object-level. | Future is the first row and the page explains the exact diagnostic. It avoids unsupported counts. | No measured flip/correctness answer is present. Mechanism is therefore partially, not fully, answered. | **3 to weak 4**. | Include real Full-vs-P+C flip/correctness if available; otherwise accept limited score movement. |

## Coverage Matrix Against True Intent

| True-intent area | Status | Strict evidence |
|---|---|---|
| Mechanism / Future effectiveness | **Partially resolved** | The one-page has a Future mechanism row and measured-only rule. It does not provide measured flip/correctness. |
| Grounding DINO / detector attribution | **Partially resolved** | The page has fixed-anchor/random/no-Current/P+C/Full control design and detector-assisted boundary. It lacks measured attribution values. |
| Efficiency / cost / P+C versus Full default | **Mostly resolved** | P+C practical/default and Full quality/offline are explicit. Exact end-to-end cost remains measured-only. |
| Recent baselines and fairness | **Partially resolved** | ONLY, VHD/VHR, and HALC are named under matched-protocol wording. No numeric superiority is claimed. |
| Claim scope / generality / novelty / attention | **Resolved as wording, partly unresolved as evidence** | The page is appropriately cautious. It cannot fully solve moderate novelty or limited scope. |

## Unresolved Problems

1. **The one-page is safe but mostly non-numeric.**
   This protects credibility, but it limits score movement for KrEs and M8du.

2. **Future mechanism is still a diagnostic promise unless logs exist.**
   M8du asked for how often Future flips decisions and whether the flips are correct. The current page says how the authors will measure it, not the measured answer.

3. **Detector attribution remains a design-level answer.**
   KrEs may accept the boundary as honest, but can still say the causal attribution is not empirically closed.

4. **Cost is framed, not fully quantified.**
   The page separates proposal cost from decode ITL, but does not give measured proposal/total/VRAM/batch values.

5. **The table is dense.**
   Visual inspection shows it is readable and not clipped, but it is close to the upper limit of table density. Later additions could easily make it unreadable.

6. **Figure 2 phrase is low value.**
   If space becomes tight, cut the Figure 2 sentence before cutting mechanism, detector, cost/default, or boundary wording.

## Follow-up Questions For The Author Team

1. Are any measured Full-vs-P+C flip/correctness logs available before final submission? If yes, replace the mechanism protocol text with one compact measured row.

2. Are any measured detector-control outputs available? If yes, include one compact attribution ordering; if no, keep the detector-assisted limitation exactly.

3. Is proposal time measured, or should the final page keep only decode-ITL-safe wording?

4. Are ONLY/VHD/VHR/HALC matched outputs actually available? If no, keep the current fairness-protocol wording and no numeric superiority.

5. Can the table be shortened by cutting the Figure 2 sentence or reducing protocol prose if measured rows are added?

6. Will the team freeze this one-page draft unless real measured values arrive? My reviewer recommendation is yes.

## Expected Table And Numeric Plausibility Check

The official one-page avoids exact unverified target values. This is the strongest numeric decision in v8.

Internal planning targets remain plausible because:

- Full is only modestly stronger than P+C and slower.
- Future effects are framed as sparse and quality-oriented.
- Detector-control ordering is conservative.
- Recent-baseline framing is modest and matched-protocol based.
- Cost totals are not used without measurement.

Reviewer-facing numeric policy is satisfied:

| Candidate content | Current one-page action | Reviewer judgment |
|---|---|---|
| Future flip/correctness | Protocol only unless logs are complete | Safe, but not fully score-moving. |
| Detector controls | Control design plus limitation | Safe, but attribution remains partial. |
| Cost | Proposal/decode separation and P+C/Full regimes | Good framing; measured totals still absent. |
| Recent baselines | Named matched protocol only | Honest; direct comparison still limited. |
| Scope/attention | Explicit boundary | Strong. |

No exact unverified statistics should be added unless real logs are ready.

## One-page Rebuttal Compression Risk

The one-page now exists and is compliant at the page-count level. Remaining compression risk is moderate:

- The table is dense but readable.
- The bottom boundary paragraph is readable and not clipped.
- There is unused lower-page whitespace, but horizontal table density is still the limiting factor.
- Adding measured rows without shortening prose will likely hurt readability.

If adding numbers, shorten:

- Figure 2 sentence;
- repeated "if logs are complete" wording;
- long recent-baseline protocol phrase.

Do not cut:

- detector-assisted boundary;
- P+C practical/default and Full quality/offline;
- measured-only rule;
- attention-as-operational-feature wording.

PC-rule compliance:

- The PDF is one page and self-contained.
- It does not rely on Official Comment or supplement.

## Next Required Action

If no real measured results are available, the current one-page PDF is close to freeze-ready after minor wording polish.

If real measured results are available, update the one-page in this priority order:

1. measured Future flip/correctness;
2. measured detector-control attribution;
3. measured proposal/total cost;
4. measured recent-baseline comparison.

Do not create another broad `review_v9` unless it directly updates the one-page PDF/TEX or adds real measured evidence.

## LOCAL_TASKS Update

This audit closes `Manual reviewer audit at 22:25` in `LOCAL_TASKS.md`.

Output path:

- `papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_2225_latest_20260530_review_v8.md`

Primary input:

- `papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_2221_review_v8.md`

Official one-page artifacts:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf`

Evidence boundary:

- One-page PDF was checked with `pdfinfo`, `pdftotext`, LaTeX log warnings, and a rendered preview.
- The PDF is one page.
- No raw experimental logs were validated.
- No PDF/TEX edit, author-response generation, experiment rerun, or OpenReview action was performed by this reviewer audit.
