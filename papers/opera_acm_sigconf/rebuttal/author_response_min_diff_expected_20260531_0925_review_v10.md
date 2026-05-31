# CHORD Author Response, All-in-One Working Draft v10

This document is written from the author-team perspective. It is a complete working response to the remaining reviewer concerns, not a reviewer score sheet. It preserves the already-solved content from v9 and answers the latest strict audit of v9 plus the 00:43 one-page PDF/TEX. That audit confirms the one-page response is weak-accept-level and compliant, but evidence-limited. Therefore v10 does not expand the scientific master, does not invent new expected numbers, and does not modify the one-page PDF/TEX without real measured evidence. The author-team decision is to freeze the 00:43 one-page draft as the current safest official rebuttal candidate, pending replacement of wording-only slots with real measurements if they arrive.

## 0G. Additions From The 03:00 Audit Of v9

The latest audit gives a stable verdict: the polished one-page response is compliant, self-contained, readable, and reviewer-safe because it avoids exact unverified statistics. The remaining weakness is not wording; it is missing measured mechanism, detector attribution, cost, and recent-baseline evidence. Our response is to freeze the current one-page draft unless real measurements arrive.

### Current Official Candidate

The current official rebuttal candidate is:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260531_0043.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260531_0043.pdf`
- visual preview: `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260531_0043_preview.png`

This candidate remains the recommended upload candidate after human review if no real measured values arrive. It should not be replaced by the older five-page scientific master.

### Direct Answers To The 03:00 Follow-Up Questions

| 03:00 follow-up question | Author-team answer | Action for one-page PDF |
|---|---|---|
| Can measured Full-vs-P+C admission logs be produced before final submission? | Not in the current author-side artifact. If the engineer provides real logs, we will replace the Future protocol text with exact flip count, corrected unsupported count, harmful count, sample size, and parser boundary. | No edit now. Keep wording-only Future diagnostic. |
| Can detector-control results be produced under fixed same-anchor, uniform/random-anchor, no-Current, P+C, and Full? | Not in the current author-side artifact. If measured outputs arrive, insert one compact attribution ordering and cut lower-priority prose. | No edit now. Keep detector-assisted limitation. |
| Is proposal time measured separately from decode ITL? | Not in the current author-side artifact. Until measured logs exist, the one-page should keep only proposal/decode separation and P+C/Full operating regimes. | No edit now. Do not add total latency, VRAM, or batch claims. |
| Are ONLY, VHD/VHR, and HALC matched outputs available with provenance? | Not in the current author-side artifact. The current matched-protocol wording is the safe maximum without logs. | No edit now. Do not add numeric superiority. |
| If one measured row arrives, what prose should be cut? | Cut Fig. 2 phrase first, then shorten repeated measured-only caveats, then compress baseline-protocol wording. Do not cut Future-first ordering, detector-assisted boundary, P+C/Full default, or operational-attention scope. | Applies only if real measured row arrives. |
| Is the final upload definitely the 00:43 one-page PDF or a later derivative? | Current recommendation: use `author_response_onepage_expected_20260531_0043.pdf` after human review unless real measurements arrive and justify a later derivative. Do not upload the five-page master. | Freeze current candidate unless measured evidence arrives. |

### Expected-Table Reasonability Check For v10

No expected table values change in v10. The latest audit explicitly validates the current numeric policy: expected values are useful internal targets but should not appear as official measured facts. The one-page PDF already follows this policy.

The expected targets remain reasonable for engineering comparison because:

- Full is expected to be slightly stronger than P+C but slower, matching the mechanism narrative.
- P+C is expected to be practical/default and close to strong baselines, not implausibly dominant.
- Relevant anchors are expected to help more than zero/noisy anchors, matching detector-attribution logic.
- Latency expectations remain derivable from proposal time plus decode ITL times token count.

But the official one-page must continue to use stricter measured-only rules:

- no Future flip/correctness counts without paired logs;
- no detector-control deltas without measured control outputs;
- no proposal/total/VRAM/batch values without measured cost logs;
- no ONLY/VHD-VHR/HALC numeric comparison without matched outputs and provenance.

### Freeze Policy

If no real measurements arrive, the next action is final human copy review and upload preparation for `author_response_onepage_expected_20260531_0043.pdf`. Further broad `review_v*` expansion is counterproductive because it increases inconsistency risk without adding evidence. If real measurements arrive, update only the one-page PDF/TEX and then create a short author-response note explaining the measured replacement.

## 0F. Additions From The 22:25 Audit Of v8

The latest audit says the one-page PDF is compliant, self-contained, readable, and safer than prior expected-table drafts because it avoids unverified exact values. Its remaining concerns are: the page is mostly non-numeric, Future and detector attribution remain protocol-level unless logs exist, and the table is dense. Our author-team response is to polish the one-page draft without adding unsupported numbers.

### New Polished One-Page Artifacts

This iteration copies the first one-page draft and produces a polished timestamped version:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260531_0043.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260531_0043.pdf`

The polish is conservative:

1. keep the one-page, self-contained structure;
2. keep Future mechanism as the first row;
3. keep detector-assisted/no-detector-independence boundary;
4. keep P+C practical/default and Full quality/offline;
5. keep ONLY/VHD-VHR/HALC as matched-protocol wording only;
6. keep exact expected statistics out of the official page.

### Direct Answers To 22:25 Follow-Up Questions

| 22:25 follow-up question | Author-team answer |
|---|---|
| Are measured Full-vs-P+C flip/correctness logs available? | Not in this author-side artifact. Therefore the one-page remains protocol-only for Future and does not report exact flip counts, rates, CI, or p-values. |
| Are measured detector-control outputs available? | Not in this author-side artifact. Therefore the one-page keeps the control design and detector-assisted limitation rather than reporting attribution deltas. |
| Is proposal time measured? | Not in this author-side artifact. Therefore the page keeps proposal/decode separation and P+C/Full regimes, but does not report exact proposal/total/VRAM/batch numbers. |
| Are ONLY/VHD/VHR/HALC matched outputs available? | Not in this author-side artifact. Therefore the page keeps fairness-protocol wording only and no numeric superiority. |
| Can the table be shortened if measured rows are added? | Yes. If measured rows arrive, first cut the Figure 2 sentence and repeated "if logs are complete" wording before cutting mechanism, detector, cost/default, or boundary text. |
| Should the team freeze this draft unless real measured values arrive? | Yes. This is now the author-team policy. The next edit should be either measured-value insertion or final copy polish, not broad Markdown expansion. |

### Expected-Table Reasonability Check For v9

No expected table values are changed. The latest audit does not find a numerical contradiction. It says the official page is strongest precisely because it excludes exact expected values. The internal expected tables remain valid as engineering targets, but the official one-page PDF must stay measured-only for exact statistics.

Current official-page numeric policy:

- Future flip/correctness: wording-only unless measured paired logs exist.
- Detector attribution: wording-only unless measured control outputs exist.
- Proposal/total/VRAM/batch: wording-only unless measured timing/memory logs exist.
- Recent baselines: wording-only unless matched outputs and provenance exist.
- Scope, P+C/Full default, k/m rationale, and attention boundary: include as wording.

This policy directly addresses reviewer skepticism: it answers the concerns without risking fabricated-looking precision.

## 0E. Additions From The Latest Audit: One-Page Draft Produced

The latest audit's actionable request is: create the strict one-page rebuttal PDF/TEX, or at minimum stop expanding author-response Markdown. We accept that assessment. In this iteration the author team creates the first one-page draft using only evidence-eligible wording and no exact expected experimental numbers.

### New One-Page Draft Artifacts

The one-page draft target for this iteration is:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2221.pdf`

The draft is intentionally conservative. It does not report expected Future flip counts, detector-control deltas, CI, p-values, VRAM, batch OOM, or recent-baseline numerical wins. Instead, it states the diagnostic protocols, the measured-only rule, and the claim boundary. This makes the page weaker than a fully measured rebuttal but much safer than mixing expected values with official claims.

### Direct Answers To The Latest Audit

| Latest audit request | Author-team answer in v8 | One-page draft action |
|---|---|---|
| Create the strict one-page PDF/TEX draft. | Accepted. The draft is created as a separate TEX/PDF artifact with the `20260530_2221` stem. | Done in this run. |
| Do not let expected values enter the official page as measured facts. | Accepted. The draft uses no exact expected metrics. | Exact expected metrics are omitted. |
| Preserve Future mechanism as the first score-moving block. | Accepted. Future Full-vs-P+C admission diagnostics are the first table row. | Included as diagnostic protocol with measured-only slot. |
| Preserve detector-assisted boundary if detector controls are not measured. | Accepted. The draft explicitly says CHORD is detector-assisted and does not claim detector independence. | Included. |
| Preserve P+C practical/default and Full quality/offline. | Accepted. This is explicitly stated in the cost/default row. | Included. |
| Preserve recent-baseline fairness without unsupported wins. | Accepted. ONLY, VHD/VHR, and HALC are included as matched-protocol wording only. | Included without numeric claims. |
| Preserve k/m rationale compactly. | Accepted. `k=5,m=3` is quality-oriented; P+C or `k=5,m=2` is practical under latency constraints. | Included as one compact sentence. |
| Avoid spending space on figures. | Accepted. Fig. 2 redraw is one phrase only. | Included only if space allows in final text. |

### Expected-Table Reasonability Check For v8

No expected numeric target is changed in v8. The reason is methodological: the latest audit does not identify a table inconsistency; it identifies a submission-readiness gap. The correct response is to compile a one-page artifact that excludes unverified exact values.

The expected tables remain useful for engineering comparison because their ordering is conservative:

- Full remains only modestly stronger than P+C and slower.
- P+C remains close to recent baselines rather than unrealistically dominant.
- Detector-relevant anchors remain stronger than zero/noisy anchors.
- Cost remains derivable from proposal time plus decode ITL times token count.

But the official one-page draft generated in this run follows the stricter rule: exact expected values are omitted unless real measurements are available. If real results later arrive, the one-page draft can be revised by replacing wording-only slots with measured numbers.

## 0D. Additions From The 21:37 Strict Audit Of v6

The 21:37 audit's core criticism is operational, not conceptual. It says v6 is a good internal response-control document, but reviewers will only see the official one-page response. Our author-team answer is to make the first one-page artifact concrete and to freeze the eligibility policy.

### Exact Next Artifact

The next one-page rebuttal draft should use this file stem:

`author_response_onepage_expected_20260530_2142`

Expected files after the next PDF/TEX action:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142.tex`
- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142.pdf`

If we cannot compile the one-page PDF immediately, we should first create a one-page evidence-eligibility ledger with the same stem:

- `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142_eligibility.md`

This v7 response does not itself compile the official PDF because it is still a Markdown author-response heartbeat artifact. It specifies the exact next PDF/TEX target and the allowed payload.

### One-Page Evidence Eligibility Ledger

| Final-page item | Current status | Official one-page action | Reviewer concern addressed |
|---|---|---|---|
| Opening concession | Wording-ready. | **Include.** "We agree aggregate scores alone do not isolate mechanism, attribution, or deployment cost." | All reviewers; shows we understood the criticism. |
| Claim scope | Wording-ready. | **Include.** "CHORD is detector-assisted, base-MLLM-training-free, and scoped to object-grounded hallucination." | yx8u, KrEs. |
| Future Full-vs-P+C flip/correctness | Expected unless paired logs are available. | **Include measured only; otherwise wording-only diagnostic.** | M8du. |
| CHAIR continuation / CHAIR-S | Expected unless paired captions are evaluated. | **Include only if measured and compact; otherwise omit exact CI/p-value.** | M8du, yx8u. |
| Detector controls | Expected unless same-anchor/random/no-Current/P+C/Full logs are available. | **Include measured attribution row if available; otherwise include control design plus detector-assisted boundary.** | KrEs, M8du. |
| Proposal time / end-to-end latency | Proposal/VRAM/batch expected unless measured; decode ITL exists from paper context. | **Include only safe cost accounting.** Use measured proposal/total/VRAM only if available; otherwise state separation of proposal and decode ITL. | jjVG, KrEs, ve3y. |
| P+C vs Full default | Wording-ready. | **Include.** "P+C is the practical/default regime; Full is quality-oriented/offline." | ve3y, yx8u. |
| `k/m` sensitivity | Expected table; wording-ready. | **Include one Pareto sentence.** "`k=5,m=3` is quality-oriented; P+C or `k=5,m=2` is practical when latency is constrained." | jjVG. |
| ONLY / VHD-VHR / HALC | Numeric rows expected unless matched outputs exist. | **Use fairness-protocol wording only unless measured.** | jjVG, KrEs. |
| Figure 2 redesign | Wording-ready. | **One phrase only if space allows.** "We will redraw Fig. 2 into sequential Past/Current/Future stages." | jjVG. |
| Generality pilots | Expected/pilot only. | **Drop from official page unless real and necessary.** Keep claim narrow instead. | yx8u, KrEs. |
| Exact expected CI, p-values, VRAM, batch OOM, recent-baseline wins | Expected. | **Drop unless measured.** | Prevents overclaiming. |

### Direct Answers To 21:37 Follow-Up Questions

| 21:37 follow-up question | Author-team answer |
|---|---|
| What is the exact path of the first strict one-page rebuttal TEX/PDF draft? | Use `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142.tex` and `.pdf`. If compilation is delayed, create `author_response_onepage_expected_20260530_2142_eligibility.md` first. |
| Which rows in that one-page draft are measured, wording-only, and dropped? | Measured-only: Future flip/correctness, detector attribution rows, proposal/VRAM/batch/recent-baseline numeric comparisons. Wording-only: claim scope, P+C/Full default, k/m Pareto, fairness protocol, detector-assisted boundary. Drop unless measured: exact expected CI, p-values, VRAM, batch OOM, recent-baseline wins, broad generality pilots. |
| If Future flip/correctness is not measured, will official page remove exact flip percentages, counts, CI, and p-values? | Yes. It should keep only diagnostic wording and no exact expected Future counts. |
| If detector controls are not measured, will official page explicitly state detector-assisted and no detector independence? | Yes. This is mandatory wording, not optional. |
| Is proposal time measured, or is only decode ITL safe? | Until engineering confirms proposal-time logs, only submitted decode ITL is safe as measured context. Proposal/total/VRAM/batch values must be omitted or explicitly marked pending outside the official page. |
| Are ONLY/VHD/HALC results measured under matched protocol? | Unless real matched outputs and provenance exist, the official page uses fairness-protocol wording only and no numeric superiority. |
| Will the one-page PDF preserve P+C practical/default and Full quality/offline? | Yes. This sentence is mandatory because it resolves the efficiency trade-off without pretending Full is cheap. |
| Should reviewer heartbeat keep producing new audits before a new author artifact appears? | No. The next useful audit should target the one-page draft or eligibility ledger. Re-auditing unchanged v6/v7 is low value. |

### Proposed One-Page Content, Text-Only Draft

This is the compact payload that should be converted into TEX/PDF next. Exact numeric slots are intentionally conditional.

> We thank the reviewers for identifying that aggregate benchmark scores alone do not isolate mechanism, detector attribution, and deployment cost. We therefore add targeted diagnostics and narrow CHORD's claim: it is a detector-assisted, base-MLLM-training-free admission-time verifier for object-grounded hallucination mitigation; attention is used as an operational scoring feature rather than a causal explanation.
>
> **Mechanism.** To isolate the Future term, we compare Full CHORD against P+C at the admission level and count rollout-induced token changes, corrected unsupported mentions, and harmful changes. If paired logs are complete, this row reports exact flip/correctness counts; otherwise we include only the diagnostic protocol and omit exact expected counts.
>
> **Detector attribution.** To test whether Grounding DINO alone explains the gain, we use fixed same-anchor, uniform/random-anchor, no-Current, P+C, and Full controls. If logs are complete, this row reports the compact attribution ordering; otherwise we state the control design and explicitly do not claim detector independence.
>
> **Cost, baselines, and hyperparameters.** We separate proposal cost from decode ITL. P+C is the practical/default regime, while Full is quality-oriented/offline. `k=5,m=3` is the quality setting; P+C or `k=5,m=2` is the practical setting under latency constraints. Recent methods ONLY, VHD/VHR, and HALC are compared only under matched backbone, split, prompt family, parser, seeds, and validation budget; numeric superiority is reported only for completed matched runs.
>
> We will also redraw Fig. 2 into sequential Past/Current/Future stages. The final paper will remove detector-independent or broad relation/composition claims unless supported by measured evidence.

### Expected-Table Reasonability Check For This Iteration

The 21:37 audit does not identify a new numeric inconsistency in the expected tables. It identifies a submission-risk boundary: expected numbers remain useful as internal targets but must not enter the one-page official response as measured facts.

Therefore v7 preserves all v6 expected-table values and changes only the production policy:

- no exact Future flip rate, corrected/harmful count, CI, or p-value in the final page unless paired logs exist;
- no exact detector-control delta unless the control runs are measured;
- no exact proposal time, VRAM, batch OOM, or total latency unless measured;
- no exact ONLY/VHD/HALC comparison unless matched outputs and implementation provenance exist;
- use wording-only commitments for scope, default regime, k/m rationale, and fairness protocol.

This is the most reviewer-safe path because it answers all five reviewer classes while avoiding the appearance that expected values are being submitted as empirical results.

## 0C. Heartbeat Update: No New Audit Since v5

The latest available reviewer/audit input remains `strict_reviewer_audit_2123_latest_20260530_review_v4.md`. The previous author response, v5, already answered that audit by adding:

1. a concrete final one-page payload decision;
2. direct answers to the latest follow-up questions;
3. a final one-page evidence eligibility sheet;
4. a pre-submission stop rule;
5. an expected-table reasonability check.

Because there is no newer strict audit, the author-team decision is to preserve v5 unchanged in substance and avoid adding new expected numeric targets. Repeatedly extending the all-in-one response without a new reviewer concern would increase the risk of bloat and make the final one-page compression harder. The next meaningful rebuttal artifact should be a strict one-page PDF/TEX draft, not another expansion of the five-page master.

### Current Author-Team Answer To The Latest Audit

| Latest audit issue | Current answer | Action status |
|---|---|---|
| v4 did not yet make the final one-page payload decision concrete enough. | v5 made the payload concrete: mechanism first, detector attribution second, cost/baseline and boundary third. | Resolved in v5 and preserved here. |
| Expected rows could leak into the official page as if measured. | v5 requires exact values only for completed paired/matched runs; otherwise wording-only or drop. | Resolved as a stop rule. |
| M8du still needs mechanism evidence if available. | Future Full-vs-P+C flip/correctness is top priority and should be the first numeric row if logs exist. | Awaiting real logs; no expected counts should be used as final facts. |
| KrEs still needs detector attribution if available. | Same-anchor, uniform/random, no-Current, P+C, and Full controls are second priority; detector-assisted boundary is mandatory. | Awaiting real logs; wording boundary is ready. |
| One-page draft does not yet exist. | The next author-side artifact should be `author_response_onepage_expected_YYYYMMDD_HHMM.tex/.pdf`. | Next action; not performed in this heartbeat. |

### Expected-Table Check For This Heartbeat

No expected values are changed in v6. This is intentional and scientifically safer. The expected tables remain internal forward-looking targets only. Their role is to guide the engineer and future replacement, not to create new official claims.

The current expected-table policy remains:

- preserve conservative target ordering: Full slightly stronger than P+C and slower;
- preserve detector-attribution logic: relevant anchors strongest, zero/noisy anchors weaker;
- preserve recent-baseline modesty: P+C near strong baselines, Full quality-oriented;
- preserve latency derivability from proposal time plus decode ITL times token count;
- remove, replace, or narrow any row when real measurements disagree;
- never report expected CI, p-value, VRAM, batch, flip counts, or recent-baseline wins as official measured values.

### Immediate Next Step

The next useful action is not another all-in-one response iteration unless a new `strict_reviewer_audit_*_latest_*.md` appears. The next useful author artifact is the one-page rebuttal draft, using this payload order:

1. opening concession and narrowed scope;
2. one compact table: Future mechanism, detector attribution, cost/baseline fairness;
3. one k/m Pareto sentence;
4. one boundary sentence: detector-assisted, base-MLLM-training-free, object-grounded scope, operational attention, P+C practical, Full quality/offline.

This v6 document therefore functions as a convergence marker: v5 answered the latest audit; v6 confirms no new audit appeared and protects against unnecessary expansion.

## 0B. Additions From The Latest Audit Of v4

The latest audit raises one central operational concern: v4 gives the right rule for measured-vs-expected content, but it does not yet make the final one-page payload decision concrete enough. Our author-team answer is to convert the rule into an actionable final-page decision table. The official ACM MM response must remain a single strict one-page PDF; therefore we should not attempt to move the five-page scientific master into the final page. We should instead use one compact evidence table plus one boundary paragraph.

### Author Decision: Final One-Page Payload

| One-page block | Final-page status | What we can safely say now | What must not be said unless measured |
|---|---|---|---|
| Mechanism: Full vs P+C admission changes | **Top priority. Include exact numbers only if paired logs are available.** | "We add an admission-level diagnostic comparing Full to P+C and report how often rollout changes token admission and whether the change reduces unsupported object mentions." | Do not report exact flip rate, corrected/harmful counts, CI, or p-value from expected tables without paired outputs. |
| Detector attribution: DINO vs decoding rule | **Second priority. Include measured row if available; otherwise include control design plus boundary.** | "We isolate detector contribution with same-anchor, uniform/random-anchor, no-Current, P+C, and Full controls; our claim remains detector-assisted." | Do not claim detector independence, second-proposer robustness, or numeric attribution deltas unless those runs are measured. |
| Efficiency and default setting | **Must include as wording; include exact total/VRAM only if measured.** | "P+C is the practical/default regime; Full is quality-oriented/offline. We separate proposal cost from decode ITL and will report end-to-end cost." | Do not use expected proposal time, VRAM, batch OOM, or total latency as measured deployment facts. |
| Recent baselines | **Include fairness protocol; include numeric comparisons only if real matched runs exist.** | "ONLY, VHD/VHR, and HALC are handled under matched parser, prompt family, split, seeds, and validation budget; official implementations/checkpoints are used when available, otherwise sanity-checked reproductions are used." | Do not claim numeric superiority over ONLY/VHD/VHR/HALC without real matched outputs and implementation provenance. |
| k/m sensitivity | **Include as one Pareto sentence.** | "`k=5,m=3` is a quality-oriented setting; P+C or `k=5,m=2` is the practical setting when latency is constrained." | Do not include a full sweep table unless measured and space permits, which it likely will not. |
| Scope and novelty boundary | **Must include.** | "CHORD is a detector-assisted, base-MLLM-training-free admission-time verifier for object-grounded hallucination mitigation; attention is an operational scoring signal, not a causal explanation." | Do not claim detector-free, detector-agnostic, broad relation/composition generality, or a fundamentally new paradigm. |

This is now the final payload rule for the one-page rebuttal. The five-page master may retain expected target values as internal guidance, but the one-page official rebuttal must treat those values as unavailable until replaced by real logs.

### Direct Answers To The Latest Follow-Up Questions

| Latest follow-up question | Author-team answer | Final one-page wording |
|---|---|---|
| Which of the three evidence blocks can be backed by measured logs today: Future mechanism, detector attribution, and cost/recent-baseline fairness? | The response document itself should not assert that any expected row is measured. Each block must be checked against engineering logs before final PDF generation. If only one measured block is available, choose Future mechanism first because it directly answers M8du's core question. If detector attribution is measured, include it second because it is KrEs's strongest objection. Cost/baseline fairness can be wording-only if exact runs are pending. | "We prioritize measured admission-level and attribution diagnostics; unmeasured diagnostics are stated as planned camera-ready additions rather than reported results." |
| For each final-page candidate row, what is the status: include measured, include wording only, or drop? | Mechanism counts: include measured only. Detector controls: include measured only, otherwise wording. Cost: wording plus measured decode/proposal if available. Recent baselines: wording unless real matched numbers exist. k/m: wording. Generality pilots and relation/composition rows: drop from official page unless measured and needed for scope boundary. | "Exact numbers below are included only for completed paired/matched runs; otherwise we state the diagnostic and narrow the claim." |
| If Future flip/correctness is not measured, what wording avoids pretending the mechanism has been empirically validated? | Do not say "we find", "we show", or give exact percentages. Say "we will add" or "we add the diagnostic protocol" only if final paper can include it; otherwise present no exact row. | "To isolate rollout, we add a paired Full-vs-P+C admission diagnostic that counts rollout-induced token changes and corrected vs harmful outcomes; exact counts are reported only for completed paired runs." |
| If detector attribution rows are not measured, will the final page explicitly say detector-assisted and no detector independence? | Yes. This sentence is mandatory because it preserves yx8u's support and reduces KrEs's overclaiming concern. | "CHORD is detector-assisted; the controls isolate the decoding rule under fixed/noisy anchors, but we do not claim detector independence." |
| Are ONLY/VHD/HALC direct numbers measured with implementation provenance? | Until engineering provides outputs, no numeric row should be used. The official page can still answer jjVG/KrEs by stating the matched-comparison protocol and promising camera-ready inclusion under the one-page constraint, but numeric superiority must be omitted unless measured. | "We will add ONLY, VHD/VHR, and HALC under a matched protocol: same backbone, split, prompt family, parser, seeds, and validation budget; numeric claims are included only for completed matched runs." |
| What is the final one-page draft path and when will it be audited? | This v5 response does not itself create the final one-page PDF. The next author-side artifact should be `author_response_onepage_expected_YYYYMMDD_HHMM.tex/.pdf`, derived from the payload rule above and audited separately before any OpenReview upload. | No wording needed in official page; this is an internal production requirement. |
| If only one numeric block can fit, which one is prioritized? | Future mechanism first. It answers the most precise reviewer question: how often Future changes decisions and whether those changes are beneficial. Detector attribution is second. Cost/baseline fairness is third and can be summarized in text if necessary. | The one-page table should order blocks as: Mechanism -> Detector attribution -> Cost/baseline fairness. |
| Will the final page preserve P+C as practical default and Full as quality/offline? | Yes. This distinction must stay even under heavy compression because it answers latency concerns without pretending Full is cheap. | "P+C is the practical/default regime; Full adds rollout for quality-oriented/offline inference." |

### Final One-Page Skeleton To Use Next

This is the exact structure we should use when converting the master into a one-page PDF:

1. **Opening concession and scope, one sentence.** "We agree that aggregate benchmark scores alone do not isolate mechanism, detector attribution, or deployment cost; we therefore add targeted diagnostics and narrow CHORD's claim to detector-assisted, object-grounded hallucination mitigation."
2. **One compact table with three blocks.** The table should have columns `Concern`, `Reviewer need`, `Response/evidence`, and `Boundary`. It should contain at most one row each for Future mechanism, detector attribution, and cost/recent baselines.
3. **One Pareto/default sentence.** "`k=5,m=3` is the quality-oriented setting; P+C or `k=5,m=2` is the practical setting under latency constraints."
4. **One final boundary sentence.** "CHORD is base-MLLM-training-free but detector-assisted; attention is used as an operational scoring feature, not a causal explanation; relation/composition generality is outside the main claim."

No Figure 2 redraw should be placed in the final one-page PDF unless the text becomes unreadable without it. A figure is lower value than mechanism and attribution evidence.

### Final One-Page Evidence Eligibility Sheet

| Candidate content | Eligibility for final one-page | Reason |
|---|---|---|
| Measured Full-vs-P+C flip/correctness | Include measured. | Best answer to M8du and strongest mechanism evidence. |
| Expected Full-vs-P+C flip/correctness | Drop exact values; wording only. | Exact expected percentages look fabricated if not measured. |
| Measured CHAIR-S paired continuation | Include if compact. | Supports open-ended hallucination benefit, but secondary to admission flips. |
| Expected CHAIR-S CI/p-value | Drop exact values. | CI/p-value without logs is high-risk. |
| Measured same-anchor/random/no-Current attribution | Include measured. | Best answer to KrEs's "DINO does the work" concern. |
| Expected attribution deltas | Wording only. | Control design is useful; fake precision is not. |
| Measured proposal/decode/total latency | Include compact. | Answers cost directly and supports P+C/Full distinction. |
| Expected VRAM/batch/OOM | Drop exact values unless measured. | Deployment claims are easy to challenge. |
| Measured ONLY/VHD/HALC matched results | Include one compact row only if provenance-backed. | Addresses related-work completeness. |
| Expected ONLY/VHD/HALC wins | Drop numeric wins; keep fairness protocol. | Numeric superiority without logs will hurt credibility. |
| k/m sweep | Wording only. | Reviewer needs basis for k/m; a full table is too costly for one page. |
| Figure 2 promise | One phrase only. | Presentation issue is lower priority than evidence. |
| Generality pilots on Qwen/LLaVA-NeXT/relation/composition | Drop unless measured and essential. | The main claim is object-grounded; broad generality is not required and can reopen scope risk. |

### Pre-Submission Stop Rule

The official one-page PDF should not be submitted until the following are true:

1. Every number in the one-page PDF is traceable to measured logs or is explicitly absent.
2. No expected value appears without an expected/internal marker in the master, and no expected value appears as a final official fact.
3. The one-page PDF states detector-assisted scope, P+C practical default, Full quality/offline use, and attention-as-operational-feature.
4. The page contains no Official Comment dependency, no hidden supplement dependency, and no overflow evidence outside the one page.
5. The one-page PDF has been visually checked for legible font, no overlapping table text, and no clipped content.

### Expected-Table Reasonability Check For This Iteration

This v5 iteration does not change the expected numeric targets from v4. That is intentional. The latest audit does not identify a new arithmetic contradiction; it identifies a final-page evidence-eligibility risk. Therefore the correct action is to preserve the current expected targets as internal engineering guidance and prevent them from leaking into the official one-page rebuttal as measured claims.

The expected targets remain reasonable as targets because their ordering and effect sizes are conservative:

- Full is only modestly stronger than P+C and clearly slower.
- P+C remains near strong recent baselines rather than unrealistically dominating them.
- Relevant anchors benefit more than zero/noisy anchors.
- Detector controls are designed to separate anchor quality from decoding coordination.
- Latency remains derivable from proposal time plus decode ITL times generated token count.
- CI/p-value rows remain expected and must be replaced or dropped unless real logs are available.

The most important numeric policy for the next artifact is stricter than the master policy: the one-page rebuttal should contain fewer numbers, not more, unless engineering has verified them. This is how we reduce reviewer skepticism instead of inviting a second round of questions.

All numeric rows below are still expected-result targets unless explicitly replaced by real experiment outputs. They are useful because they define the shape of evidence we need, but they must not be treated as measured data. When real results arrive, the correct rule is replacement plus claim narrowing if needed, not forcing the real numbers to match the expected table.

## 0A. Additions From The Latest Audit

The latest audit says the 17:44 master is now a strong internal scientific master, but the official one-page rebuttal remains borderline until we decide which rows are evidence-eligible. Our author-team answer is to separate final-page content into three classes:

1. **Include as measured**: rows backed by real logs, exact paired samples, or values already in the submitted paper.
2. **Include as wording only**: expected or pending diagnostics that are scientifically important but not yet measured.
3. **Drop from final one-page**: precise expected numbers that would look fabricated or overconfident without raw results.

This rule protects the rebuttal from the main remaining reviewer risk: using precise expected tables as if they were completed experiments.

### Final One-Page Eligibility Matrix

| Candidate final-page item | Status now | One-page decision | Exact wording to use if real values are not ready |
|---|---|---|---|
| Future Full-vs-P+C flip rate and corrected/harmful counts | Expected unless engineering provides paired output logs. | **Include as measured only.** | "We add an admission-level Full-vs-P+C flip/correctness diagnostic; if unavailable by submission, we will not report exact flip counts." |
| CHAIR-S Full-vs-P+C deltas and bootstrap CI | Expected unless paired CHAIR captions are evaluated. | **Include CI only if measured; otherwise wording only.** | "Open-ended CHAIR continuation is evaluated under the same parser; exact CI will be reported only if paired runs are complete." |
| Detector controls: same-anchor, uniform/random, P+C, Full | Expected targets but central to KrEs/M8du. | **Include measured rows if available; otherwise include control design and claim boundary.** | "We isolate the detector via same-anchor, uniform/random, and no-Current controls; we keep the claim detector-assisted and do not claim detector independence." |
| Proposal time / VRAM / batch | Proposal/VRAM/batch are expected; decode ITL is submitted context. | **Include measured cost if available; otherwise include cost-accounting promise plus P+C/Full regimes.** | "We separate proposal time from decode ITL and present P+C as practical, Full as quality/offline." |
| ONLY / VHD-VHR / HALC numeric comparison | Expected unless real matched runs exist. | **Do not include numeric wins unless measured.** | "We compare against ONLY, VHD/VHR, and HALC under official or sanity-checked implementations with shared parser, prompts, split, seeds, and validation budget." |
| k/m sweep | Expected table but reviewer-visible completeness issue. | **Include as a one-sentence Pareto claim, not a table, unless measured.** | "`k=5,m=3` is quality-oriented; P+C or `k=5,m=2` is the practical setting." |
| Claim boundary | Wording commitment. | **Must include.** | "CHORD is detector-assisted and base-MLLM-training-free; attention is an operational signal; object-grounded hallucination is the claim scope." |
| Figure 2 cleanup | Presentation commitment. | **One phrase only.** | "We will redraw Fig. 2 as sequential Past/Current/Future stages." |

### Final One-Page Payload Rule

The final one-page rebuttal should contain at most three evidence blocks and one boundary sentence:

1. **Mechanism block**: use exact Future flip/correctness numbers only if measured; otherwise state the diagnostic design and omit p-values/counts.
2. **Attribution block**: use measured same-anchor/uniform/random/P+C/Full values if available; otherwise state the controls and explicitly narrow the detector claim.
3. **Cost/baseline block**: use measured proposal/VRAM/batch and recent-baseline values if available; otherwise state end-to-end accounting and matched-baseline protocol without numeric superiority.
4. **Boundary sentence**: detector-assisted, base-MLLM-training-free, object-grounded scope, attention as operational feature, P+C practical and Full quality/offline.

This is the safest answer to the reviewer audit: the one-page response should not be a compressed version of all five pages. It should be a compressed version of the evidence that is safe to submit.

## 0. Additions From The Latest Audit

The latest audit of v2 raised five author-action questions. Our answers are:

| Audit question | Author-team answer | Master action | One-page action |
|---|---|---|---|
| Are the v2 values measured, expected, pending, or removable? | Treat every number marked `E` as expected. Only submitted decode ITL values and paper-baseline numbers already in the paper are pre-existing measured context. Real experiment values must replace `E` before final submission if used as evidence. | Add an evidence-status table, or keep `E` marks on every expected number. | Include only measured values or explicitly bounded revision commitments; do not let expected values look like completed experiments. |
| Should flip rates or counts be changed? | Preserve counts and correct rates: `96+42=138`, `138/3000=4.6%`; `102+44=146`, `146/3000=4.9%`. | Update master flip rates to `4.6%` and `4.9%`; keep "about 4-5%" wording. | Use "4-5%" rather than exact rates if space is tight. |
| Should detector-control rows be changed? | Yes. Same-anchor non-CHORD and random-anchor controls should not contradict the note that detector controls provide small gains but remain below real-anchor P+C. | Set same-anchor to `0.823 E` and random anchors to `0.821 E`, or rewrite the note. | Keep only the ordering: detector-only/random/uniform < real-anchor P+C < Full. |
| Can recent-baseline fairness fit in one page? | Yes, as a single dense fairness clause: official/sanity-checked implementations, same parser, prompt family, split, seeds, and validation budget. | Keep full detail in master. | One clause is mandatory because Official Comment is not visible and cannot serve as overflow. |
| What if real measurements do not match expected targets? | Replace the expected rows and narrow the claim. If Future flips vanish or hurt, emphasize P+C; if detector controls weaken, emphasize detector-assisted boundary; if recent baselines win, present CHORD as a trade-off rather than superiority. | Add this rule as an internal evidence boundary. | Do not submit expected rows as if measured. |

### Evidence-Status Matrix For The Current Tables

| Evidence item | Current status | Use in 5-page master | Use in final one-page PDF |
|---|---|---|---|
| Submitted decode ITL values: Greedy `19.73`, OPERA `21.69`, P+C `27.24`, Full `37.31` | Existing measured context from the paper/table. | Keep. | Can keep if space allows. |
| Proposal time `118 E`, VRAM/batch rows | Expected until engineering confirms hardware log. | Keep with `E`; replace when measured. | Use only if measured or phrase as "we will report end-to-end proposal+decode cost." |
| Future flip/corrected/harmful rows | Expected targets; arithmetic must be fixed now. | Keep with `E`, using `4.6%/4.9%`. | Use only measured values; otherwise compress as planned diagnostic. |
| CHAIR-S deltas and InstructBLIP CHAIR CI | Expected targets. | Keep as expected reliability targets. | Include only if measured; if not, mention CHAIR continuation qualitatively and avoid p-values. |
| Detector attribution rows | Expected targets but logically essential. | Keep and mark expected; fix ordering. | Include measured ordering if available; otherwise state the control design, not numerical superiority. |
| ONLY/VHD/HALC matched rows | Expected unless actual matched runs exist. | Keep as expected target/fairness plan. | Include only measured rows or cite fairness protocol without numeric claims. |
| Claim-boundary sentence | Wording commitment, not experiment. | Keep. | Must keep. |

This status matrix should prevent a future draft from overstating expected results. It also tells the engineer exactly which real outputs are needed to upgrade the one-page rebuttal from "well-planned" to "evidence-backed."

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

The 17:44 master has already fixed the known arithmetic/table-ordering mistakes. The next action should not be another broad 5-page expansion. It should be a final-page eligibility pass:

1. Mark every candidate final-page row as `include measured`, `include wording only`, or `drop unless measured`.
2. Replace expected values with real outputs as they arrive.
3. Build the strict one-page rebuttal from only evidence-eligible rows.
4. Keep the 17:44 5-page master as the internal scientific reference.
5. Do not submit anything to OpenReview until the one-page version is separately audited.

This v10 response is the author-team all-in-one decision document for final-page eligibility, one-page draft polish, and freeze decision. It should guide the next compression step, not encourage further proliferation of expected numeric rows.

Internal generation note: this v10 document was created by copying v9 first, then additively incorporating the 03:00 strict audit of v9 and the 00:43 one-page PDF/TEX. It preserves v9's polished one-page target, measured-only official-page policy, and earlier reviewer answers, then adds a freeze decision: keep `author_response_onepage_expected_20260531_0043.pdf` as the current safest official candidate unless real measured evidence arrives.
