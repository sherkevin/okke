# One-Page Rebuttal Evidence Eligibility Ledger

This is the production ledger for the first strict one-page ACM MM rebuttal draft. It is written from the author-team perspective. The official rebuttal is a single one-page PDF; Official Comment must not be used as overflow.

## Target Draft

- TEX: `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142.tex`
- PDF: `papers\opera_acm_sigconf\rebuttal\author_response_onepage_expected_20260530_2142.pdf`

## Payload Rule

The one-page rebuttal should contain:

1. one opening concession and narrowed scope sentence;
2. one compact evidence table with mechanism, detector attribution, and cost/baseline fairness;
3. one k/m Pareto sentence;
4. one final claim-boundary sentence.

It should not carry the full five-page scientific master.

## Eligibility Table

| Candidate content | Eligibility | One-page action | Reason |
|---|---|---|---|
| Opening concession | Wording-ready | Include | Shows we understand that aggregate scores alone do not isolate mechanism, attribution, or cost. |
| Claim scope | Wording-ready | Include | Keeps yx8u support and reduces KrEs overclaiming risk. |
| Future Full-vs-P+C flip/correctness | Expected unless paired logs exist | Include measured only; otherwise wording-only | Best answer to M8du, but exact expected counts are unsafe. |
| CHAIR continuation / CHAIR-S | Expected unless paired captions exist | Include only if measured and compact | Useful but secondary to admission-level mechanism. |
| Detector controls | Expected unless control logs exist | Include measured row if available; otherwise control design plus boundary | Best answer to KrEs detector-attribution concern. |
| Proposal time / end-to-end latency | Proposal/VRAM/batch expected unless measured | Include safe cost accounting only | Avoid mixing measured decode ITL with expected total cost. |
| P+C vs Full default | Wording-ready | Include | Resolves efficiency concern: P+C practical, Full quality/offline. |
| k/m sensitivity | Expected table, wording-ready | Include one sentence | Answers jjVG without spending table space. |
| ONLY / VHD-VHR / HALC | Numeric rows expected unless matched outputs exist | Use fairness-protocol wording only unless measured | Addresses related-work completeness without unsupported wins. |
| Figure 2 redraw | Wording-ready | One phrase only if space allows | Lower priority than evidence. |
| Generality pilots | Expected/pilot | Drop unless measured and essential | Broad claims reopen scope risk. |
| Exact expected CI, p-values, VRAM, batch OOM, recent-baseline wins | Expected | Drop unless measured | Prevents expected values from becoming official factual claims. |

## Text-Only One-Page Draft Payload

We thank the reviewers for identifying that aggregate benchmark scores alone do not isolate mechanism, detector attribution, and deployment cost. We therefore add targeted diagnostics and narrow CHORD's claim: it is a detector-assisted, base-MLLM-training-free admission-time verifier for object-grounded hallucination mitigation; attention is used as an operational scoring feature rather than a causal explanation.

Mechanism. To isolate the Future term, we compare Full CHORD against P+C at the admission level and count rollout-induced token changes, corrected unsupported mentions, and harmful changes. If paired logs are complete, this row reports exact flip/correctness counts; otherwise we include only the diagnostic protocol and omit exact expected counts.

Detector attribution. To test whether Grounding DINO alone explains the gain, we use fixed same-anchor, uniform/random-anchor, no-Current, P+C, and Full controls. If logs are complete, this row reports the compact attribution ordering; otherwise we state the control design and explicitly do not claim detector independence.

Cost, baselines, and hyperparameters. We separate proposal cost from decode ITL. P+C is the practical/default regime, while Full is quality-oriented/offline. `k=5,m=3` is the quality setting; P+C or `k=5,m=2` is practical under latency constraints. Recent methods ONLY, VHD/VHR, and HALC are compared only under matched backbone, split, prompt family, parser, seeds, and validation budget; numeric superiority is reported only for completed matched runs.

We will also redraw Fig. 2 into sequential Past/Current/Future stages. The final paper will remove detector-independent or broad relation/composition claims unless supported by measured evidence.

## Stop Rule

Do not submit the one-page PDF until every exact number in it is measured or removed. If real experimental outputs disagree with expected tables, replace the expected values and narrow the claim.
