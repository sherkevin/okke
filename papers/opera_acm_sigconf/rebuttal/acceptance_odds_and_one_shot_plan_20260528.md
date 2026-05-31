# CHORD Acceptance Odds and One-Shot Rebuttal Plan, 2026-05-28

Scope: planning and decision analysis only. This is not a rebuttal draft.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record. Earlier simulated or practice reviews are excluded and should not guide the rebuttal.

## Probability Estimate

My current estimate:

| Scenario | Acceptance odds | Interpretation |
|---|---:|---|
| Text-only rebuttal, no new evidence | 25-35% | Too much of the criticism is about missing diagnostics, attribution, and cost accounting. Explanations alone probably do not move KrEs or M8du enough. |
| Moderate rebuttal: add citations, clarify novelty, promise figure/camera-ready edits, but little new data | 35-45% | Likely protects the two Weak Accepts and may move jjVG, but still leaves the knowledgeable Borderline and Weak Reject with core evidence concerns. |
| Strong rebuttal: compact new diagnostics for future rollout, detector attribution, end-to-end latency, and k/m robustness | 50-60% | This is the realistic target. It can plausibly move M8du from Borderline to Weak Accept and jjVG from Borderline to Weak Accept; KrEs may remain skeptical but less decisive. |
| Exceptional rebuttal: above evidence plus credible HALC/ONLY/VHR positioning and one stronger backbone or baseline result | 60-70% | Possible only if the extra evidence is clean and not rushed. Even then, novelty/general-scope concerns cap the upside. |

Best single-number estimate if we execute the strong plan well: about 55%, i.e. five to six tenths. I would not estimate above 70% because the novelty concern is structural, not just a missing-experiment issue.

## What The Reviewers Are Really Saying

| Reviewer | Surface comments | Deeper concern | What they need to believe after rebuttal | Movement potential |
|---|---|---|---|---|
| jjVG | Cost, Figure 2 clutter, k/m ablation, missing ONLY/VHD | The submission looks promising but incomplete and maybe not fully benchmarked against current work. | The authors have a fairer related-work/baseline framing, the defaults are robust, the cost is transparent, and presentation will be fixed. | High. Low confidence and concrete requests make this reviewer movable. |
| KrEs | Novelty unclear, detector dependence, missing full latency/memory, limited baselines/models | The method may be an engineered bundle whose gains come from Grounding DINO and extra compute rather than a new decoding contribution. | CHORD's coordinated admission rule adds value beyond detector metadata and rollback; the overhead is honestly accounted for; missing baselines are handled. | Medium. Needs evidence, not rhetorical novelty claims. |
| yx8u | Incremental novelty, external detector, attention reliability, narrow scope, latency, overstrong terms | The paper is acceptable only if claims are toned down and limitations are honest. | The authors understand the boundaries: base-MLLM training-free, detector-assisted, attention as operational signal, not causal explanation. | Protect rather than convert. Keep this WA from dropping. |
| ve3y | Moderate novelty and overhead | This is a useful practical paper, but not a paradigm shift. | The paper gives a clear operating frontier and does not oversell. | Protect. A concise evidence-backed response is enough. |
| M8du | Future mechanism not validated, detector attribution, thin baselines, hyperparams, robustness | The end-to-end numbers are not enough to prove the proposed mechanism. | The future term measurably changes decisions in correct cases; gains are not just Grounding DINO; latency and hyperparameters are transparent. | Very high. This reviewer explicitly says they may raise the score. |

## Root-Cause Diagnosis

The paper is not being rejected because reviewers dislike the topic. All reviewers agree the topic fits ACM MM and is timely. The real blockers are:

1. Attribution: reviewers cannot tell whether the gains come from CHORD's admission rule or from external object proposals and extra compute.
2. Mechanism: reviewers cannot see when future rollout actually changes decisions and whether those changes are correct.
3. Completeness: missing recent methods make the empirical comparison look selectively scoped.
4. Cost realism: ITL is useful but not the same as end-to-end deployability.
5. Claim calibration: "training-free", "structural temporal collapse", and "harmonic arbitration" sound broader than current evidence.
6. Generality: two 7B backbones and object-heavy benchmarks make the method look narrow.

The rebuttal should therefore be designed as a small evidence package, not a debate.

## One-Shot Response Plan

| Priority | Concern cluster | One-shot evidence/action | Reviewers covered | Acceptance criterion |
|---:|---|---|---|---|
| P0 | Future rollout mechanism | Table: Full vs Past+Current flip rate, correctness of flips, harmful flips, by POPE/CHAIR where feasible. | M8du, KrEs, yx8u | Shows future is not decorative. If future helps mainly on CHAIR/open-ended cases, say that explicitly. |
| P0 | Detector attribution | Ablations: real anchors, no anchors/uniform weights, random anchors, Past+Future without current anchors, and same-anchor non-CHORD control if feasible. | KrEs, M8du, yx8u, ve3y | Demonstrates gains are not simply Grounding DINO doing the task. |
| P0 | End-to-end efficiency | Table: detector proposal time, decode ITL, total latency per answer, peak VRAM, batch-size note. | KrEs, M8du, jjVG, ve3y | Makes cost transparent; recommends Past+Current as latency-oriented regime and Full as quality-oriented regime. |
| P0 | Recent related work | Add ONLY, Vision-aware Head Divergence/VHR, HALC positioning. Direct comparison only if implementation/numbers are verifiable. | jjVG, KrEs, M8du | Removes "missing obvious work" objection. |
| P1 | k/m robustness | Small grid k={3,5,7}, m={1,2,3,4}; report quality and latency together. | jjVG, M8du | Shows defaults are reasonable and not cherry-picked. |
| P1 | Novelty framing | Define novelty as coordinated admission-time verification, not the raw use of rollback/detector/attention/rollout. | KrEs, yx8u, ve3y, M8du | Makes "integration" defensible as a controlled inference policy rather than a loose bundle. |
| P1 | Attention reliability | Clarify attention is an operational score, not explanation; add last-4 window ablation if available. | yx8u, KrEs | Prevents overclaim and reduces architecture-sensitivity criticism. |
| P1 | Generality | Add one stronger backbone or explicitly bound scope. | KrEs, yx8u, M8du | Helps, but should not displace P0 diagnostics. |
| P2 | Figure 2 | Commit to cleaner sequential lanes. | jjVG | Easy presentation win. |

## Reviewer-Specific Conversion Strategy

| Reviewer | Target outcome | Minimum required answer |
|---|---|---|
| jjVG | Borderline -> Weak Accept | Related-work patch for ONLY/VHD, k/m sweep, end-to-end cost note, Figure 2 cleanup. |
| KrEs | Weak Reject -> Borderline or neutralized WR | Detector attribution, novelty separation, end-to-end cost, stronger baseline positioning. |
| yx8u | Preserve Weak Accept | Tone down overclaims, qualify training-free, acknowledge detector and attention boundaries, show evidence not rhetoric. |
| ve3y | Preserve Weak Accept | Show the practical operating frontier and transparent overhead. |
| M8du | Borderline -> Weak Accept | Answer the five explicit questions directly, especially future flip/correctness and detector controls. |

## ACM MM 2026 Rebuttal Format Notes

Official public sources checked:

- ACM MM 2026 Call for Technical Papers says authors may optionally submit a rebuttal in OpenReview after receiving reviews.
- The same page says the rebuttal must maintain anonymity.
- It also says rebuttal cannot include links to external material such as code or videos.
- ACM MM 2026 Important Dates lists rebuttal deadline information. The public page currently lists Main Track rebuttal as `04-June`; if the OpenReview/email instruction says `28-May - 04-June`, follow OpenReview/email as the controlling operational instruction.
- The public Author Instructions page points to OpenReview and reiterates double-blind/anonymity rules, but does not publish a rebuttal character limit or a PDF template.
- OpenReview's general rebuttal-stage documentation says venues can configure the number of rebuttals as one per paper, one per review, or multiple per paper, so the actual ACMMM 2026 form must be checked directly before drafting.
- OpenReview's default rebuttal form is a Markdown text field with `maxLength: 2500`, but venues can override it. Therefore, do not assume ACM MM uses either a one-page PDF or exactly 2500 characters until the logged-in ACMMM 2026 form is inspected.
- Consolidated demand note: `rebuttal_demand_20260529.md`.

What must be checked inside the actual OpenReview form before drafting:

1. Whether the response is one global response per paper or one response/comment per review.
2. Character limit for each response field.
3. Whether the response is text-only or allows a PDF upload.
4. Whether tables are allowed as plain text/Markdown.
5. Whether references count toward the character limit.
6. Whether authors can edit submitted responses until deadline.

Operational assumption until verified: write as if there is a strict short OpenReview text limit, target <=2500 characters, no external links, no author-identifying information, and no revised PDF upload.

## Recommended Response Shape Under Short Limit

If OpenReview allows one global response:

1. Open with a compact evidence summary, not a polite preamble.
2. Use concern clusters: mechanism, detector attribution, cost, hyperparameters, related work, presentation.
3. Refer to reviewers by ID only when necessary.
4. Put numbers in compact semicolon-separated mini tables.

If OpenReview allows one response per reviewer:

1. Give M8du the most detailed mechanism/attribution answer.
2. Give KrEs novelty + detector + cost evidence.
3. Give jjVG k/m + related work + Figure 2.
4. Keep yx8u and ve3y focused on boundaries and claim calibration.

## Stop Conditions Before Writing Rebuttal

Do not start drafting final prose until these are known:

1. OpenReview response format and character limit.
2. Whether future flip/correctness diagnostics are available and favorable enough.
3. Whether detector attribution results support the claim.
4. Whether end-to-end latency can be reported without looking worse than expected.
5. Whether any HALC/ONLY/VHR comparison is verifiable, or only a related-work positioning table is safe.

## Bottom Line

This paper is currently not dead. It is a borderline submission with a plausible path to acceptance, but the path is narrow: we need to turn "nice engineered combination" into "measured, attributable, bounded inference-time verifier." The highest-return move is to satisfy M8du and jjVG while making KrEs less able to argue that the evaluation is incomplete. A strong one-shot rebuttal puts the odds around 55%; a mostly verbal rebuttal probably stays below 40%.
