# Reviewer Root-Concern Map for Submission 8826

Scope: decision-facing reading of the five official reviews. This is not rebuttal prose. It separates surface requests from the evidence that would actually change reviewer confidence.

## Source Files

- Official review archive: `papers/opera_acm_sigconf/rebuttal/reviews_20260528.md`
- Existing pre-rebuttal analysis: `papers/opera_acm_sigconf/rebuttal/pre_rebuttal_analysis_20260528.md`
- Main source checked: `papers/opera_acm_sigconf/sample-sigconf.tex`
- Supplement checked: `papers/opera_acm_sigconf/supplementary.tex`
- Local diagnostic slice checked: `remote_chiro_patch/tests/ablations_random_0_64/*.json`

## Global Read

The reviews are not rejecting the topic. Fit is strong and the basic idea is considered reasonable. The real blocker is trust in attribution:

1. Is CHORD more than OPERA plus Grounding DINO plus rollout?
2. Does the future term actually change token admission in correct ways?
3. Are gains from the decoding rule, or from extra object proposals?
4. Is the cost reported honestly end to end?
5. Are missing recent baselines hiding a weaker comparison?

The response should therefore be evidence-first. A rhetorical novelty argument alone will not move the knowledgeable reviewers.

## Reviewer-Level Read

### jjVG

- Rating / confidence: Borderline, low confidence.
- Surface requests: cost, Figure 2 clarity, k/m rationale, ONLY and VHD/VHR related work.
- Real concern: the paper looks promising but incomplete. This reviewer is not trying to disprove the method; they need obvious holes closed so they can safely support it.
- Best movement lever: concise table or paragraph for k/m, explicit ONLY/VHD/VHR positioning, and a clear cost caveat. Figure cleanup is useful but not score-defining.
- Risk: over-answering theory while leaving named papers and k/m unexplained.

### KrEs

- Rating / confidence: Weak Reject, knowledgeable.
- Surface requests: novelty, Grounding DINO role, end-to-end latency, memory, batch size, stronger baselines/models.
- Real concern: attribution failure. They think the contribution may be an engineered mixture whose gains could be explained by external detection or by known components.
- Best movement lever: detector-control ablations plus end-to-end cost accounting. The response needs to isolate the decoding rule from the proposer.
- Minimum credible evidence: real anchors vs no anchors/random anchors/Past+Future without anchors, plus detector time and decode time reported separately.
- Risk: saying "training-free" without qualifying "for the base MLLM" will irritate this reviewer.

### yx8u

- Rating / confidence: Weak Accept, knowledgeable.
- Surface requests: novelty, detector dependence, attention reliability, broader backbones/benchmarks, latency, terminology.
- Real concern: claim discipline. They already accept the paper as useful, but they see the framing as stronger than the evidence.
- Best movement lever: protect the Weak Accept by conceding limits and adding diagnostics. Do not oversell structural temporal collapse or universal attention reliability.
- Minimum credible evidence: attention/layer-window framing as an operational feature, not an explanation; detector failure cases; honest scope limits.
- Risk: a defensive rebuttal can convert this from "useful but limited" into "overclaimed."

### ve3y

- Rating / confidence: Weak Accept, familiar.
- Surface requests: incremental novelty, external proposer complexity, runtime overhead.
- Real concern: practical value versus cost. This reviewer likes the framework but needs the trade-off to feel honest.
- Best movement lever: emphasize CHORD as an operating-point family: Past+Current for latency-sensitive use, Full CHORD for quality-oriented open-ended generation.
- Minimum credible evidence: complete efficiency table and a clear statement that Full CHORD is not the deployment-cheap setting.
- Risk: spending too much rebuttal space on this reviewer. They are already supportive; avoid creating new doubt.

### M8du

- Rating / confidence: Borderline, knowledgeable, explicitly willing to raise.
- Surface requests: future flip rate/correctness, detector controls, end-to-end latency, hyperparameter analysis, detector robustness.
- Real concern: mechanism validation. They are not satisfied by final benchmark gains because the paper claims a mechanism: current plus future admission verification.
- Best movement lever: future-arbitration diagnostic. Report how often Full changes Past+Current decisions, whether those changes help, and where the gains concentrate.
- Minimum credible evidence: Full vs Past+Current flip rate, correct-flip and wrong-flip counts, plus detector attribution controls.
- Risk: answering with only main benchmark numbers. That directly misses their question.

## Shared Blockers

### P0: Mechanism of the future term

Surface wording: "Does future rollout do anything?"

Actual gate: prove that future arbitration changes admitted tokens and that those changes are useful, especially on open-ended hallucination. Aggregate POPE/CHAIR/MMBench scores are not enough.

Needed evidence:
- Full vs Past+Current answer-level difference rate.
- Token-admission flip rate where diagnostics are available.
- Correct-flip and wrong-flip counts.
- Breakdown for POPE and CHAIR if possible.

Current local boundary:
- The existing random 64-sample diagnostic is not positive proof. In that slice, `anchor_current_only` is slightly better than `chord_full_default`, and many variants match OPERA. Treat it as warning evidence only.

### P0: Detector attribution

Surface wording: "What is the role of Grounding DINO?"

Actual gate: prove CHORD is not mainly Grounding DINO doing object recognition outside the MLLM.

Needed evidence:
- Full CHORD with real anchors.
- Full CHORD with no anchors or uniform visual weights.
- Random-box anchors matched for count/area.
- Past+Future without current anchors.
- If feasible, same detector metadata with no CHORD reranking.
- Failure cases where the detector misses or corrupts anchors.

### P0: End-to-end cost

Surface wording: "Latency is high."

Actual gate: reviewers suspect the reported ITL excludes detector setup and may undersell deployment cost.

Needed evidence:
- Detector proposal time per image/query.
- Decode-stage ITL.
- Total answer latency under fixed generation length.
- Peak VRAM.
- Batch-size note, even if only batch size 1 versus a small batch.

Safe framing:
- Full CHORD is quality-oriented and slower.
- Past+Current is the latency-sensitive operating point.

### P0/P1: Missing recent baselines

Surface wording: "Add ONLY, VHD/VHR, HALC."

Actual gate: reviewers want confidence that the comparison set is not cherry-picked.

Needed evidence:
- At minimum, related-work and comparison-axis table covering ONLY, VHD/VHR, HALC.
- Direct numbers only if the implementation and evaluation are verifiable.

Risk:
- Do not dismiss missing baselines as space limitation. Named omissions must be acknowledged.

### P1: k/m robustness

Surface wording: "Why k=5 and m=3?"

Actual gate: if the method is training-free, reviewers still need to know whether it is brittle to hand-tuned decoding knobs.

Needed evidence:
- k in {3, 5, 7}.
- m in {1, 2, 3, 4}.
- Quality and latency together.

### P1: Novelty

Surface wording: "Incremental combination."

Actual gate: make the contribution a coordination rule with observable behavior, not a renamed bundle of existing components.

Safe novelty claim:
- CHORD contributes admission-time coordination of retrospective rollback, query-conditioned current support, and short-horizon continuation stability.
- Each component has prior cousins; the contribution is the coupled decision policy and its operating frontier.

Unsafe novelty claim:
- Do not imply every component is new.
- Do not imply formal causality for "structural temporal collapse."

## Practical Rebuttal Priority

1. Future mechanism diagnostic.
2. Detector-control ablations.
3. End-to-end cost table.
4. k/m sweep.
5. Related-work patch for ONLY, VHD/VHR, HALC.
6. Figure 2 cleanup and terminology softening.

If only one reviewer can be moved, target M8du. If only one rejection risk can be reduced, target KrEs. Protect yx8u and ve3y by being honest, not by overclaiming.

