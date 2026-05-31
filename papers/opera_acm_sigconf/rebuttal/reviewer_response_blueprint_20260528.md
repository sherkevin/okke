# CHORD Reviewer Response Blueprint, 2026-05-28

Scope: strategic response blueprint only. This is not final rebuttal text.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record. Earlier simulated or practice reviews are excluded.

## Overall Strategy

The response should not read like a defense of every original choice. It should read like a concise evidence update that directly removes the main decision blockers.

The global message should be:

1. We agree the original submission under-explained attribution, cost, and recent related work.
2. We have added targeted diagnostics showing what CHORD contributes beyond its components.
3. We now distinguish two operating regimes:
   - Past+Current: latency-oriented, strongest practical setting for discriminative hallucination control.
   - Full CHORD: quality-oriented, especially useful when open-ended generation allows early errors to compound.
4. We qualify the claim boundary:
   - training-free for the base MLLM, but detector-assisted with frozen external perception;
   - attention is used as an operational grounding score, not as a causal explanation;
   - broader relation/attribute reasoning generality remains a limitation.

## What To Concede

Concede these directly because they are true and reviewers will not accept denial:

- Full CHORD has nontrivial inference overhead.
- The submitted main table did not include enough recent hallucination baselines.
- The submitted paper did not sufficiently isolate Grounding DINO's role.
- The future rollout mechanism was supported mostly by end-to-end scores, not by direct flip/correctness diagnostics.
- `training-free` needs qualification because the system uses a frozen external proposer.
- Figure 2 can be made clearer.

Conceding these points is not weakness. It makes the response credible and opens space for the evidence that follows.

## What To Push Back On

Push back only where we can ground it in evidence:

- Not "just a detector": show no/random-anchor and same-anchor controls.
- Not "just OPERA": show Past vs Past+Current vs Past+Future vs Full and explain the admission-time coordination rule.
- Not "uncontrolled overhead": show detector time, decode ITL, total latency, and the Past+Current/Full operating trade-off.
- Not "hyperparameter cherry-picking": show k/m sweep.
- Not "missing all recent work": add ONLY, VHR/VHD, HALC positioning and direct numbers only if verifiable.

## Core Response Modules

| Module | Purpose | Evidence needed | Reviewer concerns covered |
|---|---|---|---|
| M1. Mechanism diagnostic | Prove future rollout is doing a measurable job, not just adding compute. | Full vs Past+Current flip rate; correct/harmful flip percentages; POPE and CHAIR breakdown. | M8du, KrEs, yx8u |
| M2. Detector attribution | Prove gains are not simply Grounding DINO. | Real anchors vs no anchors/uniform weights/random anchors; Past+Future without current anchors; same-anchor non-CHORD control if feasible. | KrEs, M8du, yx8u, ve3y |
| M3. Cost accounting | Remove suspicion that latency is understated. | Detector proposal time, decode ITL, total answer latency, peak VRAM, batch-size note. | KrEs, M8du, jjVG, ve3y |
| M4. Hyperparameter robustness | Answer k=5/m=3 criticism. | k={3,5,7}, m={1,2,3,4} small grid with quality and ITL. | jjVG, M8du |
| M5. Related-work completeness | Remove "missing obvious recent work" objection. | Add ONLY, VHR/VHD, HALC discussion; compare if reliable. | jjVG, KrEs, M8du |
| M6. Claim calibration | Prevent WA reviewers from dropping. | Revised wording promises: training-free qualifier, attention qualifier, softened terminology, limitation on relation/attribute hallucinations. | yx8u, ve3y, KrEs |
| M7. Presentation fix | Resolve easy presentation complaint. | Promise cleaner Figure 2 lanes. | jjVG |

## Recommended Ordering In The Rebuttal

If there is one global response field, use this order:

1. Start with a compact summary of new evidence: mechanism, detector controls, end-to-end cost, k/m, related work.
2. Explain novelty in one sentence: CHORD is a coordinated admission-time verifier, not a standalone detector or unrestricted search.
3. Give the mechanism diagnostic first because M8du explicitly asks for it and is willing to raise the score.
4. Give detector attribution second because it is the core concern for KrEs and M8du.
5. Give cost accounting third because it affects all skeptical reviewers.
6. Give k/m and related work next.
7. End with camera-ready changes: Figure 2 cleanup, terminology softening, limitations.

If OpenReview has per-reviewer response fields:

- M8du: answer all five questions directly; this is the most important conversion target.
- KrEs: lead with novelty separation, detector attribution, and end-to-end cost.
- jjVG: lead with k/m sweep, ONLY/VHD/HALC related-work update, and Figure 2 cleanup.
- yx8u: lead with claim calibration and limitation honesty.
- ve3y: reinforce operating frontier and practical deployment transparency.

## Reviewer-by-Reviewer Blueprint

### Reviewer jjVG

Current state: Borderline, low confidence. They already accept the motivation and effectiveness but see missing completeness and cost issues.

Goal: convert to Weak Accept.

Must answer:

- Cost is real, but now fully reported.
- k=5 and m=3 are justified by a sweep.
- Related work now covers ONLY and VHD/VHR.
- Figure 2 will be simplified.

Do not over-explain novelty here. This reviewer needs clean completeness fixes.

### Reviewer KrEs

Current state: Weak Reject, knowledgeable. This is the hardest reviewer.

Goal: at minimum neutralize the rejection; ideally move to Borderline.

Must answer:

- CHORD's novelty is the coordinated admission-time verification policy.
- Grounding DINO is not the whole reason for gains, supported by detector controls.
- End-to-end latency and memory are now transparent.
- Recent baselines are acknowledged and positioned.

Tone:

- Respect the criticism.
- Avoid saying "we already did this" unless the evidence is in the response.
- Do not claim a fundamentally new paradigm; claim a measured and useful inference-time verifier.

### Reviewer yx8u

Current state: Weak Accept, knowledgeable. They like the work but worry about overclaiming.

Goal: keep them from dropping.

Must answer:

- We qualify training-free as base-MLLM training-free.
- We do not treat attention as causal explanation.
- We acknowledge detector quality and broader hallucination types as limitations.
- We soften strong terminology in camera-ready.

This reviewer rewards honesty. Do not oversell.

### Reviewer ve3y

Current state: Weak Accept, familiar. They find the paper solid but incremental.

Goal: preserve support.

Must answer:

- CHORD is useful as a practical inference-time framework even if novelty is integration-oriented.
- The operating frontier is clearer now.
- Runtime cost is explicitly reported.

This can be short.

### Reviewer M8du

Current state: Borderline, knowledgeable, explicitly willing to raise.

Goal: convert to Weak Accept.

Must answer their five questions in order:

1. Future term: report how often Full flips Past+Current and how often flips are correct.
2. Detector dependency: report Past+Future without anchor term and same-anchor/non-CHORD or no/random-anchor controls.
3. Latency: report decoder-only and end-to-end latency separately.
4. Hyperparameters: report k/m analysis.
5. Detector robustness: report failure/noisy-anchor cases or at least a structured diagnostic.

This reviewer is the rebuttal's center of gravity.

## Evidence Thresholds

The response is strong enough only if:

- Future diagnostics show either clear positive value or a credible narrower claim such as "future helps mainly on open-ended CHAIR rather than POPE yes/no."
- Detector controls show real anchors outperform no/random anchors or that CHORD still helps under uniform anchors.
- End-to-end cost is not hidden; even if high, it is presented as an operating-point trade-off.
- k/m sweep does not reveal extreme fragility.
- Missing related work is addressed honestly.

If any P0 result is unfavorable, adjust the claim instead of forcing the original story:

- If future term weak on POPE: position Past+Current as the recommended POPE setting and Full CHORD as open-ended/CHAIR-oriented.
- If detector controls show Grounding DINO dominates: present CHORD as detector-assisted admission control and stop claiming detector-independent mechanism.
- If latency is high: emphasize transparent selectable regimes, not deployment cheapness.

## What Not To Say

- Do not say "due to space limitations" as the main reason for missing baselines.
- Do not argue that reviewers misunderstood the method.
- Do not claim "negligible overhead."
- Do not call the method "fully training-free" without qualifier.
- Do not promise unsupported comparisons to ONLY/VHR/HALC.
- Do not use weak small-slice diagnostic results as if they prove the mechanism.
- Do not make the rebuttal sound like a list of future promises; include numbers wherever possible.

## Best One-Shot Narrative

The cleanest narrative is:

"The reviewers correctly identified three missing pieces: attribution, cost, and current related-work coverage. We added targeted diagnostics. They show CHORD should be understood as a bounded admission-time verifier with two operating points. Past+Current is the efficient setting; Full CHORD adds rollout for quality-oriented open-ended generation. The camera-ready will make this boundary explicit, add detector and hyperparameter controls, include recent methods, and clarify limitations."

This is the narrative to support with tables. Do not submit this paragraph directly until the numbers are known.
