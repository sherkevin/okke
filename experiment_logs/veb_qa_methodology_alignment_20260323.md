# VEB-QA Methodology Alignment (2026-03-23)

## Purpose

This note distills the useful parts of the `VEB-QA` first-principles analysis into a version that is directly actionable for the current `UniGround` engineering path.

It is not a new abstract theory note detached from code. It is the contract that should constrain:

- `POPE`-oriented method design
- runtime implementation choices
- training objectives
- experiment interpretation

The main rule is simple:

> The answer should not be allowed to bypass visual evidence and be determined directly by question-triggered prior.

## Core Diagnosis

For `POPE`, the failure mode is not "text output" itself. The real failure is that the host model can still follow a shortcut:

- evidence path: `(X, Q) -> E -> Y`
- prior shortcut: `Q -> Z -> Y`

where:

- `X` is the image
- `Q` is the question
- `E` is the minimal question-relevant visual evidence
- `Z` is question-triggered prior / world-knowledge bias

The current `token-local frontier reweighting` view is too weak as the primary method story. It operates too late and too locally.

For `POPE`, the correct control object is:

- queried object `o`
- object-present hypothesis `H_yes(o)`
- object-absent hypothesis `H_no(o)`

This means the method should be formulated as an object-level verifier, not as a generic decode-time token patcher.

## What We Keep From VEB-QA

The following ideas are strong and should become part of our official method:

1. Evidence bottleneck
   - The final yes/no decision should be constrained by object-conditioned visual evidence.
   - The answer should not be justified by question prior alone.

2. Prior-conflict view of hallucination
   - The important error is not generic accuracy loss.
   - The important error is when the model outputs the prior-favored answer despite conflicting visual evidence.

3. Evidence-first decision pipeline
   - For `POPE`, the system should first establish evidence for object presence / absence, then apply a decision bias at the answer stage.

4. Support-aware backoff
   - If evidence is weak or contradictory, the method should reduce override strength instead of forcing a hard answer based on noisy evidence.

## What We Do NOT Claim

The original `VEB-QA` report assumes a stronger architecture than our current system. We should not over-claim.

We do NOT currently implement:

- a fully re-architected answer head that can never access prior variables
- a strict end-to-end latent evidence mask with full theoretical guarantees
- a proof that prior shortcut is completely eliminated

Instead, our practical claim should be:

> We impose an external visual evidence bottleneck over the final object-level yes/no decision, so that host prior cannot dominate the answer without being checked against object-conditioned evidence.

This is weaker than strict causal isolation, but faithful to our current external-control engineering route.

## Engineering Translation

For the current `UniGround` implementation, the method should be reinterpreted as:

1. Parse object query from `Q`
   - `o = h(Q)`

2. Retrieve minimal visual evidence from image
   - `E = Retrieve(X, o)`
   - detector regions are sparse evidence candidates, not generic auxiliary features

3. Score paired object hypotheses
   - `e_pos = s(E, Q, H_yes(o))`
   - `e_neg = s(E, Q, H_no(o))`
   - optional uncertainty signal `e_unk`

4. Intervene only at the answer decision stage
   - use verifier evidence to steer the `yes/no` margin
   - do not treat arbitrary top-k frontier tokens as the main object of control

5. Back off when evidence is weak
   - abstention should reduce override strength
   - the method should not hard-flip decisions when evidence is ambiguous

## Current Working Form

The current practical form should be understood as:

- host score gap:
  - `g_host = score_host(yes) - score_host(no)`
- evidence score gap:
  - `g_evid = e_pos - e_neg`
- final decision control:
  - `g_final = g_host + Delta(g_evid, support)`

Where `Delta` should be asymmetric:

- more willing to repair wrong `no -> yes`
- more conservative when pushing `yes -> no`

This is motivated by our observed `POPE` failure pattern:

- precision is already high
- recall is too low
- therefore the current system is too eager to preserve or create conservative `no` answers

## Required Runtime Constraints

To stay aligned with the theory, the runtime must satisfy:

1. `object_label` is explicit runtime state
   - not recovered only from truncated prefix text

2. trigger runs at the answer step
   - not as a generic entropy gate over arbitrary decode steps

3. retrieval is object-conditioned
   - evidence retrieval should be tied to queried object semantics

4. verifier only controls answer labels
   - `yes/no` or `A/B` in binary settings

5. audit must measure prior-vs-evidence flips
   - especially `host_no_to_yes` and `host_yes_to_no`

## Training Alignment

The training objective for the next stage should emphasize:

1. answer supervision
   - binary presence/absence correctness

2. prior-conflict counterfactuals
   - same question, visually changed object status, answer must flip

3. support calibration
   - weak evidence should reduce intervention confidence

4. object-presence negatives
   - negatives should not just be generic semantic mismatches
   - they must explicitly teach the difference between "queried object absent" and "language prior suggests present"

## Experiment Interpretation Rules

For `POPE`, we interpret the three splits as follows:

- `random`
  - main general benchmark
- `popular`
  - measures whether frequency prior still dominates
- `adversarial`
  - strongest prior-conflict benchmark and the most theory-relevant split

The method is only convincing if gains are not isolated to a single split.

Minimum meaningful signal:

- `random` improves or at least does not regress
- `popular` is not below base
- `adversarial` clearly improves

## Immediate Engineering Priority

Before any full-volume run, the current engineering objective is:

1. stabilize the evidence-bottleneck verifier implementation
2. raise recall without collapsing precision
3. make `popular` match or beat `base`
4. continue treating `adversarial` as a first-class target, not an afterthought

## One-Sentence Method Contract

The method should now be described as:

> An external object-conditioned visual evidence bottleneck that constrains the final yes/no decision, reducing hallucination by preventing question prior from dominating the answer without visual support.
