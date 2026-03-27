# EKKO Methodology Specification

## Title 
EKKO: Mitigating Hallucinations in Multimodal Large Language Models via Evaluative Knowledge-Kernel Oracles

## Abstract
Fundamentally, Multimodal Large Language Models remain text-native autoregressive systems. While they accept various modalities, non-text inputs often serve merely as conditioning signals rather than first-class languages of admission. Consequently, hallucination is a structural temporal collapse: premature semantic commitments enter the autoregressive prefix before establishing robust causal links to the input context. Existing decode-time interventions primarily address retroactive failures by penalizing historical textual inertia. Crucially, they fail to anticipate future evolutionary trajectories, allowing locally plausible tokens to hijack the generation into text-dominated autoregression. In this paper, we reconceptualize hallucination mitigation as a chronological admission bottleneck and propose EKKO, a novel inference-time decoding criterion. Using vision-language generation as our primary testbed, EKKO mandates a rigorous tripartite spatiotemporal authorization for token admission. Beyond surviving retrospective scrutiny against historical inertia and securing query-conditioned visual anchor authorization at the current step, a candidate must pass a short-horizon prospective rollout. This forward-looking verification evaluates the commitment's causal trajectory, suppressing risky tokens that would collapse into pure textual autoregression. By elevating decoding to structural trajectory forecasting, EKKO enforces persistent visual grounding, significantly outperforming baselines across multiple MLLM benchmarks

## 1. Document Purpose

This document defines the methodology of `EKKO` as the authoritative design reference for subsequent implementation. The goal is to provide a rigorous, self-consistent, and development-ready specification that preserves the effective core of `OPERA` while introducing a differentiated method contribution suitable for a standalone paper.

This specification is intended to guide:

- algorithm implementation
- ablation design
- parameter selection
- logging and diagnostics
- future paper writing

The document prioritizes conceptual precision over implementation convenience. Where a design choice is uncertain, the document recommends a conservative default that minimizes regression risk relative to `OPERA`.

---

## 2. Method Positioning

`EKKO` is not defined as `OPERA+` in the weak sense of adding a small heuristic to an existing decoder. Instead, `EKKO` is a new decode-time candidate admission framework built on three temporally distinct evidence streams:

- `past`: whether the trajectory has already entered a summary-token-driven semantic inertia regime
- `current`: whether the candidate token is presently supported by query-relevant visual evidence
- `future`: whether admitting the candidate causes subsequent short-horizon continuation to remain vision-led or collapse into text-led continuation

The central claim of EKKO is:

> A candidate token should not be admitted into the prefix solely because it is locally plausible. It should be preferred when its history is not already corrupted by summary-token inertia, its current step is supported by query-relevant visual regions, and its short future continuation remains vision-dominant rather than text-inertial.

This makes EKKO a harmony-based candidate scoring framework:

- harmony across time: past, current, future
- harmony across visual evidence: GINO anchors, query relevance, host-model attention

The name `EKKO` reflects exactly this design principle.

---

## 3. Relationship to OPERA

### 3.1 What EKKO preserves

EKKO preserves the following parts of `OPERA` as first-class components:

- beam-search decoding
- candidate-level attention-based reweighting
- retrospective rollback / allocation mechanism
- history-based protection against semantic inertia

These are treated as proven engineering assets rather than disposable baselines.

### 3.2 What EKKO changes

EKKO introduces two substantive changes:

1. The visual support used at the current step is no longer defined by uniform aggregation over the entire image token span. Instead, it is upgraded to a query-conditioned soft anchor-aware visual support signal.
2. Candidate evaluation is expanded beyond past and current signals by adding a short-horizon future trajectory analysis, computed by continuing each candidate with greedy decoding and measuring whether the continuation remains more vision-led or text-led.

### 3.3 Why EKKO is not merely OPERA+

`OPERA` uses:

- historical rollback pressure
- current-step image-related attention

`EKKO` uses:

- historical rollback pressure
- current query-conditioned regional visual support
- future continuation-based vision-vs-text tendency

The additional future term and anchor-aware current term alter the criterion for candidate admission rather than merely tuning the original penalty.

---

## 4. High-Level Architecture

At each decoding step, EKKO scores each candidate token by combining three evidence streams:

1. `Past signal`
   Derived from the preserved OPERA retrospection / rollback logic.

2. `Current signal`
   Derived from candidate-specific attention to query-relevant visual regions, using soft anchor weighting built from GINO and query matching.

3. `Future signal`
   Derived by rolling out each candidate for a short fixed horizon with greedy decoding and comparing continuation attention mass on weighted visual tokens versus text tokens.

These three signals are fused into a unified candidate score.

---

## 5. The Three Time Lines

### 5.1 Past: historical inertia

The past signal answers:

> Has the trajectory already drifted into a summary-token-dominated inertial regime?

This part of EKKO preserves OPERA's historical logic:

- maintain a history window of decoding states
- record historical rollback locations
- detect repeated concentration patterns associated with semantic inertia
- trigger rollback and candidate reallocation when the inertial pattern is sufficiently consistent

The past module acts as a historical correction mechanism. It is responsible for preventing a trajectory that has already become self-reinforcing from continuing unchecked.

In the first version of EKKO, this module must remain unchanged except for software integration changes required by the added current and future modules.

### 5.2 Current: local query-conditioned visual support

The current signal answers:

> At the present decoding step, is this candidate supported by the visually relevant regions for the actual query?

This module begins from OPERA's current candidate reweighting mechanism, but changes how visual support is aggregated.

Instead of treating all image tokens equally, EKKO introduces:

- region proposals from GINO
- query-conditioned soft relevance matching over those proposals
- token-level soft weights over visual tokens derived from the matched proposals

The current signal therefore measures not just "is the model looking at the image", but "is the model looking at the query-relevant parts of the image".

### 5.3 Future: short-horizon continuation tendency

The future signal answers:

> If this candidate enters the prefix now, will the subsequent short continuation remain vision-dominant or become text-inertial?

This module is the primary new contribution of EKKO.

For each current-step candidate:

- temporarily append the candidate to the prefix
- continue decoding for a fixed horizon `m` with greedy search
- at each future step, measure:
  - weighted visual attention sum
  - text attention sum
- aggregate these over the horizon
- produce a future tendency score

The future signal is not intended to judge whether the continuation is the true gold answer. It is used to characterize the trajectory induced by the candidate:

- image-grounded continuation
- or language-inertia continuation

This is a trajectory judgment, not a semantic verifier.

---

## 6. GINO Anchor Design

### 6.1 Role of GINO

`GINO` is used as an external visual prior for identifying potentially important object regions. Its role is not to replace the host model's visual representation and not to hard-mask irrelevant visual tokens.

GINO provides:

- a set of anchor boxes `b_i`
- objectness or confidence `s_i`
- optional text labels / phrases / attributes if available

These anchors define candidate visual regions of interest.

### 6.2 Query-conditioned anchor matching

GINO alone is insufficient because not every salient object is relevant to the question. Therefore, EKKO applies query-conditioned matching.

Each anchor receives a query relevance score `r_i`.

The first implementation should use the following hybrid strategy:

1. Extract the main target nouns or phrases from the query.
2. Perform hard filtering against the anchor-associated phrase or class name where possible.
3. Within the filtered set, assign soft relevance via text similarity or phrase overlap.
4. If hard filtering fails, back off to similarity-only soft relevance.

This yields a continuous relevance value:

- `r_i in [0, 1]`

### 6.3 Why soft weighting instead of masking

EKKO does not hard-mask non-matching visual tokens for three reasons:

1. Detector recall is imperfect; masking can suppress useful visual context when anchors are incomplete.
2. Query matching can be noisy; a rigid mask increases brittleness.
3. OPERA's original full-image context is sometimes genuinely helpful; preserving it avoids catastrophic regressions.

Thus EKKO uses soft enhancement:

- matched anchors are strengthened
- unmatched anchored regions are unchanged or very slightly attenuated
- unanchored image tokens remain unchanged

### 6.4 Pre-processing, asynchrony, and cache policy

The anchor pipeline must be fully decoupled from autoregressive decoding.

Image-region detection is a static visual operation. For a given image, the object boxes do not change across decoding steps, and the user query is already known before generation begins. Therefore:

1. `GINO` box extraction must run before autoregressive decoding starts.
2. Query-to-anchor matching must also be computed before decoding starts.
3. The decode loop must perform only:
   - cache lookup
   - loading of precomputed anchor metadata
   - token-weight construction
   - soft multiplication with model attention

The required pre-processing outputs are:

- anchor boxes `b_i`
- confidence scores `s_i`
- optional detector phrases / labels
- query-conditioned relevance `r_i`
- token-membership mapping `M_ij`

These should be cached by `(image_id or image_path, query, detector thresholds, max_boxes)` so the decoder never reruns detection per token.

Normative engineering rule:

- running the detector inside the token-generation loop is forbidden
- running the detector once per candidate token is forbidden
- decode-time behavior must be pure lookup plus soft weighting

For multi-request serving, the detector stage may be executed asynchronously ahead of decode, but the decode worker must never stall on repeated per-step detector inference.

---

## 7. Token-Level Visual Weight Construction

Let `v_j` denote visual token `j` in the host sequence.

Let `M_ij = 1` if token `v_j` lies inside anchor `b_i`, and `0` otherwise.

Define the token weight:

`w_j = 1 + alpha_anchor * max_i (M_ij * r_i * s_i)`

Recommended first-version interpretation:

- if token `j` is not covered by any matched anchor, `w_j = 1`
- if token `j` is covered by a strongly matched, high-confidence anchor, `w_j > 1`

First-version conservative policy:

- matched-anchor tokens: enhanced
- anchored but unmatched tokens: unchanged
- unanchored tokens: unchanged

This makes the first version "enhance only", which is safer than explicit suppression.

Optional later-stage variant:

- anchored but unmatched tokens: slight decay `0.97 ~ 0.99`

This should not be used in the first implementation unless ablation demonstrates benefit.

---

## 8. Current-Step Score

### 8.1 OPERA current score

OPERA's current signal is derived from candidate-specific attention and reweights candidate scores at the present decoding step.

We denote the original OPERA local score as:

- `S_current_opera(c)`

where `c` is a candidate token.

### 8.2 EKKO anchor-aware visual support

For each candidate `c`, define a weighted visual support:

- `V_anchor(c) = sum_j w_j * attn(c -> v_j)`

where:

- `w_j` is the token weight from the GINO/query module
- `attn(c -> v_j)` is the attention mass from the candidate-induced attention pattern to token `v_j`

This score should be computed using the same candidate-specific attention extraction path already used by OPERA, differing only in the aggregation rule.

### 8.3 Conservative integration

To ensure EKKO does not collapse the effective OPERA current signal, the first version should not replace the original current score entirely.

Instead:

- `S_current_EKKO(c) = S_current_opera(c) + lambda_cur * V_anchor(c)`

This preserves OPERA's baseline effect while allowing region-aware enhancement.

For terminology alignment, the current-step contribution should be named:

- `K_kernel(c) = lambda_cur * V_anchor(c)`

where `K_kernel(c)` is the `Knowledge-Kernel` contribution and `V_anchor(c)` remains the underlying measured visual-support statistic.

### 8.4 Attention extraction protocol

Attention extraction is a first-class engineering requirement rather than an implementation detail.

EKKO must use the same candidate-specific attention path as the OPERA host, but with an explicit extraction rule:

- default layer policy: average the last `2-4` decoder layers
- recommended first default: average the last `4` layers when memory permits, otherwise the last `2`
- head policy: average over all heads inside the selected layers
- token policy: use the candidate-induced attention row corresponding to the currently evaluated token

This policy applies to both:

- current-step `V_anchor(c)`
- future-step `V_tau(c)` and `T_tau(c)`

Modern MLLMs often enable FlashAttention by default, which frequently suppresses explicit attention-weight outputs. Therefore, the implementation must explicitly specify one of the following strategies:

1. dynamically disable FlashAttention only for the last few layers used by EKKO attention extraction, while keeping earlier layers on the fast backend; or
2. register a model-specific hook that reliably exposes the needed attention weights from the final decoding layers.

Global disabling of FlashAttention for the entire model should be treated as a fallback, not the preferred path.

Normative engineering rule:

- do not implement current or future scoring by repeatedly calling a generic `model.generate()` black box inside a candidate loop
- the implementation must operate inside custom beam-search control where attention tensors and KV-cache are directly accessible

---

## 9. Future Continuation Module

### 9.1 Purpose

The future module estimates the trajectory tendency induced by a candidate token.

It asks:

> After candidate `c` is admitted, does the continuation continue to reference the weighted visual regions, or does it quickly become text-led continuation?

### 9.2 Rollout design

Future rollout must be implemented as a KV-cache-sharing batched simulation rather than a serial candidate loop.

For each decoding step:

1. Obtain the current OPERA top-`k` candidates.
2. Compute `V_anchor(c)` for each candidate using the current-step attention extraction path.
3. Apply the first filter:
   - if `V_anchor(c) = 0`, drop candidate `c` from the future-rollout set immediately
   - such candidates may still keep their OPERA score, but they do not consume future-rollout budget
4. For the surviving candidates, construct a rollout batch:
   - append the candidate tokens to the shared current prefix
   - expand the current `past_key_values` along the candidate dimension
   - run one GPU-parallel forward pass with batch size equal to the number of surviving candidates, not one serial pass per candidate
5. Use the first parallel forward pass as the rollout bootstrap step.
6. Continue greedy rollout for horizon `m` only on the active batch dimensions.
7. At each future step `tau`, record:
   - weighted visual attention sum
   - text attention sum
8. Aggregate the horizon into a future score.

Important:

- the continuation horizon is fixed
- the continuation is not interpreted as the final answer itself
- the continuation is only a trajectory analysis tool
- `past_key_values` must be reused rather than rebuilt from scratch for each candidate
- implementation must avoid `for c in candidates: model.generate(...)`

This is a hard requirement, not an optimization suggestion. The canonical implementation must break the naive `O(k x m)` serial-host-overhead pattern by reusing KV-cache and parallelizing candidate rollout on GPU.

### 9.3 Why fixed horizon

QA generation does not have a reliable semantic end marker. There is no universal internal event corresponding to "the answer is now complete". There are only engineering stop conditions such as:

- EOS token
- stop string
- max token length

Because EKKO is not trying to measure answer completeness, but rather short-horizon future tendency, a fixed horizon is more stable than "generate until answer end".

### 9.4 Visual and text sums

At continuation step `tau`, define:

- `V_tau(c) = sum_j w_j * attn_tau(c -> v_j)`
- `T_tau(c) = sum_q attn_tau(c -> t_q)`

where:

- `v_j` are visual tokens
- `t_q` are text-prefix tokens

The future aggregate uses sums, not means.

This choice is deliberate:

- sums preserve the total amount of visual commitment over the short trajectory
- means can hide trajectories that only look visual in the first step and then collapse into pure language inertia

### 9.5 Future score

Recommended first version:

- `F_future(c) = sum_{tau=1..m} V_tau(c) - lambda_txt * sum_{tau=1..m} T_tau(c)`

Diagnostic companion metric:

- `R_future(c) = sum V_tau(c) / (sum V_tau(c) + sum T_tau(c) + eps)`

Use:

- `F_future` for actual reranking
- `R_future` for logging and diagnostics

For terminology alignment, the future contribution should be named:

- `O_rollout(c) = lambda_fut * F_future(c)`

where `O_rollout(c)` is the `Oracle Rollout` contribution and `F_future(c)` remains the underlying measured future-trajectory statistic.

### 9.6 Branch-and-bound, early stopping, and cascade pruning

Future rollout must not be implemented as a rigid "always run `m` steps for everyone" loop.

Two pruning stages are required:

1. `Knowledge-Kernel gate`
   - if `V_anchor(c) = 0`, candidate `c` is excluded from rollout immediately
   - this is the first branch-and-bound filter

2. `Oracle early-stop gate`
   - after rollout step `tau = 1`, compute the visual ratio
   - `R_1(c) = V_1(c) / (V_1(c) + T_1(c) + eps)`
   - if `R_1(c) < tau_abort`, terminate that candidate's rollout immediately and assign a very low future score

Recommended first default:

- `tau_abort = 0.1`

Cascade pruning requirement:

- rollout must maintain an active-branch mask
- once a branch fails the early-stop criterion, it must be removed from the next rollout batch compaction
- later rollout steps should therefore run on a shrinking active batch rather than continuing to spend compute on dead branches

This is the required engineering realization of branch-and-bound in EKKO: zero-support candidates are screened before rollout, and low-visual-ratio branches are cut off after the first future step.

---

## 10. Fusion of Past, Current, and Future

### 10.1 Fusion principle

The three time lines are not redundant:

- past corrects historical inertia
- current measures query-relevant local visual support
- future measures post-admission trajectory tendency

Therefore the final candidate score should combine all three.

### 10.2 Proposed fused score

Let:

- `S_opera(c)` be the original candidate score produced by OPERA's decoding path
- `V_anchor(c)` be the current-step anchor-aware visual support
- `F_future(c)` be the continuation-based future score
- `K_kernel(c) = lambda_cur * V_anchor(c)` be the `Knowledge-Kernel` contribution
- `O_rollout(c) = lambda_fut * F_future(c)` be the `Oracle Rollout` contribution

To align the theory with an A*-style implementation interface, define:

- `g(c) = S_opera(c) + K_kernel(c)`
- `h(c) = O_rollout(c)`
- `f_EKKO(c) = g(c) + h(c)`

Equivalently:

- `f_EKKO(c) = S_opera(c) + lambda_cur * V_anchor(c) + lambda_fut * F_future(c)`

Implementation naming rule:

- low-level measurements may keep names such as `V_anchor` and `F_future`
- score-contribution variables and comments should use `K_kernel`, `O_rollout`, `g(c)`, `h(c)`, and `f_EKKO(c)` to match the methodology

This formulation preserves OPERA as the historical backbone while packaging EKKO's additional information streams in a code structure that is consistent with best-first admission logic.

### 10.3 Rollback precedence

Rollback remains higher priority than future reranking.

If OPERA decides rollback is necessary, EKKO must:

- execute rollback first
- discard future scoring results computed on invalidated states
- recompute future scores after rollback if needed

This prevents conflicts between historical correction and future tendency estimation.

---

## 11. Detailed Processing Flow

For each decoding step:

0. Before decoding starts:
   - run `GINO` preprocessing
   - compute query-conditioned anchor relevance
   - cache anchor metadata and token-membership lookup tables

1. Run the standard OPERA step to obtain:
   - candidate tokens
   - historical state
   - rollback decision variables

2. If rollback triggers:
   - perform OPERA rollback
   - restore earlier state
   - restart the current step

3. If no rollback:
   - compute anchor-aware current visual score `V_anchor(c)` for each top-k candidate
   - form `K_kernel(c) = lambda_cur * V_anchor(c)`
   - drop candidates with `V_anchor(c) = 0` from the future-rollout set
   - pack surviving rollout candidates into one batch and reuse the current `past_key_values`
   - run one parallel bootstrap forward pass on GPU
   - continue greedy rollout only on active batch branches
   - after rollout step 1, early-stop any branch with `R_1(c) < tau_abort`
   - compact the rollout batch after each pruning event
   - accumulate `F_future(c)` and form `O_rollout(c) = lambda_fut * F_future(c)`
   - compute final `f_EKKO(c) = g(c) + h(c)`
   - rerank the top-k candidates using `f_EKKO(c)`
   - continue beam search with the reranked candidates

This preserves the OPERA control flow while cleanly appending EKKO-specific logic.

---

## 12. Parameter Specification

### 12.1 Preserved OPERA parameters

- `num_beams = 5`
- `num_attn_candidates = 5`
- `scale_factor = 50`
- `threshold = 15`
- `penalty_weights = 1.0`

These should remain identical to the stable OPERA baseline in the first implementation.

### 12.2 EKKO parameters

- `alpha_anchor = 0.5`
  - anchor enhancement strength

- `lambda_cur = 0.25`
  - weight for current anchor-aware support

- `lambda_fut = 0.5`
  - weight for future trajectory score

- `lambda_txt = 1.0`
  - text-inertia subtraction inside future score

- `future_horizon = 4`
  - number of greedy continuation steps per candidate

- `future_topk = 5`
  - number of candidates from current beam state that receive future scoring

- `tau_abort = 0.1`
  - first-step visual-ratio early-stop threshold for rollout pruning

- `attention_layers = last 2-4 layers`
  - layer window used for attention extraction

- `attention_head_reduce = mean over heads`
  - default head reduction rule for EKKO attention statistics

- `detector_box_threshold = 0.25`
- `detector_text_threshold = 0.25`
- `max_boxes = 10`

These are recommended starting defaults and must be treated as ablation targets rather than fixed truths.

---

## 13. Why EKKO Should Improve on OPERA

EKKO is expected to improve over OPERA for two main reasons.

### 13.1 Better spatial specificity

OPERA measures whether a candidate attends to the image.  
EKKO measures whether a candidate attends to the query-relevant parts of the image.

This is especially important for:

- POPE
- object-centric QA
- caption hallucination tied to specific entities

### 13.2 Better trajectory discrimination

OPERA current scoring is local.
EKKO future scoring is short-horizon trajectory-aware.

This allows EKKO to distinguish:

- candidates whose current step looks visually plausible but quickly collapse into text inertia
- candidates whose future continuation remains image-grounded

Thus EKKO should be better at identifying "dangerous entry tokens" that appear locally valid but induce non-visual continuation.

---

## 14. Failure Modes and Safety Defaults

The first version must include strong fallback behavior.

### 14.1 Detector failure

If GINO produces no anchors:

- set all visual token weights to 1
- reduce to OPERA-like full-image aggregation

### 14.2 Query matching failure

If no anchors match query terms:

- preserve detector boxes for diagnostics
- set all weights to 1 in the scoring path

### 14.3 Future rollout failure

If continuation rollout fails numerically or structurally:

- set `F_future = 0`
- continue with OPERA + current anchor-aware scoring only

### 14.4 Full fallback

If:

- `lambda_cur = 0`
- and `lambda_fut = 0`

then EKKO must reduce exactly to the underlying OPERA implementation.

This fallback property is mandatory.

---

## 15. Ablation Plan

To validate EKKO rigorously, the following variants must be run:

1. `OPERA`
2. `OPERA + anchor current only`
3. `OPERA + future only`
4. `EKKO full`
5. `EKKO full without GINO`
6. `EKKO full without future`
7. `EKKO full with horizon m = 2 / 4 / 6`
8. `EKKO full with alpha_anchor = 0.25 / 0.5 / 0.75`
9. `EKKO full with lambda_fut = 0.25 / 0.5 / 0.75`

This ablation structure is required to isolate whether improvements come from:

- anchor-aware current support
- future continuation tendency
- or their interaction

---

## 16. Logging and Diagnostics Requirements

Every EKKO run should log:

- selected candidate token
- top-k candidate tokens before and after EKKO rerank
- anchor boxes and their query relevance
- token-level visual weight statistics
- current `V_anchor`
- future `sum V_tau`
- future `sum T_tau`
- future score `F_future`
- diagnostic ratio `R_future`
- whether rollback occurred at the step
- number of candidates removed by the `V_anchor = 0` gate
- number of rollout branches removed by early stopping
- active rollout batch size at each future step
- selected attention layers and head-reduction policy
- whether FlashAttention was partially disabled or attention was obtained through hooks

This logging is essential both for debugging and for paper-ready interpretability.

---

## 17. Implementation Modules

The implementation should be separated into the following modules:

### 17.1 `opera_core`

Contains preserved OPERA logic:

- beam search
- current candidate extraction
- rollback / retrospection

### 17.2 `knowledge_kernel_evaluator`

Contains:

- GINO anchor extraction
- query-conditioned anchor relevance scoring
- mapping from anchors to visual-token weights
- preprocessing, asynchronous scheduling, and cache lookup for anchor metadata
- current-step `Knowledge-Kernel` evaluation `K_kernel(c)`

### 17.3 `oracle_rollout_simulator`

Contains:

- shared-KV batched candidate continuation
- accumulation of weighted visual attention sums
- accumulation of text attention sums
- branch-and-bound filtering
- early stopping and active-batch compaction
- `Oracle Rollout` score output

### 17.4 `EKKO_fusion`

Contains:

- fusion of `S_opera`, `K_kernel`, `O_rollout`
- A*-style variable packaging `g(c)`, `h(c)`, `f_EKKO(c)`
- candidate reranking
- fallback handling

This modularization is strongly recommended to keep EKKO analyzable and debuggable.

---

## 18. Final Method Definition

EKKO is a decode-time candidate admission framework that preserves OPERA's historical rollback and local candidate reweighting while extending candidate evaluation along three coordinated dimensions:

- `past`, via summary-inertia-aware rollback
- `current`, via query-conditioned soft anchor-aware visual support
- `future`, via fixed-horizon continuation analysis that measures whether a candidate induces image-grounded continuation or text-inertial continuation

These three signals are fused into a unified candidate score, where visual support is no longer measured uniformly over the entire image token span, but instead harmonized with external region anchors and query relevance.

The method is designed to remain backward-compatible with OPERA when its new weights are zero, while enabling a more spatially precise and temporally complete decoding rule.

---

## 19. Development Rule

Any implementation that claims to instantiate EKKO must satisfy all of the following:

- preserve OPERA rollback semantics unless explicitly ablated
- use soft anchor weighting, not hard visual token masking, in the first implementation
- use fixed-horizon greedy continuation for future scoring
- use shared-KV batched rollout rather than serial per-candidate `generate()` calls
- precompute and cache GINO anchors before autoregressive decoding starts
- use explicit attention extraction from the last `2-4` layers, with a documented FlashAttention-handling path
- apply rollout branch pruning via `V_anchor = 0` filtering and first-step early stopping
- use attention sums, not only averages, in the future module
- allow exact fallback to OPERA

If any of these conditions are violated, the implementation must be documented as a variant, not as the canonical EKKO method.
