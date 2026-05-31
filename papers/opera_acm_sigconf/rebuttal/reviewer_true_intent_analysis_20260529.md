# CHORD Reviewer True-Intent Analysis, 2026-05-29

Scope: this is an internal decision document for rebuttal prioritization. It reads the five official reviewers as decision-makers, not as a list of surface complaints. It is not rebuttal prose and not new experimental evidence.

Canonical source: `reviews_20260528.md` only. Existing planning files are used as secondary context, but no simulated review artifacts are treated as evidence.

## Hard Answer: Does This Guarantee A High Score?

No. Even a perfect response to this document cannot guarantee a high score or acceptance.

Reason: the reviews contain real, not merely rhetorical, constraints. KrEs and M8du are not asking for better wording only; they are asking for missing diagnostics. yx8u and ve3y already accept the paper conditionally but still cap the novelty as moderate/incremental. jjVG is low-confidence and may move with concrete fixes, but low confidence also means their final score can be sensitive to the meta-review discussion. Therefore the best realistic goal is not "certain high score"; it is:

1. move M8du from Borderline to Weak Accept if mechanism diagnostics are credible;
2. move jjVG from Borderline to Weak Accept if the obvious completeness gaps are closed;
3. soften KrEs from Weak Reject to Borderline or weak support if detector attribution and end-to-end cost are isolated;
4. preserve yx8u and ve3y by avoiding overclaiming.

If all P0 evidence is clean, the score profile could plausibly become something like WA/Borderline/WA/WA/WA or WA/WA/Borderline/WA/WA. That is a strong rebuttal outcome. It is not a promise of high confidence acceptance, because novelty and baseline scope may still cap the paper.

If the evidence is weak, this document should force us to downgrade the rebuttal claims rather than hide the weakness. A weaker but honest rebuttal is still preferable to an aggressive rebuttal that loses the two Weak Accepts.

## Can This Document Solve Their Doubts?

It can solve only the doubts that are addressable by rebuttal evidence or disciplined wording. It cannot erase fundamental limits of the submitted paper.

| Doubt | Can be solved in rebuttal? | What would count as solved | Residual risk even after solving |
|---|---:|---|---|
| Future term mechanism is inferred, not validated | Yes, if new diagnostics are clean | Full vs Past+Current flip/correctness, harmful-flip count, CHAIR continuation benefit | If flips are rare, Future remains a quality-mode rather than core-mechanism story. |
| Detector/Grounding DINO attribution unclear | Partly | real/no/random anchor controls, Past+Future, same-anchor non-CHORD | If controls are weak, must narrow to detector-assisted heuristic. |
| End-to-end cost is underspecified | Yes | proposal time, decode ITL, total latency, peak VRAM, batch setting | Full CHORD may still be too slow; then P+C must become the practical regime. |
| k=5, m=3 look hand-tuned | Yes | k/m sweep with Pareto analysis | If another setting dominates, the default must be reframed or updated. |
| Missing ONLY/VHD/VHR/HALC | Partly | exact citations/axis table; direct numbers only if reproducible | If no direct runs, some reviewers may still see evaluation as incomplete. |
| Novelty is incremental | Only partly | show coupled admission policy has behavior not reducible to parts | Ingredient-level novelty remains moderate; this caps upside. |
| Scope is only two 7B models and object-heavy benchmarks | Only partly | one stronger-backbone/broader-scope diagnostic, or explicit limitation | Without extra runs, generality remains a fair limitation. |
| Attention reliability | Mostly by wording, not proof | attention as operational feature, not causal explanation | Does not prove architecture-general grounding. |

This is why the document should be used as a rebuttal control surface, not as a guarantee checklist.

## Evidence Hierarchy Used In This Document

| Level | Evidence type | How it is used |
|---|---|---|
| 1 | Raw official review text, copied in Appendix A | Highest authority for reviewer intent. |
| 2 | Submitted paper/supplement facts and measured submitted tables | Used to judge what can already be claimed. |
| 3 | Existing local 64-sample diagnostic | Warning evidence only, never positive support for Full/Future. |
| 4 | Inference about hidden intent | Clearly labeled as interpretation, not fact. |
| 5 | Expected experiment plans | Used only to decide what to run next, not as rebuttal evidence. |

## Executive Judgment

The five reviews are not saying "the problem is unimportant" or "CHORD is clearly wrong." The paper has strong fit, generally accepted motivation, and two supportive Weak Accepts. The real threat is that reviewers do not yet trust the attribution chain:

1. They see CHORD as a plausible engineering combination, but not yet as a validated mechanism.
2. They suspect Grounding DINO and added compute may explain too much of the gain.
3. They worry the paper's language is stronger than the evidence can support.
4. They see missing recent baselines and k/m sensitivity as signs of an incomplete evaluation.

The rebuttal should therefore not be a defense of "our method is novel and good." It should be a controlled credibility repair:

- Admit that aggregate benchmark scores alone do not isolate the mechanism.
- Add compact diagnostics showing what Future, Current, detector anchors, and latency each contribute.
- Reframe CHORD as a detector-assisted admission-time verifier family, with Past+Current as the practical regime and Full CHORD as the quality-oriented regime.
- Avoid any language that sounds like universal causality, detector independence, or hidden deployment efficiency.

## The Real Committee Dynamic

| Reviewer | Current stance | Real role in the committee | What they need to vote upward or stay supportive | Main danger |
|---|---:|---|---|---|
| jjVG | Borderline, confidence 1 | Low-confidence completeness checker | Obvious gaps closed: k/m, related work, cost, Figure 2 | If named omissions remain, they safely stay Borderline. |
| KrEs | Weak Reject, confidence 3 | Attribution prosecutor | Evidence that gains are not just Grounding DINO + known decoding tricks, and that cost is complete | If attribution is not isolated, they keep Reject. |
| yx8u | Weak Accept, confidence 3 | Supportive but claim-sensitive expert | Honest limitations, detector-assisted wording, attention as operational signal | Overclaiming can turn support into skepticism. |
| ve3y | Weak Accept, confidence 2 | Supportive practical-value reviewer | Clear operating-point trade-off and realistic runtime story | If Full is sold as cheap, trust erodes. |
| M8du | Borderline, confidence 3 | Convertible mechanism auditor | Direct answer to whether Future flips decisions correctly and whether detector controls pass | If answered well, this is the clearest score-upside reviewer. |

The likely rebuttal objective is not to convert every reviewer to Strong Accept. It is to move M8du and possibly jjVG to Weak Accept, soften KrEs from hard Weak Reject toward Borderline, and avoid losing yx8u/ve3y. The rebuttal should be optimized for that coalition.

## Reviewer jjVG: The Low-Confidence Completeness Checker

### What They Say On The Surface

- Cost is higher than baselines.
- Figure 2 is cluttered.
- k=5 and m=3 need rationale and ablation.
- ONLY and Vision-aware Head Divergence/VHD/VHR are missing.
- The paper is otherwise well written and the idea seems reasonable.

### What They Really Mean

jjVG is not deeply attacking the method. Their confidence is very low, and their review reads like a checklist for whether the submission is complete enough to support. They are likely uncomfortable making a strong technical judgment, so they anchor on visible, concrete omissions:

- "Did the authors compare to the recent methods I recognize?"
- "Did they justify the hyperparameters they chose?"
- "Is the diagram understandable?"
- "Are they hiding the cost?"

This reviewer needs permission to stop worrying that the paper is under-polished or evaluation-incomplete. They do not need a deep philosophical novelty argument; they need the obvious holes closed.

### Hidden Acceptance Gate

The hidden gate is completeness, not mechanism. If the rebuttal includes a concise k/m table, a related-work/baseline positioning row for ONLY and VHD/VHR, and a transparent latency caveat, jjVG can plausibly move from Borderline to Weak Accept. Figure 2 cleanup is a low-cost reassurance signal, not the primary gate.

### What Will Persuade Them

| Evidence / wording | Why it works |
|---|---|
| A compact k/m sensitivity table or Pareto statement | Shows the default was not arbitrary. |
| Explicit ONLY and VHD/VHR positioning | Removes the visible missing-paper objection. |
| Detector/decode/end-to-end cost separated | Makes the cost concern feel answered. |
| "We will simplify Figure 2 into sequential stages" | Fixes a concrete presentation complaint. |

### What Will Fail

- Spending scarce rebuttal budget on abstract novelty rhetoric.
- Saying related work was omitted due to space.
- Only promising future camera-ready changes without one concrete table or named citation.
- Ignoring Figure 2 entirely; this would leave an easy, visible complaint untouched.

### Rebuttal Priority For jjVG

Medium-high. This is a movable Borderline reviewer, but not the strongest intellectual center of the review set. Answer them with compact evidence and named fixes, then move on.

## Reviewer KrEs: The Attribution Prosecutor

### What They Say On The Surface

- Novelty is unclear because CHORD combines OPERA-style rollback, Grounding DINO, attention, and rollout.
- Grounding DINO's role is under-analyzed.
- End-to-end latency, memory, and batch-size behavior are missing.
- Baselines and model scope are limited.

### What They Really Mean

KrEs does not reject the topic, implementation direction, or empirical promise. Their review repeatedly says the framework is meaningful and reasonably structured. The rejection comes from a strict attribution standard:

- "If the method is a combination of known parts, what is the actual contribution?"
- "If an external detector is added, how do I know the detector is not doing the work?"
- "If latency is reported only as decode ITL, how do I know deployment cost is not being hidden?"
- "If recent stronger baselines are absent, how do I know the comparison is not cherry-picked?"

This reviewer is prosecuting the causal chain of the paper. They need isolation, not persuasion. If the rebuttal sounds rhetorical, they will read it as evasion.

### Hidden Acceptance Gate

The hidden gate is attribution credibility. KrEs needs at least two of the following three P0 items to become clean:

1. Detector controls: real anchors vs no/uniform/random anchors, Past+Future without current anchors, and ideally same-anchor non-CHORD.
2. Cost accounting: proposal time, decode ITL, total latency, peak VRAM, batch setting.
3. Novelty axes: CHORD's coupled admission policy contrasted against OPERA, VCD, DoLa, HALC, ONLY, and VHD/VHR.

The most important of these is detector control. Without it, KrEs can maintain Weak Reject even if other reviewers are satisfied.

### What Will Persuade Them

| Evidence / wording | Why it works |
|---|---|
| Real/no/random anchor ablation | Directly isolates Grounding DINO attribution. |
| Past+Future without current anchors | Tests whether Future alone explains gains or whether Current matters. |
| Same-anchor non-CHORD control | Tests whether detector metadata alone can reproduce the win. |
| End-to-end latency and VRAM table | Removes the suspicion of hidden cost. |
| "Training-free for the base MLLM; detector-assisted at inference time" | Shows claim discipline. |

### What Will Fail

- Saying "Grounding DINO is only used once" without measuring its cost and attribution.
- Claiming CHORD is "training-free" without qualification.
- Using only aggregate POPE/CHAIR/MMBench improvements.
- Arguing that combining known parts is inherently novel without behavioral isolation.
- Reporting expected tables or placeholder ranges as if they are measured.

### Rebuttal Priority For KrEs

Highest for risk control. KrEs may not become an accept, but reducing their objection from "evidence insufficient" to "moderate novelty but addressed controls" is crucial. The rebuttal must be written so that even a skeptical reader sees the authors understand the attribution problem.

## Reviewer yx8u: The Supportive Claim-Discipline Expert

### What They Say On The Surface

- CHORD is well motivated and practically relevant.
- The "support now and stability next" framing is reasonable.
- Weaknesses: incremental novelty, external detector dependence, attention reliability, limited scope, latency, and overstrong terminology.

### What They Really Mean

yx8u already wants to accept the paper, but only under a bounded interpretation. They are not asking the authors to prove a grand new paradigm. They are asking the authors to stop overstating:

- "Training-free" is incomplete if the method depends on a frozen external detector.
- Attention weights should not be sold as faithful causal explanations.
- The method may be architecture- and detector-dependent.
- Two 7B backbones and object-heavy benchmarks do not justify broad generality.
- Terms like "structural temporal collapse" and "harmonic arbitration" sound stronger than the formal evidence.

This reviewer is a guardrail. They protect acceptance if the authors are honest; they may downgrade if the rebuttal becomes defensive or inflated.

### Hidden Acceptance Gate

The hidden gate is trustworthiness of claims. yx8u needs to see that the authors will revise the narrative to match evidence:

- detector-assisted, not detector-free;
- operational attention feature, not causal explanation;
- two operating regimes, not one universally best method;
- object/open-ended hallucination scope, not universal hallucination mitigation.

### What Will Persuade Them

| Evidence / wording | Why it works |
|---|---|
| Detector failure strata | Shows the authors know where the method breaks. |
| Attention-window note framed as operational | Avoids the attention-as-explanation trap. |
| Claim-softening commitments | Protects the paper from overclaiming. |
| Stronger-backbone/scope note | Either adds evidence or bounds the claim honestly. |

### What Will Fail

- Trying to prove "attention is explanation" in a short rebuttal.
- Saying the method is universally training-free.
- Claiming broad relation/attribute/reasoning hallucination generality without experiments.
- Overusing branded terms instead of standard technical language.

### Rebuttal Priority For yx8u

High for preservation, not conversion. The rebuttal should avoid triggering a downgrade. A disciplined limitation paragraph may be more valuable for yx8u than another aggressive claim of novelty.

## Reviewer ve3y: The Supportive Practical-Value Reviewer

### What They Say On The Surface

- The problem is important and relevant.
- CHORD is conceptually coherent and technically structured.
- The main limitations are incremental novelty, external proposer complexity, and runtime overhead.
- The paper is already a Weak Accept.

### What They Really Mean

ve3y is broadly convinced that the paper is useful. Their concern is whether the practical trade-off is presented honestly. They can accept incremental novelty if the method provides a useful decoding perspective and measured trade-off. They do not need the authors to prove a theoretical breakthrough.

Their question is:

"Given the extra engineering and runtime, is this a practically interpretable method with a clear operating mode?"

This reviewer is likely to reward a clean Past+Current vs Full framing:

- Past+Current: lower-cost, latency-aware regime.
- Full: slower, quality-oriented regime for open-ended hallucination suppression.

### Hidden Acceptance Gate

The hidden gate is deployment honesty. ve3y needs the authors to avoid pretending Full CHORD is cheap. If the rebuttal clearly separates practical and quality modes, this reviewer likely remains supportive.

### What Will Persuade Them

| Evidence / wording | Why it works |
|---|---|
| Operating-point table | Matches their practical evaluation mindset. |
| End-to-end cost table | Shows runtime honesty. |
| "Full is not the deployment-cheap setting" | Builds trust. |
| Detector-assisted limitation | Acknowledges engineering complexity. |

### What Will Fail

- Over-indexing on novelty while ignoring runtime.
- Presenting Full CHORD as the default deployment solution.
- Hiding detector proposal time.
- Spending too much rebuttal space on this reviewer while KrEs/M8du remain unresolved.

### Rebuttal Priority For ve3y

Medium for preservation. They are already favorable. Keep them comfortable through honest cost framing; do not let the rebuttal accidentally create new doubts.

## Reviewer M8du: The Convertible Mechanism Auditor

### What They Say On The Surface

- Final benchmark scores do not validate the mechanism.
- Future rollout may not actually do anything.
- The detector may be doing the work.
- Baselines and ablations are too thin.
- Generality and detector robustness are not tested.
- They explicitly say they are willing to raise the score if concerns are addressed.

### What They Really Mean

M8du is the most important score-upside reviewer. Their review is not hostile; it is diagnostic. They believe the paper could be acceptable, but they do not trust the mechanism yet.

Their central objection is:

"You claim Past, Current, and Future coordinate to improve token admission. Show me the admission-level evidence."

This is more specific than generic ablation. They do not want only "Full beats P+C in aggregate." They want:

- How often does Full change P+C decisions?
- When it changes them, how often is that correction right?
- Are harmful flips bounded?
- Does the effect concentrate in open-ended hallucination or specific object cases?
- If Grounding DINO anchors are removed or randomized, does the claim still hold?

M8du is asking for mechanism validation, not just more scores.

### Hidden Acceptance Gate

The hidden gate is direct admission-change evidence. M8du can move upward if the rebuttal contains:

1. Full vs Past+Current flip/correctness table.
2. Corrected vs harmful flip counts.
3. Detector attribution controls.
4. k/m or robustness evidence showing the behavior is not a hand-tuned artifact.

The future diagnostic is the first sentence of the answer to M8du. If the rebuttal starts with general benchmark improvements, it misses the point.

### What Will Persuade Them

| Evidence / wording | Why it works |
|---|---|
| Full vs P+C corrected/harmful flip counts | Directly answers their explicit question. |
| CHAIR/open-ended continuation result | Shows Future matters where longer continuations compound. |
| Detector controls | Prevents mechanism evidence from being dismissed as Grounding DINO. |
| k/m sensitivity | Shows mechanism is not a fragile parameter artifact. |
| Failure examples included, not hidden | Builds credibility. |

### What Will Fail

- Repeating Table 1 aggregate metrics.
- Saying "Future improves CHAIR" without showing how decisions change or where it helps.
- Hiding the existing 64-sample warning diagnostic.
- Reporting only positive qualitative examples.
- Treating detector robustness as future work without any stratification.

### Rebuttal Priority For M8du

Highest for score movement. If only one reviewer can be converted, it is M8du. The rebuttal should be structured so M8du can quickly see their five questions answered.

## Shared True Needs Across Reviewers

The surface complaints differ, but the underlying needs collapse into five gates.

| Gate | Reviewers | What they are really asking | Minimum credible response |
|---|---|---|---|
| Mechanism validation | M8du, KrEs, yx8u | Does Future/Current change decisions in useful ways, or are gains inferred from aggregate scores? | Full vs P+C flip/correctness, CHAIR continuation benefit, examples with failure cases. |
| Attribution isolation | KrEs, M8du, yx8u, ve3y | Is this CHORD, or mostly Grounding DINO plus known decoding? | Real/no/random anchors, Past+Future, same-anchor non-CHORD. |
| Cost honesty | KrEs, M8du, jjVG, ve3y, yx8u | Is the reported latency end-to-end, and is deployment cost being undersold? | Proposal time, decode ITL, total latency, VRAM, batch setting. |
| Completeness of evaluation | jjVG, KrEs, M8du | Are recent methods and hyperparameters omitted because they weaken the story? | ONLY/VHD/VHR/HALC positioning; k/m sweep. |
| Claim discipline | yx8u, ve3y, KrEs | Are the authors overstating novelty, training-free status, attention reliability, or generality? | Detector-assisted wording, operational attention wording, scoped limitations. |

## What Each Reviewer Needs To Hear First

| Reviewer | First answer in rebuttal should sound like | Why |
|---|---|---|
| jjVG | "We add compact k/m evidence, recent-method positioning, cost clarification, and will simplify Fig. 2." | They need concrete completeness fixes. |
| KrEs | "We agree attribution and full cost must be isolated; we add detector controls and end-to-end accounting." | They need evidence, not rhetorical defense. |
| yx8u | "We will revise wording to detector-assisted/base-MLLM training-free and treat attention as operational." | They need claim discipline. |
| ve3y | "We clarify Past+Current as the practical regime and Full as the quality-oriented regime with explicit cost." | They need honest practical positioning. |
| M8du | "We directly measure how often Full changes P+C and whether those changes help or hurt." | They need mechanism validation. |

## What The Rebuttal Must Not Do

| Bad move | Who it harms | Why |
|---|---|---|
| Lead with "reviewers misunderstood our novelty" | KrEs, yx8u, M8du | Sounds defensive and avoids evidence gaps. |
| Use only aggregate benchmark numbers | M8du, KrEs | Does not answer mechanism or attribution. |
| Say bare "training-free" repeatedly | KrEs, yx8u | External detector makes that claim incomplete. |
| Claim attention proves grounding causally | yx8u | Directly triggers known attention-reliability concern. |
| Hide detector proposal time | KrEs, M8du, ve3y | Confirms suspicion of incomplete cost. |
| Treat the 64-sample diagnostic as positive evidence | M8du, KrEs | It is warning evidence and includes a harmful Full-vs-P+C flip. |
| Promise direct ONLY/VHD/VHR/HALC superiority without matched runs | jjVG, KrEs, M8du | Creates a new unsupported claim. |
| Spend too much space on Figure 2 | KrEs, M8du | Fixable but not central to acceptance. |

## Rebuttal Priority Implied By Reviewer Intent

The correct priority is not the order of surface comments. It is the order of decision gates:

1. Future mechanism diagnostic, because M8du directly asks for it and KrEs doubts component necessity.
2. Detector attribution controls, because KrEs can maintain Weak Reject without them.
3. End-to-end cost table, because all skeptical reviewers suspect hidden overhead.
4. k/m robustness, because it moves jjVG and answers M8du's "training-free but tuned" concern.
5. Recent baseline/related-work positioning, because named omissions are easy to hold against the paper.
6. Claim softening and limitations, because yx8u/ve3y support depends on honesty.
7. Figure 2 cleanup, because it is a visible low-cost fix but not a core scientific gate.

## If The Rebuttal Has Very Little Space

If the final OpenReview field is short, the response should not try to answer every reviewer separately. It should use one compact evidence-first structure:

1. One sentence acknowledging the shared concern: aggregate scores alone do not isolate mechanism, detector attribution, or end-to-end cost.
2. One compact table with three rows: Future mechanism, detector controls, cost.
3. One sentence for k/m and recent baselines.
4. One sentence for claim discipline and camera-ready edits.

Do not spend the opening on general appreciation or restating the paper. Every sentence should either answer M8du/KrEs or protect yx8u/ve3y.

## Skeptical Self-Audit Of This Intent Reading

This section asks whether the document may be over-interpreting the reviewers.

| Interpretation claim | Raw-review support | Risk of over-reading | Corrective rule |
|---|---|---|---|
| jjVG is mainly a completeness checker | Low confidence, concrete requests for cost, Figure 2, k/m, and named related work | They may still care about technical novelty more than written | Do not spend all effort on jjVG; answer concisely. |
| KrEs is the main attribution prosecutor | Explicitly asks for Grounding DINO analysis, end-to-end cost, stronger baselines, novelty clarity | They may require more evidence than rebuttal space allows | Prioritize detector controls and cost; avoid rhetorical novelty defense. |
| yx8u is supportive but claim-sensitive | Weak Accept plus long list on detector, attention, scope, latency, terminology | They may still downgrade if new evidence is weak | Use limitation language even if results are positive. |
| ve3y values practical framing | Weak Accept, good presentation score, explicit runtime/overhead concern | They may be less influential than knowledgeable reviewers | Preserve with honest operating-point framing; do not over-optimize for them. |
| M8du is the clearest conversion target | Borderline and explicitly willing to raise score if concerns are addressed | They may still require broader baselines/generality, not only Future diagnostics | Answer all five questions, but lead with Future flip/correctness. |

Conclusion: the document likely captures the committee's decision structure, but it should be treated as a high-confidence inference, not a guarantee. The safest rebuttal strategy remains evidence-first and claim-disciplined.

## What Would Make This Document Wrong?

| Possible contrary reality | Signal that would reveal it | Rebuttal adjustment |
|---|---|---|
| The meta-review values novelty more than all empirical fixes | Meta-review or discussion focuses on "incremental combination" despite new tables | Lead with novelty axes and contribution framing earlier. |
| Reviewers cannot consider new evidence due to format/venue constraints | OpenReview form disallows tables or long response | Compress to one evidence table and state exact measured numbers in text. |
| Detector controls fail | Random/no-anchor/same-anchor controls match CHORD | Narrow claims aggressively; shift to honest limitation and P+C practical result. |
| Future mechanism diagnostic fails | Full does not improve corrected flips or CHAIR | Stop defending Future as central; present Full as weak and P+C as main method. |
| Related-work omissions dominate discussion | Reviewers focus on ONLY/VHD/VHR/HALC despite diagnostics | Add citations/axis table immediately; avoid unsupported superiority claims. |

## Decision Tree For Rebuttal Emphasis

| Real experiment outcome | Lead message | What to downgrade |
|---|---|---|
| Future and detector controls both pass | "New diagnostics validate CHORD as a coupled admission policy." | Still avoid universal causality and broad generality. |
| Future weak, detector controls pass | "Current/detector-assisted P+C is the practical core; Full helps mainly on open-ended cases if CHAIR supports it." | Downgrade Future mechanism claim. |
| Future passes, detector controls weak | "Future improves continuation under detector-assisted setup." | Downgrade object-resonant/detector-independent attribution. |
| Cost is high but transparent | "P+C is practical, Full is quality-oriented." | Downgrade deployment efficiency claim. |
| k/m sweep finds a better setting | "Submitted default is not universal; camera-ready default will be updated." | Downgrade robustness claim. |

## Final Mental Model

The committee's real question is:

"Is CHORD a disciplined, measurable, detector-assisted admission policy with honest cost and bounded claims, or is it an over-named combination of OPERA, Grounding DINO, attention heuristics, rollout compute, and incomplete baselines?"

The rebuttal wins only by making the first interpretation easier to believe than the second.

## Appendix A: Raw Official Review Text

This appendix copies the official reviewer text from `reviews_20260528.md` into the same file as the intent analysis, so later rebuttal decisions can be checked against first-hand evidence. The inference sections above should always be audited against this appendix.

### Raw Review: jjVG

OpenReview source header, preserved from the user-provided official review export:

```text
Official Review of Submission8826 by Reviewer jjVG
Official Reviewby Reviewer jjVG13 May 2026, 17:25 (modified: 28 May 2026, 21:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer jjVG, AuthorsRevisions
```

Metadata:

- Fit: 4: Large audience
- Technical Quality: 2: Medium
- Technical Presentation: 3: Fair
- Rating: 3: Borderline
- Confidence: 1: Not my area

Strengths:

> The paper is generally well-written and easy to follow.
>
> Calibrating hallucinations by leveraging three complementary signals (past, current, and future) is reasonable.
>
> The proposed method achieves better accuracy than the baseline methods (OPERA, VCD, DoLa) on three benchmarks (POPE, CHAIR, MMBench), showing the effectiveness.

Weaknesses:

> According to Table 1, the proposed method introduces higher computational and time costs compared with the baseline methods.
>
> The pipeline in Figure 2 is relatively difficult to follow, as the connecting lines between components are quite cluttered and could be better organized.
>
> The method adopts top-k candidate reranking with k = 5 and a rollout horizon of m = 3. It would be helpful if the authors could explain the basis for choosing these values. In addition, an ablation study on k and m would better show the robustness of the proposed method to hyperparameter choices.
>
> The paper does not discuss or compare with several recent and closely related training-free methods [1, 2] for mitigating hallucinations in LVLMs. Their absence weakens the completeness of both the related work and the experimental evaluation.
>
> [1] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models, ICCV 2025
>
> [2] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence, ACL2025

Review:

> This paper proposes a training-free decoding method for calibrating hallucinations in large multimodal language models. Specifically, the proposed method evaluates candidate tokens based on three signals: past signal for historical protection, current signal for object-resonant grounding, and future signal for short-horizon rollout evaluation. Experiment results on three different benchmarks show that the proposed method achieves higher accuracy than baseline methods. However, it also incurs higher computational and time costs in inference stage. In addition, the paper lacks sufficient discussion and empirical comparison with closely related methods, which limits the completeness of the evaluation.

Fit justification:

> This paper focuses on the calibration of hallucinations in large-scale multimodal language models.

### Raw Review: KrEs

OpenReview source header, preserved from the user-provided official review export:

```text
Official Review of Submission8826 by Reviewer KrEs
Official Reviewby Reviewer KrEs13 May 2026, 17:03 (modified: 28 May 2026, 21:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer KrEs, AuthorsRevisions
```

Metadata:

- Fit: 5: Perfect match
- Technical Quality: 2: Medium
- Technical Presentation: 2: Poor
- Rating: 2: Weak Reject
- Confidence: 3: Knowledgeable

Strengths:

> 1. The motivation is clear. The paper focuses on hallucination as a decode-time token admission problem, which is relevant and meaningful for MLLMs.
>
> 2. The framework is reasonably complete, combining rollback-based past protection, object-resonant current grounding, and bounded future rollout verification.
>
> 3. The method is training-free for the base MLLM and can be plugged into different backbones without parameter updates.
>
> 4. Experiments on POPE, CHAIR, and MMBench with LLaVA-1.5 and InstructBLIP show consistent improvements in hallucination-related metrics.

Weaknesses:

> 1. The novelty is not very clear to me. The method seems to combine several existing ideas, including OPERA-style rollback, Grounding-DINO-based object proposals, attention-based visual grounding, and short rollout reranking. The paper should better clarify what is truly new in CHORD beyond this combination.
>
> 2. The role of Grounding DINO needs more analysis. Since CHORD uses an external object proposer, it is hard to tell whether the gains mainly come from the proposed decoding strategy or from the extra grounding information. More ablations, such as removing the proposer, using random anchors, or replacing it with another proposer, would make the claim stronger.
>
> 3. The efficiency advantage is not fully convincing. Full CHORD almost doubles the per-token latency compared with greedy decoding, and it is unclear whether the reported latency includes the one-time cost of Grounding DINO. The paper should report end-to-end latency, memory cost, and behavior under different batch sizes.
>
> 4. The experiments are still a bit limited. The paper only evaluates two 7B backbones and a small set of hallucination benchmarks. More recent MLLMs and stronger recent hallucination mitigation baselines, such as HALC or other grounding/lookahead-based methods, should be included.

Review:

> 1. This paper proposes CHORD, a training-free decoding framework for reducing hallucinations in multimodal large language models. The topic is important and well aligned with the scope of ACM Multimedia.
>
> 2. The main idea is to verify token admission from three perspectives: past rollback protection, current visual grounding, and future rollout stability. The framework is clearly structured, and the results on POPE, CHAIR, and MMBench show consistent improvements over several decoding baselines.
>
> 3. The technical novelty appears somewhat limited. CHORD mainly combines several existing components, including rollback decoding, external object detection, attention-based grounding, and rollout-based reranking. The newly introduced concepts are intuitive and practically motivated, but their necessity and individual contributions are not yet sufficiently justified.
>
> 4. The empirical evidence needs further strengthening. The paper should more clearly isolate the effect of Grounding DINO, report full end-to-end efficiency including the cost of external modules and rollouts, and compare against stronger recent hallucination-mitigation baselines. CHORD is a meaningful and practical method, but the current evidence is not yet sufficient to fully support acceptance.

Fit justification:

> The paper is well aligned with ACM Multimedia because it studies hallucination mitigation in multimodal large language models, involving visual grounding, multimodal decoding, object-level evidence, and efficient inference-time control.

### Raw Review: yx8u

OpenReview source header, preserved from the user-provided official review export:

```text
Official Review of Submission8826 by Reviewer yx8u
Official Reviewby Reviewer yx8u07 May 2026, 17:43 (modified: 28 May 2026, 21:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer yx8u, AuthorsRevisions
```

Metadata:

- Fit: 4: Large audience
- Technical Quality: 3: Good
- Technical Presentation: 4: Good
- Rating: 4: Weak Accept
- Confidence: 3: Knowledgeable

Strengths:

> The paper addresses an important and timely problem: hallucination mitigation in Multimodal Large Language Models. The focus on decode-time intervention is practically meaningful because it does not require additional model training or parameter updates. The proposed CHORD framework is conceptually clear: it treats token admission as a verification problem and combines three signals, namely past rollback protection, current object-resonant grounding, and future short-horizon rollout arbitration. This "support now and stability next" formulation is a reasonable and intuitive perspective for reducing visually unsupported generations. The paper also has a relatively complete methodological design. CHORD reuses OPERA-style rollback protection, introduces query-conditioned anchors from Grounding DINO, aggregates late-layer decoder-to-vision attention, and performs bounded rollout with shared KV-cache to control inference overhead. The method is training-free and can be plugged into existing MLLMs such as LLaVA-1.5 and InstructBLIP, which improves its practical applicability. The experimental section reports results on POPE, CHAIR, and MMBench, covering both object hallucination, open-ended caption hallucination, and general multimodal ability retention. The paper also evaluates two backbones, LLaVA-1.5-7B and InstructBLIP-7B, and compares against relevant decode-time baselines, including OPERA, VCD, and DoLa. The reported results show consistent improvements in POPE F1, CHAIR hallucination ratios, and MMBench accuracy, although at the cost of higher latency.

Weaknesses:

> The main weakness is that the novelty is incremental relative to existing decode-time hallucination mitigation methods. CHORD combines several known ideas: OPERA-style rollback, visual grounding via external detectors, attention-based scoring, and lookahead/rollout-based candidate evaluation. While the integration is reasonable, the paper does not sufficiently demonstrate that the proposed combination constitutes a fundamentally new algorithmic contribution beyond a carefully engineered decoding heuristic. Second, the method relies on an external object proposer, Grounding DINO, which makes the "training-free" claim somewhat incomplete. Although CHORD does not update the MLLM parameters, it depends on an additional pretrained perception model. This introduces extra computational cost, dependency on detector quality, and possible failure when the query-relevant object is missed or poorly localized. The paper acknowledges this limitation, but the experimental analysis of proposer failure cases is insufficient. Third, the reliance on decoder attention as a grounding signal is not fully justified. It is known that attention weights are not always reliable explanations of model decisions. CHORD uses the last four decoder blocks as an empirical attention window, but the paper does not provide enough systematic evidence that this choice generalizes across architectures, model scales, or different visual encoders. The method may therefore be sensitive to architecture-specific attention behavior. Fourth, the experimental scope is limited. The paper evaluates only two 7B-scale models. Stronger evidence would require more recent and stronger MLLMs, such as Qwen2-VL, InternVL, LLaVA-NeXT, or other stronger open-source LVLMs. The method should also be tested on more diverse hallucination benchmarks, including relation hallucination, attribute hallucination, and long-form visual question answering, not only POPE and CHAIR. Fifth, the latency overhead is substantial. On LLaVA-1.5, ITL increases from 19.73 ms/token for greedy decoding to 37.31 ms/token for Full CHORD. On InstructBLIP, ITL increases from 16.47 ms/token to 35.86 ms/token. This nearly doubles decoding latency, which weakens the practical appeal of the full method, especially for real-time or large-scale deployment. Finally, some claims are still stronger than the evidence supports. Terms such as "structural temporal collapse," "object-resonant decoding," and "harmonic temporal arbitration" are intuitive but not rigorously formalized. The paper would be stronger if it used more standard terminology and provided deeper causal or diagnostic analysis of when and why future rollout helps.

Review:

> This paper proposes CHORD, a training-free decoding framework for mitigating hallucinations in multimodal large language models. At each decoding step, the method reranks top-k candidate tokens using three signals: a past signal inherited from OPERA-style rollback protection, a current signal measuring query-conditioned visual support, and a future signal obtained through short-horizon rollout. The core idea is that a candidate token should not only be locally probable, but also visually supported at the current step and likely to preserve visual grounding in the next few decoding steps. Overall, CHORD is a well-motivated and practically relevant decode-time framework with encouraging results. However, its novelty is moderate, the dependence on an external detector is under-analyzed, the evaluation is somewhat narrow, and the latency cost is high.

Fit justification:

> Multimodal hallucination mitigation in MLLMs is a highly active topic at the intersection of vision-language models, decoding strategies, and trustworthy AI, squarely within ACM MM's machine learning, computer vision, and multimodal application tracks. Not a perfect match because the technical core leans more toward NLP-style decoding than multimedia retrieval/generation specifically.

### Raw Review: ve3y

OpenReview source header, preserved from the user-provided official review export:

```text
Official Review of Submission8826 by Reviewer ve3y
Official Reviewby Reviewer ve3y07 May 2026, 10:25 (modified: 28 May 2026, 21:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer ve3y, AuthorsRevisions
```

Metadata:

- Fit: 5: Perfect match
- Technical Quality: 4: Excellent
- Technical Presentation: 4: Good
- Rating: 4: Weak Accept
- Confidence: 2: Familiar

Strengths:

> The paper addresses an important and timely problem in multimodal large language models: hallucination mitigation during decoding, which is highly relevant for reliable vision-language systems. The proposed CHORD framework is conceptually well structured, decomposing token admission into historical protection, current grounding verification, and short-horizon future arbitration. This decomposition is intuitive and technically coherent. A major strength is that the method is training-free and can be directly plugged into existing MLLMs without parameter updates, which improves practical applicability and deployment flexibility. The paper evaluates the method on multiple widely used hallucination benchmarks, including POPE, CHAIR, and MMBench, across two different MLLM backbones (LLaVA-1.5 and InstructBLIP). The experiments analyze both hallucination reduction and latency trade-offs, which provides a more realistic evaluation of inference-time methods compared with reporting a single performance metric.

Weaknesses:

> The overall framework is somewhat incremental relative to prior decode-time hallucination control methods such as OPERA and VCD. The main contribution primarily comes from combining several verification signals rather than introducing a fundamentally new decoding paradigm. The method depends on external object proposal models (e.g., Grounding DINO), which introduces additional engineering complexity and may limit robustness when visual grounding quality is poor. Although the paper discusses latency trade-offs, the computational overhead is still nontrivial due to rollout verification and candidate reranking. More detailed runtime comparisons under practical deployment settings would strengthen the work.

Review:

> This paper proposes CHORD, a training-free decoding framework for mitigating hallucinations in multimodal large language models through bounded trajectory verification. The method combines three key components: rollback-based historical protection, query-conditioned object resonance, and short-horizon future rollout arbitration. Instead of treating token generation as a purely local decoding decision, CHORD reformulates token admission as a verifier-style grounded decision process. Overall, the paper addresses an important and practically relevant problem. Hallucination remains one of the key limitations of current MLLMs, particularly in open-ended generation tasks, and inference-time mitigation strategies are attractive because they avoid costly retraining or alignment procedures. The proposed framework is technically coherent and reasonably motivated. However, several limitations remain. First, the novelty is moderate. The proposed framework builds heavily upon existing decode-time intervention paradigms, especially OPERA-style rollback protection and verification-based decoding. The main contribution lies more in the integration and decomposition of verification signals than in introducing a fundamentally new inference framework. Second, although latency analysis is discussed, the additional computational overhead introduced by rollout arbitration and candidate reranking remains substantial. More detailed efficiency studies under realistic deployment scenarios would further strengthen the practical claims. Overall, I think the paper presents a solid and practically relevant inference-time hallucination mitigation framework with good experimental validation and clear presentation. While the methodological novelty is somewhat incremental, the work offers meaningful empirical insights and a useful decoding perspective for multimodal generation systems. Therefore, I lean toward a weak accept recommendation.

Fit justification:

> The paper strongly aligns with ACM Multimedia topics, particularly multimodal reasoning, vision-language models, trustworthy AI, multimodal generation, and inference-time optimization for MLLMs.

### Raw Review: M8du

OpenReview source header, preserved from the user-provided official review export:

```text
Official Review of Submission8826 by Reviewer M8du
Official Reviewby Reviewer M8du06 May 2026, 23:55 (modified: 28 May 2026, 21:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer M8du, AuthorsRevisions
```

Metadata:

- Fit: 4: Large audience
- Technical Quality: 3: Good
- Technical Presentation: 4: Good
- Rating: 3: Borderline
- Confidence: 3: Knowledgeable

Strengths:

> This paper addresses a meaningful problem, namely how to reduce hallucinations in MLLM generation. The proposed decode-time framework is easy to understand. It combines information from past rollback, current visual grounding, and future rollout to judge whether a token should be admitted. I find the modular design helpful, since each part has a relatively clear role. The empirical results are also fairly convincing, with improvements reported on two MLLM backbones and on benchmarks such as POPE, CHAIR, and MMBench. Another positive aspect is that the paper discusses the quality and latency trade-off instead of only emphasizing the best result.

Weaknesses:

> (1) The paper argues that combining current visual grounding with future rollout verification reduces hallucination, but I'm not seeing the mechanism actually validated. Final benchmark scores and a handful of qualitative examples don't tell me how often the future term changes the admitted token, how often those changes are correct, or whether the gain concentrates in specific hallucination types. Without that breakdown the mechanism is mostly inferred from end-to-end numbers.
>
> (2) Novelty is modest. The four ingredients all have close cousins in prior decoding and grounding work, and the paper doesn't really argue for what coordinating Past, Current and Future unlocks that the parts don't.
>
> (3) The baseline set is thin for a decode-time hallucination paper. Greedy, OPERA, VCD, and DoLa is the whole comparison, and there's been more recent work in this line that should be in here. The ablations are also narrow, mostly about which of Past, Current and Future is on, rather than the choices that probably matter more.
>
> (4) Generality is barely tested. Two 7B backbones and mostly object-hallucination benchmarks (POPE, CHAIR). The pipeline is built around object-level anchors, so the open question is whether any of this transfers to relation, attribute, compositional, or reasoning-heavy hallucinations, and the paper doesn't try.

Review summary:

> CHORD is a training-free decode-time method for hallucination mitigation in MLLMs. The decoding rule combines three terms when scoring each candidate token, namely a past term that down-weights tokens flagged by prior rollbacks, a current term that grounds candidates against query-conditioned object anchors from an external detector, and a future term that uses a short-horizon rollout for verification. Evaluated on POPE, CHAIR, and MMBench with LLaVA-1.5-7B and InstructBLIP-7B, with gains reported over the decoding baselines they compare to.

Questions for authors:

> (1) I can't tell from the ablations whether the future rollout term actually does anything. How often does Past+Current+Future flip the admitted token relative to Past+Current alone, and when it flips, how often is the new choice correct?
>
> (2) The detector dependency makes the central claim hard to evaluate. Most of the win could just be Grounding DINO doing the work. I'd want Past+Future without the anchor term, and a vanilla-decoding baseline that uses the same anchors, before I'd believe the decoding rule is the real source of the gain.
>
> (3) Is the reported latency end-to-end? If it doesn't include Grounding DINO anchor generation the practical cost is being undersold. Decoder-only and end-to-end numbers reported separately would clear this up.
>
> (4) For training-free tasks, I still need hyperparameter analysis to determine whether the model depends on specific hyperparameters.
>
> (5) Detector robustness is not tested. What happens when Grounding DINO misses key objects or returns noisy anchors?
>
> I am willing to raise my score if the authors can adequately address these concerns in the rebuttal.

Fit justification:

> Hallucination mitigation in MLLMs is right in the middle of the trustworthy multimodal generation space, and the method's footprint overlaps with what a lot of the audience here is currently working on. One caveat is that the contribution sits at the inference-time decoding layer rather than at the representation level, which makes the reach a bit narrower than the strongest MM submissions, but it's clearly within scope.
