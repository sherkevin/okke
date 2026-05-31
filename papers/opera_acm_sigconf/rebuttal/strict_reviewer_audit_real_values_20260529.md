# Strict Reviewer Audit Under Real-Results Assumption, 2026-05-29

Audited rebuttal: `author_response_min_diff_expected_20260529.pdf`

Intent reference: `reviewer_true_intent_analysis_20260529.md`

Assumption: all table numbers are real measured results. Stale `E` marks and expected-result notes are ignored for scientific judgment, though they must be removed before submission.

## Formal Verdict

As a strict reviewer, I would score this rebuttal response **3/5: Borderline, leaning upward**.

If forced to decide after rebuttal discussion, I would be open to supporting acceptance, but I would not call the response fully satisfactory. It answers the most important mechanism and detector-attribution questions better than the original paper, but it still leaves enough unresolved issues that a knowledgeable reviewer, especially KrEs, can reasonably maintain skepticism.

This is not a "bad" rebuttal. It is a materially useful rebuttal. But it does **not** satisfy all reviewer needs.

## Does It Satisfy All Reviewer Needs?

No.

It satisfies the following core needs reasonably well:

- mechanism evidence for Future versus P+C;
- detector attribution controls beyond aggregate benchmark scores;
- honest cost decomposition for batch-1;
- k/m sensitivity;
- claim calibration around detector-assisted, base-MLLM training-free decoding.

It does not fully satisfy:

- stronger recent baseline comparison;
- broader backbone/benchmark generality;
- batch-size behavior;
- quantified detector-failure strata;
- statistical reliability/significance of the new diagnostics;
- exact definitions/protocols for new diagnostic metrics.

## Reviewer-Specific Strict Assessment

| Reviewer | Original score | Strict post-rebuttal score | Assessment |
|---|---:|---:|---|
| jjVG | 3 | 4 | Their concrete checklist is mostly answered: k/m, cost, recent-work mention, Figure 2 revision. Low confidence makes upward movement plausible. |
| KrEs | 2 | 3 | The response improves attribution evidence, but still does not fully address stronger recent baselines, alternate proposer robustness, batch-size behavior, or broader MLLMs. I would not confidently move them to accept. |
| yx8u | 4 | 4 | The response preserves trust through claim discipline. But their generality and attention-reliability concerns are bounded rather than resolved. |
| ve3y | 4 | 4 | The practical framing is adequate. They likely stay weak accept, not stronger. |
| M8du | 3 | 4 | The Future flip/correctness table directly answers their main question. They are the clearest conversion. |

Composite result: two likely retained weak accepts, one likely converted weak accept, one low-confidence reviewer likely moved up, and one knowledgeable weak reject softened to borderline. That is good for rebuttal dynamics, but not enough to say "all concerns are solved."

## What The Rebuttal Explains Well

### Future mechanism

The strongest part is Table 1. Reporting Full-vs-P+C flip rate and corrected/harmful counts directly targets M8du's core objection. The numbers are plausible and useful: Full changes only about 4% of decisions, and helpful flips exceed harmful flips by more than 2x. This supports a bounded mechanism claim.

Remaining weakness: the rebuttal does not define `corrected` and `harmful` precisely enough, does not give confidence intervals, and does not show whether the effect is stable across seeds or prompt perturbations.

### Detector attribution

Table 2 is the second strongest part. Same-anchor non-CHORD, uniform anchors, random anchors, Past+Future without Current, P+C real anchors, and Full real anchors are the right family of controls. These rows make it harder to claim that Grounding DINO alone explains the gain.

Remaining weakness: `same-anchor non-CHORD` is underspecified. I cannot tell exactly what scoring rule it uses and whether it is a fair detector-only baseline. There is also no alternate detector/proposer comparison and no quantified detector-failure strata.

### Cost honesty

Table 3 is a meaningful improvement because it reports proposal time, decode ITL, total time, tokens, and peak VRAM. It also correctly frames P+C as practical and Full as quality-oriented.

Remaining weakness: all cost evidence is batch-1. KrEs explicitly asked for behavior under different batch sizes. Full CHORD remains more than 2x Greedy in total latency and roughly 2x OPERA, so practical deployment is still a real limitation, not just a clarified trade-off.

### k/m robustness

Table 4 is useful and does address jjVG and M8du. It shows diminishing returns for larger m/k.

Remaining weakness: the table weakens the default choice. k=5,m=2 is very close to k=5,m=3 but cheaper. A strict reviewer would ask why k=5,m=3 remains the default rather than a quality setting.

### Claim calibration

The response is appropriately more modest: contribution as coordinated token admission, detector-assisted rather than detector-free, attention as an operational feature rather than causal explanation.

Remaining weakness: this is wording discipline, not new technical novelty. It prevents downgrade but does not remove the incremental-novelty critique.

## Unresolved Problems

1. **Recent baselines remain insufficient.** The rebuttal says ONLY, VHD/VHR, and HALC will be added, but does not provide matched numerical comparison. For a weak reject reviewer, this is not fully satisfactory.

2. **Generality is not demonstrated.** No Qwen2-VL, InternVL, LLaVA-NeXT, relation hallucination, attribute hallucination, compositional hallucination, or reasoning-heavy VQA evidence is added.

3. **Detector failure is discussed but not quantified.** The response says zero-anchor, relevant-anchor, noisy/missed-object cases will be reported, but no counts or metrics appear in the rebuttal.

4. **Protocol definitions are missing.** Corrected/harmful flip definitions, CHAIR continuation protocol, same-anchor non-CHORD implementation, and timing methodology need more precision.

5. **Batch-size behavior is absent.** The response only reports batch 1 despite the review explicitly asking for behavior under different batch sizes.

6. **Efficiency remains costly.** P+C is more practical than Full, but it is still substantially slower than OPERA/Greedy. Full CHORD is not deployment-light.

7. **The novelty critique is only reframed.** The response admits component-level novelty is limited, but does not introduce a new algorithmic argument that would make KrEs fully withdraw the "combination of existing ideas" concern.

## Questions I Would Ask In Discussion

1. What exact rule defines a corrected flip and a harmful flip in Table 1?
2. Are Table 1 corrected/harmful counts statistically significant, and are they stable across seeds?
3. How is same-anchor non-CHORD implemented?
4. Why should I believe detector controls are fair without an alternate proposer or detector-failure strata?
5. What are the zero-anchor, missed-anchor, and noisy-anchor sample counts and metrics?
6. Why is k=5,m=3 the recommended setting when k=5,m=2 is close and cheaper?
7. Does total latency include detector preprocessing, GPU synchronization, and all rollout branches?
8. How does latency scale at batch size 2/4/8?
9. Why are ONLY, VHD/VHR, and HALC not directly compared under the same backbone and split?
10. What evidence supports transfer to relation, attribute, or compositional hallucinations?

## Score

Formal score: **3/5 Borderline**.

Lean: **positive borderline**, because the response likely converts M8du and jjVG and preserves yx8u/ve3y.

Why not 4: a strict knowledgeable reviewer can still maintain that the evaluation is narrow, recent baselines are not experimentally addressed, detector failure is not quantified, batch-size behavior is missing, and novelty remains incremental.

Minimum changes needed for me to move to 4:

1. Add a compact recent-method positioning/comparison table for ONLY, VHD/VHR, HALC.
2. Add quantified detector-failure strata.
3. Add one sentence or row on batch-size scaling, or explicitly bound the claim to batch-1 inference.
4. Define corrected/harmful flips and same-anchor non-CHORD precisely.
5. State k=5,m=3 as quality-oriented, not universal default.

