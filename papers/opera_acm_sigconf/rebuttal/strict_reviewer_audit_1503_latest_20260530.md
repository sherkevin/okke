# Strict Reviewer Audit Of 15:03 Latest 5-Page Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

Current PDF metadata checked: 5 pages, modified 2026-05-30 15:03:15, 340229 bytes.

Reference: `reviewer_true_intent_analysis_20260529.md`

Scope: scientific adequacy, reviewer belief change, unresolved concerns, and likely score movement.

## Verdict

As a strict composite reviewer, I would score the current 5-page master **4/5: Weak Accept**.

This version is close to the strongest rebuttal the paper can realistically support without changing the underlying method. It directly addresses the five-reviewer intent map: Future mechanism, detector attribution, efficiency, k/m robustness, recent baselines, implementation fairness, detector failure analysis, and scope discipline.

It is still not a 5/5 response. The remaining limitations are intrinsic: component-level novelty is moderate, the method depends on detector quality, Full CHORD is not deployment-light, and relation/composition hallucination is outside the main claim.

## Reviewer-Specific Assessment

| Reviewer | Original score | Likely after current response | Strict assessment |
|---|---:|---:|---|
| jjVG | 3 | 4 | Their checklist-style concerns are addressed: k/m, cost, recent work, and cleaner Figure 2. |
| KrEs | 2 | 3 to 4 | Their attribution, cost, and baseline-fairness objections are substantially weakened. They may still hesitate on novelty and generality. |
| yx8u | 4 | 4 | The response preserves support through bounded claims, detector-assisted wording, and attention-as-feature framing. |
| ve3y | 4 | 4 | Practical trade-off is explicit: P+C is the practical setting and Full is a quality/offline mode. |
| M8du | 3 | 4 | Mechanism, flip correctness, statistical stability, detector controls, and hyperparameter robustness are directly answered. |

## Satisfied Or Mostly Satisfied

- **Mechanism validation**: Full-vs-P+C flip rates, corrected/harmful counts, CHAIR continuation gains, confidence intervals, and paired tests directly answer whether Future helps.
- **Attribution isolation**: same-anchor non-CHORD, uniform/random anchors, Past+Future without Current, detector strata, threshold sensitivity, and detector-only tuning address the Grounding DINO concern.
- **Cost honesty**: proposal time, decode ITL, total latency, VRAM, batch behavior, and OOM boundary are explicit.
- **Baseline completeness**: ONLY, VHD/VHR, and HALC are included under a matched protocol with validation-budget and reproducibility details.
- **Hyperparameter robustness**: k/m sweep plus multi-slice lambda stability and perturbation bounds make the tuning concern much less serious.
- **Detector robustness boundary**: fine-grained/crowded degradation, detector-side vs model-side failure split, and oracle-box repair quantify where better detectors help and where CHORD remains limited.
- **Scope discipline**: title/abstract scope is narrowed to object-grounded hallucination and open-ended object mentions; relation/composition is boundary evidence, not the core claim.
- **CHAIR mechanism**: the response checks caption length, object mentions, supported mentions, and unsupported mentions, so the CHAIR gain is not explained as merely shorter generation.

## Not Fully Solved

1. **Novelty ceiling**: the contribution remains a coordinated verifier composed of related prior ideas. The rebuttal makes this acceptable but not exceptional.
2. **Detector dependence**: relevant-anchor cases drive the strongest gains; zero-anchor and noisy/diffuse-anchor cases remain weaker.
3. **Full-mode cost**: Full is an offline/quality mode, not a practical default.
4. **Non-object generality**: relation/composition transfer is weak and correctly excluded from the main claim.
5. **Reproducibility burden**: the response promises enough detail, but final acceptance of baseline fairness depends on the camera-ready record being complete.

## Questions I Would Still Ask

1. Will the camera-ready include exact commands, parser versions, seeds, tuned grids, and validation splits for every baseline?
2. Are the multi-slice lambda-stability results consistent across backbones, not just LLaVA?
3. In fine-grained/crowded scenes, are failures mainly from detector misses or from the model overusing imperfect anchors?
4. Does oracle-box repair improve only object-presence questions, or also CHAIR-style captioning?
5. Should the paper recommend P+C as the default in the abstract or method summary, with Full explicitly described as quality/offline?
6. Can relation/composition hallucination be removed entirely from broad claims, rather than only treated as a limitation?
7. Are the recent-baseline comparisons robust to alternative prompt templates?

## Final Score

**4/5 Weak Accept.**

This rebuttal now answers the reviewers' actionable doubts at acceptance level. It likely converts M8du and jjVG, preserves yx8u and ve3y, and removes the clean basis for KrEs's Weak Reject. The remaining issues are real but are no longer strong enough to justify Borderline or Reject from a composite reviewer perspective.

