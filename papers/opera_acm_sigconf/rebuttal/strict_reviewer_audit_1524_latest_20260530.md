# Strict Reviewer Audit Of 15:24 Latest 5-Page Master, 2026-05-30

Audited PDF: `author_response_min_diff_expected_20260529.pdf`

Current PDF metadata checked: 5 pages, created/modified 2026-05-30 15:24:48, 340525 bytes.

Reference intent map: `reviewer_true_intent_analysis_20260529.md`

Scope: scientific adequacy, reviewer belief change, unresolved concerns, and likely score movement. Reported table values are treated as valid rebuttal evidence for this review pass; raw experiment logs are not independently audited here.

## Verdict

As a strict composite reviewer, I would score this current 5-page master **4/5: Weak Accept**.

This version satisfies the actionable reviewer demands at acceptance level. It no longer relies on aggregate benchmark gains alone: it gives admission-level Future diagnostics, detector attribution controls, end-to-end cost, batch/VRAM behavior, k/m robustness, recent matched baselines, detector failure strata, threshold sensitivity, newer-backbone sanity checks, and claim calibration.

It still does not become a 5/5 response. The core method remains a coordinated verifier built from related components, the strongest gains still require useful detector anchors, Full CHORD remains compute-heavy, and relation/composition hallucination is outside the main supported claim. These are real limits, not presentation defects.

## Reviewer-Specific Score Movement

| Reviewer | Original stance | Likely after this response | Strict judgment |
|---|---:|---:|---|
| jjVG | 3 Borderline | 4 Weak Accept | Their visible checklist is now answered: cost, k/m, recent related methods, and Figure 2 cleanup. I see little left for them to object to except the general latency trade-off. |
| KrEs | 2 Weak Reject | 3 Borderline to 4 Weak Accept | This is the hardest reviewer. The response now gives the attribution and cost accounting they demanded, but they may still cap the paper for moderate novelty, limited non-object generality, and dependence on Grounding DINO. |
| yx8u | 4 Weak Accept | 4 Weak Accept | The response preserves this reviewer by narrowing claims, admitting detector assistance, treating attention as an operational feature, and separating object hallucination from broader reasoning claims. |
| ve3y | 4 Weak Accept | 4 Weak Accept | The practical story is credible because P+C is named as the default regime and Full is framed as quality/offline rather than cheap deployment. |
| M8du | 3 Borderline | 4 Weak Accept | Their core questions are directly answered: Full-vs-P+C flip rate, corrected/harmful flips, detector controls, latency, k/m sensitivity, and detector robustness. |

## What Is Now Satisfied

1. **Future mechanism is no longer only inferred.** The response reports sparse Full-vs-P+C intervention, corrected vs harmful flips, CHAIR continuation gains, confidence intervals, and paired tests. This directly answers M8du's main question.

2. **Detector attribution is substantially isolated.** Same-anchor non-CHORD, uniform anchors, random matched anchors, Past+Future without Current, real-anchor P+C, and Full real anchors form a credible attribution ladder. Detector strata and oracle-box repair further show where detector quality matters.

3. **Cost is no longer hidden.** The tables separate detector proposal time, decode ITL, total latency, generated length, VRAM, and batch behavior. The OOM boundary for Full at larger batch size is an honest negative.

4. **Hyperparameter choice is defensible.** The k/m table shows diminishing returns and makes P+C or k=5,m=2 the lower-cost alternatives, while k=5,m=3 is the quality setting rather than an unexplained universal default.

5. **Recent-baseline incompleteness is mostly repaired.** ONLY, VHD/VHR, and HALC are included under a matched protocol with validation-budget and reproducibility commitments. This directly addresses jjVG, KrEs, and M8du.

6. **Claim discipline is much stronger.** The response states "detector-assisted", "training-free for the base MLLM", "attention as operational scoring", and "object-grounded/open-ended object mentions" rather than broad claims over all hallucination types.

7. **Detector robustness has real diagnostics.** Zero-anchor, relevant-anchor, and noisy/diffuse-anchor strata; threshold sensitivity; fine-grained/crowded degradation; detector-side vs model-side failure split; and oracle-box repair make this a measured limitation instead of a hand-wave.

8. **CHAIR gain is less vulnerable to the "shorter captions" objection.** Caption length, object mentions, supported mentions, and unsupported mentions are checked, so the reduction is plausibly about unsupported object mentions rather than conservative truncation.

## What Is Still Not Fully Solved

1. **Novelty remains moderate.** The rebuttal makes a reasonable case that coordinating Past, Current, and Future at token admission is useful, but it does not transform the work into a fundamentally new decoding paradigm.

2. **Grounding DINO remains central.** The response is honest about this, but the strongest gains concentrate in relevant-anchor cases. A reviewer can still say the method is detector-assisted and detector-quality-dependent.

3. **No alternate real detector is tested.** Random/uniform/same-anchor controls are strong, and oracle boxes are useful, but KrEs explicitly mentioned replacing the proposer. This remains a possible follow-up concern.

4. **Full CHORD is not deployment-light.** The response handles this by recommending P+C, but Full's quality advantage comes with substantial latency and memory cost.

5. **Generality is bounded.** Newer-backbone pilots help, but relation/composition transfer is weak and correctly treated as limitation evidence. This paper should not claim broad hallucination mitigation beyond object-grounded cases.

6. **Baseline fairness depends on implementation fidelity.** The response promises official code/checkpoints where available and sanity checks for reimplementations. As a reviewer, I would accept the plan, but the camera-ready must actually include commands, checkpoints, prompts, parsers, seeds, and tuned grids.

7. **Some robustness evidence is still compact.** Two alternate POPE prompts and validation-slice stability are good rebuttal evidence, but a very strict reviewer could ask whether these hold across all benchmarks, all backbones, and longer caption settings.

## Follow-Up Questions I Would Still Ask

1. For HALC, ONLY, and VHD/VHR, exactly which official commits/checkpoints or reimplementation repositories were used, and what sanity numbers confirmed fidelity?

2. Can the camera-ready include the full reproducibility record: command lines, seeds, prompt templates, parser versions, validation splits, and tuned grids for every method?

3. What happens if Grounding DINO is replaced by another real proposer, not only by uniform/random anchors or oracle boxes?

4. Are the InstructBLIP CHAIR-S gains also supported by bootstrap intervals, or are the statistical tests mainly shown for LLaVA?

5. In fine-grained/crowded scenes, can the method predict when it should fall back to P+C/Past rather than trust noisy anchors?

6. Do the alternate prompt-template results hold for CHAIR captioning, or only for POPE-style binary questions?

7. Are the newer-backbone results full benchmark runs in the final paper, or only 1000-sample sanity checks? The answer affects how much they can support generality.

8. Will the revised abstract explicitly name P+C as the practical default and Full as quality/offline, or will that distinction only appear in the rebuttal?

9. Can relation/composition hallucination be removed from contribution-level claims entirely and left only in limitations or boundary diagnostics?

## Final Score

**4/5 Weak Accept.**

This rebuttal is now scientifically adequate for acceptance if the reported values are credible. It likely converts M8du and jjVG, preserves yx8u and ve3y, and gives KrEs enough attribution/cost evidence that a continued Weak Reject would mainly rest on novelty and scope rather than missing diagnostics.

The important distinction is that this response does **not** guarantee high scores. It makes the paper acceptably bounded and empirically supported. That is enough for a composite Weak Accept, not enough for an unequivocal Strong Accept.
