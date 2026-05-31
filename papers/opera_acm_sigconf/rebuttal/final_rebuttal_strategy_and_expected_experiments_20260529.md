# CHORD Final Rebuttal Strategy and Expected Experiments, 2026-05-29

Boundary: this is a strategy and expected-results plan, not measured evidence. Any table values below are target/expected patterns for deciding what to run and how to write. They must be replaced by actual measured results before submission.

Authoritative expected-table guide: `reviewer_audited_expected_result_tables_20260529.md` supersedes the compact experiment matrix below for executable protocols, expected ranges, pass/warn/fail thresholds, and reviewer-style audits.

Canonical review source: `reviews_20260528.md` only.

Format assumption: draft as short OpenReview text, target <=2500 characters unless the logged-in ACMMM 2026 form shows a different limit. No external links, no author-identifying text, and no reliance on revised PDFs or new external material.

## Strategic Thesis

The rebuttal should not argue that every component is novel by itself. It should argue that CHORD's contribution is a measured admission-time verifier that coordinates three checks under one decoding decision:

1. Past prevents locally degenerate rollback-prone choices.
2. Current asks whether the candidate has visual support now.
3. Future asks whether accepting the candidate preserves grounded continuation.

The reviewer-facing claim should be bounded: CHORD is detector-assisted and training-free for the base MLLM; Past+Current is the latency-oriented regime; Full CHORD is the quality-oriented regime for open-ended or continuation-sensitive generation.

## What Reviewers Really Need To Believe

| Reviewer | Surface request | Deeper concern | Belief we must change | Evidence that changes it |
|---|---|---|---|---|
| jjVG | k/m ablation, recent methods, Figure 2, cost | The paper looks incomplete but not fatally flawed. | The missing items are easy to fix and the default settings are not cherry-picked. | Compact k/m sweep, related-work/baseline positioning, end-to-end cost table, Figure 2 rewrite commitment. |
| KrEs | Novelty, Grounding DINO role, full latency, stronger baselines | CHORD may be an engineered bundle whose gains come from detector metadata and extra compute. | The admission rule adds value beyond detector proposals, and its cost is transparent. | Detector attribution controls, same-anchor non-CHORD control, cost/VRAM table, axes-based novelty comparison. |
| yx8u | Incremental novelty, detector dependence, attention reliability, scope, latency | The work is acceptable only if claims are disciplined. | Authors understand the boundaries and will revise claims accordingly. | Explicit claim calibration, detector-assisted wording, attention-as-operational-signal wording, limitation statement. |
| ve3y | Moderate novelty and overhead | The method is useful, but practical cost must be honestly framed. | CHORD offers a clear quality-latency frontier. | Two-regime summary: Past+Current for latency, Full for quality; end-to-end cost table. |
| M8du | Future mechanism, detector controls, latency, hyperparams, robustness | The mechanism is inferred from aggregate scores, not validated. | The authors can isolate when Future and detector grounding change decisions and whether those changes are correct. | Full vs Past+Current flip/correctness table; detector robustness/failure analysis; k/m sweep. |

## How I Would Write The Rebuttal

Write it as an evidence package, not a debate. The response order should be:

1. **Opening:** acknowledge the shared core issue: reviewers want isolation, not just final scores.
2. **Mechanism:** answer M8du first: report Full vs Past+Current decision changes, corrected flips, harmful flips, and CHAIR/POPE split behavior.
3. **Detector attribution:** answer KrEs/M8du: report real/no/random anchors, Past+Future without current grounding, and same-anchor non-CHORD if feasible.
4. **Cost:** separate detector proposal time, decode ITL, total latency, VRAM; concede Full is slower and position Past+Current vs Full as two operating points.
5. **Robustness and missing work:** give k/m sweep, recent related work positioning, Figure 2 cleanup, and claim calibration.
6. **Close:** state exact revisions: detector-assisted wording, attention-as-operational signal, related-work additions, figure redraw, limitations.

Do not write:

- "All concerns are addressed" unless every P0 experiment is measured.
- "Training-free" without "for the base MLLM" or "detector-assisted".
- "Future always helps"; say it is most useful for continuation-sensitive/open-ended generation if the diagnostics support that.
- "Grounding DINO cost is negligible" unless end-to-end measurements support it.

## Compressed Rebuttal Skeleton

Use this only after replacing placeholders with measured values:

> We thank the reviewers. The main concern across reviews is not the motivation, but whether CHORD's gains come from the coordinated admission rule rather than detector metadata or extra compute. We therefore added four targeted diagnostics. First, comparing Full CHORD with Past+Current under identical candidates, Full changes [x]% of admissions/answers; among changed cases, [a] are corrected and [b] are harmful, with gains concentrated in [open-ended/CHAIR or adversarial POPE]. This supports our revised claim that Future is a bounded continuation-stability signal, not a universal yes/no correction. Second, detector controls show [real anchors] > [random/no anchors], while a same-anchor non-CHORD control remains below Full CHORD, indicating that anchors alone do not explain the gain. Third, we report proposal time, decode ITL, total latency, and VRAM separately; Full CHORD is the quality-oriented setting, while Past+Current is the latency-oriented setting. Fourth, a k/m sweep shows k=5,m=3 is near the quality-latency Pareto frontier. We will revise the paper to state "detector-assisted, training-free for the base MLLM", add ONLY/VHD/VHR/HALC positioning, redraw Figure 2, and tone down attention/novelty claims.

## Final Experiment Matrix With Expected Results

| Priority | Experiment | Reviewers answered | Protocol | Metrics to report | Expected / target result | If target fails |
|---|---|---|---|---|---|---|
| P0-E1 | Future mechanism: Full vs Past+Current flip/correctness | M8du, KrEs, yx8u | Same backbone, same prompts, same top-k candidates. Run POPE adversarial and CHAIR subset/full; compare Past+Current vs Full. | answer/admission flip rate; corrected flips; harmful flips; neutral flips; delta F1; delta CHAIR_S/I; examples by hallucination type | POPE answer flip rate may be low, but corrected flips should exceed harmful flips. CHAIR/open-ended should show clearer benefit, consistent with submitted Full > Past+Current by about 2 CHAIR_S points. | Narrow claim: Future is not key for binary POPE; use Full mainly for open-ended continuation stability. |
| P0-E2 | Detector attribution controls | KrEs, M8du, yx8u | Compare Full real anchors, no/uniform anchors, random anchors matched by count/area, Past+Future without current anchors, and same-anchor non-CHORD if feasible. | POPE Adv F1; CHAIR_S/I; false-positive rate; anchor count; failure cases | Real query-conditioned anchors should beat no/random anchors; same-anchor non-CHORD should not match Full; Past+Future should trail Full, showing current grounding and future are complementary. | If random/no anchors match Full, detector/admission attribution is weak; rewrite claim around detector-assisted heuristic and do not overclaim CHORD-specific causality. |
| P0-E3 | End-to-end cost accounting | jjVG, KrEs, yx8u, ve3y, M8du | Measure Greedy, OPERA/Past, Past+Current, Full under batch=1 and, if feasible, batch>1. Include one-time Grounding DINO proposal. | detector proposal time; decode ITL; total answer latency; generated tokens; peak VRAM | Full remains slower, but cost is transparent. Past+Current should be the practical latency regime; Full should be justified only when quality gain matters. Proposal time should be separately reported, not hidden inside ITL. | If overhead is too high, concede deployment limitation and push Past+Current as default practical mode. |
| P0-E4 | k/m robustness sweep | jjVG, M8du | Sweep k in {3,5,10}; m in {1,2,3,4} on a fixed validation subset. Keep all other hyperparameters fixed. | quality metric; latency; harmful flip rate; memory if available | k=5,m=3 should be near the quality-latency Pareto frontier. k=10 should add cost with little gain; m=4 should show diminishing returns; k=3 or m=1 should be weaker. | If another setting dominates, report it honestly and say the revision will update the default or reframe k=5,m=3 as the submitted operating point. |
| P0-E5 | Detector robustness / failure stratification | KrEs, yx8u, M8du | Stratify samples by 0 anchors, 1 anchor, multi-anchor, noisy/diffuse anchors, and missed key object. | F1/CHAIR by stratum; qualitative failure cases; fallback behavior | CHORD should improve most when relevant anchors exist; when anchors are missing/noisy, Past/Future should reduce but not eliminate failures. | If zero/noisy-anchor cases collapse, state this as a limitation and avoid robustness overclaim. |
| P1-E6 | Recent method positioning: ONLY, VHD/VHR, HALC | jjVG, KrEs, M8du | Add conceptual axes table; run direct baselines only if code/settings are reproducible quickly. | intervention point; external detector; train-free status; overhead; matched metrics if run | At minimum, related work becomes complete. Empirical comparison is included only if matched and reliable. | Do not report unmatched numbers. Use conceptual comparison only. |
| P1-E7 | One stronger backbone or non-object hallucination probe | KrEs, yx8u, M8du | If time permits, run a small subset on Qwen2-VL/LLaVA-NeXT or attribute/relation hallucination benchmark. | same primary metrics plus failure modes | Expected trend should hold but may be smaller; this mainly reduces the "two 7B backbones only" concern. | If unavailable, explicitly bound scope to two 7B backbones and object/open-ended hallucination benchmarks. |
| P1-E8 | Figure 2 and terminology cleanup | jjVG, yx8u, ve3y | Redraw Figure 2 as sequential lanes: candidates -> Past -> Current -> Future -> admission. Replace overloaded terms with standard wording. | camera-ready change list | Reviewers see presentation issues are fixable and not methodological flaws. | If no figure can be submitted during rebuttal, commit to camera-ready revision only. |

## Final Compact Table To Include If Space Allows

If OpenReview preserves Markdown and the limit permits one table, use a compressed version like this with actual values:

| Concern | New diagnostic | Result | Rebuttal interpretation |
|---|---|---|---|
| Future mechanism | Full vs Past+Current flips on POPE/CHAIR | `flip x%; corrected a; harmful b; delta metric c` | Future is a bounded continuation-stability signal; strongest for open-ended cases. |
| Detector attribution | real/no/random anchors + same-anchor non-CHORD | `real > random/no; same-anchor control below Full` | Gains are not explained by detector metadata alone. |
| Efficiency | proposal/decode/total/VRAM | `proposal t; ITL t; total t; VRAM m` | Full is quality mode; Past+Current is latency mode; cost is transparent. |
| Hyperparameters | k/m sweep | `k=5,m=3 near Pareto; larger k/m diminishing` | Defaults are not arbitrary. |

This table is the rebuttal's highest-value payload. If the actual form is <=2500 characters, prioritize this table over long reviewer-by-reviewer prose.
