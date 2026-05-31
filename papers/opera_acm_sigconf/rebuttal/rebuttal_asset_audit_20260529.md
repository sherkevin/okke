# CHORD Rebuttal Asset Audit, 2026-05-29

Scope: audit of current project assets for supporting rebuttal to the real official reviews in `reviews_20260528.md`. Simulated review files are excluded.

Bottom line: the project can support a conservative, evidence-constrained rebuttal, but it cannot yet support a strong comprehensive rebuttal against all reviewer concerns. The appendix is useful and should be actively cited, but it does not close the most important missing evidence: future-term mechanism diagnostics, detector attribution, end-to-end cost, and k/m robustness.

## Verified Assets

| Asset | Path | Status | Rebuttal value |
|---|---|---|---|
| Official reviews | `papers/opera_acm_sigconf/rebuttal/reviews_20260528.md` | Real source of reviewer concerns | Canonical input only. |
| Main paper source/PDF | `papers/opera_acm_sigconf/sample-sigconf.tex`, `sample-sigconf.pdf` | 10-page current paper; PDF metadata confirms CHORD title | Supports submitted results, two operating regimes, main method framing. |
| Supplement/appendix source/PDF | `papers/opera_acm_sigconf/supplementary.tex`, `supplementary.pdf` | 2-page appendix; PDF metadata confirms supplementary title | Supports procedure, variants, proposal extraction, failure modes, protocol, and latency-boundary explanations. |
| Compile guide | `papers/opera_acm_sigconf/COMPILE.md` | Present | Confirms main and supplementary build targets. |
| Rebuttal plan | `papers/opera_acm_sigconf/rebuttal/official_review_response_plan_20260528.md` | Present | Useful execution plan, not evidence. |
| Actual evidence note | `papers/opera_acm_sigconf/rebuttal/actual_evidence_tables_20260528.md` | Present | Separates verified evidence from unavailable measurements. |
| Existing fixed-slice ablations | `remote_chiro_patch/tests/ablations_random_0_64/*.json` | Present | Useful warning diagnostic only; not strong rebuttal evidence. |
| CHORD implementation/tests | `remote_chiro_patch/chord`, `remote_chiro_patch/tests` | Present; local unit tests pass | Supports code sanity and theoretical invariants, not full benchmark claims. |

## Current Verification

- `pdfinfo papers/opera_acm_sigconf/sample-sigconf.pdf`: CHORD title, 10 pages, created 2026-04-02.
- `pdfinfo papers/opera_acm_sigconf/supplementary.pdf`: supplementary title, 2 pages, created 2026-04-03.
- `$env:PYTHONPATH='D:\Shervin\OneDrive\Desktop\breaking\remote_chiro_patch'; python -m pytest remote_chiro_patch/tests/test_chord_config.py remote_chiro_patch/tests/test_chord_current_score.py remote_chiro_patch/tests/test_future_rollout.py remote_chiro_patch/tests/test_chord_decode_integration.py -q`: `14 passed in 5.22s`.
- `nvidia-smi`: unavailable locally, so no new local GPU benchmark can be run here.

## What The Appendix Actually Supports

| Reviewer concern | Appendix support | Strength | How to use in rebuttal |
|---|---|---|---|
| Pipeline/procedure clarity | Supplement lists the stepwise decode-time procedure and CHORD variants. | Medium | Use as basis for a concise response, but still promise Figure 2 cleanup. |
| Grounding DINO dependency | Supplement states proposal extraction is once per example, cached, with thresholds and max boxes. It also lists missed-anchor, diffuse-proposal, query-mismatch failures. | Medium | Good for transparency and limitation; not enough for attribution. |
| k=5, m=3 defaults | Supplement records exact default configuration. | Low-medium | Useful for reproducibility; does not answer robustness. Need sweep. |
| Latency criticism | Supplement states reported ITL is decode-stage generation time, not fully amortized end-to-end latency. | Medium for honesty, weak for persuasion | Use to avoid overclaiming. Need detector-time/VRAM/total-latency table. |
| Operating point interpretation | Supplement states Past+Current is latency-sensitive and Full is quality-oriented/open-ended. | Medium-high | This is the strongest appendix asset; use directly. |
| Qualitative evidence boundary | Supplement distinguishes trace-backed cases from final-decision case. | Medium | Helps with claim discipline. |

Key point: the appendix helps defend honesty and reproducibility. It does not by itself satisfy reviewers who asked for additional empirical isolation.

## Concern-by-Concern Support Level

| Concern cluster | Current support | Verdict | Needed next evidence |
|---|---|---|---|
| Novelty beyond component combination | Main paper has "admission-time verifier" framing; appendix has variants. | Partially supportable | Add novelty axes table vs OPERA, VCD, DoLa, HALC, ONLY, VHD/VHR. |
| Future rollout mechanism | Main paper has submitted family tables and qualitative/design figures. Existing 64-sample POPE diagnostic is weak and even shows one harmful Full-vs-P+C flip. | Not strong enough | Full vs Past+Current flip/correctness diagnostic, ideally POPE adversarial plus CHAIR. |
| Detector attribution | Appendix is transparent about Grounding DINO and failure modes. Existing `alpha=0/no-GINO-weight` 64-sample slice is inconclusive. | Not strong enough | real/no/random anchor controls; same-anchor non-CHORD control; failure cases. |
| Efficiency/end-to-end cost | Main paper reports ITL; appendix admits decode-stage only; `run_eval_pipeline.py` can track ITL and peak VRAM for some pipelines. | Partially supportable but reviewer concern remains | Detector proposal time, total answer latency, peak VRAM, batch-size note. |
| k/m hyperparameters | Default values documented in main paper and appendix. | Not enough | k/m sweep with quality and latency. |
| Recent baselines | Bib contains HALC, Qwen2-VL, LLaVA-NeXT, InternVL. Review-named ONLY and VHD/VHR are not currently in the checked bib hits. | Not enough | Add ONLY and VHD/VHR citations/discussion; run baselines only if reproducible. |
| Attention reliability | Main paper/appendix frame last-four layers as empirical operating window, not universal. | Partially supportable | Add/foreground attention-window note; avoid causal explanation language. |
| Generality | Main results cover LLaVA-1.5 and InstructBLIP on POPE/CHAIR/MMBench. | Weak for reviewer request | Either add one stronger backbone/scope note, or clearly limit claims. |
| Figure 2 readability | Figure asset exists; appendix has procedural steps. | Easy to address | Redraw Figure 2 or promise camera-ready sequential-lane revision. |

## Usable Now

These can be used in rebuttal immediately:

1. "CHORD has two operating regimes": Past+Current for latency-sensitive use, Full for quality-oriented open-ended generation.
2. "The submitted ITL is decode-stage latency, not fully amortized end-to-end latency": this is explicitly in the appendix.
3. "The method is training-free for the base MLLM but detector-assisted": appendix and main method support this, and the wording should be corrected.
4. "Grounding DINO is invoked once and cached": supported by main paper and appendix.
5. "Known proposer failure modes are acknowledged": missed anchors, diffuse proposals, query mismatch.
6. "Code-level invariants are sanity-checked": 14 local tests pass.

## Unsafe To Claim Now

These should not be claimed unless new measured evidence is produced:

1. Future rollout improves admission decisions at flip/correctness level.
2. Detector attribution is resolved.
3. Grounding DINO overhead is negligible or fully included in submitted ITL.
4. k=5 and m=3 are robust across a meaningful grid.
5. CHORD compares favorably to ONLY or VHD/VHR empirically.
6. CHORD generalizes to stronger recent backbones or non-object hallucination types.
7. The existing 64-sample POPE slice is positive evidence for Full CHORD.

## Rebuttal Readiness Verdict

| Rebuttal ambition | Current asset readiness | Recommendation |
|---|---|---|
| Conservative response preserving Weak Accepts | Enough | Use appendix heavily, be honest about boundaries, promise camera-ready clarifications. |
| Move M8du from Borderline to Weak Accept | Not enough | Need future flip/correctness diagnostic and k/m sweep. |
| Move KrEs from Weak Reject | Not enough | Need detector attribution and end-to-end cost. |
| Fully answer all reviewers with strong evidence | Not enough | Need P0 experiments: future mechanism, detector controls, cost accounting. |

## Immediate Next Work

1. Run or recover Full vs Past+Current flip/correctness diagnostics on POPE adversarial and CHAIR.
2. Run detector attribution controls: no/uniform anchors, random anchors, same-anchor non-CHORD if possible.
3. Measure detector proposal time, total answer latency, and peak VRAM.
4. Run small k/m sweep.
5. Patch related-work table for ONLY, VHD/VHR, HALC.
6. Redraw Figure 2 or prepare a camera-ready figure-edit note.

