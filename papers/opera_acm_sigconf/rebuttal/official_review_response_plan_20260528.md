# CHORD Official Review Response Plan, 2026-05-28

Scope: plan for responding to the real official reviews in `reviews_20260528.md` only. This is not rebuttal prose.

Core principle: do not argue that reviewers misunderstood the paper. Close the exact evidence gaps that make them doubt the contribution.

## One-Line Strategy

Position CHORD as an admission-time verifier family, not as a single magic decoder: Past+Current is the latency-aware setting; Full CHORD is the quality-oriented setting. Then prove this with mechanism diagnostics, detector attribution, and complete cost accounting.

## Score-Movement Targets

| Target | Current score | Desired movement | Why this target matters | Main key |
|---|---:|---:|---|---|
| M8du | 3 Borderline | 4 Weak Accept | Explicitly says they may raise score if concerns are addressed. | Mechanism diagnostics: future flip rate and correctness. |
| jjVG | 3 Borderline, confidence 1 | 4 Weak Accept | Low-confidence reviewer with mostly concrete fixable concerns. | k/m sweep, recent-method citations, cost caveat, Figure 2 cleanup. |
| KrEs | 2 Weak Reject | 3 Borderline or soft 4 | Main rejection risk; knowledgeable and focused on attribution. | Detector controls, end-to-end latency, novelty framing. |
| yx8u | 4 Weak Accept | Preserve 4 | Supportive but sensitive to overclaiming. | Claim discipline, detector limitation, attention reliability. |
| ve3y | 4 Weak Accept | Preserve 4 | Supportive and presentation-friendly. | Practical runtime trade-off and concise operating-point framing. |

## Master Plan Table

| Priority | Reviewers | Surface complaint | Real blocker | Concrete action | Output artifact | Acceptance criterion | Rebuttal stance | Risk if skipped |
|---|---|---|---|---|---|---|---|---|
| P0 | M8du, KrEs, yx8u | Future rollout is not validated. | Reviewers do not believe the future term actually changes admission decisions in useful ways. | Run a Full CHORD vs Past+Current diagnostic on POPE adversarial and CHAIR subset/full split if feasible. Count answer-level differences, token-admission flips, correct flips, wrong flips, and hallucination-type concentration. | `rebuttal/future_mechanism_diagnostic_202605xx.md` plus JSON/CSV logs. | Table reports flip rate, correct/wrong flip counts, and examples; no unsupported claim that future explains all gains. | "We agree aggregate scores alone are insufficient; we add a direct admission-change diagnostic." | M8du likely stays Borderline; KrEs keeps "components not justified." |
| P0 | KrEs, M8du, yx8u, ve3y | Grounding DINO may be doing the work. | Attribution is not isolated; CHORD could be external detection plus generic reranking. | Run detector-control ablations: real anchors, no anchors/uniform weights, random boxes matched by count/area, Past+Future without current anchors, and if implementable a detector-aware non-CHORD baseline. | `rebuttal/detector_attribution_ablation_202605xx.md`. | Real-anchor CHORD beats no/random controls; if not, response honestly narrows claim to detector-assisted decoding. | "Grounding DINO is a frozen evidence provider; gains require the admission policy." | Weak Reject concern remains unresolved. |
| P0 | KrEs, M8du, jjVG, ve3y | Latency is high and possibly incomplete. | They suspect ITL excludes detector cost and undersells deployment overhead. | Report separate detector proposal time, decode ITL, total answer latency, peak VRAM, and batch-size behavior. Include Past+Current and Full CHORD. | `rebuttal/end_to_end_efficiency_202605xx.md`. | Table separates one-time proposal cost from decode cost and shows total latency under fixed max-new-token setting. | "Full CHORD is slower; Past+Current is the latency-sensitive operating point." | Reviewers interpret efficiency discussion as incomplete or evasive. |
| P0/P1 | jjVG, KrEs, M8du | Missing recent methods: ONLY, VHD/VHR, HALC. | Baseline set looks cherry-picked or stale. | Add related-work comparison axes table. If runnable, add at least one direct baseline result; if not runnable, explicitly cite, position, and explain incompatibility/implementation boundary. | `rebuttal/recent_baseline_positioning_202605xx.md`; later `sample-base.bib` / related-work patch. | Covers ONLY, VHD/VHR, HALC with method type, dependency, overhead, benchmark overlap, and relation to CHORD. | "We add the missing discussion and clarify scope; direct comparison is included only where reproducible." | jjVG may not move; KrEs/M8du keep baseline-thin objection. |
| P1 | jjVG, M8du | Why k=5 and m=3? | Training-free method may be brittle to hand-tuned decoding knobs. | Run small grid: k in {3,5,7}; m in {1,2,3,4}; report quality plus ITL. Prefer POPE adversarial and one CHAIR slice/full split. | `rebuttal/km_sensitivity_202605xx.md`. | Shows either robustness plateau or honest cost-quality curve. | "k and m are cost knobs; default is a middle point, not a tuned secret." | Hyperparameter concern remains easy reason to withhold support. |
| P1 | KrEs, yx8u, ve3y, M8du | Novelty is incremental. | They see a bundle of known components, not a new algorithmic claim. | Reframe contribution around the coupled admission rule: retrospective rollback + current object support + short-horizon stability. Add a comparison table against OPERA, VCD, DoLa, HALC, ONLY, VHD/VHR. | `rebuttal/novelty_axes_table_202605xx.md`; later main-paper text patch. | Table makes clear which prior methods lack admission-time coordination of current and future grounding. | "Individual ingredients have precedents; the contribution is the coupled verifier and operating frontier." | Defensive novelty claims will backfire. |
| P1 | yx8u | Decoder attention reliability. | Attention may not be a faithful grounding explanation and may be architecture-specific. | Frame attention as an operational scoring feature, not causal explanation. Add layer-window diagnostic if available: last-1 vs mid-4 vs last-4. | `rebuttal/attention_window_note_202605xx.md`. | Makes no universal attention-explanation claim; includes evidence or clear limitation. | "We use attention as a decode-time feature; we do not claim it is a universal explanation." | yx8u may downgrade for overclaiming. |
| P1 | yx8u, KrEs, M8du | Only two 7B backbones and object-heavy benchmarks. | Generality beyond object hallucination is not proven. | If compute allows, run one stronger current backbone on a small standard split. If not, state camera-ready scope limitation and avoid broad claims. | `rebuttal/generality_boundary_202605xx.md`. | Either one new backbone result or a precise limitation paragraph. | "We do not claim universal model-scale generality; this paper validates a controlled decode-time policy." | Overbroad claims weaken Weak Accepts. |
| P2 | jjVG | Figure 2 clutter. | Presentation friction for a low-confidence reviewer. | Redraw Figure 2 as four lanes: Stage 0 anchors, Stage 1 rollback, Stage 2 candidate scoring, Stage 3 admission. | New `figure2` draft or camera-ready TODO. | Reviewer can follow data flow without crossing lines. | "We will revise Figure 2 for readability." | Low-cost concern remains unresolved. |
| P2 | yx8u | Terminology too strong. | Terms sound rhetorical without formal proof. | Soften "structural temporal collapse" and "harmonic temporal arbitration" in rebuttal/camera-ready. Keep operational definition only. | Text patch list. | No claim of formal causality or universal mechanism. | "We will revise wording to emphasize operational rather than causal terminology." | Supportive reviewers may see overclaiming. |

## Rebuttal Structure

| Section | Purpose | Evidence to include | Reviewers addressed |
|---|---|---|---|
| Opening paragraph | Acknowledge shared concerns and state new evidence categories. | One sentence each for mechanism, detector attribution, cost, related work. | All |
| Mechanism diagnostics | Prove future rollout is not just decorative. | Full vs Past+Current flip/correctness table. | M8du, KrEs, yx8u |
| Detector attribution | Show gains are not only Grounding DINO. | Real/no/random anchor controls; failure cases. | KrEs, M8du, yx8u, ve3y |
| Efficiency | Report complete deployment cost honestly. | Detector time, decode ITL, total latency, VRAM, batch note. | KrEs, M8du, jjVG, ve3y |
| Recent methods and novelty | Patch missing related work and clarify contribution. | ONLY, VHD/VHR, HALC axes table; maybe direct result if reproducible. | jjVG, KrEs, M8du |
| Hyperparameters and scope | Reduce easy objections. | k/m sweep; limitation on stronger backbones and non-object hallucination. | jjVG, M8du, yx8u |
| Closing | Commit camera-ready edits. | Figure 2 redesign, terminology softening, related-work additions. | All |

## Experiment Queue

| Order | Experiment | Minimal version | Strong version | Stop rule |
|---:|---|---|---|---|
| 1 | Future mechanism diagnostic | POPE adversarial fixed split, Full vs Past+Current answer changes. | Add token-level flip logs and CHAIR object-level changes. | If full split is too slow, use a fixed reproducible subset and clearly label it. |
| 2 | Detector attribution | no-anchor/uniform, random-anchor, Past+Future-only. | Add same-detector non-CHORD control and alternate proposer. | Do not report a control whose implementation changes unrelated decoding behavior. |
| 3 | End-to-end cost | batch size 1, detector time, decode ITL, total latency, VRAM. | Add small batch-size sweep. | If detector runs on CPU/GPU differ, report device explicitly. |
| 4 | k/m sweep | k={3,5,7}, m={1,2,3} on POPE adversarial. | Add m=4 and CHAIR slice. | Stop if trends are flat enough or cost becomes clearly prohibitive. |
| 5 | Recent baseline handling | Add citation/axis table for ONLY, VHD/VHR, HALC. | Run one or more baselines on overlapping setup. | Do not fabricate direct comparisons; mark non-runnable methods as related-work-only. |
| 6 | Presentation fixes | Figure 2 redesign and terminology patch list. | New camera-ready figure plus revised abstract/conclusion wording. | Keep rebuttal concise; avoid spending response budget on figure details. |

## Reviewer-by-Reviewer Response Matrix

| Reviewer | What to say first | Evidence to show | What not to say |
|---|---|---|---|
| jjVG | Thank for concrete concerns; we add missing k/m, recent methods, cost clarification, and will simplify Figure 2. | k/m sweep, ONLY/VHD/VHR/HALC positioning, cost table. | Do not spend many words on philosophical novelty. |
| KrEs | We agree attribution and cost need clearer isolation; we add controls and full accounting. | Detector-control ablations, end-to-end cost, novelty axes table. | Do not claim Grounding DINO is irrelevant. |
| yx8u | We accept scope/claim-discipline concerns and will soften terminology. | Attention-window note, detector failure cases, limitation paragraph. | Do not defend attention as faithful causal explanation. |
| ve3y | We clarify practical operating points and complete runtime reporting. | Past+Current vs Full cost-quality table. | Do not imply Full CHORD is cheap. |
| M8du | We directly answer whether future changes admissions and whether those changes are correct. | Future flip/correctness table, detector controls, k/m sweep. | Do not answer with only final benchmark scores. |

## Wording Rules

1. Use "training-free for the base MLLM" rather than bare "training-free" when discussing external frozen modules.
2. Say "admission-time verifier" or "coupled decode-time policy"; avoid implying every component is new.
3. Say "operational attention feature" rather than "attention explanation."
4. Say "quality-oriented Full CHORD" and "latency-aware Past+Current."
5. Do not claim relation/attribute/long-form hallucination generality without new evidence.
6. Do not dismiss missing baselines as space constraints.

## Minimum Viable Rebuttal Package

If time is short, the minimum package is:

1. Future flip/correctness diagnostic.
2. Detector-control ablation.
3. End-to-end efficiency table.
4. k/m sweep.
5. Missing related-work/novelty axes table.

This package directly targets both Borderline reviewers and the Weak Reject. Figure and terminology edits can be promised for camera-ready after the evidence tables.

