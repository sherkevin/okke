# CHORD Pre-Rebuttal Analysis, 2026-05-28

Scope: this document archives the analysis before rebuttal drafting. It is not a rebuttal draft and should not be pasted into OpenReview as-is.

Canonical review boundary: only `reviews_20260528.md` is treated as the real official review record. Any earlier simulated or practice reviews in the workspace are excluded from this analysis.

## Inputs Reviewed

- Raw official review archive, canonical source: `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reviews_20260528.md`
- Main paper PDF/source: `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\sample-sigconf.pdf`, `sample-sigconf.tex`
- Supplement/source: `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\supplementary.pdf`, `supplementary.tex`
- Text extraction for search: `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\sample-sigconf_20260528.txt`
- Local diagnostic ablation files: `D:\Shervin\OneDrive\Desktop\breaking\remote_chiro_patch\tests\ablations_random_0_64\*.json`
- External related-work sources checked:
  - ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models, ICCV 2025 / arXiv 2507.00898: https://arxiv.org/abs/2507.00898
  - Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence, ACL 2025: https://aclanthology.org/2025.acl-long.175/
  - HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding, ICML 2024: https://arxiv.org/abs/2403.00425 and https://openreview.net/forum?id=EYvEVbfoDp
  - ACM Multimedia 2026 official dates page for process context: https://2026.acmmm.org/site/important-dates.html

## Executive Assessment

The review situation is recoverable but only with evidence-heavy rebuttal. The score profile is two Weak Accepts, two Borderlines, and one Weak Reject. The highest-leverage target is Reviewer M8du because they explicitly say they are willing to raise the score if the concerns are addressed. Reviewer jjVG is low confidence and asks mostly concrete fixable questions. Reviewer KrEs is the main risk: the response must reduce their concerns about novelty, detector dependence, efficiency, and baseline completeness.

The rebuttal should not be defensive. It should open by acknowledging the shared concern pattern and then report new evidence or precise planned camera-ready changes. The central message should be: CHORD is not just "OPERA + detector + rollout"; the contribution is a coordinated admission-time verifier, and the paper will substantiate that claim through mechanism diagnostics, detector controls, and end-to-end cost accounting.

## Reviewer Concern Map

| Reviewer | Score / Confidence | Highest-signal concerns | Rebuttal leverage |
|---|---:|---|---|
| jjVG | Borderline / 1 | Higher cost, cluttered Figure 2, missing k/m rationale and ablation, missing ONLY/VHD | High. Mostly concrete additions; low confidence means concise evidence can move them. |
| KrEs | Weak Reject / 3 | Novelty unclear, Grounding DINO dependence, missing end-to-end efficiency, limited baselines/models | Medium-high. Needs hard evidence, not only wording. |
| yx8u | Weak Accept / 3 | Incremental novelty, external detector, attention reliability, narrow scope, latency, overstrong terminology | Protect WA by conceding boundaries and adding diagnostics. |
| ve3y | Weak Accept / 2 | Incremental relative to OPERA/VCD, external proposer, runtime overhead | Protect WA with short evidence-backed framing. |
| M8du | Borderline / 3 | Future mechanism not validated, detector could explain gains, thin baselines, generality, hyperparameters | Very high. This is the most actionable reviewer and explicitly open to raising. |

## Cross-Review Issue Clusters

### P0. Mechanism evidence for the future term

Concern: Reviewers do not just ask whether Full CHORD improves final benchmark scores. They ask whether future rollout actually changes admitted tokens, when those changes are correct, and whether the gain concentrates in particular hallucination types.

Current paper evidence:
- Main Table 3 compares Past, Past+Current, Past+Future, and Full CHORD.
- The paper argues that Past+Current is the efficient POPE regime and Full CHORD is better for open-ended CHAIR-style generation.
- Qualitative figures show representative current/future behavior, but not aggregate mechanism counts.

Gap:
- No token-admission flip-rate table.
- No correctness breakdown for flips relative to Past+Current.
- No benchmark-specific breakdown showing where future helps or hurts.

Recommended evidence:
- Add a "Future arbitration diagnostic" table:
  - Full vs Past+Current answer-level difference rate.
  - Token-admission flip rate at monitored decision positions.
  - Percentage of flips that correct a hallucination.
  - Percentage of flips that introduce an error.
  - Separate POPE yes/no, CHAIR object mentions, and possibly MMBench retention cases.
- If time is tight, prioritize POPE adversarial and CHAIR. CHAIR matters because the paper's own current framing says future is mainly useful for longer open-ended outputs.

Risk:
- The local 64-sample random-slice diagnostic is not positive enough to use as rebuttal evidence. In that slice, Full differs from Past+Current on only one answer and is worse on that case. Treat this as a warning that the proper diagnostic must be run on the real evaluation split and should not be overclaimed.

### P0. Detector dependence and Grounding DINO attribution

Concern: Multiple reviewers suspect the improvement may come mainly from Grounding DINO rather than the decoding rule.

Current paper evidence:
- The implementation section says Grounding DINO is called once before autoregressive decoding and anchors are cached.
- The limitations mention proposer dependence.
- The current-score design explains query-conditioned anchors.

Gap:
- No control where anchors are removed, randomized, or replaced.
- No "same anchors but no CHORD reranking" baseline.
- No failure-case analysis when the proposer misses the relevant object.

Recommended evidence:
- Add detector-control ablations:
  - Full CHORD with real anchors.
  - CHORD without anchors / uniform visual-token weights.
  - CHORD with random boxes matched for count/area.
  - Past+Future without the current anchor term.
  - Greedy or OPERA with the same detector metadata available but no CHORD reranking, if implementable.
  - Optional alternate proposer if already easy to run.
- Add a small failure-case table:
  - Missed object anchor.
  - Noisy anchor.
  - Correct anchor but language prior still dominates.

Rebuttal framing:
- Qualify "training-free" precisely as "training-free for the base MLLM / no parameter update"; do not imply the system is free of external frozen perception models.
- Position Grounding DINO as frozen test-time evidence provider, not the novelty itself.

### P0. End-to-end efficiency and deployment cost

Concern: Reviewers object that Full CHORD nearly doubles ITL and may omit Grounding DINO cost.

Current paper evidence:
- Main Table 1 reports ITL in ms/token.
- Supplementary protocol says latency is decode-stage generation time after model inputs are assembled, under fixed proposal policy.
- Implementation says the detector is invoked once and cached.

Gap:
- Reviewers need explicit end-to-end latency, detector cost, memory, and possibly batch-size behavior.
- Current wording can be interpreted as underselling cost because ITL excludes one-time proposal generation.

Recommended evidence:
- Add a compact cost table:
  - Detector proposal time per image/query.
  - Decode ITL.
  - Total answer latency for fixed max/new-token settings.
  - Peak VRAM.
  - Batch size 1 and a small batched setting if feasible.
- Report separately for Past+Current and Full CHORD. Past+Current is likely the best practical operating point and should be defended as such.

Rebuttal framing:
- Be candid: Full CHORD is quality-oriented and slower; Past+Current is the recommended latency-efficient setting.
- Do not claim Full CHORD is deployment-cheap. Claim that the paper now reports the complete trade-off so users can select an operating point.

### P0. Missing recent baselines and related work

Concern: jjVG explicitly names ONLY and Vision-aware Head Divergence. KrEs and M8du mention HALC and other recent grounding/lookahead methods.

Current paper evidence:
- `sample-base.bib` already includes HALC, Qwen2-VL, LLaVA-NeXT, and InternVL entries.
- Main paper says HALC was checked in the same pipeline but not included in the main table.
- ONLY and Vision-aware Head Divergence are not currently in the bib/source.

Research notes:
- ONLY is a direct recent competitor because it is a training-free LVLM hallucination method with one-layer intervention and single-query efficiency claims.
- Vision-aware Head Divergence / VHR is direct because it is training-free, attention-head based, and claims negligible additional time overhead.
- HALC is direct because it is plug-and-play decoding for object hallucination and uses local/global focal-contrast behavior.

Recommended evidence:
- At minimum, add related-work citations and an axes table:
  - Whether method is training-free for base LVLM.
  - Uses external detector or internal attention.
  - Uses contrastive decoding, rollback, head intervention, grounding, rollout/lookahead.
  - Overhead type.
  - Benchmarks/backbones reported.
- If implementations are feasible, run ONLY, VHR, or HALC on at least the closest overlapping benchmark/backbone. If not feasible before rebuttal, do not fabricate a comparison; say the related-work discussion and camera-ready comparison table will be added, and report HALC only if existing pipeline numbers are verifiable.

### P1. Hyperparameter robustness for k and m

Concern: jjVG and M8du ask why k=5 and m=3, and whether the method depends on these values.

Current paper evidence:
- Main and supplement state k=5 and m=3 as defaults.
- Supplement lists all default hyperparameters.

Gap:
- No sweep table.

Recommended evidence:
- Run a small grid: k in {3, 5, 7}, m in {1, 2, 3, 4}.
- Use POPE adversarial plus one open-ended slice if time allows.
- Report quality and ITL together; m is directly a cost knob.

Likely framing:
- k=5 captures plausible alternatives without broad search.
- m=3 is a short-horizon verifier, not a search horizon; larger m likely increases latency with diminishing returns.
- If the sweep is flat, emphasize robustness; if not flat, state the cost-quality trade-off honestly.

### P1. Novelty and terminology

Concern: The paper is seen as a combination of existing ideas. Some terminology may sound stronger than the evidence supports.

Current paper evidence:
- The paper already says "structural temporal collapse" is operational rather than causal.
- Contributions frame CHORD as coordinated past/current/future admission verification.

Gap:
- The novelty is not crisply separated from OPERA, VCD, HALC, ONLY, and VHR.

Recommended response direction:
- Define novelty as the coordination rule, not each raw component:
  - OPERA is retrospective protection after prefix collapse.
  - Detector/grounding alone supplies object evidence but does not decide token admission trajectory.
  - Lookahead alone can be text-biased or generic.
  - CHORD binds rollback, query-conditioned object support, and short-horizon stability into one admission-time candidate verifier.
- Reduce or soften terms such as "structural temporal collapse" and "harmonic temporal arbitration" in camera-ready if reviewers view them as rhetorical.

### P1. Attention reliability

Concern: yx8u questions whether decoder attention is a reliable grounding signal across architectures.

Current paper evidence:
- The paper uses late-layer aggregation and includes a design-evidence figure.
- Supplement notes last-four layers are empirically chosen and not universal.

Gap:
- No systematic last-1/mid-4/last-4 quantitative ablation in the main response.

Recommended evidence:
- Add or cite an existing layer-window ablation if the numbers are available.
- Frame attention as an operational scoring feature, not a faithful explanation of model decisions.

### P1. Generality beyond two 7B models and object benchmarks

Concern: Reviewers ask about Qwen2-VL, InternVL, LLaVA-NeXT, relation/attribute/compositional hallucinations, and long-form VQA.

Current paper evidence:
- The paper evaluates LLaVA-1.5-7B and InstructBLIP-7B on POPE, CHAIR, and MMBench.
- The bibliography has newer model entries but not results.

Recommended response:
- If compute permits, add one stronger backbone, preferably LLaVA-NeXT or Qwen2-VL, on a small but standard hallucination benchmark.
- If not feasible, explicitly mark this as scope limitation and avoid claiming broad model-scale generality.
- For hallucination types, HALC and other object-hallucination work can be used to frame existence/attribute/relation distinctions; CHORD should not claim it has fully solved non-object hallucination without evidence.

### P2. Figure 2 clarity

Concern: Only jjVG raises figure clutter.

Recommended response:
- Commit to revising Figure 2 into sequential lanes:
  - Stage 0 anchors.
  - Stage 1 rollback.
  - Stage 2 current/future candidate scoring.
  - Stage 3 final admission.
- This is low effort and should be acknowledged briefly.

## Recommended Experiment Priority

1. Future mechanism diagnostic: Full vs Past+Current flip rate and flip correctness.
2. Detector attribution ablations: no anchors, random anchors, Past+Future without anchor term, and same-detector non-CHORD control if possible.
3. End-to-end cost table: detector time, decode ITL, total latency, VRAM, batch-size note.
4. k/m sweep: k={3,5,7}, m={1,2,3,4}, reported with both quality and latency.
5. Baseline/related-work patch: ONLY, VHR/VHD, and HALC positioning; direct experiments only if verifiable.
6. Figure 2 cleanup and terminology softening for camera-ready.

## Suggested Rebuttal Organization, Not Wording

Use concern clusters instead of one answer per reviewer:

1. Novelty and what CHORD contributes beyond combining components.
2. New mechanism diagnostics for future rollout.
3. Detector attribution and robustness.
4. Full efficiency accounting.
5. Hyperparameter and baseline coverage.
6. Figure/readability and camera-ready edits.

The response should lead with concrete new numbers once available. If the new numbers are not ready, the safest near-term action is to say which analyses are being run and avoid making unsupported claims.

## Claims to Avoid

- Do not say CHORD is simply "training-free" without qualification. Say it does not update the base MLLM and uses a frozen external proposer.
- Do not say future rollout is the main reason for all gains unless the flip/correctness diagnostic supports it.
- Do not use the local 64-sample random-slice ablation as positive proof.
- Do not dismiss missing baselines as a space issue. Reviewers named specific recent work, so the final response needs citations and either comparisons or a careful explanation.
- Do not start the rebuttal from polished rhetoric. Start from evidence tables.

## Current Bottom Line

The strongest rebuttal path is not to argue that reviewers misunderstood the paper. It is to concede the exact missing evidence, add compact diagnostics, and reposition CHORD as a controllable inference-policy family with two regimes: Past+Current for latency-sensitive use and Full CHORD for quality-oriented open-ended generation. If only three items can be done before rebuttal, do future mechanism diagnostics, detector attribution, and end-to-end latency first.
