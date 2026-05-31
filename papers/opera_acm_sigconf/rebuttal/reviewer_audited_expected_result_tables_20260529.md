# CHORD Reviewer-Audited Expected Result Tables, 2026-05-29

Scope: this file is a planning and experiment-guidance artifact for the ACM MM rebuttal. It does not report new measurements. Every range below is an expected/target interpretation band to guide the next real experiments. Do not cite these ranges as results unless the corresponding run has been executed and the raw artifacts are saved.

Canonical review source: `papers/opera_acm_sigconf/rebuttal/reviews_20260528.md` only.

Primary evidence anchors used here:

- Submitted main paper tables in `sample-sigconf.tex`.
- Supplement protocol in `supplementary.tex`.
- Existing evidence boundary in `rebuttal/actual_evidence_tables_20260528.md`.
- Existing asset audit in `rebuttal/rebuttal_asset_audit_20260529.md`.

Hard boundary: the existing 64-sample POPE random diagnostic is warning evidence only. It showed Full vs Past+Current changed 1/64 answers, corrected 0, and introduced 1 harmful answer. Any final rebuttal claim about future rollout must therefore come from a larger, protocol-matched diagnostic, preferably POPE adversarial plus CHAIR/open-ended diagnostics.

## 1. Reviewer-To-Table Map

| Reviewer pressure | Reviewers | Table(s) that answer it | Minimum output needed before using in rebuttal |
|---|---:|---|---|
| Does the Future term actually change correct decisions? | M8du, KrEs, yx8u | Table A | Full vs Past+Current flip rate, corrected flips, harmful flips, and metric deltas on POPE adversarial and CHAIR. |
| Is the gain mostly Grounding DINO? | KrEs, M8du, yx8u, ve3y | Table B, Table E | Real-anchor, no-anchor, random-anchor, Past+Future, and same-anchor non-CHORD controls. |
| Is latency end-to-end and practical? | jjVG, KrEs, yx8u, ve3y, M8du | Table C | Detector proposal time, decode ITL, total latency, generated length, peak VRAM, batch size. |
| Are k=5 and m=3 robust or cherry-picked? | jjVG, M8du | Table D | Fixed-split k/m sweep with quality-latency Pareto rule. |
| What happens when detector anchors fail? | KrEs, yx8u, M8du | Table E | Stratified results by anchor availability and failure type. |
| Are recent baselines and novelty positioned fairly? | jjVG, KrEs, M8du | Table F | Axes table for OPERA, VCD, DoLa, HALC, ONLY, VHD/VHR, and CHORD; direct numbers only if reproducible. |
| Does the method generalize beyond two 7B backbones/object hallucination? | KrEs, yx8u, M8du | Table G | Optional stronger-backbone or broader-hallucination smoke result, or explicit scope limitation. |

## 2. Submitted Numeric Anchors

These are measured submitted-paper numbers and may be cited as existing evidence. They are not the new rebuttal diagnostics requested by reviewers.

| Model | Regime | Adv. F1 | CHAIR_S | CHAIR_I | MMBench | Decode ITL ms/token |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-v1.5-7B | Greedy | 0.8036 | 0.2291 | 0.2046 | 68.63 | 19.73 |
| LLaVA-v1.5-7B | OPERA | 0.8042 | 0.2284 | 0.1996 | 68.65 | 21.69 |
| LLaVA-v1.5-7B | Past | 0.8196 | 0.2042 | 0.1927 | 68.84 | 24.38 |
| LLaVA-v1.5-7B | Past+Current | 0.8318 | 0.1745 | 0.1852 | 69.15 | 27.24 |
| LLaVA-v1.5-7B | Past+Future | 0.8251 | 0.1794 | 0.1873 | 69.46 | 34.62 |
| LLaVA-v1.5-7B | Full | 0.8453 | 0.1548 | 0.1754 | 69.78 | 37.31 |
| InstructBLIP-7B | Greedy | 0.8327 | 0.2140 | 0.3380 | 69.34 | 16.47 |
| InstructBLIP-7B | OPERA | 0.8327 | 0.2180 | 0.3320 | 69.27 | 16.90 |
| InstructBLIP-7B | Past | 0.8413 | 0.1842 | 0.3158 | 69.64 | 21.43 |
| InstructBLIP-7B | Past+Current | 0.8517 | 0.1543 | 0.2946 | 70.08 | 24.51 |
| InstructBLIP-7B | Past+Future | 0.8485 | 0.1651 | 0.3042 | 70.43 | 32.55 |
| InstructBLIP-7B | Full | 0.8651 | 0.1347 | 0.2795 | 70.82 | 35.86 |

Submitted deltas that anchor the expected ranges:

| Contrast | LLaVA delta Adv. F1 | LLaVA delta CHAIR_S | LLaVA delta ITL | InstructBLIP delta Adv. F1 | InstructBLIP delta CHAIR_S | InstructBLIP delta ITL |
|---|---:|---:|---:|---:|---:|---:|
| Past+Current minus Greedy | +0.0282 | -0.0546 | +7.51 | +0.0190 | -0.0597 | +8.04 |
| Full minus Greedy | +0.0417 | -0.0743 | +17.58 | +0.0324 | -0.0793 | +19.39 |
| Full minus Past+Current | +0.0135 | -0.0197 | +10.07 | +0.0134 | -0.0196 | +11.35 |

## 3. Execution Locks Before Any New Run

Use these locks to keep the later result table defensible.

| Lock | Required value / behavior | Why it matters |
|---|---|---|
| Review source | Official `reviews_20260528.md` only | Avoid mixing simulated-review artifacts into rebuttal decisions. |
| Model set | LLaVA-v1.5-7B and InstructBLIP-7B first | Matches submitted paper; avoids a moving target. |
| Benchmark protocol | POPE standard splits: 3000 each; CHAIR: 5000 caption samples; MMBench dev parquet | Matches supplement. Shorter slices are diagnostic only. |
| CHORD submitted default | k=5, m=3, alpha=0.5, lambda_cur=0.25, lambda_fut=0.05, lambda_txt=1.0, epsilon=0, tau_abort=0.0 | Current archival scripts may have older defaults; final diagnostics must match the paper default. |
| Grounding DINO default | box threshold 0.25, text threshold 0.2, max_boxes=8 | Required for detector-attribution claims. |
| Determinism | do_sample=False, fixed prompt/parser, fixed split order, saved raw JSON/JSONL | Flip/correctness statistics are invalid if prompts or sample order drift. |
| Diagnostics | Save per-sample id, label, P+C answer, Full answer, candidate ids before/after, v_anchor, f_future, anchor_count, proposal time, generation time, peak VRAM | Needed to answer mechanism, detector, and cost criticisms. |
| Script preflight | Do not interpret `remote_chiro_patch/run_chord_pope_ablation.py` defaults as submitted CHORD unless the output spec records the submitted values above | The current fixed-slice runner defaults differ from the paper settings. Patch/add exact named specs before final rebuttal runs. |

Command skeletons:

```powershell
# Baseline/latency rows supported by run_eval_pipeline.py.
python run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method base --pope_split adversarial --mini_test 512 --max_new_tokens 8 --output_json outputs/rebuttal/base_llava_pope_adv512.json
python run_eval_pipeline.py --model llava-v1.5-7b --dataset pope --method opera --pope_split adversarial --mini_test 512 --max_new_tokens 8 --output_json outputs/rebuttal/opera_llava_pope_adv512.json

# CHORD POPE diagnostic skeleton. Before final use, add/verify exact submitted specs:
# p_c_submitted, past_f_submitted, full_submitted, no_anchor_submitted, random_anchor_submitted.
python remote_chiro_patch/run_chord_pope_ablation.py --split adversarial --limit 512 --offset 0 --suite spec --model llava-1.5 --max-new-tokens 8 --batch-size 1 --output-dir remote_chiro_patch/tests/rebuttal_p0_adv512
```

## 4. Table A: Future Mechanism Expected Results

Goal: answer M8du's direct question: "How often does Full flip relative to Past+Current, and when it flips, how often is it correct?"

| ID | Required run | Contrast | Expected planning band | Pass threshold | Warn threshold | Fail threshold | Rebuttal interpretation |
|---|---|---|---|---|---|---|---|
| A1 | POPE adversarial, full 3000 if possible; minimum fixed 512/1024 if deadline-bound | Full vs Past+Current | Binary answer flip rate likely low: 1-8 percent. Submitted Full-P+C Adv. F1 margin is about +0.013 on both models, so a modest net correction is realistic, not a large flip story. | Corrected flips > harmful flips by >=5 per 1000 samples, harmful flips <=40 percent of all flips, and delta Adv. F1 >= +0.006 on both or at least one model with no drop on the other. | Corrected flips only slightly exceed harmful flips, or delta Adv. F1 is +0.002 to +0.006. | Corrected flips <= harmful flips, or repeats the existing 64-sample pattern at larger scale. | If pass, say future helps selected admissions. If warn, say future has limited binary-POPE effect and use CHAIR for open-ended claim. If fail, do not claim future improves answer-level decisions. |
| A2 | CHAIR fixed 1000 minimum; full 5000 preferred | Full vs Past+Current | Submitted Full-P+C CHAIR_S gain is about -0.020 on both models. Expect -0.012 to -0.025 on a matched larger sample. | CHAIR_S improves by >=0.012 and CHAIR_I does not worsen by >0.005; MMBench/retention not harmed if measured. | CHAIR_S improves by 0.005-0.012 or only one model is clean. | CHAIR_S improves <0.005 or worsens. | If pass, future is mainly an open-ended continuation stabilizer. This is the safest pro-Future framing. |
| A3 | POPE adversarial and CHAIR same samples | Past+Future vs Past+Current vs Full | Past+Future should trail Full because current and future are complementary; submitted Full beats Past+F by +0.0202 Adv. F1 and -0.0246 CHAIR_S on LLaVA, +0.0166 and -0.0304 on InstructBLIP. | Full beats both one-missing variants on either Adv. F1 or CHAIR_S with no major retention/cost surprise. | Full only beats on CHAIR but not POPE. | Past+Future or Past+Current dominates Full on both POPE and CHAIR. | If fail, present P+C as the default and Full as not justified for rebuttal. |
| A4 | Per-flip trace table, 10-20 examples | Full-corrected vs Full-harmful flips | Expected examples should show grounded candidate promoted or unstable continuation suppressed; harmful cases should be admitted as limitations. | At least 5 clean corrected examples with candidate trace and anchor evidence, plus 2 harmful examples. | Examples exist but trace is incomplete. | Only cherry-picked final answers, no trace. | Use only trace-backed examples; do not imply token-level trace for cases without it. |
| A5 | Existing 64-sample random slice | Full vs Past+Current | Already observed: 1/64 flip, 0 corrected, 1 harmful. | Not a pass condition. | Treat as a warning to verify on adversarial/open-ended settings. | If larger runs match this, future claim fails. | This must stay in the internal evidence boundary, not the rebuttal headline. |

Reviewer-safe expected result table layout after running A1/A2:

| Model | Split/task | Contrast | N | Flip rate | Corrected flips | Harmful flips | Delta Adv. F1 / Delta CHAIR_S | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LLaVA | POPE-Adv | Full - P+C | measured | expected 1-8% | measured | measured | target >= +0.006 | pass/warn/fail |
| LLaVA | CHAIR | Full - P+C | measured | n/a | n/a | n/a | target <= -0.012 CHAIR_S | pass/warn/fail |
| InstructBLIP | POPE-Adv | Full - P+C | measured | expected 1-8% | measured | measured | target >= +0.006 | pass/warn/fail |
| InstructBLIP | CHAIR | Full - P+C | measured | n/a | n/a | n/a | target <= -0.012 CHAIR_S | pass/warn/fail |

## 5. Table B: Detector Attribution Controls

Goal: show whether CHORD's decoding rule contributes beyond simply injecting Grounding DINO proposals.

| ID | Control | Exact contrast | Expected planning band | Pass threshold | Fail threshold | Claim allowed |
|---|---|---|---|---|---|---|
| B1 | No-anchor / uniform-anchor control | P+C real anchors vs P+C uniform weights | Current term submitted contribution is roughly P+C-Past: +0.012 Adv. F1 and -0.030 CHAIR_S on LLaVA; +0.010 and -0.030 on InstructBLIP. Real anchors should recover a meaningful fraction of this. | Real anchors beat uniform by >=0.006 Adv. F1 or >=0.012 CHAIR_S on both/most matched runs. | Uniform matches real within +/-0.003 Adv. F1 and +/-0.005 CHAIR_S. | If pass: query-conditioned anchors matter. If fail: detector weighting attribution is weak. |
| B2 | Random-anchor matched control | P+C real anchors vs random boxes matched by count/area | Random anchors may regularize attention but should not match query-conditioned anchors. | Real beats random by >=0.006 Adv. F1 or >=0.012 CHAIR_S; random has higher false positives or worse CHAIR. | Random equals or beats real. | If fail: do not claim object-resonant grounding; rewrite as attention regularization at best. |
| B3 | Remove current, keep future | Full vs Past+Future | Submitted Full beats Past+F on both models, so current support should still matter when future is present. | Full improves CHAIR_S by >=0.010 or Adv. F1 by >=0.006 over Past+F. | Past+F matches Full. | If pass: current and future are complementary. If fail: Full coordination claim is weak. |
| B4 | Same-anchor non-CHORD baseline | Anchor-only rerank, or vanilla/OPERA generation with anchors computed but not used in CHORD scoring | If detector alone explains gains, this baseline will approach P+C/Full. Expected safe result is below P+C on the main quality metric. | P+C/Full exceed same-anchor non-CHORD by >=0.006 Adv. F1 or >=0.012 CHAIR_S. | Same-anchor baseline matches Full or P+C. | If fail: say gains are detector-assisted and not isolated to CHORD decoding. |
| B5 | Alternate proposer, if feasible | Grounding DINO vs another reproducible proposer | Trend may hold but absolute quality may shift with proposal quality. | Directional gains remain with lower/higher anchors, or failure is explained by proposal recall. | Alternate proposer collapses without explanation. | Optional. Use only if clean; otherwise state detector-dependence limitation. |
| B6 | Anchor-count normalized analysis | Same method, strata by anchor count and relevance | P+C/Full should improve most with 1-8 relevant anchors; zero-anchor cases should fall back near Past. | Relevant-anchor strata show gains; zero/noisy anchors do not create catastrophic false positives. | No relationship between anchor quality and outcome, or noisy anchors improve as much as relevant ones. | Supports honest detector-robustness discussion. |

Reviewer-safe detector table layout after running:

| Model | Task | Method/control | Real anchors? | Random/no anchor? | Adv. F1 | CHAIR_S | False positives | Anchor failure rate | Decision |
|---|---|---|---|---|---:|---:|---:|---:|---|
| LLaVA | POPE-Adv | P+C submitted | yes | no | measured | n/a | measured | measured | pass/warn/fail |
| LLaVA | POPE-Adv | P+C uniform | no | uniform | measured | n/a | measured | measured | pass/warn/fail |
| LLaVA | POPE-Adv | P+C random | no | random matched | measured | n/a | measured | measured | pass/warn/fail |
| LLaVA | CHAIR | Full / P+C controls | mixed | mixed | n/a | measured | n/a | measured | pass/warn/fail |

## 6. Table C: End-To-End Efficiency Expected Results

Goal: answer whether the reported ITL hides the detector and whether Full is practical.

Submitted decode-stage anchors:

| Model | Method | Submitted decode ITL | New fields required | Expected interpretation |
|---|---|---:|---|---|
| LLaVA | Greedy | 19.73 | total latency, peak VRAM, generated tokens | Baseline. No detector proposal. |
| LLaVA | OPERA | 21.69 | total latency, peak VRAM, generated tokens | Rollback-only baseline. |
| LLaVA | Past+Current | 27.24 | detector proposal time, decode ITL, total latency, peak VRAM | Practical CHORD regime if total cost remains transparent and not extreme. |
| LLaVA | Full | 37.31 | detector proposal time, decode ITL, total latency, peak VRAM | Quality mode; not real-time default. |
| InstructBLIP | Greedy | 16.47 | total latency, peak VRAM, generated tokens | Baseline. |
| InstructBLIP | OPERA | 16.90 | total latency, peak VRAM, generated tokens | Rollback-only baseline. |
| InstructBLIP | Past+Current | 24.51 | detector proposal time, decode ITL, total latency, peak VRAM | Practical CHORD regime if detector is amortized. |
| InstructBLIP | Full | 35.86 | detector proposal time, decode ITL, total latency, peak VRAM | Quality mode. |

Required efficiency result table:

| Method | Batch | Task | N | Proposal ms/img | Decode ITL ms/token | Avg gen tokens | Total ms/sample | Peak VRAM GB | Ratio vs Greedy total | Rebuttal wording |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Greedy | 1 | POPE/CHAIR | measured | n/a | measured | measured | measured | measured | 1.00x | baseline |
| OPERA/Past | 1 | POPE/CHAIR | measured | n/a | measured | measured | measured | measured | measured | rollback-only |
| P+C | 1 | POPE/CHAIR | measured | measured | measured | measured | measured | measured | target: CHAIR <=1.8x, POPE may be higher | practical regime if transparent |
| Full | 1 | POPE/CHAIR | measured | measured | measured | measured | measured | measured | target: CHAIR <=2.6x, POPE may be much higher | quality regime only |
| P+C / Full | 2 or 4 if feasible | POPE/CHAIR | measured | measured | measured | measured | measured | measured | measured | optional batch behavior |

Pass/warn/fail:

| Criterion | Pass | Warn | Fail |
|---|---|---|---|
| Transparency | All rows include proposal time, decode ITL, total time, peak VRAM, batch size, generated tokens. | One field missing but limitation is explicit. | Detector time or VRAM omitted while making practical claims. |
| P+C practicality | On CHAIR/open-ended, P+C total <=1.8x Greedy total and peak VRAM <=1.25x, or overhead is clearly amortized. | P+C total 1.8-2.2x or VRAM 1.25-1.4x. | P+C >2.2x on CHAIR or missing total time. |
| Full practicality | Full total <=2.6x Greedy total on CHAIR and quality gain is clean. | Full 2.6-3.2x; frame as quality mode only. | Full >3.2x on CHAIR or VRAM prevents normal batch use. |
| POPE short-output accounting | Explicitly states detector dominates short answers and reports both decoder-only and end-to-end. | Reports end-to-end but no decomposition. | Uses decode ITL to imply full system latency. |

Formula to include in the result script/report:

```text
total_ms_per_sample = proposal_ms_per_image + generation_elapsed_ms
decode_itl_ms_per_token = generation_elapsed_ms / generated_tokens
ratio_vs_greedy_total = total_ms_per_sample(method) / total_ms_per_sample(greedy)
```

## 7. Table D: k/m Robustness Expected Results

Goal: show k=5 and m=3 are not an arbitrary cherry-picked operating point.

Required grid:

- k in {3, 5, 10}
- m in {1, 2, 3, 4}
- Include m=0 / no future as the Past+Current reference, even if not shown in the reviewer table.
- Fixed model first: LLaVA-v1.5-7B on POPE adversarial plus CHAIR subset. If time allows, repeat the Pareto-near rows on InstructBLIP.

Expected trend table:

| Setting family | Expected quality trend | Expected latency trend | Reviewer interpretation |
|---|---|---|---|
| k=3, m=1/2 | Lower cost but often misses useful alternatives; likely below k=5 on Adv. F1/CHAIR. | Lowest Full-like cost. | Useful lower-cost point if quality is close. |
| k=3, m=3/4 | May recover some future signal but still candidate-limited. | Moderate. | If this dominates k=5,m=3, default should change. |
| k=5, m=1 | Should be close to P+C plus weak future signal. | Lower than submitted Full. | Tests whether m=3 is needed. |
| k=5, m=2 | Expected near Pareto if future helps early. | Lower than m=3. | Strong alternative if within epsilon of m=3. |
| k=5, m=3 | Submitted default; expected near quality-latency frontier, not necessarily strictly best. | Submitted Full anchor: 37.31 / 35.86 ms/token decode ITL. | Defensible only if Pareto-near. |
| k=5, m=4 | Expected diminishing returns. | Higher than m=3. | If it improves strongly, submitted horizon is under-tuned. |
| k=10, m=1/2 | More candidates but more forward passes. | Higher, may be unstable. | Useful only if quality jump is clear. |
| k=10, m=3/4 | Expected high cost with small quality gain. | Highest. | Should not be default unless quality jump is large. |

Pareto decision rule:

| Outcome | Rule | Rebuttal action |
|---|---|---|
| Pass | k=5,m=3 is non-dominated, or no other setting improves Adv. F1 by >=0.006 or CHAIR_S by >=0.010 while reducing total latency by >=10 percent. | Say the default is Pareto-near under the fixed validation split. |
| Warn | k=5,m=2 is within 0.003 Adv. F1 / 0.005 CHAIR_S and at least 10 percent faster. | Say m=3 is the submitted quality-oriented setting; lower horizon is an efficiency alternative. |
| Fail | Another setting dominates k=5,m=3 by quality and cost. | Do not defend k=5,m=3 as robust. Say camera-ready will update the default or frame it as submitted operating point only. |

Reviewer-safe table layout after running:

| k | m | Adv. F1 | CHAIR_S | Decode ITL | Total ms/sample | Delta vs k5m3 quality | Delta vs k5m3 total | Pareto status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 3 | 2 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 3 | 3 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 3 | 4 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 5 | 1 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 5 | 2 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 5 | 3 | measured | measured | measured | measured | 0 | 0 | submitted default |
| 5 | 4 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 10 | 1 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 10 | 2 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 10 | 3 | measured | measured | measured | measured | measured | measured | dominated/Pareto |
| 10 | 4 | measured | measured | measured | measured | measured | measured | dominated/Pareto |

## 8. Table E: Detector Robustness And Failure Strata

Goal: avoid claiming robustness that the detector cannot support.

Required per-sample fields: anchor_count, max_anchor_confidence, mean_anchor_area, matched query phrase, fallback_used, missed_key_object label if manually tagged for a small subset, prediction correctness, hallucinated object category for CHAIR.

| Stratum | Expected behavior | Pass threshold | Fail threshold | Claim allowed |
|---|---|---|---|---|
| 0 anchors / fallback | CHORD should behave close to Past or OPERA; large gains are not expected. | No catastrophic false-positive increase over Past; fallback explicitly counted. | False positives spike or Full claims rely on zero-anchor cases. | "Bounded by proposal quality." |
| 1 relevant anchor | Often enough for binary object questions; current support should help precision. | P+C improves precision or reduces CHAIR_S vs Past. | No improvement over Past and no diagnostic explanation. | "Query-conditioned anchors help when present." |
| 2-4 relevant anchors | Best expected stratum for object grounding. | Strongest or near-strongest gains. | No relationship between relevant anchors and gains. | Main detector-support evidence. |
| 5-8 anchors, diffuse/noisy | Risk of over-broad support; future may reduce text-dominated continuation but detector noise remains. | Noisy strata do not erase aggregate gains; harmful flips are shown. | Noisy anchors match relevant anchors or cause high harmful-flip rate. | Honest limitation, not robustness claim. |
| Missed key object | CHORD cannot reliably recover missing evidence. | Failure examples are acknowledged; P+C/Full do not overclaim. | Rebuttal hides missed-object failures. | "Detector-assisted, not detector-immune." |
| Wrong localized object | Could create hallucination by reinforcing wrong object. | Harmful examples are counted and bounded. | Harmful flips dominate corrected flips. | Limitation and future work. |

Reviewer-safe table layout:

| Stratum | N | Anchor failure rate | Past Adv. F1 / CHAIR_S | P+C Adv. F1 / CHAIR_S | Full Adv. F1 / CHAIR_S | Harmful flip rate | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 anchors | measured | measured | measured | measured | measured | measured | fallback |
| 1 relevant | measured | measured | measured | measured | measured | measured | current helps? |
| 2-4 relevant | measured | measured | measured | measured | measured | measured | strongest expected |
| noisy/diffuse | measured | measured | measured | measured | measured | measured | limitation |
| missed key object | measured | measured | measured | measured | measured | measured | limitation |

## 9. Table F: Recent Baseline And Novelty Positioning

Goal: remove the "missing obvious recent work" objection without fabricating baselines. Direct numbers are allowed only when implementation, prompts, and evaluation are reproducible under matched settings.

| Method/work | What must be verified before final text | Intervention stage | External detector? | Rollout/lookahead? | Closest relation to CHORD | Safe rebuttal use |
|---|---|---|---|---|---|---|
| Greedy | Already submitted | None | No | No | Baseline | Numeric comparison already submitted. |
| OPERA | Already submitted | Rollback / attention intervention | No | No | Closest past-signal ancestor | Numeric comparison already submitted. |
| VCD | Already submitted | Visual contrastive decoding | No separate detector in submitted framing | No | Contrastive decode-time baseline | Numeric comparison already submitted. |
| DoLa | Already submitted | Layer-contrast decoding | No | No | Text-side decode-time baseline | Numeric comparison already submitted. |
| HALC | Paper already cites/checks; verify exact existing run before reporting numbers | Plug-and-play hallucination mitigation / focal contrast | Check implementation | No explicit CHORD-style current+future verifier | Direct hallucination baseline | If numbers are verifiable, include; otherwise position conceptually and say main table kept closest matched baselines. |
| ONLY | Reviewer-named; add exact citation and verify implementation/license | One-layer intervention, training-free LVLM hallucination method per reviewer title | Verify | Likely no CHORD-style rollout; verify | Recent efficient training-free comparator | Must add related-work axis row. Direct numbers only if runnable. |
| VHD/VHR | Reviewer/planning-named; verify exact acronym/title before final text | Vision/head-divergence style attention intervention; verify | Verify | Verify | Recent attention/head baseline | Must not cite from memory. Add only after exact paper identity is verified. |
| CHORD-P+C | Submitted | Admission-time verifier: Past + Current | Yes, detector-assisted | No | Practical CHORD regime | Use as latency-oriented operating point. |
| Full CHORD | Submitted | Admission-time verifier: Past + Current + Future | Yes, detector-assisted | Yes, bounded m-step | Quality-oriented CHORD regime | Use only with future/cost diagnostics. |

Expected outcome:

| If direct baseline runs are available | If only positioning is available |
|---|---|
| Report matched numbers only if prompts, split, model, and parser are identical; otherwise do not compare numerically. | Provide axes table and say direct empirical comparison will be added only where reproducible. This is weaker than numbers but safer than unsupported claims. |

## 10. Table G: Optional Stronger-Backbone / Broader-Scope Probe

This is P1, not required before the P0 rebuttal evidence. It is useful only if P0 tables are already clean.

| Probe | Minimum run | Expected planning band | Pass | Fail / action |
|---|---|---|---|---|
| Qwen2-VL / LLaVA-NeXT / InternVL small subset | POPE adversarial 512 or CHAIR 500, one backbone | Direction should be similar but effect may shrink because stronger base models leave less room. | No collapse; P+C or Full improves the main metric without large latency surprise. | If unavailable or noisy, explicitly scope current evidence to two 7B backbones. |
| Attribute/relation hallucination subset | Small diagnostic only | CHORD may help less because anchors are object-centric. | Any clean improvement is useful but not central. | If weak, state CHORD primarily targets object/open-ended hallucination. |
| Long-form VQA/captioning | 100-500 examples | Future should help more where early weak admissions compound. | Supports Full as quality mode. | If no benefit, keep Full claim limited to submitted CHAIR trend. |

## 11. One Compact Rebuttal Table If Space Allows

Use this only after the real experiments have been executed. Replace every `measured` cell with real values.

| Concern | Diagnostic | Expected reviewer-safe result | If measured result is weak |
|---|---|---|---|
| Future mechanism | Full vs P+C on POPE-Adv and CHAIR | Low but positive corrected flip margin on POPE; clearer CHAIR_S gain near submitted Full-P+C margin (~0.02). | Say future is mainly useful for open-ended continuation, not binary POPE. |
| Detector attribution | Real vs uniform/random anchors; Full vs Past+F; same-anchor non-CHORD | Real anchors and CHORD scoring outperform no/random/same-anchor controls. | Reframe as detector-assisted heuristic; do not claim detector-independent gain. |
| Cost | Proposal time + decode ITL + total latency + VRAM | P+C is practical regime; Full is quality regime; detector time is reported separately. | Concede deployment limitation and emphasize P+C. |
| k/m | k in {3,5,10}, m in {1,2,3,4} | k=5,m=3 is Pareto-near, or lower-cost alternative is identified. | Update default or stop defending k=5,m=3. |
| Robustness | Anchor strata | Gains concentrate when relevant anchors exist; missed/noisy anchors are explicit limitations. | State detector-quality limitation directly. |
| Related work | ONLY/VHD/VHR/HALC axes | Recent work is acknowledged; direct numbers only if matched. | Do not invent comparisons; promise camera-ready discussion. |

## 12. Reviewer-Style Audit Rounds

### Round 1: KrEs / M8du Attack

Attack:

- The previous experiment matrix could still let detector attribution leak into CHORD attribution.
- Future expected results sounded too optimistic given the existing 64-sample warning.
- Current fixed-slice script defaults do not match submitted CHORD.

Revisions made in this artifact:

- Added Table B controls: uniform anchors, random matched anchors, Past+Future, same-anchor non-CHORD, and optional alternate proposer.
- Added explicit 64-sample warning boundary in Table A, with fail rule if larger runs repeat it.
- Added execution lock requiring k=5, m=3, lambda_fut=0.05, max_boxes=8, text threshold 0.2, and a script preflight before interpreting results.
- Replaced exact-looking future numbers with expected ranges and pass/warn/fail thresholds.

### Round 2: yx8u / ve3y Attack

Attack:

- The method may overclaim "training-free" despite using Grounding DINO.
- Attention should not be described as causal grounding proof.
- Full CHORD latency is nearly doubled and can be impractical.

Revisions made:

- All wording now uses "training-free for the base MLLM" or "detector-assisted" in the claim policy.
- Tables require operational diagnostics, not causal explanation claims.
- Table C requires detector proposal time, total latency, VRAM, and batch size; Full is framed as quality mode unless end-to-end data supports stronger wording.

### Round 3: jjVG Attack

Attack:

- The table set is too broad for a short OpenReview rebuttal.
- k/m defaults need a robustness rule, not just a sweep.
- Missing recent baselines may remain a visible defect.

Revisions made:

- Added the one compact rebuttal table in Section 11.
- Added a precise Pareto decision rule for k=5,m=3.
- Added Table F with safe baseline positioning and a no-fake-numbers rule for ONLY, VHD/VHR, and HALC.

### Round 4: Statistical / Execution Audit

Attack:

- A table with expected ranges can still be misleading if the final run is a small slice.
- Flip counts can be unstable when the answer parser changes.
- Result files may be hard to audit later.

Revisions made:

- Every table distinguishes full preferred runs from deadline-bound diagnostic slices.
- Execution locks require fixed prompts, parsers, split order, raw JSON/JSONL, sample ids, labels, and per-sample diagnostics.
- Pass thresholds are stated per 1000 samples where possible, so small-slice evidence cannot be overstated.

## 13. Stop Rules And Red Flags

| Red flag | What it means | Required action |
|---|---|---|
| Full vs P+C corrected flips <= harmful flips on large POPE adversarial and CHAIR gain <0.005 | Future term is not supported. | Remove future-mechanism claim; present P+C as main rebuttal evidence. |
| Random/no anchors match real anchors | Detector attribution is not isolated. | Do not claim object-resonant grounding; rewrite as detector-assisted attention heuristic. |
| Same-anchor non-CHORD matches Full/P+C | The detector may explain most gains. | Concede and narrow novelty claim. |
| P+C total latency >2.2x Greedy on CHAIR | Practical regime is weaker than submitted narrative. | Lead with transparency; do not claim deployment efficiency. |
| k=5,m=3 is dominated | Hyperparameter choice is not robust. | Update default or state it was only the submitted operating point. |
| ONLY/VHD/VHR identity or implementation is unverified | Related-work comparison risk. | Add conceptual positioning only after exact citation verification; no numerical claim. |
| Any result depends on a non-official review artifact | Rebuttal planning contamination. | Discard that rationale and reroute through `reviews_20260528.md`. |

## 14. Claim Policy For The Final Rebuttal

Allowed if P0 tables pass:

- "CHORD is training-free for the base MLLM and detector-assisted at inference time."
- "Past+Current is the latency-oriented operating point; Full CHORD is the quality-oriented operating point."
- "New diagnostics isolate future rollout by reporting Full-vs-P+C flips and corrected/harmful changes."
- "New controls separate real query-conditioned anchors from no/random-anchor and same-anchor baselines."
- "End-to-end latency, detector proposal time, and peak VRAM are now reported separately."

Not allowed unless directly measured:

- "Future rollout causally explains the submitted gains."
- "Grounding DINO overhead is negligible."
- "CHORD is detector-independent."
- "k=5,m=3 is universally optimal."
- "CHORD empirically outperforms ONLY/VHD/VHR/HALC."
- "The 64-sample random-slice diagnostic supports Full CHORD."

## 15. Recommended Execution Order

1. Patch or verify the CHORD diagnostic runner so exact submitted specs are recorded in every output JSON.
2. Run Table A on LLaVA POPE adversarial fixed 512/1024 first; inspect corrected/harmful flips before spending full-run compute.
3. Run Table B controls on the same fixed samples; stop if random/no-anchor controls match real anchors.
4. Run Table C efficiency on Greedy, OPERA/Past, P+C, and Full with proposal-time and VRAM logging.
5. Run Table D k/m sweep on the fixed validation subset; expand only Pareto-near settings.
6. Run CHAIR subset/full for Table A/B/E once POPE controls are not obviously failing.
7. Add Table F related-work positioning after exact citation verification; run direct baselines only if reproducible.
8. Draft rebuttal from the measured compact table, not from the expected ranges in this file.

