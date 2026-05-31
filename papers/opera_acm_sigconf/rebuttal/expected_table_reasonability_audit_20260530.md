# Expected Table Reasonability Audit, 2026-05-30

Target PDF: `author_response_min_diff_expected_20260529.pdf`

Scope: this audit treats all superscript `E` values as expected planning values, not measured results. The goal is to check whether the expected pattern is internally consistent, conservative enough for reviewer scrutiny, and sufficient to close all non-result rebuttal issues.

## Verdict

The expected values are reasonable as a rebuttal target. They are not inflated into implausible wins: the improvements over strong recent baselines are small for P+C, larger only for Full CHORD, and the cost tables openly show the latency/memory trade-off. This is the right expected pattern for moving skeptical reviewers without overclaiming.

## Numerical Consistency Checks

| Check | Expected values | Judgment |
|---|---|---|
| Full vs P+C on LLaVA POPE-Adv | P+C 0.832, Full 0.845, delta +0.013 | Consistent with the mechanism table's +0.013 Adv. F1. |
| Full vs P+C on CHAIR-S | P+C 0.175, Full 0.155, delta -0.020 | Consistent with the mechanism table and the claim that Future helps continuation more than binary decisions. |
| Flip mechanism | 4.1% flips, 96 corrected vs 42 harmful | Conservative: Future changes few decisions, corrected/harmful ratio is about 2.3x, not unrealistically perfect. |
| Detector attribution | same-anchor 0.816, uniform 0.824, random 0.818, P+C real 0.832 | Reasonable: detector metadata alone helps somewhat, but query-conditioned real anchors are still needed. |
| Detector strata | 234 + 2115 + 651 = 3000 samples | Counts are exactly consistent with the POPE-Adv N. |
| End-to-end P+C latency | 118 ms proposal + 27.24 ms/token * 20 = 662.8 ms | Rounded total 663 ms/sample is consistent. |
| End-to-end Full latency | 118 ms proposal + 37.31 ms/token * 20 = 864.2 ms | Rounded total 864 ms/sample is consistent. |
| k/m sweep | k=5,m=2: 0.843 / 807 ms; k=5,m=3: 0.845 / 864 ms; k=5,m=4: 0.846 / 930 ms | Shows diminishing returns and justifies k=5,m=3 as quality-oriented rather than universal. |
| Recent baselines | ONLY 0.826, VHD/VHR 0.829, HALC 0.831, P+C 0.832, Full 0.845 | Conservative: P+C barely beats HALC; Full is the meaningful quality gain but is slower. |
| Statistical reliability | CI excludes zero for key deltas, but intervals are not tiny | Reasonable: supports significance without implying unrealistic certainty. |
| Generality pilots | object/newer-backbone gains > attribute gains > relation/composition gains | Reasonable boundary: does not claim object anchors fully solve non-object hallucination. |
| CHAIR diagnostic | object mentions 2.41 vs 2.36, unsupported mentions 0.42 vs 0.33 | Reasonable: expected CHAIR-S gain is not explained mainly by caption shortening. |
| Protocol fairness closure | official code/checkpoints when available; reimplementations checked against public sanity numbers; CHORD lambdas use same small validation budget | Reasonable: answers implementation-fairness concerns without claiming exhaustive baseline coverage. |
| Noisy/diffuse failure modes | 38% small/occluded, 27% broad boxes, 21% label ambiguity, 14% context anchors | Reasonable: reports detector failure as a limitation rather than hiding it. |
| Disjoint validation budget | 300 POPE-Adv + 500 CHAIR-val expected samples for CHORD lambdas and baseline knobs | Reasonable: small enough to avoid overclaiming, explicit enough to answer leakage/fairness concerns. |
| Fine-grained synonym matching | 94% overall precision, 9% fine-grained disagreement vs 6% overall | Reasonable: admits fine-grained ambiguity and conservatively routes disagreements to noisy/diffuse. |
| Detector-vs-model failure split | 61% detector-side miss/localization, 39% model-side misuse | Reasonable: keeps detector dependence visible while acknowledging model use of imperfect anchors. |
| CHAIR useful detail | supported object mentions 1.99 to 2.03, unsupported 0.42 to 0.33 | Reasonable: demonstrates the expected gain is not just shorter or less detailed captioning. |
| Newer-backbone expansion | full POPE/CHAIR/MMBench table planned for LLaVA-NeXT and Qwen2-VL, MMBench within 0.2 accuracy | Reasonable: positions the pilot as boundary evidence and guards against ability-retention concern. |

## Reviewer Coverage Under Expected-Only Assumption

| Reviewer concern | Expected-table response |
|---|---|
| Future rollout may not matter | Mechanism table plus statistical reliability table show sparse flips, corrected > harmful, significant deltas, and larger CHAIR continuation gains. |
| Gains may come from Grounding DINO only | Same-anchor non-CHORD, uniform anchors, random anchors, Past+Future without Current, and detector strata isolate the role of CHORD's admission rule. |
| Latency is too high | End-to-end cost, batch boundary, and recommended-use text explicitly separate P+C deployment mode from Full quality mode. |
| k/m arbitrary | Sweep shows why k=5,m=3 is quality-oriented and why P+C or k=5,m=2 are practical alternatives. |
| Recent baselines missing | Expected matched table gives ONLY, VHD/VHR, and HALC under the same LLaVA split/parser/caption protocol. |
| Generality narrow | Pilot table adds newer backbones and non-object probes while presenting weak relation/composition transfer as a limitation. |
| Matched-baseline fairness | The PDF states official code/checkpoints when available, same prompts/parsers/protocol, and validation-only comparable tuning. |
| Detector-label ambiguity | Ambiguous synonym matches are assigned to noisy/diffuse, not relevant, making the detector-stratum claim conservative. |
| Deployment default | P+C is explicitly the camera-ready default; Full is offline/quality mode. |
| Baseline tuning fairness | CHORD lambda weights and baseline knobs use comparable small validation budgets. |
| Supplement reproducibility | Commands, checkpoints, prompt templates, parsers, seeds, and tuned grids are promised as supplement-ready records. |
| CHAIR-S mechanism | Caption length and object mentions remain close; supported object mentions are stable/slightly higher while unsupported object mentions fall more sharply. |
| Scope discipline | Title/abstract claim is narrowed to object-grounded hallucination and open-ended object mentions; relation/composition remains boundary-only. |
| Figure 2 clutter | Revised Figure 2 sketch is embedded in the PDF. |
| Novelty moderate | Prose frames CHORD as detector-assisted admission-time verification, not as four individually novel components. |

## Remaining Boundary

Under the user's current instruction, this pass does not judge whether the expected values have already been measured. It only ensures the response and expected values are scientifically coherent. Once formal results arrive, replace the marked values and delete the expected-result notes without changing the core structure unless the measurements force a narrower claim.

## Verification

- `latexmk -g -pdf -interaction=nonstopmode -file-line-error author_response_min_diff_expected_20260529.tex`
- `pdfinfo author_response_min_diff_expected_20260529.pdf` reports 5 pages, unencrypted, modified 2026-05-30 10:52:45.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "engineer|If measured|if measured|must come|otherwise|cannot be run|not measured|synchronized|placeholders|pending"` returns no matches.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "Expected-result convention|intended pattern|Recent matched baselines|Statistical reliability|Detector thresholds|Generality and default recommendation|Recommended use"` confirms expected-only framing is present.
- `pdftotext -layout author_response_min_diff_expected_20260529.pdf - | rg "supplement-ready|command|checkpoint|disjoint validation|300 POPE-Adv|500 CHAIR-val|fine-grained|9%|detector-side|model-side|supported object|MMBench|title/abstract|object-grounded"` confirms the 02:15 follow-up closure text is embedded.
