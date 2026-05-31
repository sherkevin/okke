# CHORD Rebuttal Coverage Matrix Against True Reviewer Intent, 2026-05-29

Scope: map the full reviewer-intent document to the minimum-difference expected-results author-response PDF. Canonical reviewer-intent source: `reviewer_true_intent_analysis_20260529.md`.

## Shared Concern Coverage

| True need from reviewer-intent analysis | Main reviewers | Required answer in PDF | Current PDF status | Remaining risk |
|---|---|---|---|---|
| Future mechanism validation | M8du, KrEs, yx8u | Full vs Past+Current flip rate, corrected/harmful flips, CHAIR continuation delta. | Covered by Future table; needs final expected-values polish and real-result replacement later. | If real corrected flips do not exceed harmful flips, Future claim must be narrowed. |
| Detector / Grounding DINO attribution | KrEs, M8du, yx8u, ve3y | Real anchors vs uniform/random/no-anchor, Past+Future, same-anchor non-CHORD, detector-failure limitation. | Covered by detector control table and prose; should explicitly mention zero/noisy-anchor fallback. | KrEs may remain skeptical without real measured controls. |
| End-to-end cost honesty | KrEs, M8du, jjVG, ve3y, yx8u | Proposal time, decode ITL, total latency, generated length, peak VRAM, batch setting. | Covered by efficiency table; needs final table wording that Full is quality mode, P+C is practical mode. | Full CHORD remains slow; do not oversell. |
| k=5, m=3 robustness | jjVG, M8du | Compact k/m sweep or Pareto-near table with concrete values. | Partly covered in prose; should become an expected-value table. | If omitted from PDF, jjVG has an easy unresolved complaint. |
| Recent baselines / related work | jjVG, KrEs, M8du | Explicit ONLY, VHD/VHR, HALC positioning; direct numbers only if matched and reproducible. | Covered in prose; should include a concise axis sentence or table. | Do not claim superiority without real matched runs. |
| Claim discipline | yx8u, ve3y, KrEs | Detector-assisted wording, attention as operational feature, object/open-ended hallucination scope, toned-down terminology. | Covered in novelty/presentation prose; must keep this language in final PDF. | Overclaiming can lose current Weak Accepts. |
| Figure 2 clarity | jjVG | Commit to redraw into sequential lanes. | Covered in presentation section. | Low risk if mentioned explicitly. |
| Generality / stronger models | KrEs, yx8u, M8du | Acknowledge two-7B/object-heavy scope; optionally add smoke result only if real. | Covered as limitation language, not as new evidence. | Cannot fully solve without extra experiments. |

## Reviewer-Specific Checklist

| Reviewer | Must hear first | PDF section/table that should answer it | Adequate for minimum-difference expected version? |
|---|---|---|---|
| M8du | "We directly measure Full vs P+C flips and whether changes help or hurt." | Section 2 Future table; reviewer close. | Yes, if table has concrete expected values and small expected-note. |
| KrEs | "We isolate detector attribution and report true end-to-end cost." | Section 3 detector controls; Section 4 efficiency table. | Yes for format; real evidence still pending engineer run. |
| yx8u | "We bound the claim: detector-assisted, operational attention, limited scope." | Sections 1, 5, reviewer close. | Yes. |
| ve3y | "P+C is practical mode; Full is quality mode, not cheap." | Section 4 efficiency table and prose. | Yes. |
| jjVG | "We add k/m, related methods, cost split, and Figure 2 cleanup." | Section 5 k/m/related work/presentation. | Needs a compact expected k/m table to be fully convincing. |

## Required PDF Changes

1. Rename the visible title to `Author Response for Submission 8826: CHORD`.
2. Remove top-level "draft/internal" framing from the PDF body; keep expected-result caveats only in small notes under tables.
3. Add a compact expected-value k/m table.
4. Keep concrete expected values in every core evidence table.
5. Add exact wording that the engineer-run results will replace expected values; final PDF should delete those small notes once measured values are inserted.
6. Keep the reviewer-specific close short but complete.

## Strict Boundary

The minimum-difference PDF can guarantee final-form structure, not final factual correctness. Its expected numbers are plausible targets anchored to submitted tables and reviewer-requested diagnostics, but they remain non-measured until the engineer-run artifacts replace them.
