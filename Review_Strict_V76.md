# Review_Strict_V76

## Overall Score
Score: 4/5

## Verdict
This is a methodologically mature, refreshingly self-aware proposal. Unlike many submissions that rely on macroscopic, often inaccurate critiques of existing baselines (e.g., falsely claiming `DoLa` or `VCD` depend on global pooling), this paper proposes a narrow, falsifiable positive hypothesis: injecting token-local visual evidence at decode-time while preserving structure via VASM. Because the paper is currently in an experimental pre-registration state, my verdict is a conditional **Weak/Strong Accept (4/5)**, entirely dependent on the rigorous, unmanipulated execution of the proposed evaluation contract. If the authors execute the parity ablations strictly as defined, the resulting paper will be a high-quality contribution to ACM Multimedia, regardless of whether the `AdaptiveTopK` claim fully survives or contracts to a narrower VASM-centric claim.

## Summary
The paper introduces Token-Local Resonance Anchoring (TLRA), a decode-time intervention framework for MLLMs. It seeks to reduce hallucination by reweighting candidate tokens based on localized visual support, using Adaptive Top-$k$ resonance. To prevent the destruction of syntax and multi-token entities, it introduces Vocabulary-Anchored Semantic Masking (VASM). The paper explicitly separates a strict zero-shot probe (`TLRA_zero`) from a frozen-model calibrated plug-in (`TLRA_calib`). It commits to a strict evaluation protocol: isolating hallucination reduction, proving structure preservation (using OCR tasks explicitly as negative controls), and mandating a parity ablation against `MeanPool` and `RandomK` baselines using the exact same calibrator.

## WhatShouldBeKept
1. **The Framing of Baselines:** Keep treating `DoLa`, `VCD`, and `OPERA` strictly as legitimate competitive baselines. Do not slide back into inventing macro-flaws about them; your positive methodology is strong enough to stand on its own.
2. **The Separation of `TLRA_zero` and `TLRA_calib`:** This transparency is excellent. Acknowledging that `TLRA_calib` is a "base-model-frozen plug-in" rather than claiming it is magically "training-free" avoids the calibrator blackbox trap.
3. **The OCR Concession and Negative Control:** Retain the stance that TLRA mathematically bypasses arbitrary text-in-image tokens via VASM. Using `DocVQA` as a strict negative control (expecting flat performance, not gains) is a brilliant, logically consistent methodological boundary. Do not give in to the temptation to claim OCR capabilities later.
4. **The Parity Ablation Contract:** The mandate that `AdaptiveTopK` must beat `MeanPool` and `RandomK` using the *exact same* $\Phi_{calib}$ weights is the scientific core of the paper. Keep this front and center.
5. **The Bounded Video Pilot:** Keep video strictly in the appendix. Do not attempt an "image-video unified theory" narrative. The spatial routing claims are complex enough without adding temporal confounding variables to the main text.

## MajorWeaknesses
While the experimental plan is structurally sound, there are critical vulnerabilities in the proposed execution that could invalidate the results if not tightly controlled:

1. **The VASM Lexical Table Vulnerability:** Section 3.4 mentions a "precomputed lexical table" to identify physical entities. If this is a manually curated list of heuristics, it is unscalable and scientifically brittle. You must explicitly define an automated pipeline for this (e.g., using WordNet synsets for physical objects, or automated POS tagging of the vocabulary).
2. **Calibrator Data Leakage ($\Phi_{calib}$):** You propose training $\Phi_{calib}$ on a 50k caption subset. If this subset shares object distributions with POPE or CHAIR, `AdaptiveTopK` might not be performing true "spatial routing." Instead, the calibrator may simply be memorizing object co-occurrences, acting as a language prior. The protocol *must* include an explicit "Seen vs. Unseen" Category Leakage Audit.
3. **Hyperparameter Fragility ($\tau_{sim}$, $\tau_{evidence}$, $\alpha$):** The method introduces several temperature and scaling parameters. If these are tuned per dataset (e.g., one set of hyperparameters for POPE, another for MME), the method is invalid. You must establish a single, fixed hyperparameter set derived from a held-out validation split.
4. **The Out-of-Candidate Bound Risk:** You correctly acknowledge that post-hoc reweighting cannot recover tokens absent from the Top-$M$ set. However, if $M$ is too large (e.g., 100+), the inference cost will explode. The efficiency trade-off is currently under-specified.

## SectionBySectionComments
*   **Abstract & Intro:** Excellent scoping. The transition from the problem definition to the three design commitments is clear.
*   **Section 3.1 (`TLRA_calib`):** You must define the exact contrastive loss used to train $\Phi_{calib}$. Is it InfoNCE? What are the negative samples? This must be reproducible.
*   **Section 3.2 (Adaptive Top-$k$):** The fallback to $k_{min}$ is logical, but how is $\rho$ determined? Be careful that $\rho$ doesn't inadvertently recreate a global pool if the resolution is low.
*   **Section 3.4 (VASM):** The BPE continuation inheritance is a highly practical and novel engineering contribution for decode-time methods. Ensure you visualize a failure case of this in the appendix (e.g., where a tokenizer splits a word in a way that violates semantic boundaries).
*   **Section 4.1 (Zero-Shot Pilot):** Be prepared to accept that `TLRA_zero` might fail entirely on some base models due to deep embedding asymmetry. If it fails, execute your planned contraction to `TLRA_calib` without shame.

## RequiredRevisions
1. **Formalize VASM Construction:** Explicitly document the algorithmic generation of the VASM root-token lookup table. No manual heuristics allowed.
2. **Lock Hyperparameters:** State explicitly in the experimental protocol that $\tau_{sim}$, $\tau_{evidence}$, $\alpha$, and $M$ will be fixed across *all* primary benchmarks after tuning on a single, distinct validation set.
3. **Pre-define the Contractive Claim:** If the experiments reveal that `TLRA_AdaptiveTopK` is statistically tied with `TLRA_MeanPool`, you must commit to shrinking the paper's claim. The fallback claim—"VASM successfully protects structure during global calibrated logits adjustment"—is still a publishable finding. Do not force a fake spatial-routing narrative if the data does not support it.

## SuggestedFiguresTablesExperiments
To help you finalize the ongoing experiments, enforce the following structure:

*   **Table 1 (Hallucination - POPE/CHAIR):** You *must* include the Average Generation Length (AGL) column side-by-side with CHAIR/POPE metrics. If TLRA cuts AGL by >10-15% compared to the Base model, reviewers will rightfully accuse the method of artificially inflating precision simply by truncating outputs.
*   **Table 2 (Structure - MMBench/MME/DocVQA):** For `DocVQA`, add a specific metric for OCR-token overlap to prove mathematically that VASM masked these tokens, resulting in the expected flat line (negative control).
*   **Table 3 (Local Evidence Parity - FREAK):** Break this down into "Objects Seen in $\Phi_{calib}$" vs. "Objects Unseen in $\Phi_{calib}$". If `AdaptiveTopK` only beats `MeanPool` on *seen* objects, your spatial routing hypothesis is actually a data-leakage artifact.
*   **Figure 1 (Systemic Cost Pareto):** Plot Time-Per-Output-Token (TPOT) on the X-axis and POPE F1 on the Y-axis. You must plot the Base model, `VCD`, `DoLa`, and `TLRA` at $M \in \{10, 50\}$. Reviewers need to see exactly how much your localized extraction costs in latency.
*   **Failure Case Figure:** Show a specific visual example where the required entity token ranked 51st in the base model's logits, but $M$ was set to 50, proving the hard mathematical limit of your intervention.

## AcceptanceOutlook
The outlook is highly positive *if and only if* you stick to the uncompromised execution of this experimental protocol. If the results require you to downgrade the local-evidence claim to a MeanPool+VASM claim, do so honestly; the paper will survive on methodological rigor. If you attempt to manipulate hyperparameters per dataset or hide AGL truncation to force a SOTA narrative, the paper will be rejected. Proceed with the strict execution as planned.