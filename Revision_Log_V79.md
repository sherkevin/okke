# Revision_Log_V79
- **Fixed VASM Autoregressive Paradox (Critical Flaw):** Replaced the vague decode-time POS tagging with an *offline, pre-computed static dictionary* derived from WordNet, coupled with dynamic $O(1)$ state-machine updates for BPE inheritance. This directly resolves the reviewer's concern about catastrophic latency and tagger inaccuracy on incomplete prefixes.
- **Clarified Methodological Identity of `TLRA_calib`:** Explicitly acknowledged that using 50k captions makes `TLRA_calib` a lightweight Test-Time Adaptation (TTA) / PEFT method. This removes the "unfair advantage" critique when comparing against pure zero-shot baselines like DoLa/VCD.
- **Formalized the Evaluation Protocol:** 
    - Added the hard Average Generation Length (AGL) column to Table 1 to audit truncation artifacts.
    - Explicitly designated DocVQA in Table 2 as a *flat-line* safety check (expecting zero improvement).
    - Structured Table 3 as the definitive Unseen Parity Ablation to trigger the Fallback Clause if needed.
    - Formalized Figure 1 to map TPOT (ms) vs POPE F1 on A100 GPUs for $M \in \{10, 50, 100\}$.
    - Added the sequence probability graph outline for the Out-of-Candidate failure mode analysis.
- **Retained Core Strengths:** Maintained the highly praised framing of baselines, the Fallback Clause, and the negative control concession for OCR tasks.