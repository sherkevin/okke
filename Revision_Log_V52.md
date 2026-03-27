# Revision_Log_V52
- **Clarified Hybrid Claim:** Explicitly revised the Abstract and Intro to concede that `BRA_calib` makes this a *hybrid test-time adaptation*, distinctly separating it from purely zero-shot regularizers (DoLa/VCD/OPERA).
- **Mathematical Rigor Added:** Formally documented the Compositional InfoNCE loss equation in Section 3.1, including explicit formulations for the Graph Veto ($\mathbb{I}_{graph}$) and Semantic Veto ($\mathbb{I}_{sem}$) indicator functions altering the denominator.
- **BPE-to-WordNet Formalization:** Replaced vague BPE handling with a concrete "Greedy Prefix-Trie Bitmask" heuristic in Section 3.3, explaining exactly how subword fragments tentatively inherit the $\gamma=1$ mask during step-by-step generation.
- **Rolling Decay Sensitivity:** Added a parameter sensitivity discussion for window size $W$ and target rate $\tau_{target}$ in Section 3.2.
- **Experimental Execution Blueprint Hardened:** 
  - Added MMMU(Hard) stratification (Math vs. Biology) to track dictionary variance.
  - Shifted the "Visual Proof of Locality" (heatmap comparing `BRA_calib` vs `BRA_MeanPool`) from the Appendix to a mandatory Main-Text requirement in Chain C.
  - Expanded Defense Line 4 to explicitly track Inter-Token Latency (ms/token overhead).
- **Retained Highlights:** Strictly maintained the 2D limitation, the respectful framing of global regularizers, the Compositional Coherence logic, and the critical "Inaction Boundary" scatterplot.