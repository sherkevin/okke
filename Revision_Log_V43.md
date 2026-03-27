# Revision_Log_V43
- **Tone & Framing:** Completely stripped the dramatic "meta-writing" (e.g., "ruthless", "the ultimate crucible", "zero-shot illusion"). Rewrote Intro and Abstract to sound objective and mature, while preserving the core math and hypotheses.
- **Data Leakage Mitigation:** Explicitly updated the 5k pairs for $\Phi_{calib}$ to be sourced from *Visual Genome subsets strictly disjoint from POPE/CHAIR evaluation sets* to resolve the critical COCO overlap flaw pointed out by the reviewer.
- **Evaluation Upgrades:** 
  - Added POPE False Positive Rate (FPR) to Table 1 to prove we aren't just universally suppressing visual generation.
  - Formalized the metrics for FREAK (Accuracy/ANLS) and DocVQA (ANLS) in Chain C.
  - Added Figure 2 details (side-by-side heatmaps of `lm_head` vs $\Phi_{calib}$).
- **Latency Honesty:** Acknowledged the $\mathcal{O}(M \cdot N_v \log k)$ decode bottleneck directly in the text. Added BF16 and FlashAttention-2 specifications, and committed to plotting a vanilla baseline horizontal line in Figure 3.
- **Module Pruning:** Renamed the theatrical "Arrogance-Triggered Entropy Gate" to "Confidence-Conditioned Gating" and maintained the scientific 2% ablation mandate.
- **Preserved Highlights:** Kept the strong `Base + 5k LoRA` control, the 2D spatial focus, the DoLa/VCD/OPERA "orthogonal regularizer" framing, the `BRA_MeanPool` baseline comparison, and the highly praised VASM Trigger Rate (OOV vs Accuracy Drop) scatterplot.