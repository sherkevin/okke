# Revision_Log_V2
- **Title Fixed**: Replaced "A-OSP" with "Bi-directional Resonance Anchoring (BRA)" to resolve the fatal disconnect pointed out by the reviewer.
- **Space Mismatch Resolved**: Explicitly redefined $\bar{z}_j$ as $h_L^{(v_j)}$ (the last hidden state corresponding to visual tokens just before unembedding) to ensure cosine similarity is mathematically valid in the same latent space.
- **BPE Collapse Fixed**: Introduced the "Sub-word Momentum Integration (SMI)" strategy to pass resonance scores to sub-word fragments, directly addressing the reviewer's most critical methodology critique.
- **Equation Corrected**: Changed the logit modification from absolute value multiplier ($-\alpha |L_{orig}|$) to a translationally invariant additive shift ($+\alpha S_{res} - \beta(1-S_{res})$) to avoid flipping negative logits.
- **Trigger Mechanism Adjusted**: Removed the hard entropy threshold ($\Delta E_t > \epsilon$) which fails on "blindly confident" hallucinations. Replaced with continuous application relying on the improved equation and topological orthogonality.
- **Tone Purified**: Removed all aggressive and subjective rhetoric ("无脑", "死结", "降维打击"). Replaced with objective academic terminology.
- **Experiment Redesign**: Converted all "placeholder" absolute claims into a rigorous "Evaluation Protocol / Hypothesis" format, directly mapping to the reviewer's suggested 5 tables/figures.

**Retained Highlights:**
- The theoretical critique of the "Pooling Paradox" (MeanPool destroying 2D-RoPE).
- The core intuition of Unembedding Space Topological Orthogonality.
- The $Z_\Delta$ design for video action hallucination.

**Requires Future Experimental Validation:**
- The actual latency profiling (ITL/VRAM) on RTX 5090.
- Running VisualWebBench to prove the zero-pooling supremacy vs. RepE MeanPool.
- Density distribution plotting (KDE) of $S_{res}$ to physically prove topological orthogonality.