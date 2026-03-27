# Revision_Log_V47
- **Fixed Fatal VASM/DocVQA Contradiction**: Expanded VASM (Section 3.3) into a Dual-Tier Architecture, supplementing the WordNet dictionary with deterministic Regex triggers (numbers, dates, capitalized entities). This ensures DocVQA (Chain C) is mathematically viable to evaluate, directly addressing the reviewer's strongest critique.
- **Baseline Parity Adjusted**: Updated Chain A (Section 4.1) so the `Base + 5k LoRA` control utilizes an auxiliary InfoNCE spatial objective matching $\Phi_{calib}$'s calibration difficulty, preventing unfair baseline comparisons.
- **Clarified Zero-Shot Claims**: Explicitly removed ambiguous language; firmly branded `TLVI_calib` as a "hybrid test-time adaptation" to avoid false advertising of a "training-free" method.
- **Enhanced Experimental Tables/Ablations**: 
  - Added an `Intervention Trigger Rate (%)` column to Table 2 (Chain C) to validate that performance deltas aren't noise.
  - Added a formal ablation on the prefill threshold percentile ($\theta_{active}$) to empirically justify the 85th percentile.
  - Added TTFT vs Context Length ($N_v$) scaling to Section 4.4 and Appendix E.
- **Expanded Appendix D**: Added a planned failure case tracking BPE Dynamic Continuation breakdown (greedy string matching failure) per reviewer request.