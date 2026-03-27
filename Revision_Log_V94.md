# Revision_Log_V94
- **CFG Baseline Added:** Integrated zero-shot Visual Classifier-Free Guidance into the core evaluation contract (Section 1, 4.2, Table 1).
- **VASM Re-framed as a Hack:** Removed absolute claims of "strictly preserving language structure." Explicitly defined VASM as an English-centric "crutch" and set a <5% degradation pass/fail threshold on MMBench-CN (Section 3.4, 4.3).
- **The Top-M Ceiling (Hijacking Problem) Quantified:** Introduced the Hijacking CDF (Figure B) to explicitly track how often the GT token is pushed out of the intervention window (Section 4.5).
- **Textual PPL Audit Added:** Added WikiText PPL to Stage 0 (Table 1) to scientifically prove `ContinuousAdd_Gated` destroys grammar while logit-routing preserves it.
- **OPERA Clarified:** Corrected the framing of OPERA to specify its "over-trust penalty on attention matrices" (Section 1).
- **Temperature Scaling Fixed:** Added $/ T$ adjustment to the logit spread anchor $\Delta_L$ to prevent boundary collapse at high temperatures (Section 3.3).
- **Native Separation Scatter Plot (Figure A):** Planned the specific visualization requested by the reviewer to prove manifold bleed (Section 4.3).