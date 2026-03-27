# Revision_Log_V115
- **Fixed the BPE Lookahead Fallacy:** Replaced "Pre-Commitment Nucleus Scanning" with "Deterministic Vocabulary Trie Masking" to explicitly address the reviewer's critique on subword collateral damage (protecting `_really` while penalizing `_refrigerator`).
- **Upgraded the Negation Evaluation:** Subdivided the Negation Trap audit into Explicit vs. Implicit negations, formally acknowledging the kill-switch heuristic will likely fail on the latter.
- **Added System Overhead Metrics:** Introduced Tokens/Second and Peak GPU memory to Table 3 to address the critical latency blindspot for late-fusion methods.
- **Clarified Baseline Symmetry:** Explicitly separated "Training-Free Decoders" (DoLa, VCD) from "Trained Adapters/Prompts" (TLRA, DINO) in the experimental tables.
- **Integrated VASM:** Formally named the $\tau$-abort mechanism Visual-Aware Syntax Maintenance (VASM) and designed MMBench/MME ablations to test its structural preservation.
- **Retained Core Strengths:** Maintained the frozen $W_{out}$ anchor, Dynamic Pooling, Prompt-Injected DINO baseline, and the intellectual honesty regarding Relational Exacerbation.