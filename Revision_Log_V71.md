# Revision_Log_V71
- **Scope Refinement:** Explicitly scoped the Abstract and Intro to "physical entity grounding in natural scenes," directly addressing the reviewer's concern about the OCR contradiction and overclaiming.
- **Leakage Defense Upgraded:** Formalized the "Vocabulary Leakage Audit" (Seen vs. Unseen POPE entities) and strict dataset deduplication as mandatory parts of Evidence Chain A.
- **Transparency on Cost & Heuristics:** Acknowledged the heavy external prior/cost of GroundingDINO and CLIP-ViT-L/14 in Section 3.1. Acknowledged that $L_{char} \ge 4$ is a tokenizer-dependent heuristic.
- **BPE-CSR Evaluation:** Promoted BPE-CSR tracking to a continuous distribution audit across intervention strengths ($\alpha$) (Figure 1 in Exp section).
- **Statistical Significance Requirement:** Added the hard $p < 0.05$ requirement for the gap between AdaptiveTopK and MeanPool to Evidence Chain C.
- **Failure Modes Detailed:** Expanded the Limitations section to explicitly cover the $M=10$ candidate bounding risk, WordNet English-only brittleness, and the prefill pruning blindspot. 
- **Video Pilot Restrained:** Formally scoped the video experiment as a "Bounded Secondary Video Pilot" focused strictly on proving temporal local evidence, avoiding any distraction from the main 2D image contract.