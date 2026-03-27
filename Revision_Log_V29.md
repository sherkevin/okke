# Revision_Log_V29
- **Fixed Data Contamination Risk:** Explicitly added a "Zero Data Leakage Guarantee" stating the 5k COCO calibration images are strictly disjoint from POPE/CHAIR/DocVQA test sets.
- **Added Fair Calibration Baseline:** Introduced a `Base + 5k LoRA` baseline in Table 1 to definitively prove performance gains originate from decode-time local evidence rather than mere exposure to calibration data.
- **Added OOV Tracking:** Committed to tracking the exact percentage of ground-truth targets defaulting to $\gamma=0$ in Chain B to address VASM dictionary exhaustion transparency.
- **Refined Negative Sampling:** Added a "semantic negative constraint" to the InfoNCE loss to prevent penalizing valid overlapping objects in dense scenes.
- **Added Hyperparameter Sensitivity:** Formalized Figure 2 to strictly track performance across Top-$M$ sizes and $\theta_{noise}$ thresholds.
- **Maintained Strengths:** Retained the 2D bounding constraint (kept video excluded), preserved the DoLa/VCD "orthogonal regularizer" framing, and kept the highly praised VASM BPE inheritance logic.