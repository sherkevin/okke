# Revision_Log_V38

- 本版不是在 `V37` 上继续局部打补丁，而是一次**正式收敛重置**。
- 新的正式 benchmark 体系从 `标杆V1` 开始，当前对应文件为 `论文大纲_标杆V1.md`。
- `论文大纲_V38.md` 由 `论文大纲_标杆V1.md` 直接复制生成，作为新的正式最新迭代起点。
- 本次收敛的核心变化：
  - benchmark 与迭代版分离：benchmark 改用 `标杆VX` 编号
  - 正式命名统一到 `TLRA`
  - 主线收缩为 image-only
  - `TLRA_zero` 被降为 Stage 0 viability probe
  - 后续 reviewer/scientist 迭代必须围绕新的 benchmark 合同继续推进
- 从这一版开始，若继续自动迭代，应把 `V38` 视为新的 latest iteration base，而不是继续沿用旧链路自动生成的 `V39+` 草案。
# Revision_Log_V38
- **Entropy Gate Paradox Fixed:** Completely redesigned the logic to a "Dual-Condition Entropy Gate." High-confidence visual nouns now *force* intervention rather than bypassing it, directly directly confronting the "arrogance of language priors" as requested by the reviewer.
- **`BRA_calib` Identity Defined:** Explicitly specified $\Phi_{calib}$ as a lightweight 2-layer MLP (~4.2M parameters, <0.01 TFLOPs). Introduced a formal mathematical "Washout Threshold" (>2% POPE F1 degradation) to dictate exactly when `BRA_zero` fails and `BRA_calib` must be used.
- **Negative Sampling Latent Poisoning Fixed:** Updated SAM/IoU sampling to mandate that negative patches must belong to a different COCO semantic super-category, preventing contextual correlation poisoning.
- **Dense Image Median Fix:** Added a historical upper-bound clamp ($\theta_{max}$) to the moving median to ensure dense crowds don't wipe out valid patch activations.
- **Experimental Protocol Upgraded:** Added CHAIR Object vs. Attribute metric. Added baseline horizontal line to Chain B scatterplot. Added "Top-5 Worst Polysemy False Positives" table plan. Specified hardware (A100 80GB) for defense lines. Added "Threshold Dynamics" step-by-step graph to Appendix plan.
- **Retained Highlights:** Maintained the strict 2D bound, the brilliant `Base + 5k LoRA` control baseline, and the acknowledgment of DoLa/VCD/OPERA as orthogonal baselines rather than setting up a false global pooling strawman.
- **Pending:** Actual execution of the protocol, especially tracking the BS=8 OOM limits and the domain shift degradation on DocVQA.