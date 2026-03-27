# Revision_Log_V18
- 删除了所有关于 "Pooling Paradox" 的旧表述，严格采用 "token-local visual evidence vs. global pooling" 的正面且客观的 framing。
- 保留了原有的三大核心创新：`BRA_zero`/`BRA_calib` fairness boundary, decode-time local intervention, 和 VASM 的 BPE 结构保护。
- **针对 Reviewer 提出的 "Fatal Risk" (BRA_zero 语义有效性)**：在 Section 4 实验大纲中新增了 "Defense Line 1: Zero-Shot Semantic Validity Pilot"，强制要求先通过 COCO Bounding box 的 token overlap 实验自证 $W_{vocab} \cdot h_L^{(v_j)}$ 的合理性。
- **针对 Reviewer 提出的 "Naive Spatio-Temporal" 硬伤**：主动将 Video 从主线 Claims 中降级（移除了 Defense Line 4 Video 测试），在 Methodology 3.2 明确声明方法目前专注于 2D 空间，并在 Discussion 5 中将 Spatio-Temporal Dilution 作为一个必须在未来通过 temporal proximity 解决的 Limitation，保全了 Image 核心实验的逻辑严密性。
- **针对 Reviewer 对 VASM Dictionary 的质疑**：弱化了静态字典的地位，将论述重点和实验 Ablation 完全聚焦在 "BPE Continuation Inheritance" 这一更鲁棒的动态防线上，并在 Limitations 中诚实承认了字典的瓶颈和多义词缺陷。
- 全面重构了 Section 4，将实验大纲组织成严格的 "3 条证据链 + 5 道防线"，并将 AGL（防缩短作弊）、`BRA_no_VASM`（防推理崩塌）和 `BRA_MeanPool`（证明 Local 价值）明确列为必须通过的 Execution Rules。