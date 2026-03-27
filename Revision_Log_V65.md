# Revision_Log_V65
- **Fixed Prompt-Conditioned Pruning Vulnerability**: 针对审稿人指出的“Describe the image”等开放式生成任务中 prompt cross-attention 失效的问题，在 Section 3.2 引入了 **Dynamic Fallback** 机制（短 prompt 或注意力熵高时强制 $N_{active} = N_v$），并在 Figure 2 新增了 Prompt-Pruning Ratio 的消融实验计划。
- **Addressed BPE-to-Lemma Gap**: 在 Section 3.4 和 Limitations 中正式提出了 **BPE Momentum Delegation**。明确承认 VASM 只干预 BPE 切词后的首个子词，后续子词完全依赖 base LLM 的自回归惯性补全。在 Figure 3 新增了 BPE Fragmentation Failure 的可视化计划。
- **Ensured Fair Grounding of `TLRA_calib`**: 在 Section 4.1 (Chain A / Table 1) 强制加入了 `Base + TLRA_MeanPool` 基线。严格规定只有当 AdaptiveTopK 击败 MeanPool 时，才能宣称提升来源于 decode-time local intervention，而不仅仅是 CC3M 训练数据的注入。
- **Retained Core Strengths**: 保留了 OCR Paradox / OOV Concession 的坦诚声明；保留了以 AGL / PPL 为核心的长度防作弊检查；坚持将时空视频扩展作为附录的次主线。