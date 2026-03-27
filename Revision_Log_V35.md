# Revision_Log_V35
- 针对 $\Phi_{calib}$ 训练中的 InfoNCE 负样本重叠风险，引入了基于 SAM 分割和 IoU < 0.1 的严格负样本约束。
- 修复了硬编码的 0.90 entropy gate 缺陷，提出了基于验证集校准的 Dynamic Entropy Calibration (Confidence vs. Hallucination Probability)。
- 在 Section 3.1 增加了 2-layer MLP 的 fallback ablation，以防 single linear layer 欠拟合。
- 明确解释了 Threshold-Gating 选择 median 而不是 mean + $1\sigma$ 的统计原因（对稀疏/密集场景更鲁棒）。
- 承认了 VASM 在不完整解码上下文中的一词多义 (Polysemy) 问题，并将其明确定位为 brute-force superset。
- 在实验计划中强制加入了 Batch Size = 1, 4, 8 的延迟与 VRAM 评测，明确了 Out-Of-Candidate (Top-$M$) Unrecoverability 的绝对数学上限追踪。
- 在 Chain C 增加了 "Entanglement Washout" failure case 的强制对照设计。
- **保留的原有亮点**：保持了 2D 空间推理的边界（未恢复视频部分）；保留了 parameter-matched `Base + 5k LoRA` 核心基线并增加 3 random seeds；保留了 DoLa/VCD/OPERA 作为正交工具的 affirmative framing。