# Revision_Log_V50
- 针对 `Local_Zero` Strawman 批评：在摘要、引言和方法论中大幅压缩了 `Local_Zero` 的叙事比重，明确承认它主要作为一个必败的“诊断指标（diagnostic）”，核心算法逻辑实质上已经承认 `Local_Calib` 是结构性必须（structurally required）。
- 针对 InfoNCE 组合盲区 (Compositional Blindspot)：在 Section 3.1 引入了 "Compositional Coherence Margin"，利用 Visual Genome 场景图的 parent-child 关系否决破坏性负样本（如 rider/horse），防止组合关系被摧毁。
- 针对 阈值衰减回退 (Threshold Attrition Fallback)：在 Section 3.2 增加了一个硬性的操作级回退机制。如果有效激活率连续3个 token 跌破 $\epsilon$ 临界值，直接降级到 `Local_MeanPool`，确保模型在长文本生成中不会陷入彻底休眠。
- 针对 实验协议漏洞：在 Table 1 增加 `Intervention ITL Overhead (ms/token)`；在 Figure 1 的散点图上设定了硬性的 <10% 触发率（inaction boundary）；在 Table 2 (FREAK/DocVQA) 强制加入了 `Base + 5k LoRA` baseline；将 Figure 3 的横坐标修改为严格的 Generation Step $t$。
- 针对 Appendix：在 Appendix D 增加了第三类失败案例 "Compositional Destruction"。
- 保留的原有亮点：严格的 2D 场景限定、对 DoLa/VCD 正确的“全局正则化器”正交定位、以及严苛的 `Base + 5k LoRA` 证伪边界。