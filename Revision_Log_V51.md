# Revision_Log_V51
- **修复 Fallback 机制缺陷:** 针对审稿人指出 Hard Fallback Step-function 可能导致长文本语义断裂的问题，移除了粗暴的 $\epsilon$ 截断，替换为基于滑动窗口 EMA 的 Rolling Decay 机制 (Section 3.2)。
- **强化 Compositional Coherence:** 针对 Visual Genome 标注不全导致的重叠物体被 InfoNCE 错误排斥的风险，增加了基于冻结 CLIP 文本特征的 Semantic Similarity Secondary Check (Section 3.1)。
- **明确 LoRA 控制组设定:** 在实验协议中明确声明 `Base + 5k LoRA` 采用标准自回归 Next-token Prediction 训练，彻底隔绝参数化知识注入与 Decode-time Routing 的混淆 (Defense Line 1)。
- **维度匹配说明:** 明确 $\Phi_{calib}$ 是 $\mathbb{R}^{D_{model}} \rightarrow \mathbb{R}^{D_{model}}$ 的映射，作用于 LLM 侧的 contextualized 视觉 token，解决审稿人提出的维度匹配疑惑。
- **落实评测架构:** 按要求统一更名为 `BRA_zero` / `BRA_calib` / VASM，并将后续实验计划严格结构化为“三大证据链 (Chain A-C)”与“五道防线 (Defense Lines 1-5)”。同时在 Appendix 计划中加入了审稿人强烈建议的 Qualitative Heatmap 和 BPE Logit Trajectory 映射。
- **保留的原有亮点:** 严格遵守 2D Image-LLM 边界不外扩；保留 DoLa/VCD 作为 Orthogonal Global Regularizers 的公平定位；保留 10% Inaction Boundary 机制。
- **未来待验证点:** 实际执行时需重点观察 EMA 的 Decay Rate $\tau_{target}$ 是否需要按特定模型 (如 Qwen 动态分辨率) 进一步调参，以及 BPE Bypass 对 DocVQA 真实分数的打击到底有多深。