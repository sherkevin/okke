1. **严重缺乏创新性 (Marginal Novelty)**
本方法在本质上只是对已有的对比解码（Contrastive Decoding, CD）和上下文感知解码（Context-Aware Decoding, CAD）的极其微小的缝合。将全模态概率与单文本模态概率进行对比，以抑制语言先验，这是 VCD、CAD 乃至诸多 LVLM 幻觉消除工作的核心基石。本文仅仅是在对数概率差上加了一个 `ReLU` 激活函数（Eq. 4），就包装成一个新的解码策略。这种极其单薄的增量微调根本无法达到 ACM MM 级别会议对创新的要求。

2. **致命的数学与理论错误 (Fatal Mathematical Flaw)**
你在 Eq. 3 中将 $\hat{\ell}_t$ 定义为归一化后的对数概率（$\log P_t$），并在 Eq. 4 中在此基础上进行惩罚计算得到 $\tilde{\ell}_t$。然而在 Eq. 5 中，你居然对 $\tilde{\ell}_t$ 再次应用带有温度参数 $T$ 的 `Softmax` 操作！
`Softmax` 的输入应当是未归一化的原始 logits。对数概率（Log-probabilities）本身已经是经过指数归一化并取对数的结果，其数值分布（通常是绝对值较小的负数）与原始 logits 完全不同。对 Log-probs 再次强行应用 Softmax，会在数学上极其严重地扭曲原有的概率分布，不仅温度参数 $T$ 会完全失效，还会导致长尾词汇的概率被异常放大，极大地破坏生成质量。这是一个入门级的常识性错误。

3. **对 Transformer 架构的根本性误解 (Fundamental Misunderstanding of Architecture)**
在 3.2 节中，你声称“由于全模态和纯文本分支处理的序列长度不同，导致 Softmax 的配分函数（partition functions）不同，从而引起 logits 的全局尺度偏移（global scale shift）”。这完全是胡言乱语。
在现代 Transformer 中，输出层的 logits 是由顶层隐藏状态 $h_t$ 经过线性投影得到的。而 $h_t$ 在投影前必定会经过 LayerNorm 或 RMSNorm 进行标准化。因此，无论上下文序列有多长，隐藏状态的范数和最终 logits 的尺度都会被强制规范化。配分函数是在 `Softmax` 计算**时**产生的，而不是在 logits 产生前导致 logits 偏移的原因。你为了强行合理化自己使用对数概率的做法，凭空捏造了一个不存在的理论依据。

4. **关于推理开销的虚假声明 (Misleading Claims on Efficiency)**
在 3.1 节中，你声称利用 vLLM 的 Prefix Caching 可以“显著摊销内存带宽需求”，使得双分支解码“切实可行”。这是一种极具误导性的避重就轻。
Prefix Caching 仅仅在 Prefill 阶段起作用。在自回归的 Decode 阶段，生成每一个 token 时，模型依然需要对全模态和纯文本两个分支分别进行完整的线性层读取和前向传播。因为 Decode 阶段是严格的 Memory-Bandwidth Bound（受限于模型权重的读取），你的方法在生成阶段依然会实打实地导致 100% 的额外计算延迟和显存带宽开销。用 Prefill 阶段的优化来掩盖 Decode 阶段翻倍的代价，是极不严谨的学术表达。

5. **逻辑悖论与语义破坏 (Semantic Destruction via Blind Penalization)**
在 3.3 节中，你假设 $\Delta_t(w) > 0$（纯文本预测概率高于图文预测概率）就“指示了潜在的语义幻觉”。这种一刀切的假设完全忽略了 LVLM 需要依赖文本先验进行事实性知识回忆（Factual Knowledge Recall）的场景。
如果用户询问图片中某个人物的背景信息，图像只提供了人脸，详细信息必须依赖文本先验。此时文本分支的预测概率必然大于视觉分支，按照你的 Eq. 4，这种正确的事实性词汇会遭到无情的惩罚（被 ReLU 截获并扣除 $\alpha$ 倍的概率）。你的方法在试图消除视觉幻觉的同时，必然会严重摧毁模型的外部知识问答能力（VQA性能大概率会崩塌），而你在方法设计中完全没有考虑如何区分“视觉幻觉”和“合法的事实性文本先验”。

6. **超参数鲁棒性倒退 (Regression in Robustness)**
你声称摒弃全局熵等启发式触发器，改用固定的超参数 $\alpha$ 是一种“实践优势”。实际上，这是学术上的倒退。不同复杂度的 prompt、不同解码阶段对语言先验的依赖程度完全不同，采用静态的 $\alpha$ 极其脆弱，必须针对每个数据集进行繁琐的 grid-search。DoLa 等前沿工作之所以引入动态熵截断，正是为了解决固定惩罚带来的语法崩溃问题，而你却将其退化为手动调参。

**打分 (Score): 1.5 / 5 (Strong Reject)**