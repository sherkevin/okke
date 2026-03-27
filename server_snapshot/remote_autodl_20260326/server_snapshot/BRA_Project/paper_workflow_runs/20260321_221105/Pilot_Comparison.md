# Pilot Comparison

## Workflow Settings
- Model: `[L]gemini-3.1-pro-preview`
- Pilot scope: `V1 -> Review_V1 -> V2 -> Review_V2`
- Goal: validate that the adversarial workflow can improve paper quality without discarding the core idea

## Score Tracking
- `V1`: 2.0/5
- `V2`: 2.0/5
- Delta: +0.0

## Review V1 Snapshot
# Review_Strict_V1
## Overall Score
Score: 2/5

## Verdict
本文提出了一种极具潜力的无池化（Zero-Pooling）多模态幻觉干预框架，巧妙地指出了当前表征工程在面对高分辨率现代 MLLMs 时遭遇的“池化悖论”。然而，当前草稿在核心数学表达上存在严重漏洞（特征空间错位），对现代大模型 BPE 切词机制的物理兼容性思考不足，且行文带有强烈的过度主观和不严谨的攻击性色彩。鉴于实验部分目前全为占位符，本文距离达到 ACM Multimedia 的接收标准仍有极大距离。但我认可其核心洞见的价值，提供了详尽的实验与方法补强路线，若能完美执行，具备冲击顶会的潜力。

## Summary
本文旨在解决多模态大语言模型（MLLMs）在生成过程中因语言先验主导而产生的幻觉问题。作者严厉批判了现有基于隐状态减法范式导致的“池化悖论”（丢失 2D-RoPE 高分辨率空间信息）和高昂的解码期算力开销。为此，文章提出双向共振锚定（BRA）架构，将干预操作后置于大模型的 Logits 解码空间。利用解嵌空间（Unembedding Space）中功能词与视觉实体词的拓扑正交性，BRA 通过零样本的视觉-词汇最大池化匹配机制，在全分辨率视觉特征上动态重塑 Logits 概率。此外，文章还提出了针对视频模态的时序差异特征池（$Z_\Delta$）以抑制动作幻觉。

## WhatShouldBeKept
1. **“池化悖论 (Pooling Paradox)”的理论批判**：这是本文最精彩、最有价值的洞见。明确指出 MeanPool/Token Drop 会摧毁 M-RoPE 和 AnyRes 架构中的细粒度坐标系，逻辑十分自洽。
2. **转移至 Logits 空间的干预思路**：放弃在 Transformer 内部隐状态做向量减法，转而在最后一层利用 $W_{vocab}$ 的拓扑特性进行干预，极其轻量且工程友好。
3. **视频时序差异池 ($Z_\Delta$) 的设计**：直接对对齐的空间网格做帧间差分来捕捉动态（Action Semantic），简单但具有很强的物理合理性。
4. **评测基准的选择 (FREAK, VisualWebBench, VIDHALLUC)**：跳出了仅在 POPE 上刷榜的舒适区，敢于在极需高分辨率的 GUI Agent 场景和长视频动态场景中验证“零池化”的优势。

## MajorWeaknesses

**1. 标题与正文核心机制存在致命矛盾 (Fatal Disconnect)**
论文标题为“...via Adaptive Orthogonal Subspace Projection (A-OSP)”，但在正文 Section 3 中，你明确提出了“双向共振锚定 (BRA)”，且在多处（包括附录）强调“剔除了所有与 SVD 提取、特征隐状态减法...相关的过时理论”。作为一篇顶会论文，标题与方法论脱节是不可饶恕的失误。请立即修改标题以匹配你的 BRA 核心方法。

**2. 方法论深层漏洞：特征空间的严重错位 (Space Mismatch)**
在公式 $S_{res}(c) = \max \left( \cos(w_c, \bar{z}_j) \right)$ 中，存在一个致命的数学未经定义问题。$w_c$ 是 Unembedding 矩阵中的向量（属于最终的 Logits 映射空间），而 $\bar{z}_j$ 被描述为“跨模态层归一化对齐后的深层视觉特征”。
在现代 MLLMs（如 Qwen-VL）中，视觉编码器的输出（Visual Features）与语言模型的最后一层输出（Logits 空间）相隔了数十层 Transformer Block。你**绝对不能**直接计算这两个异构空间向量的余弦相似度。
*   **质疑：** 这里的 $\bar{z}_j$ 究竟是什么？是视觉 Token 刚输入 LLM 时的 Embedding？还是在 LLM 最后一层（Last Hidden State）对应的视觉 Token 位置的隐状态？如果是后者，你需要明确说明“提取第 L 层对应于视觉 Token 序列的隐状态”；如果是前者，这两个空间未经对齐，内积毫无意义。

**3. BPE 碎片化导致的方法论崩溃 (The BPE Collapse)**
你在附录 E 中诚实地指出了 BPE 碎片化的问题（如 "Tele" + "communication"）。但在我看来，这不仅仅是一个 "Failure Case"，而是**直接击穿 BRA 核心机制的系统性漏洞**。
现代大模型的绝大多数长单词或罕见实体都是由无意义的 sub-words 组成的（如 "Chihuahua" 被切分为 "Chi", "hua", "hua"）。当模型尝试生成 "Chi" 时，其对应的词向量 $w_{Chi}$ 在视觉空间中找不到任何共振（$S_{res} \approx 0$），随后你的算法会用 $-\alpha |L_{orig}|$ 严厉惩罚它。这会导致模型**永远无法生成由多 Token 组成的长视觉实体**，直接造成词汇表坍缩。不能仅在附录中一笔带过，必须在方法（Section 3）中给出缓解策略（例如：滑动窗口动量、基于前缀树的探测、或者降低对非完整词汇的惩罚权重）。

**4. 触发机制鲁棒性存疑**
干预触发条件 $\Delta E_t = |E_t - E_{t-1}| > \epsilon$（香农熵增量）。如果模型在产生幻觉时展现出“盲目的自信”（例如受到极强的语言偏置引导，输出分布极其尖锐），此时 $\Delta E_t$ 可能非常小，导致你的干预机制根本无法触发。你没有任何论据支撑“认知不确定性必然伴随幻觉发生”这一强假设。

**5. 极度不专业的行文基调 (Unprofessional Tone)**
论文中充斥着“无

## Revision Log V2 Snapshot
# Revision_Log_V2
- **Title Fixed**: Replaced "A-OSP" with "Bi-directional Resonance Anchoring (BRA)" to resolve the fatal disconnect pointed out by the reviewer.
- **Space Mismatch Resolved**: Explicitly redefined $\bar{z}_j$ as $h_L^{(v_j)}$ (the last hidden state corresponding to visual tokens just before unembedding) to ensure cosine similarity is mathematically valid in the same latent space.
- **BPE Collapse Fixed**: Introduced the "Sub-word Momentum Integration (SMI)" strategy to pass resonance scores to sub-word fragments, directly addressing the reviewer's most critical methodology critique.
- **Equation Corrected**: Changed the logit modification from absolute value multiplier ($-\alpha |L_{orig}|$) to a translationally invariant additive shift ($+\alpha S_{res} - \beta(1-S_{res})$) to avoid flipping negative logits.
- **Trigger Mechanism Adjusted**: Removed the hard entropy threshold ($\Delta E_t > \epsilon$) which fails on "blindly confident" hallucinations. Replaced with continuous application relying on the improved equation and topological orthogonality.
- **Tone Purified**: Removed all aggressive and subjective rhetoric ("无脑", "死结", "降维打击"). Replaced with objective academic terminology.
- **Experiment Redesign**: Converted all "placeholder" absolute claims into a rigorous "Evaluation Protocol / Hypothesis" format, directly mapping to the reviewer's suggested 5 tables/figures.

**Retained Highlights:**
- The theoretical critique of the "Pooling Paradox" (MeanPool destroying 2D-RoPE).
- The core intuition of Unembedding Space Topological Orthogonality.
- The $Z_\Delta$ design for video action hallucination.

**Requires Future Experimental Validation:**
- The actual latency profiling (ITL/VRAM) on RTX 5090.
- Running VisualWebBench to prove the zero-pooling supremacy vs. RepE MeanPool.
- Density distribution plotting (KDE) of $S_{res}$ to physically prove topological orthogonality.

## Review V2 Snapshot
# Review_Strict_V2

## Overall Score
Score: 2/5

## Verdict
Reject (Major Revision Required). The manuscript proposes an intriguing, zero-pooling logits-space intervention to mitigate MLLM hallucinations. However, the core methodology relies on a mathematically perilous assumption regarding the modality gap in the unembedding space, and the paper currently lacks completely executed experiments. The experimental design is ambitious and well-structured, but the method needs rigorous justification and the protocols must be physically executed before acceptance.

## Summary
The paper introduces Bi-directional Resonance Anchoring (BRA) to address MLLM hallucinations. Critiquing the "Pooling Paradox" of hidden-state interventions that destroy high-resolution 2D-RoPE coordinates, BRA shifts the intervention to the logits space. It uses a zero-shot max-pooling match between the text vocabulary unembedding matrix and the last-layer visual hidden states. It also proposes Sub-word Momentum Integration (SMI) for BPE issues and a Temporal-Difference Pool ($Z_\Delta$) for video tasks. Currently, the paper outlines a comprehensive five-stage evaluation protocol, but empirical results are strictly theoretical/planned.

## WhatShouldBeKept
1. **The "Pooling Paradox" Concept:** The critique of traditional MeanPool/PCA interventions destroying 2D-RoPE coordinates is highly insightful and a strong motivation for high-resolution MLLMs.
2. **Sub-word Momentum Integration (SMI):** The identification of BPE fragmentation causing semantic collapse in logits-based interventions is astute. SMI is a pragmatic and necessary engineering solution.
3. **The Evaluation Protocols:** The proposed 5-stage benchmark matrix (Protocols 1-5) is exceptionally well-designed. Testing across static hallucination, high-res GUI navigation, video temporal dynamics, and system overhead forms a complete proof-of-concept.
4. **Failure Case Analysis (Appendix E):** The explicit acknowledgement of abstract logic tension and BPE limitations demonstrates scholarly maturity.

## MajorWeaknesses

1. **The Modality Gap Fallacy in Unembedding Space (Critical Methodological Flaw):** 
In Section 3.2, you define $S_{res}(c) = \max \cos(w_c, h_L^{(v_j)})$, directly taking the dot product between the text unembedding matrix $W_{vocab}$ and the *last-layer visual tokens*. This assumes that visual tokens and text tokens reside in the exact same semantic manifold at Layer $L$. While MLLMs align modalities at the *input*

## Preliminary Judgment
- If `V2` improves the score or shifts criticism from fatal flaws to actionable weaknesses, the workflow is functioning.
- If the score does not improve, inspect whether the scientist over-rewrote the draft or failed to address the harshest reviewer concerns.
- Use the next rounds to focus on claim calibration, experimental closure, fairer related-work positioning, and clearer contribution boundaries.
