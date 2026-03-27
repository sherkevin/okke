# Review_Strict_V87
## Overall Score
Score: 3/5

## Verdict
这篇论文的“预注册/实验契约”式写作极其罕见且令人耳目一新。作者展现了极高的科学诚实度，主动设立了极具杀伤力的负控制（OCR测试）和基线（同数据 LoRA）。然而，作为首席主审稿人，我必须指出：尽管你们的实验防线固若金汤，但你们的**核心方法论中存在一个致命的常识性事实错误**（Causal Attention Fallacy），这将直接导致你们的动态路由设想在现代主流 MLLM 上破产或严重降速。如果能修正这一理论谬误并按计划完成实验，这将是一篇顶会级别的坚实工作。目前暂给 3 分（大修），以观后效。

## Summary
本文提出了一种名为 TLRA（Token-Local Resonance Anchoring）的混合路由干预框架，旨在缓解多模态大语言模型（MLLM）在物理实体对齐时的幻觉。TLRA 在解码时计算候选词与局部视觉 token 的匹配度，并通过动态 Top-k 机制调整 Logits。为了防止破坏语法和长实体，引入了离线预计算的语义掩码（VASM）。本文将重点放在了实验协议的严谨性上，主动切分了 Zero-shot 和 Calibrated 两个变体，并强制要求与使用相同校准数据微调的 LoRA 基线进行正面对决。

## WhatShouldBeKept
1. **The Evaluation Contract (实验契约精神)**：不提前声称 SOTA，而是设定严格的通过/证伪条件。这种写法必须保留，这是本文最大的亮点。
2. **Mandatory LoRA Baseline (强制同数据基线)**：引入 `Phi_calib` 投影器后主动放弃“免训练”标签，并强制对比 `Base + LoRA (Phi_calib data)`。这是极具审稿人视角的防御，完美堵死了“性能提升来源于数据而非路由机制”的质疑。
3. **OCR Negative Control (OCR负控制)**：主动承认 VASM 对任意文本覆盖率为零，将 DocVQA 作为负控制。
4. **Offline VASM (离线词表掩码)**：使用离线 WordNet+BPE 继承来避免在线 POS tagging 的灾难性延迟，这是一个非常务实且优雅的系统工程设计。

## MajorWeaknesses
1. **致命的事实性偏差：因果注意力悖论 (The Causal Attention Fallacy)** 
在第 3.2 节中，你们声称：“...these states are already heavily contextualized by the generated text prefix through autoregressive self-attention, meaning the visual representation adapts as the sentence unfolds.” 
**这是完全错误的。** 在现代主流的 Decoder-only MLLM（如 LLaVA, Qwen-VL）中，视觉 token 位于序列前缀（Prefill 阶段）。由于单向因果掩码（Causal Mask）的存在，视觉 token 只能注意到自身及之前的 token，**绝对不可能** attend 到后续不断生成的文本 token。因此，在开启 KV Cache 的情况下，所有视觉 token 的最后一层隐藏状态在整个生成过程中是**静态且冻结**的，根本不会 "adapts as the sentence unfolds"。
如果你们真的让视觉表示随生成动态更新，这意味着你们要么禁用了 KV Cache（导致 $O(N^2)$ 的灾难性推理延迟），要么使用了双向注意力的 Encoder-Decoder 架构（需明确限定模型范围）。

2. **`TLRA_zero` 探针的虚假人设 (The Strawman in Stage 0)**
第 3.1 节提出的 `TLRA_zero` 几乎注定失败。在绝大多数 MLLM 中，视觉编码器的输出经过 MLP 投影后，只是作为 LLM 输入层的 Embedding。在经过 32 层或更多的 Transformer 层后，最后一层输出的视觉 token state 所在的特征空间，与 LLM 词表分类头（LM Head）所期望的文本语义空间存在巨大的漂移（Embedding Asymmetry）。将其直接与词表进行点积计算 `TLRA_zero`，不仅是“可能失败”，而是“必然乱码”。这个 Stage 0 作为实验显得过于充数，不如直接一笔带过，将核心算力留给 `TLRA_calib`。

3. **延迟建模的过度乐观 ($O(M \times N_v)$ 的真实代价)**
在 3.2 节中，你们承认了 $O(M \times H \times W)$ 的复杂度。但在自回归解码的内部循环（Auto-regressive loop）中，不仅是 FLOPs 的问题。由于候选窗口 $M$ 的隐藏状态 $c$ 需要与数千个视觉 token $N_v$ 的隐藏状态做密集点积，这在 GPU 上的 Memory Bound（内存带宽受限）极高。如果在 Python 端（如 HuggingFace `LogitsProcessor`）实现，框架开销会让 `tokens_per_second` 断崖式下跌。单纯计算 FLOPs 掩盖了真实的工程瓶颈。

4. **VASM BPE 继承的隐患 (Subword Bleeding)**
离线 VASM 的 BPE 继承机制（"refriger" $\to$ "ator"）在确定性上没问题，但存在“子词污染”（Subword Bleeding）风险。例如，"ator" 这个 BPE token 也可能是 "creator" 或 "senator" 的一部分。如果全局粗暴地将 "ator" 的 $M_{VASM}$ 置为 1（或 0），会误伤其他毫不相关的词。你们目前的描述没有明确如何处理同一 subword 在不同前缀下的多态性。

## SectionBySectionComments
*   **Abstract & Intro**: 逻辑极其清晰。但对 VCD, OPERA, DoLa 等 baseline 的定性有些宽泛。请明确指出这些 baseline 属于“语言侧启发式”或“全局视觉噪声”，以凸显 TLRA “局部视觉干预”的差异化生态位。
*   **Method 3.2**: 必须立即重写！删除或修正关于“视觉表示随生成展开而适应”的描述。你必须说明你是使用 Prefill 阶段冻结的视觉 KV 状态，还是在每次解码时引入了某种跨层或逆向计算（如果是后者，立刻给出复杂度惩罚）。
*   **Method 3.4**: 需要增加一小段讨论：当一个 BPE token 属于多个不同词根的后缀时（例如复数 "-s" 或常见词缀 "-ing"），VASM 的离线构建如何进行冲突消解（Conflict Resolution）？如果强制取并集或交集，请在失败边界审计（4.5）中增加“BPE冲突导致的过度干预”。

## RequiredRevisions
1. **修正因果掩码事实错误**：在 3.2 节承认视觉 token 状态是静态的（对于 Decoder-only 模型），或者澄清你们使用的是允许双向注意力的特殊架构。这是必须修改的硬约束。
2. **细化 `Phi_calib` 的技术细节**：因为 `TLRA_calib` 依赖此模块，你需要明确它是一个简单的线性层、两层 MLP，还是类似 Q-Former 的结构。它的参数量必须极小，否则它就变成了一个全新的大模型组件，破坏了“干预”的轻量化定义。
3. **BPE 冲突消解策略**：在 VASM 章节补充说明如何处理多个词根共享同一 Subword 的情况。
4. **重新定义 Stage 0**：如果 `TLRA_zero` 只是为了证明不可行，不要在主干实验中给它太多篇幅，将其降级为附录中的预备实验（Preliminary），或者直接在正文中用数学/架构原理说明其不可行性。

## SuggestedFiguresTablesExperiments
1. **增加 Figure 2: The Attention/State Lifecycle**：鉴于 3.2 节存在的混乱，强烈建议增加一个时序图，明确表示 $h_L^{(v_j)}$ 是在哪个阶段提取的，以及它与生成的文本 token 之间的数据流。
2. **Table 1 必须补充的列**：除了 ITL (Inter-Token Latency) 和 tokens/s，必须加上 `Peak VRAM (GB)`。因为如果你的 `Phi_calib` 或密集点积需要缓存大量的中间状态，显存爆炸会击穿你的方法实用性。
3. **实验计划补充 (Calibration Scale Ablation)**：既然使用了同数据 LoRA 进行对比，建议在附录中增加一个简单的曲线：随着 `Phi_calib` 校准数据量的增加（例如 10k, 50k, 100k 条图文对），TLRA 相对于 `Base + LoRA` 的胜率变化。这能测试你的方法是否在“低资源校准”下才有效，还是能随着数据 scaling 持续保持优势。
4. **Fail Case Analysis 强求**：不要只写 Polysemy（多义词）。我要求你们必须在最终论文里展示一例“因 Top-K 候选词中根本没有正确实体，导致 TLRA 强行把权重赋给错误视觉对象”的越界失败（Out-of-candidate hijacking）。

## AcceptanceOutlook
**Borderline / Revision Required (3/5)**. 
本文的立意、防御性写作和实验契约设计属于顶会中最优质的那一档（Top 10% in framing）。但我不能放过 3.2 节中关于 MLLM 自回归工作原理的基础事实错误。如果在执行实验前，作者能够修正这一理论假设，承认视觉特征的静态属性或重新设计动态特征提取机制，并如约完成上述所有极其严苛的对比（特别是 Table 1 和 Table 3），这篇论文将转变为一篇具备极强说服力的 Solid Accept (4.5/5) 文章。期待看到完整实验数据。