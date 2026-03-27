# Revision_Log_V88
- **修正“因果注意力悖论” (Causal Attention Fallacy)**：完全接受审稿人的严厉批评。在 Abstract、Intro 和 Method 3.2 中，将视觉特征明确更正为“Static Prefill Visual States”（来自 prefill 阶段冻结的 KV cache），承认解码过程中视觉特征不随文本动态更新，并说明 `Phi_calib` 正是用来弥合这种“静态视觉 vs 动态文本”差距的关键。
- **重构 VASM BPE 继承机制**：针对审稿人指出的“子词污染”（Subword Bleeding）问题，在 3.4 节引入了“保守冲突消解”（Conservative Subword Conflict Resolution），即当一个 BPE token 同时属于实体和抽象词时，强制置 0（受保护），优先保证不破坏语法。
- **细化实验与硬件审计**：在 Table 1 和 Table 3 的指标中强制加入 `Peak VRAM`，回应审稿人对 $O(M \times N_v)$ 密集点积内存带宽（Memory Bound）的担忧。
- **明确失败案例承诺**：在 4.5 节和 5 节的 limitation 中，主动加入了审稿人要求的 “Out-of-candidate Hijacking”（越界劫持）失败案例收集计划。
- **降级 `TLRA_zero`**：在 3.1 和 4.1 节将其重命名为 Architectural Preliminary，并定性为“预期失败的数学探针”，节省篇幅。