# Revision_Log_V40

- `V40` 是新的正式 latest iteration base，不沿用旧自动链路的 `V38 / V39` 作为新起点。
- `V40` 由 `论文大纲_标杆V1.md` 直接复制生成，用于从新的 benchmark 体系重新开始 reviewer/scientist 收敛迭代。
- 本轮正式收敛动作包括：
  - benchmark 与迭代版分离：benchmark 改用 `标杆VX`
  - 方法命名统一为 `TLRA`
  - 主线收缩为 image-only
  - `TLRA_zero` 被定位为 Stage 0 viability probe
  - 后续自动迭代只应围绕 `标杆V1 -> V40 -> V41 ...` 继续
- 旧链路自动产生的 `V38 / V39` 保留为历史文件，但不再作为新的正式 latest iteration base。
# Revision_Log_V40
- **修复 Washout Threshold 的数学黑盒**: 在 Sec 3.1 补充了如何通过 Logits Reshaping、Min-Max 归一化和 Otsu's 算法对 Heatmap 进行二值化来提取 2D Zero-shot 预测框，让 IoU 的计算有了严谨的数学定义。
- **化解 Pipeline Bloat 和计算复杂度危机**: 针对 Area Chair 对“ moving median” 和字典查询的担忧，在方法和实验区明确补充了 Tensor 操作细节（GPU-native `torch.kthvalue` 避免 CPU 同步）以及严格的 Big-O 复杂度（$\mathcal{O}(N_v \log k)$ top-k 排序和 $\mathcal{O}(|\mathcal{C}_t|)$ 的 bitmask）。
- **实验大纲完全对齐“Strict Syllabus”**: 在 Table 1 中加入了强迫症级别的逐层 Ablation (`Base -> MeanPool -> Top-k -> VASM -> Entropy`)；Figure 1 增加 OOV rate 与 Bypass rate 的明细；在 Figure 2 强制规定了展示 VASM 多义词误伤的失败案例。
- **保留原稿核心亮点**: 继续强调 2D 场景限定的 defensible scope、正交基线视角（VCD/OPERA 不等于 Global Pooling），以及强悍的 `Base + 5k LoRA` 消融基线设计。
- **防御态度**: 在 Limitation 明确声明，如果真实实验跑出来，某一层 heuristic 的增益极小，作者将主动挥刀自宫（prune the mechanism），避免叠甲陷阱。