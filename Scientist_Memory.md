# Scientist_Memory

- **当前正式主线:** `TLRA_univ` 已取代“把 `Phi_calib` 强写成通用插件”的旧方向。论文现在应围绕 strict `train once, run anywhere` 展开，主学习模块是 `Psi_univ`，而不是 `Phi_calib`。
- **最关键的理论纠偏:** universality 不能建立在 model-native hidden states 或 `lm_head.weight` 上；只要学习模块读写的是模型原生几何，它就不是严格热插拔插件。
- **旧路径的新身份:** `TLRA_zero / TLRA_calib / Phi_calib` 保留，但只作为 `TLRA_internal` control，用来展示 internal ceiling，而不是 universality headline。
- **下一轮最需要继续盯紧的问题:**
  1. **Strict universality discipline:** 绝不允许为了“跑得更好”偷偷引入 per-model learned adapter，否则整条新主线失真。
  2. **Tokenizer bridge honesty:** `TLRA_univ` 可以接受 tokenizer-dependent 但 parameter-free 的桥接；不能把桥接逻辑写成隐性学习模块。
  3. **Portability vs ceiling tradeoff:** 若 `TLRA_internal_calib` 更强，而 `TLRA_univ` 更通用，论文必须正面呈现这一 tradeoff，而不是回避。
  4. **String-side structure protection:** 新的 string-side gate 必须承担原来 vocab-side VASM 的结构保护职责，否则通用性换来了结构崩塌。
  5. **Failure-boundary explicitness:** `Top-M` ceiling、prefix ambiguity、span collapse、abstention calibration 都应作为正式失败边界写入 benchmark，而不是只在实现里提及。