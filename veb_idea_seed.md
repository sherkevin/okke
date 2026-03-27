# VEB-QA 初始方法论草案（可替换）

> 如需与对话中的逐字完整稿一致，请用对话全文覆盖本文件。

## 1. 方法总述

### 方法名称
**VEB-QA**: `Visual-Evidence-Bottleneck Question Answering`  
中文可表述为：**视觉证据瓶颈问答范式**

### 核心思想
给定图像 `X`、问题 `Q`、答案 `Y`，传统 MLLM 的主要风险在于存在一条“问题语义/内部知识 -> 答案”的捷径路径，模型可在未充分核验图像的情况下输出一个语言上“像对的”答案。  
VEB-QA 的目标是显式构造一条更强的因果路径：

\[
(X,Q) \rightarrow E \rightarrow \hat Y,
\]

其中 `E` 是从图像中抽取出的**最小充分视觉证据**，并强制答案头只能访问 `E`，不能直接访问问题触发的语言先验变量 `Z`。  
最终文本只是**表达层**，而不是**事实来源**。

---

## 2. 形式化问题定义

### 2.1 随机变量
- `X, Q, Y, Z, E^*, \hat Y` 如主文定义。
- 假设 `E^*=g^*(X,Q)`，`Y \sim p^*(Y \mid E^*, Q)`。

### 2.2 幻觉定义
先验最优 `y_P(q)`，证据最优 `y_E(x,q)`，冲突集 `\mathcal{C}`，幻觉事件 `\mathcal{H}` 与风险 `R_{\mathrm{hall}}(f)` 按主文形式化。

---

## 3. 因果诊断模型
双路径：`Q \rightarrow Z \rightarrow \hat Y` 与 `(X,Q)\rightarrow E \rightarrow \hat Y`；要削弱捷径 `Z \rightarrow \hat Y`。

---

## 4. 方法：视觉证据瓶颈 VEB-QA

### 4.1 四模块
视觉编码器、证据选择器 `M=\pi_\theta(T_V,Q)`、`E=M\odot T_V`、受限答案器 `p_\theta(\hat Y\mid E,Q)`、支持性校验与拒答。

### 4.2 三原则
证据先于答案；答案依赖最小证据；支持不足时拒答。

---

## 5. 理论要点（摘要）
- **定理1**：纯可逆模态重编码不改变 Bayes 最优风险。  
- **定理2**：对数线性融合下，先验覆盖视觉当且仅当 `(1-\alpha)\Delta_P \ge \alpha \Delta_E`。  
- **定理3**：在证据瓶颈与理想证据条件下，`R_{\mathrm{hall}}` 可上界到证据抽取误差。  
- **定理4**：拒答 + 校准给出条件错误率上界。

（完整证明与展开见论文正文稿。）

---

## 6. 训练目标（摘要）
\[
\mathcal L
= \mathcal L_{\mathrm{ans}}
+ \lambda_1 \mathcal L_{\mathrm{ev}}
+ \lambda_2 \mathcal L_{\mathrm{cf}}
+ \lambda_3 \mathcal L_{\mathrm{sp}}
+ \lambda_4 \mathcal L_{\mathrm{ver}}
+ \lambda_5 \mathcal L_{\mathrm{ind}}.
\]

---

## 7. 算法 1：推理流程
编码 → 掩码 → 证据 → 答案分布 → argmax → 支持度 → 拒答或输出。

---

## 8. 实验方法论（摘要）
假设 H1–H4；模型 M0–M5；数据分层 A–E；指标含 HallRate、PCER、证据对齐、拒答质量、反事实敏感性；统计：CI、bootstrap、McNemar、多种子、多重比较校正。

---

## 9. 主结论表述建议
强调“幻觉来自缺乏视觉证据硬约束”，而非“文本输出本身”；贡献分理论/方法/实验三条。

---

## 10. 贡献点（三条）
理论因果框架；VEB-QA 方法；prior-conflict 评测协议与实证。

---

## 11. 一句话总结
幻觉的关键在于答案是否可绕过视觉证据由先验决定；以证据为唯一事实入口，瓶颈阻断 shortcut，校验与拒答控制不确定性。
