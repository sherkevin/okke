# 标杆V1 资源审计
> 日期：2026-03-21
> 依据文档：`实验方案_标杆V1.md`
> 审计目标：检查当前 **所有需要的数据和模型** 是否已经下载齐备，可否直接支持 `标杆V1` 实验合同。

## 结论

**结论不是“全部齐了”。**

当前状态可以分成三层：

1. **P0 / 当前主线基础资源大体已齐**
   - `Qwen3-VL-8B-Instruct`
   - `Qwen3-VL-2B-Instruct`
   - `COCO val2014 images`
   - `COCO annotations`
   - `POPE`
   - `CHAIR` 所需的 `instances_val2014.json`

2. **P1 / 扩展 benchmark 数据大部分已存在，但有命名和版本确认问题**
   - `MMBench_EN_hf`：存在，结构看起来可用
   - `MME_hf`：存在，结构看起来可用
   - `MMMU_hf`：存在，且是完整学科目录，不是单独的 `Hard` 子集
   - `FREAK_hf`：存在，结构看起来可用

3. **仍然缺失或尚未确认的关键资产**
   - `DocVQA`：**未发现**
   - `Base + 5k LoRA`：**未发现明确 LoRA / adapter checkpoint**
   - `MMMU (Hard Subset)`：**未发现现成 hard 子集，仅有 full MMMU**
   - `TLRA_calib` 明确注册权重：**不够明确**
     - 发现了 `V_matrix.pt`, `V_matrix_q3.pt`, `V_matrix_q3_mini.pt`
     - 但从命名和注册信息上，**还不能严谨地把它们直接判定为论文主线的 `Phi_calib` 正式资产**

所以，如果问题是：

- **“现在能不能继续推进 P0 主线（Stage 0 + POPE/CHAIR）？”**
  - 可以。
- **“现在能不能直接无阻碍完成整个 标杆V1 合同？”**
  - 不能，至少还差 `DocVQA`、`5k LoRA`，以及 `MMMU Hard` 的明确子集定义/准备。

---

## 对照 `实验方案_标杆V1.md` 的逐项核查

### 一、模型

方案里直接或隐含依赖的主模型资产：

- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-2B-Instruct`（开发/对照常用）
- `Base + 5k LoRA`
- `TLRA_calib` 所需轻量校准权重（若走 calib 分支）

当前远程状态：

| 资产 | 状态 | 说明 |
|------|------|------|
| `Qwen3-VL-8B-Instruct` | 已有 | 4 个 safetensors shard，结构完整 |
| `Qwen3-VL-2B-Instruct` | 已有 | 单文件 safetensors，结构完整 |
| `Qwen3-VL-4B-Instruct` | 已有 | 方案里未强制要求，但远程上存在 |
| `Base + 5k LoRA` | 缺失 | 未发现 `adapter_config.json` / `adapter_model.*` / 独立 LoRA 目录 |
| `TLRA_calib` 显式权重 | 未确认 | 存在 `V_matrix*.pt`，但未在注册表中被清楚标记为正式校准权重 |

### 二、Stage 0：Zero-Shot Semantic Validity Pilot

方案要求：

- 带 object annotation 的 held-out image subset
- 统计 patch lexical overlap

当前远程状态：

| 资产 | 状态 | 说明 |
|------|------|------|
| `COCO val2014 images` | 已有 | 40,504 张图片 |
| `instances_val2014.json` | 已有 | `CHAIR`/object annotation 足够支持 Stage 0 |
| 预制 held-out subset | 未发现 | 但这不是必须下载资产，可以运行时切分 |

判断：

- **Stage 0 所需原始数据已经够用**
- **缺的是切分逻辑，不是下载资产**

### 三、证据链 A：Hallucination Reduction

方案要求：

- `POPE`
- `CHAIR`

当前远程状态：

| 数据集 | 状态 | 说明 |
|--------|------|------|
| `POPE` | 已有 | 目录存在，文件数与先前评测一致 |
| `CHAIR` | 已有 | 依赖 `COCO + instances_val2014.json`，均存在 |

判断：

- **这一条已经就绪**

### 四、证据链 B：Structure and Reasoning Preservation

方案要求：

- `MMBench V1.1`
- `MME`
- `MMMU (Hard Subset)`

当前远程状态：

| 数据集 | 状态 | 说明 |
|--------|------|------|
| `MMBench_EN_hf` | 已有但需版本确认 | README 显示 dev/test split，可用；未直接写明 “V1.1” |
| `MME_hf` | 已有 | README 显示 test split，2374 样本 |
| `MMMU_hf` | 已有 full dataset | 学科目录完整，但未见现成 `Hard` subset |

判断：

- `MMBench`：**大概率可用，但最好在真正跑表前确认是否对应 V1.1 版本**
- `MME`：**已就绪**
- `MMMU Hard`：**原始全集在，但 Hard 子集未单独准备**

### 五、证据链 C：Local Evidence Value

方案要求：

- `FREAK`
- `DocVQA`

当前远程状态：

| 数据集 | 状态 | 说明 |
|--------|------|------|
| `FREAK_hf` | 已有 | README 显示 test split，1799 样本 |
| `DocVQA` | 缺失 | 在 `datasets/` 下未发现 `DocVQA`，全项目搜索也未发现对应数据目录 |

判断：

- **FREAK 已就绪**
- **DocVQA 仍未下载**

### 六、公平性增强矩阵

方案要求：

- `Base + 5k LoRA`

当前远程状态：

- 未发现明确 LoRA / adapter checkpoint
- 搜索结果只命中了 `vllm` 第三方库里的测试目录，不是项目资产

判断：

- **这一项未就绪**

---

## 远程审计证据摘要

### 已确认存在的关键路径

```text
/root/autodl-tmp/BRA_Project/models/Qwen3-VL-8B-Instruct
/root/autodl-tmp/BRA_Project/models/Qwen3-VL-2B-Instruct
/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014
/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations
/root/autodl-tmp/BRA_Project/datasets/POPE
/root/autodl-tmp/BRA_Project/datasets/MMBench_EN_hf
/root/autodl-tmp/BRA_Project/datasets/MME_hf
/root/autodl-tmp/BRA_Project/datasets/MMMU_hf
/root/autodl-tmp/BRA_Project/datasets/FREAK_hf
```

### 已确认缺失的关键路径

```text
/root/autodl-tmp/BRA_Project/datasets/DocVQA
/root/autodl-tmp/BRA_Project/lora
/root/autodl-tmp/BRA_Project/outputs
/root/autodl-tmp/BRA_Project/datasets/MMMU_Hard   (未发现现成 hard 子集)
```

### 发现但未正式确认用途的权重

```text
/root/autodl-tmp/BRA_Project/models/V_matrix.pt
/root/autodl-tmp/BRA_Project/models/V_matrix_q3.pt
/root/autodl-tmp/BRA_Project/models/V_matrix_q3_mini.pt
```

这些文件**可能**与 `TLRA_calib` / 历史 BRA 校准矩阵有关，但在当前共享注册与命名体系里并未被清楚声明，因此**不能直接算作“方案所需模型已确认齐备”**。

---

## 最终判断

### 已经够跑的部分

- `Stage 0`
- `POPE`
- `CHAIR`
- `MME`
- `FREAK`
- `MMBench`（大概率可直接跑，但建议先确认版本）
- `MMMU` 全集（但不是 hard 子集）

### 还不算齐的部分

- `DocVQA`
- `Base + 5k LoRA`
- `MMMU Hard Subset`
- `TLRA_calib` 明确注册权重身份

---

## 推荐动作

按优先级：

1. **先补 `DocVQA`**
   - 这是 `标杆V1` 主合同里明确点名的数据集，当前是真缺。

2. **确认 `MMBench_EN_hf` 是否就是 `MMBench V1.1`**
   - 如果不是，要补正确版本。

3. **为 `MMMU` 准备 `Hard` 子集过滤脚本或静态清单**
   - 目前只有 full dataset。

4. **确认 `V_matrix*.pt` 是否映射到 `TLRA_calib`**
   - 若是，需要补注册表说明和统一命名。

5. **补 `Base + 5k LoRA`**
   - 若 benchmark 真的要做公平性增强矩阵，这项不能继续空着。
