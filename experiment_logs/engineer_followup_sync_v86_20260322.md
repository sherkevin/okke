# Engineer Follow-up Sync (V86)

日期：`2026-03-22`

## 当前运行态势

- 工程师1：
  - `GPU1` 上继续承担 `Chain A` 主线长跑
  - `Chain B` 已持续落盘，主任务应继续以“闭合现有结果 + 镜像回本地”为主
  - `TLRA_calib` 仍不得扩跑，直到 `Phi_calib` 身份冻结

- 工程师2：
  - `GPU0`：`VidHalluc` appendix-only pilot
  - `GPU2`：`FREAK parity`
  - `GPU3`：`DocVQA main`
  - `Video-MME smoke` 已从 “Loaded 0 samples” 升级到 “loader/index 可审计，但底层解码栈仍不稳定”

- 工程师3：
  - `P0-1` 到 `P0-6` 已闭合
  - 下一阶段从“补代码入口”转向“补审计脚本、成本拆分、matched baseline feasibility”

## 本轮新增高优先级判断

来自 `Review_Strict_V86` 的新增高价值信号：

1. `TLRA_calib` 与训练自由 baseline 的公平性仍不闭合，需要新增 `Base + LoRA (Phi_calib data)` 作为 matched-budget 基线。
2. 必须把 `VASM` 的在线执行逻辑写清楚，并补成本拆分，不然 reviewer 会继续质疑其真实性与可用性。
3. 必须定义清楚 `h_L^{(v_j)}` 到底是哪一层的 visual state，否则方法定义不稳。

## 调度原则

- `GPU1`：继续固定给工程师1 的 `Chain A`
- `GPU2`：继续固定给工程师2 的 `FREAK parity`
- `GPU3`：继续固定给工程师2 的 `DocVQA`，工程师3 仅允许短时 smoke
- `GPU0`：只允许 appendix pilot 或短任务；若主线出现卡顿，优先让给主表补跑而不是继续消耗在视频上

## 当前最重要的产出顺序

1. 闭合 `Table 1` 与 `Table 2` 的主资产
2. 闭合 `Table 3` 的 `RandomK + Seen/Unseen`
3. 补 `VASM` / latency / leakage 的可审计附录资产
4. 视频只保留为 bounded appendix pilot，不反向抢占主线 GPU

## 最新状态刷新

- 工程师1：
  - `GPU1` 上的 `Chain A` 仍在跑干净版 `POPE base`
  - `Stage 0` 与 `Chain B` 结果已稳妥镜像回本地
  - 当前 `Table 1` 的核心缺口仍然是干净的 `POPE/CHAIR` 主结果、`AGL StdDev`、`peak_vram`

- 工程师2：
  - `GPU0/2/3` 当前都已释放
  - `DocVQA main`、`MMMU Hard`、`FREAK provisional`、`VidHalluc` closeout 已完成
  - `DocVQA` 当前存在两个严重口径问题：
    - 输出缺 `normalized_exact_match` / `anls`
    - `bra_method` 标签塌缩成 `bra_zero`
  - `MMMU Hard` 当前结果不支持强 `tlra_full > tlra_no_vasm` 叙事，且 `intervention_rate = 0.0`

- 工程师3：
  - `Category Leakage Audit`、`suffix-collapse / continuation audit`、`VASM/routing cost split`、`visual_state_provenance`、`LoRA matched-budget` 脚手架均已闭环
  - `VASM` 说明文档已明确：当前是 `offline precomputed vocab lookup + BPE continuation inheritance`，没有在线 POS tagger

## 最新高优先级判断

1. `DocVQA` 必须先修导出协议，再补跑；当前版本不能算正式负控制资产。
2. `MMMU Hard` 已可用于“当前不支持强 VASM 主张”的负面收敛判断，但还应先复核 `intervention_rate=0.0` 是否是方法未触发还是记录口径问题。
3. 现在最缺的主论文资产仍然是 `Table 1` 的干净 `POPE/CHAIR`。
4. `FREAK` 新字段链已打通，但在 projector 未冻结前仍只能保留 provisional 身份。
5. 视频线正式停止，不再分配长跑 GPU。

## 当前 GPU 策略

- `GPU1`：继续固定给工程师1 跑 `Chain A`
- `GPU3`：优先留给工程师2 进行 `DocVQA` 修复后补跑与 `MMMU Hard` 复核
- `GPU2`：留作备用主表卡；若 `Chain A` 卡住，可切过来跑 `CHAIR` 或补 `POPE` 其他方法
- `GPU0`：只给工程师3 做短时 smoke / LoRA 最小验证，不得上长跑

## 最新补充同步

- 工程师1：
  - `GPU1` 上的 clean `Chain A` 仍在运行，当前尚未产出新的 clean `POPE/CHAIR` JSON
  - 已新增 clean 资产登记表：
    - 本地：`experiment_logs/v3_table1_clean_registry_20260322.md`
    - 远端：`logs/v3_engineer_a/v3_table1_clean_registry_20260322.md`
  - 需要特别注意：工程师3已经补齐 `agl_stddev / peak_vram_gb` 输出协议，因此工程师1后续第一份新 clean JSON 必须核验是否来自修复后的代码路径

- 工程师2：
  - 当前 `GPU2/GPU3` 都空闲
  - 之前 `DocVQA` 的阻塞不再是实验设计问题，而是“远端执行版本未切到修复后代码”

- 工程师3：
  - `DocVQA` 正式导出协议 smoke 已通过：
    - `bra_method` 不塌缩
    - `normalized_exact_match` 存在
    - `anls` 存在
  - `MMMU Hard` 的 `intervention_rate=0.0` 已被 smoke 证伪，旧结果更接近记录口径/代码路径问题
  - `Chain A` 输出协议 smoke 已显示：
    - `agl_stddev` 可稳定输出
    - `peak_vram_gb` 可稳定输出

## 最新高优先级判断（覆盖旧判断）

1. `DocVQA` 已升级为 `contract-ready`，不再是 `Table 2` blocker。
2. `MMMU Hard` 修复后路径复核已完成，旧的 `intervention_rate = 0.0` 结论作废；当前主要任务是把新结果正确写入口径，而不是继续怀疑是否完全不触发。
3. `Chain A` 当前 run 继续；第一份 clean `8B` JSON 一旦落盘，仍需先检查 `agl_stddev / peak_vram_gb`。
4. `FREAK` 继续保持 `provisional`，视频线继续停掉。

## 最新结果同步

- 工程师2：
  - `DocVQA` 正式补跑已完成，现可登记为 `contract-ready`
  - `MMMU Hard` 正式复核已完成，`intervention_rate` 非零，旧口径失效

- 工程师3：
  - 已确认远端正式入口与本地修复版 SHA 一致
  - 新增 `validate_export_contract.py` 作为统一验收入口

## 现在的真正主缺口

1. `Table 1` 的 `8B` clean `POPE/CHAIR`
2. projector 冻结后的正式 `FREAK parity`
3. matched-budget `Base + LoRA` 正式结果

## 最新优先级重排

- `Table 2` 已基本脱离 blocker 身份，工程主力不应继续耗在 `DocVQA/MMMU Hard` 上。
- 当前空闲 GPU 更应该转向：
  1. `Table 1` clean 主表闭合
  2. matched-budget `Base + LoRA` 正式训练与评测
  3. projector 冻结后的 `FREAK` 正式化

## 新的时间约束

- `3月25日`：摘要提交截止
- `4月1日`：初稿提交截止

因此从现在起，所有任务优先级都要服从：

1. **先确保摘要所需数据在 `3/25` 前完备。**
2. **再继续推进 `4/1` 初稿闭合资产。**

## 新的摘要优先判断

- 对 `3/25` 摘要最关键的资产：
  1. `Table 2` 当前稳定结果整理
  2. `Table 1` 至少回收首批 clean `8B` 结果
  3. 对 `Base + LoRA` 的正式口径必须明确为：
     - 若正式 50k 训练数据入口未到位，则摘要不能写结果，只能写为 matched-budget fairness baseline 已纳入正式合同、训练入口与预检已完成

- 对 `3/25` 不应继续消耗主力 GPU 的资产：
  - 视频
  - projector 未冻结前的 `FREAK` 正式化
  - 大规模 Figure 1 全量 sweep

## 关于当前 GPU 空闲的最新判断

- 当前真正缺的不是评测数据结果，而是 **`Base + LoRA` 正式训练数据源**：
  - 缺口文件：`/root/autodl-tmp/BRA_Project/train_data/phi_calib_matched_budget_50k.jsonl`
  - 性质：训练数据源缺失，不是训练结果缺失

- 因此，若继续让 `GPU2/GPU3` 等待 LoRA 正式数据入口，会造成不必要空转。

- 新的并行原则改为：
  1. `GPU1` 继续当前 `8B POPE base`
  2. `GPU2/GPU3` 不再等待 LoRA，可立即切去独立 clean `8B Table 1` 长跑
  3. `GPU0` 与 CPU 优先用于构建并验收 `phi_calib_matched_budget_50k.jsonl`

## 新的并行重排（摘要优先）

- `GPU1`：工程师1继续 `8B POPE base`
- `GPU2`：工程师2独立启动 clean `8B POPE vcd`
- `GPU3`：工程师2独立启动 clean `8B POPE tlra_zero`
- `GPU0`：工程师3负责正式 `50k JSONL` 构建、校验、以及后续 LoRA 预备

这样做的原因：

- `Table 1` 是当前摘要与主论文的最大单点瓶颈
- `LoRA` 现在卡在数据源，不应继续占住两张空闲 GPU 等待
- `POPE/CHAIR` 各方法本身并不要求严格串行，只要 GPU 隔离、输出路径隔离、验收严格，就可以并行回收 clean 资产
