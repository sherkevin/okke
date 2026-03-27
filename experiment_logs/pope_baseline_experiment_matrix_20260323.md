# POPE Decoding Baseline — Experiment Matrix（计划 vs 实际）

Last updated: `2026-03-23`

**数据来源：** `experiment_logs/experiment_registry_latest.md` 中 *Related Baseline / Matrix Logs* 与 UniGround 行（仅作对照，非本表「解码 baseline」单元格）。

**约定：**

- **矩阵单元格** = `模型 × POPE split × method`；每格目标样本量 **3000**（与 `run_eval_pipeline.py --mini_test 3000` 一致）。
- **方法集**（与已跑 GPU0 矩阵一致）：`base`、`beam_search`、`dola`、`opera`。  
  `launch_baseline_only_fullrun.py` 默认还含 **`vcd`**；若论文表需要 VCD，需在矩阵中 **另增一列/一行** 并单独排队（当前 `pope_full_matrix` 日志未覆盖 VCD）。
- **结果 JSON 目录（远端）：** `/root/autodl-tmp/BRA_Project/logs/minitest/`  
- **矩阵 runner 日志：** `/root/autodl-tmp/BRA_Project/logs/pope_full_matrix/`

---

## 1. 总览：每模型 12 格（4 method × 3 split）

| 模型 | 已完成 | 未完成 / 需补跑 | 备注 |
| --- | ---: | ---: | --- |
| `llava-v1.5-7b` | **12** | **0** | `pope_llava_matrix_*` + `pope_llava_splits_followup_*`；汇总见 `llava_pope_baseline_main_table_20260323.md` |
| `qwen3-vl-8b` | **1** | **11** | 仅 `random`+`base` 有正式 JSON；`random`+`beam_search` 曾启动后 **stopped**；其余未跑或未产出 JSON |
| `instructblip-7b` | **0** | **12** | 登记中 **无** POPE 全量矩阵子实验行 |
| `qwen3-vl-4b` | **0** | **out-of-scope** | 用户已明确不再测该模型 baseline |
| `qwen3-vl-2b` | **0** | **out-of-scope** | 用户已明确不再测该模型 baseline |

> **说明：** UniGround 在 Qwen/LLaVA 上的全量/分 split 结果见 registry 主表，**不计入**上表「解码 baseline」完成数。

---

## 2. 明细矩阵 — `qwen3-vl-8b`（POPE 解码 baseline）

| split ↓ / method → | base | beam_search | dola | opera |
| --- | --- | --- | --- | --- |
| **random** | ✅ done → `base_pope_20260323_125009.json` | ⚠️ stopped（无 JSON，需重跑） | ⬜ not_started | ⬜ not_started |
| **popular** | ⬜ pending | ⬜ pending | ⬜ pending | ⬜ pending |
| **adversarial** | ⬜ pending | ⬜ pending | ⬜ pending | ⬜ pending |

**父日志：** `pope_full_matrix_20260323_111111.log`（矩阵在 LLaVA 优先策略后未继续完成 Qwen 余格）。

---

## 3. 明细矩阵 — `llava-v1.5-7b`（POPE 解码 baseline）

| split ↓ / method → | base | beam_search | dola | opera |
| --- | --- | --- | --- | --- |
| **random** | ✅ | ✅ | ✅ | ✅ |
| **popular** | ✅ | ✅ | ✅ | ✅ |
| **adversarial** | ✅ | ✅ | ✅ | ✅ |

**父日志：** `pope_llava_matrix_20260323_134023.log`（random）、`pope_llava_splits_followup_20260323_145038.log`（popular + adversarial）。  
**JSON 文件名索引：** `experiment_logs/llava_pope_baseline_main_table_20260323.md`。

---

## 4. 明细矩阵 — `instructblip-7b`

`instructblip-7b` 在 registry 的 baseline 矩阵区目前 **无** 对应 12 格记录，全部记为 **⬜ pending**。

| split ↓ / method → | base | beam_search | dola | opera |
| --- | --- | --- | --- | --- |
| **random** | ⬜ | ⬜ | ⬜ | ⬜ |
| **popular** | ⬜ | ⬜ | ⬜ | ⬜ |
| **adversarial** | ⬜ | ⬜ | ⬜ | ⬜ |

**建议启动方式：** `launch_baseline_only_fullrun.py` 仅 POPE 时需加 `--datasets pope`；按 split 分批时可配合 `--pope-split` 与 `--methods`。

**批量补跑（GPU0、含 manifest）：** 仓库根目录执行 `python launch_pope_baseline_pending_gpu0.py --mode launch`，默认只包含 `qwen3-vl-8b` 与 `instructblip-7b`；说明见 **`experiment_logs/POPE_baseline_GPU0_launch_RUNBOOK.md`**。

---

## 5. 明细矩阵 — `qwen3-vl-4b` / `qwen3-vl-2b`

这两组 baseline 已从当前主表计划中移除，标记为 **out-of-scope**，不再继续排队。

---

## 6. 可选扩展（未纳入当前 12 格）

| 项 | 状态 |
| --- | --- |
| **VCD**（`run_eval_pipeline` 支持） | 未在 `pope_full_matrix` 系列中批量跑满；若需要与 `launch_baseline_only_fullrun.py` 默认方法对齐，请增加「第 5 种 method」列并单独登记 JSON。 |
| **CHAIR 全量 baseline**（`--datasets chair`，默认 5000） | 与 POPE 矩阵独立；registry 未跟踪「每模型 × 全方法」CHAIR 主表完成情况；早期仅有小样本冒烟见 `minitest_50_baseline_eval_20260321.md`。 |

---

## 7. 待跑清单（仅 POPE 解码 baseline，可贴到任务队列）

**必须补跑（相对当前 registry）：**

1. **Qwen3-VL-8B：** `random` 上 `beam_search`（重跑）、`dola`、`opera`；**全部** `popular`、`adversarial` × 4 methods（共 **11** 格）。
2. **InstructBLIP-7B：** 全 **12** 格（若进主表）。
3. **Qwen3-VL-4B / 2B：** 已移出当前 baseline 计划，不再补跑。

**已完成、无需再跑：** LLaVA-1.5-7B 全 12 格。
