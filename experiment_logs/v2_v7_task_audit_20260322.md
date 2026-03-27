# V2-V7 任务实现与完成度审计

生成时间：`2026-03-22`

## 审计范围

本次审计覆盖：

- `今夜4GPU任务清单_V2.md`
- `今夜4GPU任务清单_V3.md`
- `今夜4GPU任务清单_V4.md`
- `今夜4GPU任务清单_V5.md`
- `今夜4GPU任务清单_V6.md`
- `今夜4GPU任务清单_V7.md`

审计目标有两层：

1. 判断每一版清单中的任务，是否已经在代码层面实现。
2. 判断这些任务是否已经被实际执行并形成可引用结果，且结果质量是否足以支撑论文叙述或后续任务。

说明：

- `V2~V4` 的任务主体基本相同，差异主要在巡检快照。
- `V5~V7` 也是同一套任务主体，差异主要在“风险判断”和“下一步优先级”。
- 本审计以当前工作区代码、`experiment_logs/remote_mirror/` 镜像结果、以及远端复核结果为证据基础。

## 总结结论

整体判断：

- `Stage 0`、`POPE/CHAIR`、`MMBench/MME/MMMU pilot`、`FREAK`、`DocVQA loader`、`Video-MME loader` 这些路线在代码层面都已经接通，不再属于“未实现”。
- 真正的问题不再是“代码有没有写”，而是“结果有没有完整跑出来”和“结果是否支撑原论文表述”。
- 从 `V2` 到 `V7`，任务完成度的变化主要是：
  - `DocVQA` 从资源缺失，演变为资源到位且 loader 可运行。
  - `Video-MME` 从资源未确认，演变为资源目录已存在但仍未达到可评测布局。
  - `Chain B` 从 pilot 结果初现，演变为足以支持 `VASM` 必要性的强证据。
  - `Chain C` 仍然是主要风险区，尤其是 `AdaptiveTopK` 在 `FREAK` 上没有赢过 `MeanPool`，`VidHalluc` 仍为空结果。

论文可直接使用的当前强证据：

- `Stage 0`：`TLRA_zero` 更像 viability probe，不适合直接作为主表核心身份。
- `Chain B`：`TLRA_full` 相比 `TLRA_no_VASM` 更稳，`VASM` 必要性已有较强证据。
- `Chain C`：当前证据不支持“`AdaptiveTopK` 优于 `MeanPool`”的强表述。

## 版本分组审计

### V2-V4

这三版的核心任务是同一套：

- GPU0：`Stage 0`，必要时补 `tlra_calib`
- GPU1：`POPE + CHAIR`
- GPU2：`MMBench + MME + MMMU pilot`
- GPU3：`FREAK + VidHalluc`

结论：

- 代码实现：已实现。
- 实际完成：部分完成。

关键判断：

- `GPU0 Stage 0 tlra_zero`：已完成。
- `GPU0 Stage 0 tlra_calib`：未完成。
- `GPU1 POPE`：已跑出多方法结果。
- `GPU1 CHAIR`：只有部分结果可证，仍不完整。
- `GPU2 Chain B`：已完成，并形成有效对照。
- `GPU3 FREAK`：已完成。
- `GPU3 VidHalluc`：未形成有效结果。
- `GPU3 DocVQA`：在 `V2~V4` 时段内属于未闭合。

### V5-V7

这三版的核心不再是“再下载再等待”，而是：

- 把已有实现转成正式主表证据。
- 修通 `DocVQA` 和 `VidHalluc` 这两条空结果支路。
- 用新巡检信息修正任务优先级。

结论：

- 代码实现：已实现。
- 实际完成：仍为部分完成，但比 `V2~V4` 更接近“正式论文级闭环”。

关键变化：

- `DocVQA` 已下载完成，且 20-sample smoke 已可执行。
- `Video-MME` 目录已存在，但当前仍是 zip/chunk 形态，loader 仍然 `Loaded 0 samples`。
- `FREAK` 的复核结果没有改善原风险判断。
- `V7` 已经正确把主优先级从“下载数据”切到“修通空结果并形成可读结果”。

## 任务级审计表

### GPU0 / Stage 0

状态：

- `tlra_semantic_validity_pilot.py`：已实现。
- `tlra_zero`：已完成。
- `tlra_calib`：代码支持，但缺正式 checkpoint 身份，未完成。

证据：

- `experiment_logs/remote_mirror/gpu0_stage0_tlra_zero_8b.json`

关键数据：

- `top1_overlap = 0.9688`
- `top5_overlap = 1.0`
- `top10_overlap = 1.0`
- `patch_top10_overlap = 0.0674`
- `candidate_window_overlap = 0.875`

判断：

- 图像级 overlap 很高，但 patch 级 overlap 偏低。
- 当前结果更支持把 `TLRA_zero` 作为 viability probe，而不是主表核心身份。
- 这和 `V7` 中的风险判断是一致的。

### GPU1 / Chain A

状态：

- `run_eval_pipeline.py`：已实现。
- `POPE`：已完成多方法结果。
- `CHAIR`：部分完成，主表仍不完整。

证据：

- `experiment_logs/remote_mirror/base_pope_20260322_015950.json`
- `experiment_logs/remote_mirror/vcd_pope_20260322_021057.json`
- `experiment_logs/remote_mirror/dola_pope_20260322_022732.json`
- `experiment_logs/remote_mirror/tlra_zero_pope_20260322_023336.json`
- `experiment_logs/remote_mirror/base_chair_20260322_025320.json`

关键数据：

- `POPE base`: `accuracy = 0.925`, `f1 = 0.918`, `agl = 78.55`
- `POPE vcd`: `accuracy = 0.915`, `f1 = 0.9091`, `agl = 72.2`
- `POPE dola`: `accuracy = 0.935`, `f1 = 0.9091`, `agl = 128.0`
- `POPE tlra_zero`: `accuracy = 0.925`, `f1 = 0.918`, `agl = 78.55`, `itl_ms_per_token = 22.97`
- `CHAIR base`: `chair_s = 0.1364`, `chair_i = 0.3127`, `agl = 361.75`, `notes = ["chair_agl_near_cap"]`

判断：

- `POPE` 路线已有可比结果，`tlra_zero` 在 `POPE` 上并未明显优于 `base`，但效率更好。
- `dola` 在 `POPE` 上准确率最高，但其 `agl = 128.0` 明显可疑，且日志中曾出现大量 `unknown` / 乱码风险，不能直接作为可靠主表对照。
- `CHAIR` 当前仍未完成四方法同批重跑，且 `base` 已经接近长度上限，这意味着“AGL 不再被 cap 掩盖”这一目标尚未真正完成。

### GPU2 / Chain B

状态：

- `bra_eval_matrix.py`：已实现。
- `MMBench`：已完成。
- `MME`：已完成。
- `MMMU pilot`：已完成，但仍只是 pilot，不是 `MMMU Hard` 正式主表。
- `MMMU Hard manifest`：未完成。

证据：

- `experiment_logs/remote_mirror/gpu2_mmbench_tlra_full.json`
- `experiment_logs/remote_mirror/gpu2_mmbench_tlra_no_vasm.json`
- `experiment_logs/remote_mirror/gpu2_mme_tlra_full.json`
- `experiment_logs/remote_mirror/gpu2_mme_tlra_no_vasm.json`
- `experiment_logs/remote_mirror/gpu2_mmmu_tlra_full.json`
- `experiment_logs/remote_mirror/gpu2_mmmu_tlra_no_vasm.json`

关键数据：

- `MMBench tlra_full`: `accuracy = 0.865`
- `MMBench tlra_no_vasm`: `accuracy = 0.855`
- `MME tlra_full`: `accuracy = 0.770`, `perception_score = 145.0`
- `MME tlra_no_vasm`: `accuracy = 0.725`, `perception_score = 133.75`
- `MMMU tlra_full`: `accuracy = 0.2667`, `sample_count = 30`
- `MMMU tlra_no_vasm`: `accuracy = 0.2000`, `sample_count = 30`

判断：

- `Chain B` 是目前完成度最高、论文证据价值最高的一条链。
- `tlra_full` 在 `MMBench/MME/MMMU pilot` 上都优于 `tlra_no_vasm`。
- 因此“`VASM` 是必要组件”这一判断已经有较强证据支持。
- 但 `MMMU` 仍只有 `30` 个样本，且缺 `Hard manifest`，不能写成正式 `MMMU Hard` 主表结论。

### GPU3 / Chain C + Video

状态：

- `FREAK`：已完成。
- `VidHalluc`：代码已接通，但当前结果为空。
- `DocVQA`：资源已到位，smoke 已跑通，正式结果仍待继续积累。
- `Video-MME`：资源目录已存在，但当前仍不可评测。

证据：

- `experiment_logs/remote_mirror/gpu3_freak_tlra_meanpool.json`
- `experiment_logs/remote_mirror/gpu3_freak_tlra_adaptivetopk.json`
- `experiment_logs/remote_mirror/gpu3_vidhalluc_tlra_adaptivetopk.json`
- 远端复核：`datasets/DocVQA_hf` 已完整，`datasets/video/Video-MME_hf` 已存在但仍为 chunk/zip 布局

关键数据：

- `FREAK tlra_meanpool`: `accuracy = 0.235`
- `FREAK tlra_adaptivetopk`: `accuracy = 0.225`
- `VidHalluc`: `qwen3vl2b = []`
- `DocVQA smoke (20 samples)`: baseline/bra `accuracy = 0.0 / 0.0`

判断：

- `AdaptiveTopK` 目前没有在 `FREAK` 上超过 `MeanPool`，所以这条论文主张不能写强。
- `VidHalluc` 当前只能算“bounded video pilot 仍未形成可读证据”。
- `DocVQA` 已经从“资源问题”转成“结果质量问题”：现在不是下载阻塞，而是模型在 OCR-heavy 路线上尚未给出有效表现。
- `Video-MME` 则仍然是“资源布局未达成 loader 可消费状态”的工程问题。

## 逐版本完成度结论

### V2

- 任务大部分已实现。
- 真正完成的是 `Stage 0`、`POPE`、`Chain B`、`FREAK`。
- 未完成的是 `Stage 0 calib`、完整 `CHAIR`、`DocVQA`、`VidHalluc`、`MMMU Hard`。

### V3

- 相比 `V2`，远端快照显示更多文件已落盘。
- 但从当前本地镜像看，关键未闭合项仍未闭合。

### V4

- 与 `V3` 基本相同，更多是时间推进而非任务实质变化。

### V5

- 任务重心开始转向“正式化”和“修空结果”。
- 方向判断正确，但仍未把 `Chain C` 风险压下去。

### V6

- 与 `V5` 同主体，更多是巡检续写。
- 主要价值在于积累“资源已到位 / 结果仍为空”的过程证据。

### V7

- 这是目前判断最成熟的一版。
- 它已经正确识别：
  - `TLRA_zero` 更像 probe
  - `Chain B` 有效
  - `Chain C` 风险未解除
  - `DocVQA` 已从下载问题转成 loader/结果问题
  - `Video-MME` 仍在资源布局阶段

## 对论文写作最重要的当前依据

可以写进论文内部备忘或后续实验设计依据的结论：

1. `TLRA_zero` 不应再被当成默认主方法身份，应降为 viability probe 或 appendix 辅助身份。
2. `VASM` 在当前 `MMBench/MME/MMMU pilot` 上具备明确必要性证据。
3. `AdaptiveTopK` 当前没有在 `FREAK` 上稳定优于 `MeanPool`，因此不能继续维持强主张，至少要降级为“待进一步验证”。
4. `CHAIR` 仍有长度上限污染，任何关于 caption-style hallucination 的正式结论都必须等待完整四方法同批重跑。
5. `DocVQA` 现在不是下载阻塞，而是 OCR-heavy 路线结果尚未自洽；论文中不应把它写成已闭合主证据。
6. `Video-MME` 当前不能作为已完成 video benchmark 引用，只能写成资源已进入补齐阶段。

## 后续任务建议

基于 `V2~V7` 的演化，下一批任务应按这个顺序推进：

1. 完整补齐 `GPU1 CHAIR` 四方法统一批次结果，并解除 `AGL` 上限污染。
2. 物化 `MMMU Hard` manifest，把 `Chain B` 从 pilot 升级成正式主表。
3. 把 `DocVQA` 从“可跑 smoke”推进到“可读结果 + 失败模式诊断”。
4. 重新审视 `AdaptiveTopK` 在 `FREAK` 上的设定，不要继续默认它优于 `MeanPool`。
5. 为 `VidHalluc` 和 `Video-MME` 区分清楚两类问题：
   - `VidHalluc`：结果为空，偏 loader/路径/样本可解析性问题。
   - `Video-MME`：数据布局仍未转成可直接消费的视频文件形态。

## 最终结论

如果只问一句：

`V2~V7` 中的大多数任务，代码都已经实现；但真正“很好地完成”的只有 `Stage 0（zero 分支）`、`POPE`、`Chain B`、`FREAK`。`

其余最关键的未闭合项仍然是：

- `Stage 0 calib`
- `完整 CHAIR`
- `MMMU Hard`
- `DocVQA 正式可读结果`
- `VidHalluc 可读结果`
- `Video-MME 可评测布局`

这份状态已经足够作为后续论文写作和新任务设计的依据。
