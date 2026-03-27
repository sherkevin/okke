# 今夜4GPU任务清单_V6

> 生成时间：`2026-03-22 16:04:02`
> 生成方式：本地镜像结果 + 远程自动巡检（带超时降级）

## 1. 当前完成情况

- `Stage 0`：`risky`。`Stage 0` 已有镜像结果：image top1=0.9688, patch top10=0.0674, candidate overlap=0.875。patch-level overlap 偏低，`TLRA_zero` 更像 viability probe，不应直接当主表核心。
- `Chain A`：`partial`。Chain A 已有历史镜像（`base/vcd/tlra_zero` on `POPE`，`base` on `CHAIR`），但仍需要在同一批次、同一参数上重跑主表，尤其要确保 `CHAIR` 的 `AGL` 不再被 cap 掩盖。
- `Chain B`：`partial`。Chain B pilot 已有镜像：MMBench full=0.865 vs no_vasm=0.855，MME full acc=0.770，MMMU pilot acc=0.267 (n=30)。 但当前仍是 pilot，不是 `MMMU Hard` 正式主表。
- `Chain C / video`：`risky`。FREAK pilot 已有镜像：MeanPool=0.235, AdaptiveTopK=0.225。 当前 `AdaptiveTopK` 尚未证明优于 `MeanPool`，这条主张仍未闭合。 DocVQA 镜像文件存在但结果为空，说明数据虽然补齐，当前 loader / 布局仍未真正跑通。 VidHalluc 镜像文件为空，bounded video pilot 仍需重新验证。
- `DocVQA` 资源：`complete`。 远程已复核。
- `Video-MME` 资源：`in_progress`。 远程已复核。
- `TLRA_calib` 正式 checkpoint 身份：`unresolved`。
- `MMMU Hard` manifest：`unresolved`。
- `Base + 5k LoRA`：`unresolved`。

## 2. 当前数据/远端支撑情况

- 当前远程探查：`reachable`
- `DocVQA`：exists=`True` / file_count=`223`
- `VidHalluc`：exists=`True` / file_count=`9`
- `Video-MME`：exists=`True` / file_count=`74`
- `night_v2` 文件数：`21`
- `night_v2` 文件预览：`base_pope_20260322_015950.json, docvqa_smoke_after_download.json, gpu0_stage0.log, gpu0_stage0_calib.log, gpu0_stage0_tlra_zero_8b.json, gpu1_chainA.log, gpu1_chainA_calib.log, gpu2_chainB.log`

## 3. 本轮调度结论

- 远端当前可达，因此本轮任务单以“继续推进正式主表 + 修复空结果分支”为主。
- 当前最需要优先解决的不是下载，而是把 `DocVQA` 和 `VidHalluc` 从空结果/不可复现状态推进到可读结果。
- `AdaptiveTopK` 目前尚未在 `FREAK` 镜像里稳定优于 `MeanPool`，因此 Chain C 是今夜必须复核的核心风险。 

## 4. 今夜 4 GPU 新任务清单

### GPU0：控制面 + Stage 0 判定

- 先做控制面复核：确认 SSH 恢复、`logs/night_v2/` 可列出、`V_matrix*.pt` 可见。
- 若主机恢复可达，重跑 `Stage 0` 并补 `TLRA_calib` 分支；若 `patch_top10_overlap` 仍低于 `0.10`，将 `TLRA_zero` 降为 appendix-only probe。

```bash
ssh -p 47559 root@connect.westc.seetacloud.com
cd /root/autodl-tmp/BRA_Project && mkdir -p logs/night_v3
ls -lah models/V_matrix*.pt
python tlra_semantic_validity_pilot.py --model qwen3-vl-8b --method tlra_zero --n_samples 64 --topk 10 --candidate_window 50 --output logs/night_v3/gpu0_stage0_tlra_zero_8b.json
python tlra_semantic_validity_pilot.py --model qwen3-vl-8b --method tlra_calib --projector_checkpoint /root/autodl-tmp/BRA_Project/models/REPLACE_ME.pt --n_samples 64 --topk 10 --candidate_window 50 --output logs/night_v3/gpu0_stage0_tlra_calib_8b.json
```

### GPU1：Chain A 正式主表

- 优先重跑 `POPE + CHAIR` 主表，统一使用同一批次配置和高 `chair_max_new_tokens`。
- 如果 `projector_checkpoint` 今夜冻结，追加 `tlra_calib`；否则保留 `base/vcd/dola/tlra_zero` 四方法可比主表。

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH="/root/miniconda3/bin:$PATH" && \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method base      --mini_test 200 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method vcd       --mini_test 200 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method dola      --mini_test 200 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset pope  --method tlra_zero --mini_test 200 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method base      --mini_test 150 --chair_max_new_tokens 384 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method vcd       --mini_test 150 --chair_max_new_tokens 384 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method dola      --mini_test 150 --chair_max_new_tokens 384 ; \
python run_eval_pipeline.py --model qwen3-vl-8b --dataset chair --method tlra_zero --mini_test 150 --chair_max_new_tokens 384' > logs/night_v3/gpu1_chainA.log 2>&1 &
```

### GPU2：Chain B 正式化

- 继续 `MMBench + MME + MMMU` 路线，但把重点放在 `TLRA_full` vs `TLRA_no_VASM` 的正式对照。
- 并行冻结 `MMMU Hard` manifest；在它冻结前，所有 `MMMU` 结果都只记作 pilot，不进最终主表。

```bash
CUDA_VISIBLE_DEVICES=2 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH="/root/miniconda3/bin:$PATH" && \
python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_full    --output logs/night_v3/gpu2_mmbench_tlra_full.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset mmbench --n_samples 300 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mmbench_tlra_no_vasm.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset mme     --n_samples 300 --bra_method tlra_full    --output logs/night_v3/gpu2_mme_tlra_full.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset mme     --n_samples 300 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mme_tlra_no_vasm.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu    --n_samples 200 --bra_method tlra_full    --output logs/night_v3/gpu2_mmmu_tlra_full.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset mmmu    --n_samples 200 --bra_method tlra_no_vasm --output logs/night_v3/gpu2_mmmu_tlra_no_vasm.json' > logs/night_v3/gpu2_chainB.log 2>&1 &
```

### GPU3：DocVQA 修通优先，其次 Chain C / video

- 第一优先级从“下载 DocVQA”改成“修通 DocVQA loader / layout”，先做 20-sample smoke。
- 若 DocVQA 冒烟通过，再跑 `DocVQA + FREAK`；若仍 `no_samples`，立刻回切 `FREAK` 复核与 `VidHalluc` 重试，不要空转。

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash -lc 'cd /root/autodl-tmp/BRA_Project && export PATH="/root/miniconda3/bin:$PATH" && \
python bra_eval_matrix.py --model qwen3vl2b --dataset docvqa --n_samples 20  --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_docvqa_smoke.json || exit 17 ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset docvqa --n_samples 200 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_docvqa_tlra_adaptivetopk.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset freak  --n_samples 200 --bra_method tlra_meanpool    --output logs/night_v3/gpu3_freak_tlra_meanpool.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset freak  --n_samples 200 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_freak_tlra_adaptivetopk.json ; \
python bra_eval_matrix.py --model qwen3vl2b --dataset vidhalluc --n_samples 80 --bra_method tlra_adaptivetopk --output logs/night_v3/gpu3_vidhalluc_tlra_adaptivetopk.json' > logs/night_v3/gpu3_chainC.log 2>&1 &
```

## 5. 明早前必须回收的最小产物

1. 一份新的 `Stage 0` 判定：`TLRA_zero` 是保留为主表、附录，还是仅作 viability probe。
2. 一套统一批次的 `POPE + CHAIR` 可比结果，包含 `AGL` 和至少一个效率字段。
3. 一套 `MMBench + MME + MMMU pilot/Hard-prep` 的 `TLRA_full` vs `TLRA_no_VASM` 对照。
4. 一份可读的 `DocVQA` 结果，或者一份清晰的 `DocVQA loader/layout` 失败诊断。
5. 一份 `FREAK` 新结果，用于重新判断 `AdaptiveTopK` 是否真的强于 `MeanPool`。
6. 一份 `VidHalluc` 可读结果，若仍为空则明确降级为 appendix-only pilot。

## 6. 定时巡检规则

- 本巡检器每次运行都会生成一个新的 `今夜4GPU任务清单_Vx.md`。
- 若远端不可达，下一轮自动把“控制面恢复”置顶，不会误判为实验全失败。
- 若远端恢复可达，下一轮会自动把优先级切回结果回收与空结果修复。
- 若你希望停止循环巡检，只需结束本地后台运行的 `night_4gpu_patrol.py` 进程。

