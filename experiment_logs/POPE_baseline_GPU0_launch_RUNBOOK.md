# POPE 解码 Baseline — GPU0 批量启动与登记

## 一键启动（本机 → 远端 AutoDL）

在仓库根目录：

```powershell
python launch_pope_baseline_pending_gpu0.py --mode dry-run   # 默认预览 23 个 job（Qwen8B + InstructBLIP）
python launch_pope_baseline_pending_gpu0.py --mode launch    # SSH 上载 runner + nohup
```

- 默认只包含：`qwen3-vl-8b`、`instructblip-7b`
- 已移出当前 baseline 计划：`qwen3-vl-4b`、`qwen3-vl-2b`

- **GPU：** `CUDA_VISIBLE_DEVICES=0`（全程串行，避免多进程抢显存）。
- **远端主日志：** `logs/pope_full_matrix/pope_baseline_pending_gpu0_<时间戳>.log`
- **清单（每格一行，含 JSON 路径猜测）：** 同目录 `*.manifest.tsv`
- **指标 JSON（惯例）：** `logs/minitest/{method}_pope_<时间戳>.json`

本地会备份一份 shell：`experiment_logs/remote_runners/pope_baseline_pending_gpu0_<时间戳>.sh`。

## 跑完后如何登记

1. **看主日志** 是否出现 `[MATRIX] pope_baseline_pending_gpu0 finished`。
2. **打开 manifest TSV**：每行 `exit_code` 应为 `0`；`latest_minitest_json_guess` 为该 job 完成后目录里**同 method 最新**的 `*_pope_*.json`（串行队列下即本格结果）。
3. 在 `experiment_logs/experiment_registry_latest.md` 的 **Related Baseline / Matrix Logs** 中：
   - 将父任务 `pope_baseline_pending_gpu0_*` 的 `status` 改为 `done`（或 `partial` 若有失败行）；
   - 按 manifest **逐行追加子实验行**（或合并进主表），填齐 `remote_result_json`。

可选：把 manifest 与新增 `minitest` JSON `scp` 到本机存档（例如 `pope_baseline_fullrun_sync/`）。

## 冲突提醒

若 **LLaVA round5 verifier 矩阵** 等任务也占用 **GPU0**，请先停掉其一再跑本队列，否则易 OOM 或极慢。
