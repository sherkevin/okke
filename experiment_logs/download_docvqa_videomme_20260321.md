# DocVQA / Video-MME 下载日志
> 日期：2026-03-21
> 远程主机：`root@connect.westc.seetacloud.com:47559`
> 加速方式：`source /etc/network_turbo` + `HF_ENDPOINT=https://hf-mirror.com`

## 目标

- P0 必补：`DocVQA`
- P1 尽量补：`Video-MME`

## 下载前检查

- 远程磁盘余量：`/root/autodl-tmp` 可用约 `125G`
- `huggingface_hub` 可用
- `huggingface-cli` 不在 PATH，但不影响，直接用 `huggingface_hub.snapshot_download`

## 仓库体量

- `HuggingFaceM4/DocumentVQA`：约 `9.59 GB`
- `lmms-lab/Video-MME`：约 `101.00 GB`

## 远程目标路径

- `DocVQA_hf`：`/root/autodl-tmp/BRA_Project/datasets/DocVQA_hf`
- `DocVQA` 别名：`/root/autodl-tmp/BRA_Project/datasets/DocVQA`
- `Video-MME_hf`：`/root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf`
- `Video-MME` 别名：`/root/autodl-tmp/BRA_Project/datasets/video/Video-MME`

## 状态

- `DocVQA`：**已完成**
  - 远程路径：`/root/autodl-tmp/BRA_Project/datasets/DocVQA_hf`
  - 远程别名：`/root/autodl-tmp/BRA_Project/datasets/DocVQA`
  - 文件数：`223`
  - 总大小：`9,591,618,321 bytes`（约 `9.0G`）
  - 远程汇总：`/root/autodl-tmp/BRA_Project/logs/downloads/docvqa_summary.json`

- `Video-MME`：**已启动后台下载**
  - 远程路径：`/root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf`
  - 远程别名：`/root/autodl-tmp/BRA_Project/datasets/video/Video-MME`
  - 文件数：`73`
  - 总大小：`101,002,238,065 bytes`（磁盘占用约 `95G`）
  - 远程日志：`/root/autodl-tmp/BRA_Project/logs/downloads/videomme_resume.log`
  - 远程汇总：`/root/autodl-tmp/BRA_Project/logs/downloads/videomme_summary.json`
  - 当前状态：**已完成**

## 备注

- `DocVQA` 已用低并发、可重试策略补齐
- `Video-MME` 已用单独可重试下载器完成
- 完成后需要把最终状态补登记到共享文档
