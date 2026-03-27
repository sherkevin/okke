# BRA_Project — Data & Checkpoints

根目录：`/root/autodl-tmp/BRA_Project/`（200G 数据盘）

## 一键下载

```bash
source /etc/network_turbo 2>/dev/null || true
proxy_on
bash /root/autodl-tmp/BRA_Project/download_all.sh
```

日志：`download_all.log`  
脚本特性：已存在文件跳过、`aria2c -x 16 -s 16`、Qwen3-VL 使用 `huggingface_hub.snapshot_download`（多分片，非单文件 `model.safetensors`）。

**并行下载**：在模型软链/跳过后，**COCO val、COCO annotations、HallusionBench/FREAK/MMMU/MMBench-EN/MME 的 HF snapshot、vLLM git、MVBench/VidHalluc 等会同时起多个后台任务**，最后 `wait` 再统一校验并解压 COCO。若需最快速度，请勿再使用旧版串行脚本；可结束旧进程后重新执行本脚本（`aria2c -c` 会断点续传）。

**COCO 与代理**：`images.cocodataset.org` 在开启 `proxy_on` 时可能出现 **Connection refused**，脚本已对含 `cocodataset.org` 的 URL **自动去掉代理环境变量后调用 aria2c**（直连官方源）。HuggingFace/GitHub 仍走代理。

**val2014.zip 完整性**：完整包约 **6645010738 bytes**；脚本默认 `VAL_MIN_BYTES=6630000000`，避免「>5GB 但 zip 损坏」被误判完成。可通过环境变量覆盖：`VAL_MIN_BYTES=... bash download_all.sh`。

## 持续监控直到下完（推荐长时间挂机）

```bash
nohup bash /root/autodl-tmp/BRA_Project/monitor_until_done.sh >/dev/null 2>&1 &
tail -f /root/autodl-tmp/BRA_Project/monitor_until_done.log
```

- **完成标志**：生成 `BRA_DOWNLOAD_COMPLETE.txt`；HF 成功项会在 `locks/hf_ok_*` 写入标记（仅 `snapshot_download` 全成功才落盘，避免误判）。
- **加速/稳态**：默认 **最多 2 路并行** HF（`HF_PARALLEL_MAX`，减轻 OOM 与 xet 超时）、`HF_MAX_WORKERS=6`、拉长 `HF_HUB_DOWNLOAD_TIMEOUT`、尝试安装 **`hf_transfer`**（`HF_HUB_ENABLE_HF_TRANSFER=1`）。COCO val 使用 **aria2 `-x24 -s24`** 与无限重试。
- **环境变量**：`HF_PARALLEL_MAX=3 HF_MAX_WORKERS=8 SLEEP_SEC=30` 等可按机器内存与带宽调整；xet 仍频繁超时可尝试 **`unset HF_ENDPOINT`** 走官方源后重启监控。
- **注意**：监控主进程对 `locks/monitor_main.lock` 使用 `flock`；子进程已关闭继承的 lock fd，避免孤儿 Python 占锁导致「无法启动第二个监控」。

## 目录结构

| 路径 | 内容 |
|------|------|
| `checkpoints/Qwen3-VL-8B-Instruct` | 8B 权重（若本机 A-OSP 已有则 **符号链接**，不重复下载） |
| `checkpoints/Qwen3-VL-2B-Instruct` | 2B 权重（同上） |
| `datasets/coco2014/` | `val2014.zip`、`annotations_trainval2014.zip` 及解压目录 |
| `datasets/HallusionBench_hf/` | HF：`lmms-lab/HallusionBench` |
| `datasets/HallusionBench/` | GitHub 参考仓库（可选） |
| `datasets/FREAK_hf/` | HF：`hansQAQ/FREAK` |
| `datasets/MMMU_hf/` | HF：`MMMU/MMMU`（体积大，可能较慢） |
| `datasets/MMBench_EN_hf/` | HF：`lmms-lab/MMBench_EN`（MMBench 英文子集，与论文 Table 常用口径一致） |
| `datasets/MME_hf/` | HF：`lmms-lab/MME`（MME 评测数据，lmms-eval 常用） |
| `datasets/video/` | 视频类 HF：`OpenGVLab_MVBench`、`chaoyuli_VidHalluc`（**VidHalluc 官方数据集**，勿用已失效的 `VidHalluc/VidHalluc`） |
| `third_party/vllm/` | vLLM 源码（浅克隆） |

## 完整性校验（脚本内 + 手动）

- **COCO `val2014.zip`**：下载后校验 **文件大小 ≥ 5,000,000,000 bytes**（约 6.1GiB）。
- **`annotations_trainval2014.zip`**：大小 **≥ 200,000,000 bytes**。
- **模型**：`config.json` + 至少一个 `*.safetensors` 分片。

手动示例：

```bash
stat -c%s /root/autodl-tmp/BRA_Project/datasets/coco2014/val2014.zip
unzip -t /root/autodl-tmp/BRA_Project/datasets/coco2014/val2014.zip | tail -3
```

脚本结束时若打印 **`Ready for Clone`**，表示本阶段校验通过（COCO 大小 + 目录汇总）。

## 与原草稿脚本差异说明

- Qwen3-VL **无** 单一 `model.safetensors` 主文件，必须用 **snapshot 全目录** 或 **已有完整目录软链**。
- `csit_freak/images.zip`、`VidHalluc/.../videos.tar.gz` 等 URL 易失效；本脚本改为 **HF 数据集 snapshot**（FREAK / MVBench 等），失败时日志 `[WARN]` 需人工核对仓库名。
