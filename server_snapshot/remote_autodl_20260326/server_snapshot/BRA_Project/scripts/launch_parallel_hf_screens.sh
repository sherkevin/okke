#!/usr/bin/env bash
# 在数据盘 BRA_Project 下，用多个 screen 会话并行拉取 HF 模型/数据集（续传）。
# 用法: bash scripts/launch_parallel_hf_screens.sh
# 日志目录: logs/hf_parallel/
set -euo pipefail

ROOT="/root/autodl-tmp/BRA_Project"
LOGDIR="$ROOT/logs/hf_parallel"
mkdir -p "$LOGDIR"

if ! command -v screen >/dev/null 2>&1; then
  echo "[ERR] 请先安装 screen: apt-get update && apt-get install -y screen"
  exit 1
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# 国内镜像（与 ModelScope 无关，仅 Hub API/文件走 hf-mirror）；覆盖：export HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# COCO val2014：aria2 多连接 + -c 断点续传；官方源需无代理（与 download_all.sh 一致）
start_coco_val_screen() {
  local session="bra_coco_val2014"
  local zip="$ROOT/datasets/coco2014/val2014.zip"
  local dir="$ROOT/datasets/coco2014"
  local min_bytes="${VAL_MIN_BYTES:-6630000000}"
  mkdir -p "$dir"
  if pgrep -f "aria2c.*val2014\\.zip" >/dev/null 2>&1; then
    echo "[SKIP] COCO val2014 已有 aria2c 在下载"
    return 0
  fi
  if [[ -f "$zip" ]]; then
    local sz
    sz=$(stat -c%s "$zip" 2>/dev/null || echo 0)
    if [[ "$sz" -ge "$min_bytes" ]]; then
      echo "[SKIP] val2014.zip 已达完整阈值 (${sz} >= ${min_bytes})"
      return 0
    fi
  fi
  if screen -list 2>/dev/null | grep -qE "[0-9]+\.${session}[[:space:]]"; then
    echo "[SKIP] screen 已存在: $session"
    return 0
  fi
  echo "[START] screen=$session COCO val2014.zip (aria2c 续传)"
  screen -dmS "$session" env ROOT="$ROOT" ZIP="$zip" DIR="$dir" bash -c '
set -uo pipefail
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
declare -F proxy_on >/dev/null 2>&1 && proxy_on true || true
mkdir -p "$DIR"
LOG="$ROOT/logs/hf_parallel/coco_val2014.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] BEGIN coco val2014 aria2c resume -> $ZIP"
opts=( -x 16 -s 16 -c --file-allocation=none --summary-interval=30 )
# cocodataset.org 经代理常失败，直连
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  aria2c "${opts[@]}" -d "$DIR" -o val2014.zip \
  "http://images.cocodataset.org/zips/val2014.zip"
ec=$?
echo "[$(date -Iseconds)] END coco val2014 ec=$ec"
exit "$ec"
'
}

start_screen() {
  local session="$1" logfile="$2" repo="$3" dest="$4" rtype="$5"
  if screen -list 2>/dev/null | grep -qE "[0-9]+\.${session}[[:space:]]"; then
    echo "[SKIP] screen 已存在: $session  (screen -r $session)"
    return 0
  fi
  echo "[START] screen=$session repo=$repo -> $dest"
  # shellcheck disable=SC2090
  screen -dmS "$session" env \
    ROOT="$ROOT" \
    LOGFILE="$logfile" \
    REPO="$repo" \
    DEST="$dest" \
    RTYPE="$rtype" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    HF_HUB_ETAG_TIMEOUT="$HF_HUB_ETAG_TIMEOUT" \
    HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    bash -c '
set -uo pipefail
cd "$ROOT" || exit 1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# network_turbo / proxy 可能返回非零，不能用 set -e 否则到不了 python
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
declare -F proxy_on >/dev/null 2>&1 && proxy_on true || true
python3 -m pip install -q --upgrade huggingface_hub hf_transfer 2>/dev/null || true
mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1
echo "[$(date -Iseconds)] BEGIN session=${STY:-noscreen} repo=$REPO dest=$DEST"
if python3 -u ./scripts/hf_snapshot_one.py --repo "$REPO" --dest "$DEST" --repo-type "$RTYPE" --max-workers 8; then
  echo "[$(date -Iseconds)] END OK repo=$REPO"
else
  ec=$?
  echo "[$(date -Iseconds)] END FAIL repo=$REPO exit=$ec"
  exit "$ec"
fi
'
}

start_coco_val_screen

# 模型 -> BRA_Project/models/
start_screen "bra_hf_minigpt4" "$LOGDIR/minigpt4_llama7b.log" \
  "wangrongsheng/MiniGPT-4-LLaMA-7B" "$ROOT/models/MiniGPT-4-LLaMA-7B" model

start_screen "bra_hf_llava15" "$LOGDIR/llava_1_5_7b.log" \
  "llava-hf/llava-1.5-7b-hf" "$ROOT/models/llava-1.5-7b-hf" model

start_screen "bra_hf_instructblip" "$LOGDIR/instructblip_vicuna_7b.log" \
  "Salesforce/instructblip-vicuna-7b" "$ROOT/models/instructblip-vicuna-7b" model

# 数据集 -> BRA_Project/datasets/
start_screen "bra_hf_mmbench_en" "$LOGDIR/mmbench_en.log" \
  "lmms-lab/MMBench_EN" "$ROOT/datasets/MMBench_EN_hf" dataset

start_screen "bra_hf_mme" "$LOGDIR/mme.log" \
  "lmms-lab/MME" "$ROOT/datasets/MME_hf" dataset

start_screen "bra_hf_freak" "$LOGDIR/freak.log" \
  "hansQAQ/FREAK" "$ROOT/datasets/FREAK_hf" dataset

start_screen "bra_hf_mmmu" "$LOGDIR/mmmu.log" \
  "MMMU/MMMU" "$ROOT/datasets/MMMU_hf" dataset

start_screen "bra_hf_vidhalluc" "$LOGDIR/vidhalluc.log" \
  "chaoyuli/VidHalluc" "$ROOT/datasets/video/chaoyuli_VidHalluc" dataset

start_screen "bra_hf_mvbench" "$LOGDIR/mvbench.log" \
  "OpenGVLab/MVBench" "$ROOT/datasets/video/OpenGVLab_MVBench" dataset

start_screen "bra_hf_hallusionbench" "$LOGDIR/hallusionbench.log" \
  "lmms-lab/HallusionBench" "$ROOT/datasets/HallusionBench_hf" dataset

# 进度汇总（另一个 screen，周期性 du + tail）
PROG_SESSION="bra_hf_progress"
if screen -list 2>/dev/null | grep -qE "[0-9]+\.${PROG_SESSION}[[:space:]]"; then
  echo "[SKIP] 进度 screen 已存在: $PROG_SESSION"
else
  screen -dmS "$PROG_SESSION" env ROOT="$ROOT" LOGDIR="$LOGDIR" bash -c '
set -u
MASTER="$LOGDIR/master_progress.log"
mkdir -p "$LOGDIR"
while true; do
  {
    echo "========== $(date -Iseconds) =========="
    for line in \
      "COCO_val2014_zip|$ROOT/datasets/coco2014|coco_val2014.log" \
      "MiniGPT-4-LLaMA-7B|$ROOT/models/MiniGPT-4-LLaMA-7B|minigpt4_llama7b.log" \
      "LLaVA-1.5-7b-hf|$ROOT/models/llava-1.5-7b-hf|llava_1_5_7b.log" \
      "InstructBLIP-vicuna-7b|$ROOT/models/instructblip-vicuna-7b|instructblip_vicuna_7b.log" \
      "MMBench_EN_hf|$ROOT/datasets/MMBench_EN_hf|mmbench_en.log" \
      "MME_hf|$ROOT/datasets/MME_hf|mme.log" \
      "FREAK_hf|$ROOT/datasets/FREAK_hf|freak.log" \
      "MMMU_hf|$ROOT/datasets/MMMU_hf|mmmu.log" \
      "VidHalluc|$ROOT/datasets/video/chaoyuli_VidHalluc|vidhalluc.log" \
      "MVBench|$ROOT/datasets/video/OpenGVLab_MVBench|mvbench.log" \
      "HallusionBench_hf|$ROOT/datasets/HallusionBench_hf|hallusionbench.log"
    do
      IFS="|" read -r name dest log <<<"$line"
      echo "--- $name ---"
      du -sh "$dest" 2>/dev/null || echo "  (目录不存在或无法访问)"
      incomplete=$(find "$dest" -name "*.incomplete" 2>/dev/null | wc -l)
      echo "  .incomplete 文件数: $incomplete"
      if [[ -f "$LOGDIR/$log" ]]; then
        echo "  [日志末 4 行] $LOGDIR/$log"
        tail -n 4 "$LOGDIR/$log" | sed "s/^/    /"
      else
        echo "  (尚无日志 $LOGDIR/$log)"
      fi
    done
    echo ""
  } >> "$MASTER"
  echo "[$(date -Iseconds)] snapshot -> $MASTER" | tee -a "$LOGDIR/monitor_echo.log"
  sleep "${HF_PROGRESS_INTERVAL:-90}"
done
'
  echo "[START] 进度汇总 screen: $PROG_SESSION  (主日志: $LOGDIR/master_progress.log)"
fi

echo ""
echo "==== 已调度完成（或已跳过已存在会话）===="
echo "查看会话: screen -ls | grep bra_hf"
echo "附着示例: screen -r bra_hf_llava15"
echo "主进度日志: tail -f $LOGDIR/master_progress.log"
echo "各任务日志: ls -1 $LOGDIR/*.log"
