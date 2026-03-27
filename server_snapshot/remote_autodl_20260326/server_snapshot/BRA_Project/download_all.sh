#!/usr/bin/env bash
# BRA (双向共振锚定) — 全量数据/权重下载脚本
# - 优先启用 AutoDL 代理 (proxy_on)
# - 已存在且通过完整性校验的文件跳过，避免重复下载
# - 大文件使用 aria2c -x 16 -s 16
# - Qwen3-VL 为多分片 safetensors，使用 huggingface-cli download（支持断点续传）
set -euo pipefail

# ── 1. 代理与环境 ─────────────────────────────────────────────
if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi
if declare -F proxy_on >/dev/null 2>&1; then
  proxy_on true
else
  export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
  export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
  export no_proxy="${no_proxy:-127.0.0.1,localhost}"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  export NO_PROXY="$no_proxy"
fi

# HuggingFace：默认国内镜像；若镜像 503/缺文件可临时改为官方：
#   export HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

WORKDIR="/root/autodl-tmp/BRA_Project"
CKPT="$WORKDIR/checkpoints"
DS="$WORKDIR/datasets"
COCO="$DS/coco2014"
VID="$DS/video"
THIRD="$WORKDIR/third_party"
LOG="$WORKDIR/download_all.log"

mkdir -p "$CKPT" "$COCO" "$VID" "$THIRD"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "[$(date -Iseconds)] BRA download_all.sh start"
echo "WORKDIR=$WORKDIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "========================================"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }
need_cmd aria2c
need_cmd git
need_cmd unzip
need_cmd tar
need_cmd python3

python3 -m pip install -q --upgrade huggingface_hub 2>/dev/null || true

hf_snapshot() {
  # $1=repo_id  $2=local_dir  $3=repo_type (model|dataset)
  python3 - "$1" "$2" "$3" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo_id, local_dir, rtype = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    repo_type=rtype,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("OK", repo_id, "->", local_dir)
PY
}

# ── 工具函数 ───────────────────────────────────────────────────
file_ok_min_bytes() {
  local f="$1" min="$2"
  [[ -f "$f" ]] || return 1
  local sz
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
  [[ "$sz" -ge "$min" ]]
}

aria2_get() {
  local url="$1" out_dir="$2" out_name="${3:-}"
  mkdir -p "$out_dir"
  local opts=( -x 16 -s 16 -c --file-allocation=none --summary-interval=30 )
  # COCO 官方源经学术代理常出现 Connection refused，改为直连
  local run=( aria2c "${opts[@]}" )
  if [[ -n "$out_name" ]]; then
    run+=( -d "$out_dir" -o "$out_name" "$url" )
  else
    run+=( -d "$out_dir" "$url" )
  fi
  if [[ "$url" == *"cocodataset.org"* ]]; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy "${run[@]}"
  else
    "${run[@]}"
  fi
}

# 并行子 shell 需显式导出函数（必须在定义之后）
export -f hf_snapshot aria2_get

# ── 2. 模型：若 A-OSP 已完整则符号链接，否则 HF 全量下载 ─────────
link_or_hf_model() {
  local local_name="$1" repo_id="$2"
  local dest="$CKPT/$local_name"
  local osp="/root/autodl-tmp/A-OSP_Project/models/$local_name"

  if [[ -d "$dest" && -f "$dest/config.json" ]]; then
    local n
    n=$(find "$dest" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    if [[ "$n" -ge 1 ]]; then
      echo "[SKIP] $local_name already present under $dest ($n shard(s))"
      return 0
    fi
  fi

  if [[ -d "$osp" && -f "$osp/config.json" ]]; then
    n=$(find "$osp" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    if [[ "$n" -ge 1 ]]; then
      echo "[LINK] Reusing existing model from A-OSP → $dest"
      mkdir -p "$CKPT"
      rm -rf "$dest"
      ln -sfn "$osp" "$dest"
      return 0
    fi
  fi

  echo "[HF] Downloading $repo_id → $dest"
  hf_snapshot "$repo_id" "$dest" model
}

echo "--- Models (Qwen3-VL) ---"
link_or_hf_model "Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-8B-Instruct"
link_or_hf_model "Qwen3-VL-2B-Instruct" "Qwen/Qwen3-VL-2B-Instruct"

# ── 3–8. 并行下载阶段（COCO + 全部 HF 数据集 + Git 同时进行）──────────
# 说明：原先为串行（先等 val2014 再拉其它），现改为全部后台 job + wait，总墙钟时间最短。
VAL_ZIP="$COCO/val2014.zip"
ANN_ZIP="$COCO/annotations_trainval2014.zip"
# 完整 val2014.zip 官方 6645010738 bytes；半成品可能 >6.5GiB 仍损坏，阈值须贴近完整大小
VAL_MIN_BYTES="${VAL_MIN_BYTES:-6630000000}"
HALL_DIR="$DS/HallusionBench_hf"
HALL_GIT="$DS/HallusionBench"
FREAK_DIR="$DS/FREAK_hf"
MMMU_DIR="$DS/MMMU_hf"
# MMBench：英文评测集（论文常用 MMBench-EN）；全量多语言见 lmms-lab/MMBench
MMBENCH_DIR="$DS/MMBench_EN_hf"
# MME：多模态感知/认知综合评测（lmms-eval 常用 HF 镜像）
MME_DIR="$DS/MME_hf"
VLLM_DIR="$THIRD/vllm"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
python3 -m pip install -q hf_transfer 2>/dev/null || true

declare -a BG_PIDS=()
declare -a BG_NAMES=()

bg_run() {
  local name="$1"
  shift
  echo "[PARALLEL START] $name @ $(date -Iseconds)"
  (
    set +e
    "$@"
    ec=$?
    echo "[PARALLEL END]   $name ec=$ec @ $(date -Iseconds)"
    exit "$ec"
  ) &
  BG_PIDS+=($!)
  BG_NAMES+=("$name")
}

echo "--- Phase PARALLEL: COCO zips + HF datasets + git (all at once) ---"

if file_ok_min_bytes "$VAL_ZIP" 5000000000; then
  echo "[SKIP] val2014.zip ok"
else
  rm -f "$VAL_ZIP"
  bg_run "coco_val2014" aria2_get "http://images.cocodataset.org/zips/val2014.zip" "$COCO" "val2014.zip"
fi

if file_ok_min_bytes "$ANN_ZIP" 200000000; then
  echo "[SKIP] annotations_trainval2014.zip ok"
else
  rm -f "$ANN_ZIP"
  bg_run "coco_ann" aria2_get "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "$COCO" "annotations_trainval2014.zip"
fi

if [[ -d "$HALL_DIR" ]] && [[ $(find "$HALL_DIR" -type f 2>/dev/null | wc -l) -gt 5 ]]; then
  echo "[SKIP] HallusionBench_hf"
else
  bg_run "hf_HallusionBench" hf_snapshot lmms-lab/HallusionBench "$HALL_DIR" dataset
fi

if [[ ! -d "$HALL_GIT/.git" ]]; then
  bg_run "git_HallusionBench" git clone --depth 1 https://github.com/FuxiaoLiu/HallusionBench.git "$HALL_GIT"
else
  echo "[SKIP] HallusionBench git"
fi

if [[ -d "$FREAK_DIR" ]] && [[ $(find "$FREAK_DIR" -type f 2>/dev/null | wc -l) -gt 3 ]]; then
  echo "[SKIP] FREAK_hf"
else
  bg_run "hf_FREAK" hf_snapshot hansQAQ/FREAK "$FREAK_DIR" dataset
fi

if [[ -d "$MMMU_DIR" ]] && [[ $(find "$MMMU_DIR" -type f 2>/dev/null | wc -l) -gt 10 ]]; then
  echo "[SKIP] MMMU_hf"
else
  bg_run "hf_MMMU" hf_snapshot MMMU/MMMU "$MMMU_DIR" dataset
fi

if [[ -d "$MMBENCH_DIR" ]] && [[ $(find "$MMBENCH_DIR" -type f 2>/dev/null | wc -l) -gt 3 ]]; then
  echo "[SKIP] MMBench_EN_hf"
else
  bg_run "hf_MMBench_EN" hf_snapshot lmms-lab/MMBench_EN "$MMBENCH_DIR" dataset
fi

if [[ -d "$MME_DIR" ]] && [[ $(find "$MME_DIR" -type f 2>/dev/null | wc -l) -gt 3 ]]; then
  echo "[SKIP] MME_hf"
else
  bg_run "hf_MME" hf_snapshot lmms-lab/MME "$MME_DIR" dataset
fi

if [[ ! -d "$VLLM_DIR/.git" ]]; then
  bg_run "git_vllm" git clone --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
else
  echo "[SKIP] vllm git"
fi

# VidHalluc 官方 HF 数据集为 chaoyuli/VidHalluc（非 VidHalluc/VidHalluc；部分镜像无前者映射会 404）
for repo in "OpenGVLab/MVBench" "chaoyuli/VidHalluc"; do
  SAFE="${repo//\//_}"
  TDIR="$VID/${SAFE}"
  if [[ -d "$TDIR" ]] && [[ $(find "$TDIR" -type f 2>/dev/null | wc -l) -gt 2 ]]; then
    echo "[SKIP] video $repo"
  else
    bg_run "hf_${SAFE}" hf_snapshot "$repo" "$TDIR" dataset
  fi
done

# 等待所有并行任务结束（不因单任务失败而整脚本退出）
set +e
FAIL=0
for i in "${!BG_PIDS[@]}"; do
  pid="${BG_PIDS[$i]}"
  name="${BG_NAMES[$i]}"
  wait "$pid"
  ec=$?
  if [[ $ec -ne 0 ]]; then
    echo "[WARN] Parallel job FAILED: $name (pid=$pid exit=$ec)"
    FAIL=1
  fi
done
set -e
if [[ $FAIL -ne 0 ]]; then
  echo "[WARN] One or more parallel downloads failed — check log above; COCO may still be resumable (aria2c -c)."
fi

echo "--- Phase VERIFY: COCO zip sizes ---"
if ! file_ok_min_bytes "$VAL_ZIP" "$VAL_MIN_BYTES"; then
  echo "[FAIL] val2014.zip too small or missing after parallel phase (need >= ${VAL_MIN_BYTES} B)"; exit 1
fi
if ! file_ok_min_bytes "$ANN_ZIP" 200000000; then
  echo "[FAIL] annotations_trainval2014.zip too small or missing"; exit 1
fi

echo "[UNZIP] val2014 (idempotent)"
if [[ ! -d "$COCO/val2014" ]] || [[ $(find "$COCO/val2014" -maxdepth 1 -type f 2>/dev/null | wc -l) -lt 1000 ]]; then
  unzip -q -o "$VAL_ZIP" -d "$COCO/"
fi
echo "[UNZIP] annotations"
unzip -q -o "$ANN_ZIP" -d "$COCO/" || true

# ── 9. 汇总校验 ───────────────────────────────────────────────
echo "========================================"
echo "[VERIFY] Layout:"
du -sh "$CKPT"/* 2>/dev/null || true
du -sh "$COCO"/*.zip 2>/dev/null || true
du -sh "$DS"/* 2>/dev/null | head -20
echo "[$(date -Iseconds)] Ready for Clone — BRA_Project data staged under $WORKDIR"
echo "Log: $LOG"
echo "========================================"
