#!/usr/bin/env bash
# =============================================================================
# 在已开启 VPN + 本地代理（默认 127.0.0.1:7890）时，拉取实验相关数据集全量/大样本，
# 并生成 logs/dataset_download_report.md
#
# 用法（在项目根目录）:
#   bash code/data/run_vpn_full_download.sh
#
# 环境变量:
#   PROXY_URL   默认 http://127.0.0.1:7890
#   SKIP_AMBER_GDOWN=1  跳过 AMBER 网盘大文件（~14GB）
#   USE_HF_MIRROR=1     仍走 hf-mirror（一般 VPN 下应关闭，用官方 Hub）
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/vpn_full_download_${STAMP}.log"
ln -sf "$MASTER_LOG" "${LOG_DIR}/vpn_full_download_latest.log"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ "${USE_HF_MIRROR:-0}" == "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[info] HF_ENDPOINT=$HF_ENDPOINT (mirror mode)"
else
  unset HF_ENDPOINT || true
  echo "[info] HF_ENDPOINT unset → 使用官方 huggingface.co"
fi

export TEXTVQA_DOWNLOAD_TIMEOUT_SEC="${TEXTVQA_DOWNLOAD_TIMEOUT_SEC:-14400}"
export CHAIR_SETUP_TIMEOUT_SEC="${CHAIR_SETUP_TIMEOUT_SEC:-7200}"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=============================================="
echo "VPN / 代理全量下载  $(date -Iseconds)"
echo "PROXY=$PROXY_URL"
echo "LOG=$MASTER_LOG"
echo "PROJECT=$PROJECT_ROOT"
echo "=============================================="

echo ""
echo ">>> [Phase 0] 代理连通性测试"
if curl -fsS --connect-timeout 20 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co/; then
  echo "[Phase 0] huggingface.co OK"
else
  echo "[Phase 0] WARNING: huggingface.co failed — 请确认 clash/VPN 与端口 7890"
fi
if curl -fsS --connect-timeout 15 -o /dev/null -w "google HTTP %{http_code}\n" https://www.google.com/; then
  echo "[Phase 0] google OK"
else
  echo "[Phase 0] WARNING: google failed"
fi

echo ""
echo ">>> [Phase 1] pip 依赖（gdown）"
python3 -m pip install -q gdown huggingface_hub hf_transfer 2>/dev/null || pip install -q gdown huggingface_hub hf_transfer

echo ""
echo ">>> [Phase 2] download_experiment_datasets.py --full --skip_chair"
# CHAIR 单独跑，便于观察日志与延长超时
python3 code/data/download_experiment_datasets.py --full --skip_chair

echo ""
echo ">>> [Phase 3] setup_chair.sh（COCO 标注 + CHAIR 脚本）"
set +e
timeout "${CHAIR_SETUP_TIMEOUT_SEC}" bash code/data/setup_chair.sh
CHAIR_RC=$?
set -e
echo "[Phase 3] setup_chair exit code: $CHAIR_RC"

if [[ "${SKIP_AMBER_GDOWN:-0}" != "1" ]]; then
  echo ""
  echo ">>> [Phase 4] AMBER 图像（Google Drive，体积大，可能需较长时间）"
  AMBER_IMG="${PROJECT_ROOT}/data/benchmarks/amber/images"
  mkdir -p "$AMBER_IMG"
  N_AMBER="$(find "$AMBER_IMG" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | wc -l)"
  if [[ "$N_AMBER" -ge 500 ]]; then
    echo "[Phase 4] 已有 AMBER 图像约 ${N_AMBER} 张，跳过 gdown"
  else
    set +e
    cd "$AMBER_IMG"
    # 官方打包（与 download_core_benchmarks.py 中 AMBER_IMAGE_GDRIVE_ID 一致）
    GDOWN_RC=0
    gdown "https://drive.google.com/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl" --fuzzy --remaining-ok || GDOWN_RC=$?
    set -e
    cd "$PROJECT_ROOT"
    echo "[Phase 4] gdown exit: $GDOWN_RC (失败常见：配额/需登录 cookie，请浏览器下载后解压到此目录)"
    shopt -s nullglob
    for arc in "$AMBER_IMG"/*.zip "$AMBER_IMG"/*.tar "$AMBER_IMG"/*.tar.gz "$AMBER_IMG"/*.tgz; do
      echo "[Phase 4] trying extract: $arc"
      if [[ "$arc" == *.zip ]]; then unzip -q -o "$arc" -d "$AMBER_IMG" || true; fi
      if [[ "$arc" == *.tar.gz ]] || [[ "$arc" == *.tgz ]]; then tar -xzf "$arc" -C "$AMBER_IMG" || true; fi
      if [[ "$arc" == *.tar ]] && [[ "$arc" != *.tar.gz ]]; then tar -xf "$arc" -C "$AMBER_IMG" || true; fi
    done
    shopt -u nullglob
  fi
else
  echo ""
  echo ">>> [Phase 4] SKIP_AMBER_GDOWN=1，跳过 AMBER 网盘"
fi

echo ""
echo ">>> [Phase 5] 生成状态报告"
python3 code/data/generate_dataset_status_report.py

echo ""
echo "=============================================="
echo "ALL_PHASES_DONE  $(date -Iseconds)"
echo "报告: ${LOG_DIR}/dataset_download_report.md"
echo "主日志: $MASTER_LOG"
echo "=============================================="
