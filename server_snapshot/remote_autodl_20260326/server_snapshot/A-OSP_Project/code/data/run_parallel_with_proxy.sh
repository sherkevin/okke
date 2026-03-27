#!/usr/bin/env bash
# 并行下载（走本地代理 127.0.0.1:7890，与 Clash 一致）
# 用法: bash code/data/run_parallel_with_proxy.sh
# 日志: logs/parallel_orchestrator.log ；各任务 logs/parallel_<name>.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
# VPN 下建议直连官方 Hub；若仍想用镜像: export USE_HF_MIRROR=1 并设置 HF_ENDPOINT
if [[ "${USE_HF_MIRROR:-0}" != "1" ]]; then
  unset HF_ENDPOINT || true
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export REPORT_INTERVAL_SEC="${REPORT_INTERVAL_SEC:-60}"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/parallel_stdout.log") 2>&1
echo "=== $(date -Iseconds) 启动并行下载 ==="
echo "proxy=$http_proxy"
python3 "$ROOT/code/data/run_parallel_download.py"
echo "=== $(date -Iseconds) 并行下载主进程结束 ==="
