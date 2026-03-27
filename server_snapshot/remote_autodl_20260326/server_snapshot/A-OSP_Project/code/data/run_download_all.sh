#!/usr/bin/env bash
# 完整下载所有实验数据 — 使用 HF 镜像 + 分步执行 + 失败继续
# 用法: bash code/data/run_download_all.sh [--skip-chair]

set -e
PROJECT="/root/autodl-tmp/A-OSP_Project"
cd "$PROJECT"

# 国内环境优先使用 HF 镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1 2>/dev/null || true
echo "HF_ENDPOINT=$HF_ENDPOINT"

SKIP_CHAIR=""
[[ "$1" == "--skip-chair" ]] && SKIP_CHAIR="--skip_chair"

echo "========== Step 1: Core (MMBench + AMBER) =========="
python3 code/data/download_core_benchmarks.py --mini_test --n_samples 10 || echo "[WARN] Core failed, continuing..."

echo ""
echo "========== Step 2: Cross-domain (IU X-Ray + ChartQA) =========="
python3 code/data/download_crossdomain_datasets.py --mini_test --n_samples 10 || echo "[WARN] Cross-domain failed, continuing..."

echo ""
echo "========== Step 3: Full experiment script (TextVQA, VWB, MVBench, RefCOCO, MMMU, MIRAGE, COD10K) =========="
python3 code/data/download_experiment_datasets.py --mini_test --n_samples 10 $SKIP_CHAIR || echo "[WARN] Experiment datasets failed, continuing..."

echo ""
echo "========== Step 4: CHAIR setup (if not skipped) =========="
if [[ -z "$SKIP_CHAIR" ]]; then
    bash code/data/setup_chair.sh || echo "[WARN] CHAIR setup failed"
fi

echo ""
echo "========== DONE. Check data/benchmarks/ and data/mvbench/ =========="
