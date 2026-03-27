#!/usr/bin/env bash
# 一次性修复脚本：InstructBLIP / FREAK / MVBench / VidHalluc
set -uo pipefail

export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600

M="/root/autodl-tmp/BRA_Project/models"
DS="/root/autodl-tmp/BRA_Project/datasets"
VID="$DS/video"
LOGDIR="/root/autodl-tmp/BRA_Project/logs/fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "========================================"
echo "[$(date -Iseconds)] fix_all.sh START"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "LOGDIR=$LOGDIR"
echo "========================================"

# ── 0. Kill stale monitor screen ─────────────────────────────
screen -S bra_monitor -X quit 2>/dev/null || true
pkill -f monitor_until_done.sh 2>/dev/null || true

# ── 1. Install hf_xet ────────────────────────────────────────
echo "[step1] Installing hf_xet..."
pip install -q hf_xet 2>&1 | tail -3
python3 -c "import hf_xet; print('[step1] hf_xet', hf_xet.__version__, 'OK')" 2>/dev/null \
  || echo "[step1] hf_xet install may have failed"

# ── Python helper ─────────────────────────────────────────────
run_hf_download() {
  local name="$1" repo="$2" dest="$3" rtype="${4:-dataset}"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -Iseconds)] [$name] START repo=$repo dest=$dest"
  python3 -u - "$repo" "$dest" "$rtype" >> "$log" 2>&1 <<'PY'
import os, sys, time
repo, dest, rtype = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dest, exist_ok=True)
from huggingface_hub import snapshot_download
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id=repo, repo_type=rtype, local_dir=dest, max_workers=8)
        print(f"OK {repo} -> {dest}")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL {attempt} {repo}: {e}")
        if attempt < 3:
            time.sleep(30 * attempt)
sys.exit(1)
PY
  local ec=$?
  echo "[$(date -Iseconds)] [$name] END ec=$ec  log=$log"
  return $ec
}

# ── 2. InstructBLIP: 清 incomplete，重下全量模型 ───────────────
echo ""
echo "[step2] InstructBLIP-Vicuna-7B 全量重下..."
rm -rf "$M/instructblip-vicuna-7b/.cache"
(
  run_hf_download "instructblip" "Salesforce/instructblip-vicuna-7b" "$M/instructblip-vicuna-7b" "model"
  echo "[instructblip] DONE"
) &
PIDS+=($!)
NAMES+=("instructblip")

# ── 3. FREAK: 清 incomplete，重新 snapshot（只会补缺失 shard）──
echo "[step3] FREAK 补全缺失 shard..."
rm -rf "$DS/FREAK_hf/.cache"
(
  run_hf_download "freak" "hansQAQ/FREAK" "$DS/FREAK_hf" "dataset"
  echo "[freak] DONE"
) &
PIDS+=($!)
NAMES+=("freak")

# ── 4. MVBench: 清 incomplete，重新 snapshot ─────────────────
echo "[step4] MVBench 续传缺失视频..."
rm -rf "$VID/OpenGVLab_MVBench/.cache"
(
  run_hf_download "mvbench" "OpenGVLab/MVBench" "$VID/OpenGVLab_MVBench" "dataset"
  echo "[mvbench] DONE"
) &
PIDS+=($!)
NAMES+=("mvbench")

# ── 5. VidHalluc: xet-team/VidHalluc（需 hf_xet）──────────────
echo "[step5] VidHalluc xet-team/VidHalluc 下载..."
# 清除旧的残缺目录缓存
rm -rf "$VID/chaoyuli_VidHalluc/.cache"
rm -rf "$VID/xet_team_VidHalluc" 2>/dev/null || true
mkdir -p "$VID/xet_team_VidHalluc"
(
  run_hf_download "vidhalluc_xet" "xet-team/VidHalluc" "$VID/xet_team_VidHalluc" "dataset"
  echo "[vidhalluc_xet] DONE"
) &
PIDS+=($!)
NAMES+=("vidhalluc_xet")

# ── 等待全部任务 ─────────────────────────────────────────────
echo ""
echo "[$(date -Iseconds)] All jobs launched. Waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  ec=$?
  if [[ $ec -ne 0 ]]; then
    echo "[WARN] ${NAMES[$i]} FAILED (ec=$ec)"
    FAIL=1
  else
    echo "[OK]   ${NAMES[$i]} completed"
  fi
done

echo ""
echo "========================================"
echo "[$(date -Iseconds)] fix_all.sh END  (FAIL=$FAIL)"
echo "Logs: $LOGDIR"
du -sh "$M/instructblip-vicuna-7b" "$DS/FREAK_hf" "$VID/OpenGVLab_MVBench" "$VID/xet_team_VidHalluc" 2>/dev/null
echo "========================================"
