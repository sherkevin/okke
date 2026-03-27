#!/usr/bin/env bash
# BRA 数据集完成守护：周期性检查 download_all.sh 约定的目录，断线/失败后自动重拉 HF。
# 用法: nohup bash scripts/bra_dataset_watcher.sh >> logs/dataset_watcher.log 2>&1 &
# 停止: pkill -f bra_dataset_watcher.sh
set -u

ROOT="/root/autodl-tmp/BRA_Project"
DS="$ROOT/datasets"
COCO="$DS/coco2014"
VID="$DS/video"
LOGDIR="$ROOT/logs/hf_parallel"
INTERVAL="${BRA_WATCH_INTERVAL:-180}"

mkdir -p "$LOGDIR"

[[ -f /etc/network_turbo ]] && source /etc/network_turbo 2>/dev/null || true
declare -F proxy_on >/dev/null 2>&1 && proxy_on true || true

hf_screen_running() {
  local s="$1"
  screen -list 2>/dev/null | grep -qE "[0-9]+\.${s}[[:space:]]"
}

start_hf_dataset() {
  local session="$1" logfile="$2" repo="$3" dest="$4"
  if hf_screen_running "$session"; then
    return 0
  fi
  echo "[$(date -Iseconds)] WATCHER: starting screen $session -> $repo"
  # 大文件易断：关闭 hf_transfer，降低并发，拉长超时
  screen -dmS "$session" env \
    ROOT="$ROOT" \
    LOGFILE="$logfile" \
    REPO="$repo" \
    DEST="$dest" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-180}" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_ENDPOINT="${BRA_HF_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}" \
    bash -c '
set -uo pipefail
cd "$ROOT" || exit 1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
declare -F proxy_on >/dev/null 2>&1 && proxy_on true || true
python3 -m pip install -q --upgrade huggingface_hub 2>/dev/null || true
mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1
echo "[$(date -Iseconds)] WATCHER-RETRY BEGIN repo=$REPO dest=$DEST HF_TRANSFER=0"
if python3 -u ./scripts/hf_snapshot_one.py --repo "$REPO" --dest "$DEST" --repo-type dataset --max-workers 4; then
  echo "[$(date -Iseconds)] WATCHER-RETRY END OK repo=$REPO"
else
  echo "[$(date -Iseconds)] WATCHER-RETRY END FAIL repo=$REPO exit=$?"
fi
'
}

val_ok() {
  local z="$COCO/val2014.zip"
  [[ -f "$z" ]] || return 1
  local sz
  sz=$(stat -c%s "$z" 2>/dev/null || echo 0)
  [[ "$sz" -ge 6630000000 ]]
}

ann_ok() {
  local z="$COCO/annotations_trainval2014.zip"
  [[ -f "$z" ]] || return 1
  local sz
  sz=$(stat -c%s "$z" 2>/dev/null || echo 0)
  [[ "$sz" -ge 200000000 ]]
}

need_mmbench() {
  local d="$DS/MMBench_EN_hf"
  [[ -d "$d" ]] || return 0
  if find "$d" -name "*.incomplete" -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  local n
  n=$(find "$d" -type f -path "*/data/*.parquet" 2>/dev/null | wc -l)
  [[ "$n" -lt 2 ]]
}

need_mme() {
  local d="$DS/MME_hf"
  [[ -d "$d" ]] || return 0
  if find "$d" -name "*.incomplete" -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  # 至少若干 parquet / 数据文件
  local n
  n=$(find "$d" -type f \( -name "*.parquet" -o -name "*.jsonl" -o -name "*.json" \) 2>/dev/null | wc -l)
  [[ "$n" -lt 3 ]]
}

need_vidhalluc() {
  # 与 download_all.sh 中 repo=chaoyuli/VidHalluc → 目录 video/chaoyuli_VidHalluc 一致
  local d="$VID/chaoyuli_VidHalluc"
  [[ -d "$d" ]] || return 0
  # 仅若干元数据文件不算完成；全量约数 GB+（视频）
  local bytes
  bytes=$(du -sb "$d" 2>/dev/null | awk '{print $1}')
  [[ "${bytes:-0}" -lt 500000000 ]]
}

echo "[$(date -Iseconds)] bra_dataset_watcher START interval=${INTERVAL}s"

while true; do
  echo ""
  echo "========== $(date -Iseconds) tick =========="

  if ! val_ok; then
    sz=$(stat -c%s "$COCO/val2014.zip" 2>/dev/null || echo 0)
    echo "[INFO] COCO val2014.zip size=$sz (need >=6630000000). aria2 若未运行可手动: aria2c -c -x16 -s16 ..."
    if ! pgrep -f "val2014.zip" >/dev/null 2>&1; then
      echo "[WARN] 未发现 val2014 aria2 进程；如需续传请执行 download_all.sh 或单独 aria2c -c"
    fi
  else
    echo "[OK] COCO val2014.zip complete"
  fi

  if ann_ok; then echo "[OK] COCO annotations zip"; else echo "[WARN] annotations missing/small"; fi

  if need_mmbench; then
    start_hf_dataset "bra_hf_mmbench_en" "$LOGDIR/mmbench_en.log" "lmms-lab/MMBench_EN" "$DS/MMBench_EN_hf"
  else
    echo "[OK] MMBench_EN_hf (no incomplete / has parquet)"
  fi

  if need_mme; then
    if ! hf_screen_running "bra_hf_mme"; then
      start_hf_dataset "bra_hf_mme" "$LOGDIR/mme.log" "lmms-lab/MME" "$DS/MME_hf"
    fi
  else
    echo "[OK] MME_hf"
  fi

  if need_vidhalluc; then
    start_hf_dataset "bra_hf_vidhalluc" "$LOGDIR/vidhalluc.log" "chaoyuli/VidHalluc" "$VID/chaoyuli_VidHalluc"
  else
    echo "[OK] chaoyuli_VidHalluc (VidHalluc)"
  fi

  if val_ok && ann_ok && ! need_mmbench && ! need_mme && ! need_vidhalluc; then
    echo "[$(date -Iseconds)] ALL tracked datasets satisfied — watcher keeps running (Ctrl+C / pkill to stop)."
  fi

  sleep "$INTERVAL"
done
