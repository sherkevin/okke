#!/usr/bin/env bash
# BRA 数据集：持续监控 + 自动续传，直到全部满足完成条件
export HF_ENDPOINT="https://hf-mirror.com"
export PATH="/root/miniconda3/bin:$PATH"
# 加速/稳态：HF 限并发(防 OOM)、长超时、可选 hf_transfer、成功标记文件、aria2 多连接
# 日志：monitor_until_done.log / val2014_monitor.log
set -uo pipefail

WORKDIR="/root/autodl-tmp/BRA_Project"
DS="$WORKDIR/datasets"
COCO="$DS/coco2014"
VID="$DS/video"
LOCKDIR="$WORKDIR/locks"
LOG="$WORKDIR/monitor_until_done.log"
VAL_TARGET_BYTES=6645010738
ANN_MIN_BYTES=200000000
SLEEP_SEC="${SLEEP_SEC:-45}"
# 同时进行的 HF snapshot 数（5 个并行易导致内存爆、xet 超时；默认 2）
HF_PARALLEL_MAX="${HF_PARALLEL_MAX:-2}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-6}"

mkdir -p "$LOCKDIR" "$COCO" "$VID"

MON_PID_FILE="$LOCKDIR/monitor.pid"
if [[ -f "$MON_PID_FILE" ]]; then
  old_pid="$(cat "$MON_PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[$(date -Iseconds)] monitor pid $old_pid still alive — exit."
    exit 0
  fi
fi

exec 300>"$LOCKDIR/monitor_main.lock"
if ! flock -n 300; then
  echo "[$(date -Iseconds)] another monitor holds monitor_main.lock — exit."
  exit 0
fi
echo $$ >"$MON_PID_FILE"
trap 'rm -f "$MON_PID_FILE"' EXIT

if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi
declare -F proxy_on >/dev/null 2>&1 && proxy_on true

# Hugging Face：拉长读超时（xet 大文件易 read timeout）；可按环境覆盖
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
# 若已 pip 安装 hf_transfer，可显著加速大文件（失败则自动回退）
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

exec >>"$LOG" 2>&1

echo "========================================"
echo "[$(date -Iseconds)] monitor_until_done.sh START pid=$$"
echo "WORKDIR=$WORKDIR  SLEEP_SEC=$SLEEP_SEC  HF_PARALLEL_MAX=$HF_PARALLEL_MAX  HF_MAX_WORKERS=$HF_MAX_WORKERS"
echo "HF_ENDPOINT=${HF_ENDPOINT:-<default>}  HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo "Tip: xet 仍超时可尝试 unset HF_ENDPOINT 走官方源后重启本脚本"
echo "========================================"

python3 -m pip install -q --upgrade huggingface_hub 2>/dev/null || true
python3 -m pip install -q hf_transfer 2>/dev/null || true

# 主进程用 fd 300 做 singleton flock；子进程若继承该 fd，可能在父进程退出后仍持有锁，阻塞新监控。
close_inherited_singleton_fds() {
  local fd
  for fd in 300 301 302 303 304 305; do
    eval "exec ${fd}<&-" 2>/dev/null || true
  done
}

val_bytes() {
  local f="$COCO/val2014.zip"
  [[ -f "$f" ]] || { echo 0; return; }
  stat -c%s "$f"
}

val_ok() {
  local f="$COCO/val2014.zip"
  [[ -f "$f" ]] || return 1
  local s
  s=$(stat -c%s "$f")
  [[ "$s" -ge "$VAL_TARGET_BYTES" ]] || return 1
  if ! unzip -tqq "$f" >/dev/null 2>&1; then
    echo "[val] size OK but zip test FAILED (corrupt?) — keep aria2 running"
    return 1
  fi
  return 0
}

ann_ok() {
  local f="$COCO/annotations_trainval2014.zip"
  [[ -f "$f" ]] || return 1
  [[ $(stat -c%s "$f") -ge "$ANN_MIN_BYTES" ]]
}

aria2_val_running() {
  pgrep -af "aria2c.*val2014\.zip" >/dev/null 2>&1
}

ensure_val_download() {
  if val_ok; then
    echo "[val] OK ($(val_bytes) bytes, unzip -t passed)"
    return 0
  fi
  if aria2_val_running; then
    echo "[val] aria2 already running ($(val_bytes) / $VAL_TARGET_BYTES bytes)"
    return 1
  fi
  echo "[val] starting aria2 resume (no proxy for cocodataset.org)…"
  (
    close_inherited_singleton_fds
    nohup env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
      aria2c -x 24 -s 24 -c --file-allocation=none --summary-interval=60 \
      --max-tries=0 --retry-wait=5 \
      -d "$COCO" -o val2014.zip \
      http://images.cocodataset.org/zips/val2014.zip \
      >>"$WORKDIR/val2014_monitor.log" 2>&1 &
    apid=$!
    disown "$apid" 2>/dev/null || true
    echo "[val] aria2 pid=$apid"
  )
  return 1
}

ensure_ann() {
  if ann_ok; then
    echo "[ann] OK"
    return 0
  fi
  echo "[ann] downloading…"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    aria2c -x 24 -s 24 -c --file-allocation=none --summary-interval=30 \
    --max-tries=0 --retry-wait=5 \
    -d "$COCO" -o annotations_trainval2014.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  ann_ok
}

# 后台单实例拉取（flock）；成功写入 $LOCKDIR/hf_ok_<name>，供 all_hf_complete 判定
pull_hf_bg() {
  local name="$1" repo="$2" dest="$3"
  local lock="$LOCKDIR/hf_${name}.lock"
  local marker="$LOCKDIR/hf_ok_${name}"
  mkdir -p "$dest"
  (
    close_inherited_singleton_fds
    flock -n 200 || { echo "[$name] another pull holds lock — skip"; exit 0; }
    export HF_REPO="$repo" HF_DEST="$dest" HF_OK_MARKER="$marker" HF_MAX_WORKERS="$HF_MAX_WORKERS"
    echo "[$(date -Iseconds)] [$name] snapshot_download START $repo"
    python3 -u - <<'PY'
import os, sys, time

repo = os.environ["HF_REPO"]
dest = os.environ["HF_DEST"]
marker = os.environ["HF_OK_MARKER"]
max_workers = int(os.environ.get("HF_MAX_WORKERS", "6"))

os.makedirs(dest, exist_ok=True)

def run_once():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dest,
        max_workers=max_workers,
    )

last_err = None
for attempt in range(1, 4):
    try:
        run_once()
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        print("OK", repo, "->", dest)
        sys.exit(0)
    except Exception as e:
        last_err = e
        print("FAIL", attempt, repo, repr(e))
        if attempt < 3:
            time.sleep(min(30 * attempt, 120))
sys.exit(1)
PY
    ec=$?
    echo "[$(date -Iseconds)] [$name] snapshot_download END ec=$ec"
    exit "$ec"
  ) 200>"$lock" &
  echo "[${name}] background pid=$!"
}

# 限并发启动 HF 任务，避免 OOM / 连接风暴
hf_pull_all_limited() {
  local max="$HF_PARALLEL_MAX"
  local running=0
  local name repo dest
  while (($# >= 3)); do
    name=$1 repo=$2 dest=$3
    shift 3
    while ((running >= max)); do
      wait -n
      running=$((running - 1))
    done
    pull_hf_bg "$name" "$repo" "$dest"
    running=$((running + 1))
  done
  wait || true
}

vllm_ok() {
  [[ -d "$WORKDIR/third_party/vllm/.git" ]]
}

ensure_vllm() {
  vllm_ok && { echo "[vllm] OK"; return 0; }
  mkdir -p "$WORKDIR/third_party"
  git clone --depth 1 https://github.com/vllm-project/vllm.git "$WORKDIR/third_party/vllm" || true
  vllm_ok
}

unzip_val_if_needed() {
  val_ok || return 0
  if [[ ! -d "$COCO/val2014" ]] || [[ $(find "$COCO/val2014" -maxdepth 1 -type f 2>/dev/null | wc -l) -lt 1000 ]]; then
    echo "[unzip] val2014/ …"
    unzip -q -o "$COCO/val2014.zip" -d "$COCO/"
  fi
  if ann_ok; then
    unzip -q -o "$COCO/annotations_trainval2014.zip" -d "$COCO/" || true
  fi
}

# 仅以「当次 snapshot_download 成功」写入的标记为准，避免文件数阈值误判
all_hf_complete() {
  [[ -f "$LOCKDIR/hf_ok_HallusionBench" ]] || return 1
  [[ -f "$LOCKDIR/hf_ok_MVBench" ]] || return 1
  [[ -f "$LOCKDIR/hf_ok_VidHalluc" ]] || return 1
  [[ -f "$LOCKDIR/hf_ok_MMMU" ]] || return 1
  [[ -f "$LOCKDIR/hf_ok_FREAK" ]] || return 1
  return 0
}

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo ""
  echo "########## CYCLE $cycle @ $(date -Iseconds) ##########"

  ensure_ann || true
  ensure_val_download || true

  hf_pull_all_limited \
    "HallusionBench" "lmms-lab/HallusionBench" "$DS/HallusionBench_hf" \
    "MVBench" "OpenGVLab/MVBench" "$VID/OpenGVLab_MVBench" \
    "VidHalluc" "chaoyuli/VidHalluc" "$VID/VidHalluc_VidHalluc" \
    "MMMU" "MMMU/MMMU" "$DS/MMMU_hf" \
    "FREAK" "hansQAQ/FREAK" "$DS/FREAK_hf"

  ensure_vllm || true

  if val_ok && ann_ok && all_hf_complete && vllm_ok; then
    unzip_val_if_needed
    echo "[$(date -Iseconds)] ✅ ALL DATASETS COMPLETE"
    date -Iseconds >"$WORKDIR/BRA_DOWNLOAD_COMPLETE.txt"
    du -sh "$DS"/* "$WORKDIR/third_party/vllm" 2>/dev/null || true
    echo "Marker: $WORKDIR/BRA_DOWNLOAD_COMPLETE.txt"
    exit 0
  fi

  echo "[$(date -Iseconds)] cycle $cycle: not finished yet — sleep ${SLEEP_SEC}s"
  echo "  val_bytes=$(val_bytes) / $VAL_TARGET_BYTES"
  echo "  hf_ok markers: $(ls -1 "$LOCKDIR"/hf_ok_* 2>/dev/null | xargs -r basename -a | tr '\n' ' ')"
  sleep "$SLEEP_SEC"
done
