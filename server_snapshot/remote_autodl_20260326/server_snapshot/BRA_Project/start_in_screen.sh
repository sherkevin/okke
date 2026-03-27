#!/usr/bin/env bash
# 在 screen 会话中启动下载/监控，SSH 断开后任务继续跑；可随时 screen -r 回到会话。
# 用法:
#   bash start_in_screen.sh              # 默认：monitor_until_done.sh
#   bash start_in_screen.sh monitor
#   bash start_in_screen.sh download_all
# 环境变量:
#   SESSION_NAME=bra_mon  自定义 screen 会话名（避免冲突）
set -euo pipefail

WORKDIR="/root/autodl-tmp/BRA_Project"
MODE="${1:-monitor}"

case "$MODE" in
  monitor) SCRIPT="monitor_until_done.sh" ;;
  download_all) SCRIPT="download_all.sh" ;;
  *)
    echo "用法: $0 [monitor|download_all]"
    exit 1
    ;;
esac

SESSION="${SESSION_NAME:-bra_${MODE}}"

if ! command -v screen >/dev/null 2>&1; then
  echo "[ERR] 未安装 screen。请先: apt-get update && apt-get install -y screen"
  echo "      或改用: nohup bash $WORKDIR/$SCRIPT >/dev/null 2>&1 &"
  exit 1
fi

if screen -list 2>/dev/null | grep -qE "[0-9]+\.${SESSION}[[:space:]]"; then
  echo "[WARN] screen 会话已存在: $SESSION"
  echo "       附着查看: screen -r $SESSION"
  echo "       若需重建: screen -S $SESSION -X quit   # 会先结束里面的任务，慎用"
  exit 1
fi

# 用 env 传参，避免引号嵌套；内层 bash 负责 network_turbo / proxy_on
screen -dmS "$SESSION" env \
  BRA_W="$WORKDIR" \
  BRA_S="$SCRIPT" \
  bash -c '
set -euo pipefail
cd "$BRA_W"
if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi
if declare -F proxy_on >/dev/null 2>&1; then
  proxy_on true
fi
echo "[$(date -Iseconds)] screen session='"$SESSION"' script=$BRA_S pid=$$"
echo "日志: monitor → $BRA_W/monitor_until_done.log ；download_all → $BRA_W/download_all.log"
exec bash ./"$BRA_S"
'

echo "[OK] 已在 screen 中启动: session=$SESSION  script=$SCRIPT"
echo "     附着会话:  screen -r $SESSION"
echo "     列出会话:  screen -ls"
echo "     脱离会话:  Ctrl+A 然后按 D"
echo ""
echo "说明: monitor_until_done.sh 会把输出追加到 monitor_until_done.log，"
echo "      附着后若窗口较安静，请执行: tail -f $WORKDIR/monitor_until_done.log"
