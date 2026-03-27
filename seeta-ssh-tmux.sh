#!/usr/bin/env bash
# SeetaCloud SSH inside tmux: reconnect with: tmux attach -t seetacloud
SESSION=seetacloud
HOST=root@connect.westd.seetacloud.com
PORT=23427

SSH_OPTS=(
  -p "$PORT"
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=3
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
else
  exec tmux new-session -s "$SESSION" -- ssh "${SSH_OPTS[@]}" "$HOST"
fi
