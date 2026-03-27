#!/usr/bin/env bash
# 一次性打印当前模型/数据集体积与 .incomplete 数量（无需 screen）
ROOT="/root/autodl-tmp/BRA_Project"
LOGDIR="$ROOT/logs/hf_parallel"
echo "=== $(date -Iseconds) 数据盘 HF 并行任务状态 ==="
echo ""
printf "%-26s %12s %10s %s\n" "名称" "磁盘占用" "incomplete" "路径"
while IFS= read -r dest; do
  name=$(basename "$dest")
  if [[ -d "$dest" ]]; then
    sz=$(du -sh "$dest" 2>/dev/null | awk "{print \$1}")
    inc=$(find "$dest" -name "*.incomplete" 2>/dev/null | wc -l)
  else
    sz="(无目录)"
    inc="-"
  fi
  printf "%-26s %12s %10s %s\n" "$name" "$sz" "$inc" "$dest"
done <<'PATHS'
/root/autodl-tmp/BRA_Project/datasets/coco2014
/root/autodl-tmp/BRA_Project/models/MiniGPT-4-LLaMA-7B
/root/autodl-tmp/BRA_Project/models/llava-1.5-7b-hf
/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b
/root/autodl-tmp/BRA_Project/datasets/MMBench_EN_hf
/root/autodl-tmp/BRA_Project/datasets/MME_hf
/root/autodl-tmp/BRA_Project/datasets/FREAK_hf
/root/autodl-tmp/BRA_Project/datasets/MMMU_hf
/root/autodl-tmp/BRA_Project/datasets/video/chaoyuli_VidHalluc
/root/autodl-tmp/BRA_Project/datasets/video/OpenGVLab_MVBench
/root/autodl-tmp/BRA_Project/datasets/HallusionBench_hf
PATHS
echo ""
echo "screen 会话 (bra_hf_*):"
screen -ls 2>/dev/null | grep bra_hf || echo "  (无)"
echo ""
echo "并行日志目录: $LOGDIR"
ls -la "$LOGDIR" 2>/dev/null | tail -n +2 || echo "  (尚无)"
if [[ -f "$LOGDIR/master_progress.log" ]]; then
  echo ""
  echo "=== master_progress.log 末 30 行 ==="
  tail -n 30 "$LOGDIR/master_progress.log"
fi
