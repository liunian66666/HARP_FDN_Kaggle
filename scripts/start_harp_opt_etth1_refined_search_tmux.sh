#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

session=harp_opt_etth1_refined

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SESSION_EXISTS:${session}"
    tmux list-windows -t "${session}"
    exit 0
fi

tmux new-session -d -s "${session}" -n worker0 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && PRED_LIST='96 336' bash scripts/search_HARP_FDN_optimized_ETTh1_refined_lr1e4.sh"

tmux new-window -t "${session}" -n worker1 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && PRED_LIST='192 720' bash scripts/search_HARP_FDN_optimized_ETTh1_refined_lr1e4.sh"

tmux new-window -t "${session}" -n gpu \
    "watch -n 30 nvidia-smi"

tmux new-window -t "${session}" -n summary \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && while true; do clear; echo Summary: logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4/summary.csv; echo; if [ -f logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4/summary.csv ]; then tail -20 logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4/summary.csv; else echo waiting for summary; fi; echo; latest=\$(ls -t logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4/*.log 2>/dev/null | head -1); if [ -n \"\$latest\" ]; then echo === \$latest ===; tail -60 \"\$latest\"; else echo waiting for logs; fi; sleep 30; done"

tmux select-window -t "${session}:worker0"
echo "STARTED:${session}"
