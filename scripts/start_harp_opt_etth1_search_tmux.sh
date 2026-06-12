#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

session=harp_opt_etth1_search

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SESSION_EXISTS:${session}"
    tmux list-windows -t "${session}"
    exit 0
fi

tmux new-session -d -s "${session}" -n search \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && bash scripts/search_HARP_FDN_optimized_ETTh1_batch1024.sh"

tmux new-window -t "${session}" -n gpu \
    "watch -n 30 nvidia-smi"

tmux new-window -t "${session}" -n summary \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && while true; do clear; echo Summary: logs/HARP_FDN_optimized_search_ETTh1/summary.csv; echo; if [ -f logs/HARP_FDN_optimized_search_ETTh1/summary.csv ]; then tail -20 logs/HARP_FDN_optimized_search_ETTh1/summary.csv; else echo waiting for summary; fi; echo; echo Latest log:; latest=\$(ls -t logs/HARP_FDN_optimized_search_ETTh1/*.log 2>/dev/null | head -1); if [ -n \"\$latest\" ]; then echo === \$latest ===; tail -60 \"\$latest\"; else echo waiting for logs; fi; sleep 30; done"

tmux select-window -t "${session}:search"
echo "STARTED:${session}"
