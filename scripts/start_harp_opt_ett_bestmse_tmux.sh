#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

session=harp_opt_ett_bestmse

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SESSION_EXISTS:${session}"
    tmux list-windows -t "${session}"
    exit 0
fi

tmux new-session -d -s "${session}" -n worker0 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && DATA_LIST='ETTh2 ETTm2' bash scripts/run_HARP_FDN_optimized_ETT_bestMSE_batch1024.sh"

tmux new-window -t "${session}" -n worker1 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && DATA_LIST='ETTm1' bash scripts/run_HARP_FDN_optimized_ETT_bestMSE_batch1024.sh"

tmux new-window -t "${session}" -n gpu \
    "watch -n 30 nvidia-smi"

tmux new-window -t "${session}" -n summary \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && while true; do clear; echo Summary: logs/HARP_FDN_optimized_bestMSE_ETT/summary.csv; echo; if [ -f logs/HARP_FDN_optimized_bestMSE_ETT/summary.csv ]; then tail -20 logs/HARP_FDN_optimized_bestMSE_ETT/summary.csv; else echo waiting for summary; fi; echo; latest=\$(ls -t logs/HARP_FDN_optimized_bestMSE_ETT/*.log 2>/dev/null | head -1); if [ -n \"\$latest\" ]; then echo === \$latest ===; tail -60 \"\$latest\"; else echo waiting for logs; fi; sleep 30; done"

tmux select-window -t "${session}:worker0"
echo "STARTED:${session}"
