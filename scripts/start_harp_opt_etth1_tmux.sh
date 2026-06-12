#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

session=harp_opt_etth1

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SESSION_EXISTS:${session}"
    tmux list-windows -t "${session}"
    exit 0
fi

tmux new-session -d -s "${session}" -n train \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && bash scripts/run_HARP_FDN_optimized_ETTh1.sh"

tmux new-window -t "${session}" -n gpu \
    "watch -n 30 nvidia-smi"

tmux new-window -t "${session}" -n logs \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && while true; do clear; echo Logs:; ls -lt logs/HARP_FDN_optimized/*.log 2>/dev/null | head -8; echo; latest=\$(ls -t logs/HARP_FDN_optimized/*.log 2>/dev/null | head -1); if [ -n \"\$latest\" ]; then echo === \$latest ===; tail -80 \"\$latest\"; else echo waiting for logs; fi; sleep 30; done"

tmux select-window -t "${session}:train"
echo "STARTED:${session}"
