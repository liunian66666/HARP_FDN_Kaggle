#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

session=harp_opt_auto_mse_ett

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SESSION_EXISTS:${session}"
    tmux list-windows -t "${session}"
    exit 0
fi

tmux new-session -d -s "${session}" -n ETTh2 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && DATA_NAME=ETTh2 python -u scripts/auto_search_HARP_FDN_optimized_ETT_mse.py"

tmux new-window -t "${session}" -n ETTm1 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && DATA_NAME=ETTm1 python -u scripts/auto_search_HARP_FDN_optimized_ETT_mse.py"

tmux new-window -t "${session}" -n ETTm2 \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && DATA_NAME=ETTm2 python -u scripts/auto_search_HARP_FDN_optimized_ETT_mse.py"

tmux new-window -t "${session}" -n summary \
    "cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN && while true; do clear; echo Summary: logs/HARP_FDN_optimized_auto_mse_ETT/summary.csv; echo; if [ -f logs/HARP_FDN_optimized_auto_mse_ETT/summary.csv ]; then python - <<'PY'
import csv
from pathlib import Path
p = Path('logs/HARP_FDN_optimized_auto_mse_ETT/summary.csv')
rows = [r for r in csv.DictReader(p.open()) if r.get('mse') not in {'', 'NA', None}]
print(f'completed: {len(rows)}')
for data in ['ETTh2', 'ETTm1', 'ETTm2']:
    print('\\n' + data)
    for pred in ['96', '192', '336', '720']:
        xs = [r for r in rows if r['data'] == data and r['pred_len'] == pred]
        if xs:
            b = min(xs, key=lambda r: float(r['mse']))
            print(f'  {pred}: n={len(xs):02d} best={b[\"config_id\"]} mse={float(b[\"mse\"]):.6f} mae={float(b[\"mae\"]):.6f}')
        else:
            print(f'  {pred}: pending')
PY
else echo waiting for summary; fi; sleep 60; done"

tmux select-window -t "${session}:summary"
echo "STARTED:${session}"
