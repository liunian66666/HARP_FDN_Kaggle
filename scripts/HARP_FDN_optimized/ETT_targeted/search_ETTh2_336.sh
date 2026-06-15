#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_NAME="ETTh2"
PRED_LEN="336"
CANDIDATES='
t01_mae_d96_temp07|mae|0.5|mse|0.005|0.005|0.01|-4.5|-7.0|-5.0|96|0.05|0.7|1
t02_mae_d96_temp13|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|96|0.05|1.3|1
t03_mae_d64_temp13|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.3|1
t04_mix10_d96_temp13|mix|1.0|mae|0.01|0.01|0.02|-4.0|-6.0|-5.0|96|0.05|1.3|1
t05_mix10_d64_temp13|mix|1.0|mae|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.3|1
t06_mix05_d96_valmse|mix|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|96|0.05|1.0|1
t07_mae_d96_nocalib|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|96|0.05|1.0|0
t08_mix15_d96_drop10|mix|1.5|mae|0.015|0.015|0.03|-3.5|-5.0|-4.0|96|0.10|1.0|1
'
source "${SCRIPT_DIR}/_target_ett_single_common.sh"
