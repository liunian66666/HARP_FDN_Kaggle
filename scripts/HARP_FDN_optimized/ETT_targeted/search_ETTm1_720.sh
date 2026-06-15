#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_NAME="ETTm1"
PRED_LEN="720"
CANDIDATES='
t01_mae_d64_temp13|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.3|1
t02_mae_d64_temp07|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|0.7|1
t03_mix10_d64_temp13|mix|1.0|mae|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.3|1
t04_mix15_d64_temp13|mix|1.5|mae|0.015|0.015|0.03|-3.5|-5.0|-4.0|64|0.05|1.3|1
t05_mix10_d96_temp13|mix|1.0|mae|0.01|0.01|0.02|-4.0|-6.0|-5.0|96|0.05|1.3|1
t06_mae_d64_nocalib|mae|0.5|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.0|0
t07_mae_d64_conhalf|mae|0.5|mse|0.005|0.005|0.01|-4.5|-7.0|-5.0|64|0.05|1.0|1
t08_mae_d64_mseup|mae|1.0|mse|0.01|0.01|0.02|-4.0|-6.0|-5.0|64|0.05|1.0|1
'
source "${SCRIPT_DIR}/_target_ett_single_common.sh"
