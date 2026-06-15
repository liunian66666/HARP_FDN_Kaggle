#!/usr/bin/env bash
set -euo pipefail

# 自动判断运行环境
if [[ "$(uname -s)" == *MINGW* ]]; then
    PROJECT_ROOT="D:/曹浩冉/HARP_FDN_Kaggle"
else
    PROJECT_ROOT="/kaggle/working"
fi

TASK_SCRIPTS_DIR="${PROJECT_ROOT}/scripts/HARP_FDN_optimized/ETT_targeted"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "开始串行执行指定脚本"
echo "============================================="

# 手动按顺序执行，想跳过哪个就不写哪一行
bash "${TASK_SCRIPTS_DIR}/search_ETTh2_336.sh"
bash "${TASK_SCRIPTS_DIR}/search_ETTm1_720.sh"
bash "${TASK_SCRIPTS_DIR}/search_ETTm2_96.sh"
bash "${TASK_SCRIPTS_DIR}/search_ETTm2_720.sh"

echo -e "\n✅ 所有指定脚本执行完毕"