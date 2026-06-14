#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_TAG="exchange"
DATA_ARG="custom"
ROOT_PATH="./dataset/exchange/"
DATA_PATH="exchange_rate.csv"
ENC_IN=8
SEQ_LEN=96
LABEL_LEN=48
DMODEL_SHORT=96
DMODEL_LONG=64

declare -A BATCH_CANDIDATES=(
    [96]="128 64 32 16 8"
    [192]="128 64 32 16 8"
    [336]="128 64 32 16 8"
    [720]="128 64 32 16 8"
)

source "${SCRIPT_DIR}/_common.sh"
