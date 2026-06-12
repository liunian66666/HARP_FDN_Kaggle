#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_TAG="Traffic"
DATA_ARG="custom"
ROOT_PATH="./dataset/traffic/"
DATA_PATH="traffic.csv"
ENC_IN=862
SEQ_LEN=96
LABEL_LEN=48
DMODEL_SHORT=64
DMODEL_LONG=64

declare -A BATCH_CANDIDATES=(
    [96]="24 16 8 4 2"
    [192]="24 16 8 4 2"
    [336]="16 8 4 2"
    [720]="8 4 2"
)

source "${SCRIPT_DIR}/_common.sh"
