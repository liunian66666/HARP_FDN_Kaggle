#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_TAG="Solar"
DATA_ARG="Solar"
ROOT_PATH="./dataset/Solar/"
DATA_PATH="solar_AL.txt"
ENC_IN=137
SEQ_LEN=96
LABEL_LEN=48
DMODEL_SHORT=64
DMODEL_LONG=64

declare -A BATCH_CANDIDATES=(
    [96]="96 64 32 16"
    [192]="96 64 32 16"
    [336]="64 48 32 16"
    [720]="48 32 16 8"
)

source "${SCRIPT_DIR}/_common.sh"
