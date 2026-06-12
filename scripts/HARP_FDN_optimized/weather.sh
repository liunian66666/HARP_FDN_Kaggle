#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_TAG="weather"
DATA_ARG="custom"
ROOT_PATH="./dataset/weather/"
DATA_PATH="weather.csv"
ENC_IN=21
SEQ_LEN=96
LABEL_LEN=48
DMODEL_SHORT=96
DMODEL_LONG=64

declare -A BATCH_CANDIDATES=(
    [96]="256 128 64"
    [192]="256 128 64"
    [336]="128 96 64 32"
    [720]="64 48 32 16"
)

source "${SCRIPT_DIR}/_common.sh"
