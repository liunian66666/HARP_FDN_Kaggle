#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

ROOT="/kaggle/working"
cd "${ROOT}"

: "${DATA_NAME:?DATA_NAME is required}"
: "${PRED_LEN:?PRED_LEN is required}"
: "${CANDIDATES:?CANDIDATES is required}"

MODEL_NAME="HARP_FDN_optimized"
LOG_DIR="logs/HARP_FDN_optimized_target_ETT"
SUMMARY="${LOG_DIR}/summary.csv"
mkdir -p "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
SEQ_LEN="${SEQ_LEN:-96}"
ENC_IN="${ENC_IN:-7}"
MA_TYPE="${MA_TYPE:-ema}"
ALPHA="${ALPHA:-0.3}"
BETA="${BETA:-0.3}"
# 默认双卡，也可外部覆盖
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

if [ ! -f "${SUMMARY}" ]; then
  echo "timestamp,data,pred_len,config_id,batch,learning_rate,train_objective,mse_loss_weight,vali_objective,con_cls_1,con_cls_2,con_time,relative_strength_init,posterior_calib_strength_init,correction_strength_init,mixer_d_model,mixer_dropout,support_temperature,use_posterior_calib,mse,mae,log_file" > "${SUMMARY}"
fi

is_done() {
  local cfg="$1"
  if [ "${FORCE:-0}" = "1" ]; then return 1; fi
  awk -F, -v d="${DATA_NAME}" -v p="${PRED_LEN}" -v c="$cfg" 'NR>1 && $2==d && $3==p && $4==c && $20!="NA" {found=1} END{exit found?0:1}' "${SUMMARY}"
}

run_candidate() {
  local cfg="$1" obj="$2" msew="$3" vali="$4" c1="$5" c2="$6" ct="$7" rel="$8" pcal="$9" corr="${10}" dmodel="${11}" drop="${12}" temp="${13}" use_pcal="${14}"
  if is_done "$cfg"; then
    echo "Skip done: ${DATA_NAME}-${PRED_LEN} ${cfg}"
    return 0
  fi

  local coarse_len=$((PRED_LEN / 4))
  local model_id="HARP_FDN_opt_TARGET_${DATA_NAME}_pl${PRED_LEN}_${cfg}"
  local log_file="${LOG_DIR}/${model_id}.log"

  echo "================================================================================"
  echo "TARGET ETT | data=${DATA_NAME} pred_len=${PRED_LEN} cfg=${cfg} batch=${BATCH_SIZE}"
  echo "obj=${obj} msew=${msew} vali=${vali} con=(${c1},${c2},${ct}) rel=${rel} pcal=${pcal} corr=${corr} d=${dmodel} drop=${drop} temp=${temp} use_pcal=${use_pcal}"
  echo "================================================================================"

  set +e
  python -u run_patch0608.py     --is_training 1     --root_path ./dataset/ETT-small/     --data_path "${DATA_NAME}.csv"     --model_id "${model_id}"     --model "${MODEL_NAME}"     --data "${DATA_NAME}"     --features M     --seq_len "${SEQ_LEN}"     --pred_len "${PRED_LEN}"     --enc_in "${ENC_IN}"     --des TargetMSE     --itr 1     --train_epochs "${TRAIN_EPOCHS}"     --patience "${PATIENCE}"     --batch_size "${BATCH_SIZE}"     --learning_rate "${LEARNING_RATE}"     --lradj sigmoid     --train_objective "${obj}"     --mse_loss_weight "${msew}"     --vali_objective "${vali}"     --use_vali_ratio 0     --con_cls_1 "${c1}"     --con_cls_2 "${c2}"     --con_time "${ct}"     --ma_type "${MA_TYPE}"     --alpha "${ALPHA}"     --beta "${BETA}"     --patch_len 16     --stride 8     --padding_patch end     --support_temperature "${temp}"     --support_path idx2.xlsx     --num_support 25     --coarse_len "${coarse_len}"     --use_relative_decode 1     --use_distribution_transport 1     --use_uncertainty_fusion 1     --use_posterior_calib "${use_pcal}"     --relative_strength_init "${rel}"     --relative_residual_scale_init 0.5     --transport_var_weight 0.05     --posterior_calib_strength_init "${pcal}"     --use_horizon_segment 1     --use_residual_correction 1     --use_spectral_transport 1     --correction_strength_init "${corr}"     --use_dynamic_support 1     --mixer_d_model "${dmodel}"     --mixer_layers 2     --mixer_dropout "${drop}"     --use_checkpoint 1     --light_head 1     --resume 0 \
  # ========== 新增多卡参数 ==========
  --use_gpu 1 \
  --use_multi_gpu 1 \
  --device_ids 0,1 \
  # =================================
  2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "FAILED ${DATA_NAME}-${PRED_LEN} ${cfg} status=${status}"
    return "${status}"
  fi

  local metric_line mse mae
  metric_line=$(grep -E 'mse:[0-9.eE+-]+, mae:[0-9.eE+-]+' "${log_file}" | tail -1 || true)
  mse="NA"; mae="NA"
  if [ -n "${metric_line}" ]; then
    mse=$(echo "${metric_line}" | sed -E 's/.*mse:([^,]+), mae:.*/\1/')
    mae=$(echo "${metric_line}" | sed -E 's/.*mae:([^ ]+).*/\1/')
  fi
  echo "$(date '+%F %T'),${DATA_NAME},${PRED_LEN},${cfg},${BATCH_SIZE},${LEARNING_RATE},${obj},${msew},${vali},${c1},${c2},${ct},${rel},${pcal},${corr},${dmodel},${drop},${temp},${use_pcal},${mse},${mae},${log_file}" >> "${SUMMARY}"
  echo "Recorded ${DATA_NAME}-${PRED_LEN} ${cfg}: mse=${mse} mae=${mae}"
}

while IFS='|' read -r cfg obj msew vali c1 c2 ct rel pcal corr dmodel drop temp use_pcal; do
  [ -z "${cfg}" ] && continue
  case "${cfg}" in \#*) continue ;; esac
  run_candidate "$cfg" "$obj" "$msew" "$vali" "$c1" "$c2" "$ct" "$rel" "$pcal" "$corr" "$dmodel" "$drop" "$temp" "$use_pcal"
done <<< "${CANDIDATES}"

python3 - <<PY2
import csv
from pathlib import Path
p = Path('${SUMMARY}')
rows = [r for r in csv.DictReader(p.open()) if r['data']=='${DATA_NAME}' and r['pred_len']=='${PRED_LEN}' and r['mse'] not in {'', 'NA'}]
if rows:
    best = min(rows, key=lambda r: float(r['mse']))
    print(f"Best target ${DATA_NAME}-${PRED_LEN}: {best['config_id']} mse={float(best['mse']):.6f} mae={float(best['mae']):.6f}")
PY2