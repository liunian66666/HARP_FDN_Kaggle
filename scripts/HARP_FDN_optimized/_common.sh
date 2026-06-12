#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/HARP_FDN"
cd "${ROOT}"

: "${DATA_TAG:?DATA_TAG is required}"
: "${DATA_ARG:?DATA_ARG is required}"
: "${ROOT_PATH:?ROOT_PATH is required}"
: "${DATA_PATH:?DATA_PATH is required}"
: "${ENC_IN:?ENC_IN is required}"
: "${SEQ_LEN:?SEQ_LEN is required}"
: "${LABEL_LEN:?LABEL_LEN is required}"

MODEL_NAME="HARP_FDN_optimized"
BASE_ID="${BASE_ID:-HARP_FDN_opt_${DATA_TAG}}"
LOG_DIR="${LOG_DIR:-logs/HARP_FDN_optimized/${DATA_TAG}}"
SUMMARY="${LOG_DIR}/summary.csv"

MA_TYPE="${MA_TYPE:-ema}"
ALPHA="${ALPHA:-0.3}"
BETA="${BETA:-0.3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-10}"
DMODEL_SHORT="${DMODEL_SHORT:-64}"
DMODEL_LONG="${DMODEL_LONG:-64}"
MIXER_LAYERS="${MIXER_LAYERS:-2}"
PATCH_LEN="${PATCH_LEN:-16}"
STRIDE="${STRIDE:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

mkdir -p "${LOG_DIR}"
if [ ! -f "${SUMMARY}" ]; then
    echo "timestamp,data_tag,data_arg,pred_len,config_id,batch,learning_rate,train_objective,mse_loss_weight,vali_objective,con_cls_1,con_cls_2,con_time,relative_strength_init,posterior_calib_strength_init,correction_strength_init,mixer_d_model,mixer_dropout,support_temperature,use_posterior_calib,mse,mae,log_file" > "${SUMMARY}"
fi

is_done() {
    local pred_len="$1"
    if [ "${FORCE:-0}" = "1" ]; then
        return 1
    fi
    awk -F, -v p="${pred_len}" 'NR > 1 && $4 == p && $21 != "NA" { found = 1 } END { exit found ? 0 : 1 }' "${SUMMARY}"
}

profile_for_pred() {
    local pred_len="$1"

    if [ "${pred_len}" = "720" ]; then
        config_id="mae_weak_d${DMODEL_LONG}_drop05"
        train_objective="mae"
        mse_weight="0.5"
        vali_objective="mse"
        con1="0.01"
        con2="0.01"
        cont="0.02"
        rel="-4.0"
        pcal="-6.0"
        corr="-5.0"
        dmodel="${DMODEL_LONG}"
        dropout="0.05"
        support_temp="1.0"
        posterior_calib="1"
    else
        config_id="mix15_d${DMODEL_SHORT}_drop05"
        train_objective="mix"
        mse_weight="1.5"
        vali_objective="mae"
        con1="0.015"
        con2="0.015"
        cont="0.03"
        rel="-3.5"
        pcal="-5.0"
        corr="-4.0"
        dmodel="${DMODEL_SHORT}"
        dropout="0.05"
        support_temp="1.0"
        posterior_calib="1"
    fi
}

run_one() {
    local pred_len="$1"
    local batch="$2"

    profile_for_pred "${pred_len}"

    local coarse_len=$((pred_len / 4))
    local model_id="${BASE_ID}_${config_id}_${DATA_TAG}_sl${SEQ_LEN}_pl${pred_len}_b${batch}_${MA_TYPE}"
    local log_file="${LOG_DIR}/${model_id}.log"

    echo "================================================================"
    echo "HARP_FDN_optimized | ${DATA_TAG} | pred_len=${pred_len} | batch=${batch} | config=${config_id}"
    echo "lr=${LEARNING_RATE} obj=${train_objective} mse_w=${mse_weight} con=(${con1},${con2},${cont}) d=${dmodel} drop=${dropout}"
    echo "================================================================"

    set +e
    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --model_id "${model_id}" \
        --model "${MODEL_NAME}" \
        --data "${DATA_ARG}" \
        --features M \
        --seq_len "${SEQ_LEN}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --enc_in "${ENC_IN}" \
        --des "${DATA_TAG}" \
        --itr 1 \
        --train_epochs "${TRAIN_EPOCHS}" \
        --patience "${PATIENCE}" \
        --batch_size "${batch}" \
        --learning_rate "${LEARNING_RATE}" \
        --lradj sigmoid \
        --num_workers "${NUM_WORKERS}" \
        --train_objective "${train_objective}" \
        --mse_loss_weight "${mse_weight}" \
        --vali_objective "${vali_objective}" \
        --use_vali_ratio 0 \
        --con_cls_1 "${con1}" \
        --con_cls_2 "${con2}" \
        --con_time "${cont}" \
        --ma_type "${MA_TYPE}" \
        --alpha "${ALPHA}" \
        --beta "${BETA}" \
        --patch_len "${PATCH_LEN}" \
        --stride "${STRIDE}" \
        --padding_patch end \
        --support_temperature "${support_temp}" \
        --support_path idx2.xlsx \
        --num_support 25 \
        --coarse_len "${coarse_len}" \
        --use_relative_decode 1 \
        --use_distribution_transport 1 \
        --use_uncertainty_fusion 1 \
        --use_posterior_calib "${posterior_calib}" \
        --relative_strength_init "${rel}" \
        --relative_residual_scale_init 0.5 \
        --transport_var_weight 0.05 \
        --posterior_calib_strength_init "${pcal}" \
        --use_horizon_segment 1 \
        --use_residual_correction 1 \
        --use_spectral_transport 1 \
        --correction_strength_init "${corr}" \
        --use_dynamic_support 1 \
        --mixer_d_model "${dmodel}" \
        --mixer_layers "${MIXER_LAYERS}" \
        --mixer_dropout "${dropout}" \
        --use_checkpoint 1 \
        --light_head 1 \
        --resume 0 \
        2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "FAILED ${DATA_TAG} pred_len=${pred_len} batch=${batch}; trying next batch if available."
        return "${status}"
    fi

    local metric_line
    metric_line=$(grep -E 'mse:[0-9.eE+-]+, mae:[0-9.eE+-]+' "${log_file}" | tail -1 || true)
    local mse="NA"
    local mae="NA"
    if [ -n "${metric_line}" ]; then
        mse=$(echo "${metric_line}" | sed -E 's/.*mse:([^,]+), mae:.*/\1/')
        mae=$(echo "${metric_line}" | sed -E 's/.*mae:([^ ]+).*/\1/')
    fi

    echo "$(date '+%F %T'),${DATA_TAG},${DATA_ARG},${pred_len},${config_id},${batch},${LEARNING_RATE},${train_objective},${mse_weight},${vali_objective},${con1},${con2},${cont},${rel},${pcal},${corr},${dmodel},${dropout},${support_temp},${posterior_calib},${mse},${mae},${log_file}" >> "${SUMMARY}"
    echo "Recorded ${DATA_TAG} pred_len=${pred_len}: mse=${mse} mae=${mae}"
}

run_pred() {
    local pred_len="$1"

    if is_done "${pred_len}"; then
        echo "Skip completed: ${DATA_TAG} pred_len=${pred_len}. Set FORCE=1 to rerun."
        return 0
    fi

    local candidates="${BATCH_CANDIDATES[${pred_len}]:-${DEFAULT_BATCHES:-32 16 8 4 2}}"
    local batch
    for batch in ${candidates}; do
        if run_one "${pred_len}" "${batch}"; then
            return 0
        fi
        sleep 10
    done

    echo "FAILED_ALL ${DATA_TAG} pred_len=${pred_len}"
    return 1
}

main() {
    local preds="${RUN_PRED_LENS:-96 192 336 720}"
    local pred_len
    for pred_len in ${preds}; do
        run_pred "${pred_len}"
    done

    echo
    echo "Summary: ${SUMMARY}"
    tail -n +1 "${SUMMARY}"
}

main "$@"
