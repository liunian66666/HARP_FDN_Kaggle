#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

log_dir="logs/HARP_FDN_optimized_bestMSE_ETT"
mkdir -p "${log_dir}"

summary="${log_dir}/summary.csv"
if [ ! -f "${summary}" ]; then
    echo "timestamp,data,pred_len,config_id,batch,learning_rate,train_objective,mse_loss_weight,vali_objective,con_cls_1,con_cls_2,con_time,relative_strength_init,posterior_calib_strength_init,correction_strength_init,mixer_d_model,mixer_dropout,mse,mae,log_file" > "${summary}"
fi

model_name="HARP_FDN_optimized"
base_id="HARP_FDN_opt_BESTMSE_B1024"

root_path="./dataset/ETT-small/"
seq_len=96
batch=1024
lr=0.0001
ma_type=ema
alpha=0.3
beta=0.3
enc_in=7

is_done() {
    local data_name="$1"
    local pred_len="$2"
    [ -f "${summary}" ] && awk -F, -v d="${data_name}" -v p="${pred_len}" 'NR > 1 && $2 == d && $3 == p && $18 != "NA" { found = 1 } END { exit found ? 0 : 1 }' "${summary}"
}

config_for_pred() {
    local pred_len="$1"
    case "${pred_len}" in
        96)
            config_id="best96_r04_mix15_d96_drop05"
            train_objective="mix"
            mse_weight="1.5"
            vali_objective="mae"
            con1="0.015"
            con2="0.015"
            cont="0.03"
            rel="-3.5"
            pcal="-5.0"
            corr="-4.0"
            dmodel="96"
            dropout="0.05"
            ;;
        192)
            config_id="best192_r01_mix30_d96_drop05"
            train_objective="mix"
            mse_weight="3.0"
            vali_objective="mae"
            con1="0.03"
            con2="0.03"
            cont="0.06"
            rel="-2.5"
            pcal="-3.0"
            corr="-2.2"
            dmodel="96"
            dropout="0.05"
            ;;
        336|720)
            config_id="best${pred_len}_r06_mix30_d64_drop05"
            train_objective="mix"
            mse_weight="3.0"
            vali_objective="mae"
            con1="0.03"
            con2="0.03"
            cont="0.06"
            rel="-2.5"
            pcal="-3.0"
            corr="-2.2"
            dmodel="64"
            dropout="0.05"
            ;;
        *)
            echo "Unsupported pred_len: ${pred_len}" >&2
            return 1
            ;;
    esac
}

run_one() {
    local data_name="$1"
    local pred_len="$2"

    if is_done "${data_name}" "${pred_len}"; then
        echo "Skip completed: data=${data_name} pred_len=${pred_len}"
        return 0
    fi

    config_for_pred "${pred_len}"

    local data_path="${data_name}.csv"
    local coarse_len=$((pred_len / 4))
    local model_id="${base_id}_${config_id}_${data_name}_${seq_len}_${pred_len}_${ma_type}"
    local log_file="${log_dir}/${model_id}.log"

    echo "================================================================"
    echo "BEST-MSE ETT | data=${data_name} | pred_len=${pred_len} | config=${config_id} | batch=${batch}"
    echo "lr=${lr} obj=${train_objective} mse_w=${mse_weight} con=(${con1},${con2},${cont}) d=${dmodel} drop=${dropout}"
    echo "================================================================"

    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${model_id}" \
        --model "${model_name}" \
        --data "${data_name}" \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in}" \
        --des 'BestMSE' \
        --itr 1 \
        --train_epochs 100 \
        --patience 10 \
        --batch_size "${batch}" \
        --learning_rate "${lr}" \
        --lradj 'sigmoid' \
        --train_objective "${train_objective}" \
        --mse_loss_weight "${mse_weight}" \
        --vali_objective "${vali_objective}" \
        --use_vali_ratio 0 \
        --con_cls_1 "${con1}" \
        --con_cls_2 "${con2}" \
        --con_time "${cont}" \
        --ma_type "${ma_type}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --patch_len 16 \
        --stride 8 \
        --padding_patch end \
        --support_temperature 1.0 \
        --support_path idx2.xlsx \
        --num_support 25 \
        --coarse_len "${coarse_len}" \
        --use_relative_decode 1 \
        --use_distribution_transport 1 \
        --use_uncertainty_fusion 1 \
        --use_posterior_calib 1 \
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
        --mixer_layers 2 \
        --mixer_dropout "${dropout}" \
        --use_checkpoint 1 \
        --light_head 1 \
        --resume 0 \
        2>&1 | tee "${log_file}"

    local metric_line
    metric_line=$(grep -E 'mse:[0-9.eE+-]+, mae:[0-9.eE+-]+' "${log_file}" | tail -1 || true)
    local mse="NA"
    local mae="NA"
    if [ -n "${metric_line}" ]; then
        mse=$(echo "${metric_line}" | sed -E 's/.*mse:([^,]+), mae:.*/\1/')
        mae=$(echo "${metric_line}" | sed -E 's/.*mae:([^ ]+).*/\1/')
    fi

    echo "$(date '+%F %T'),${data_name},${pred_len},${config_id},${batch},${lr},${train_objective},${mse_weight},${vali_objective},${con1},${con2},${cont},${rel},${pcal},${corr},${dmodel},${dropout},${mse},${mae},${log_file}" >> "${summary}"
    echo "Recorded summary: data=${data_name} pred_len=${pred_len} mse=${mse} mae=${mae}"
}

for data_name in ${DATA_LIST:-ETTh2 ETTm1 ETTm2}; do
    for pred_len in 96 192 336 720; do
        run_one "${data_name}" "${pred_len}"
    done
done

python - <<'PY'
import csv
from pathlib import Path

summary = Path("logs/HARP_FDN_optimized_bestMSE_ETT/summary.csv")
rows = [r for r in csv.DictReader(summary.open()) if r["mse"] != "NA"]
print("\nCompleted best-MSE runs:")
for data in ["ETTh2", "ETTm1", "ETTm2"]:
    for pred in ["96", "192", "336", "720"]:
        matches = [r for r in rows if r["data"] == data and r["pred_len"] == pred]
        if matches:
            r = matches[-1]
            print(f"{data} pred_len={pred}: mse={r['mse']} mae={r['mae']} config={r['config_id']}")
        else:
            print(f"{data} pred_len={pred}: pending")
PY
