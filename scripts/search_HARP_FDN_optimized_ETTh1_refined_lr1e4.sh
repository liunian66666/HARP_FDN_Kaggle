#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

log_dir="logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4"
mkdir -p "${log_dir}"

summary="${log_dir}/summary.csv"
if [ ! -f "${summary}" ]; then
    echo "timestamp,pred_len,config_id,batch,learning_rate,train_objective,mse_loss_weight,vali_objective,con_cls_1,con_cls_2,con_time,relative_strength_init,posterior_calib_strength_init,correction_strength_init,mixer_d_model,mixer_dropout,mse,mae,log_file" > "${summary}"
fi

model_name="HARP_FDN_optimized"
base_id="HARP_FDN_opt_REFINE_LR1e4_B1024"

data_name="ETTh1"
root_path="./dataset/ETT-small/"
data_path="ETTh1.csv"
enc_in=7

seq_len=96
batch=1024
lr=0.0001
ma_type=ema
alpha=0.3
beta=0.3

is_done() {
    local pred_len="$1"
    local config_id="$2"
    [ -f "${summary}" ] && awk -F, -v p="${pred_len}" -v c="${config_id}" 'NR > 1 && $2 == p && $3 == c && $17 != "NA" { found = 1 } END { exit found ? 0 : 1 }' "${summary}"
}

run_one() {
    local pred_len="$1"
    local config_id="$2"
    local train_objective="$3"
    local mse_weight="$4"
    local vali_objective="$5"
    local con1="$6"
    local con2="$7"
    local cont="$8"
    local rel="$9"
    local pcal="${10}"
    local corr="${11}"
    local dmodel="${12}"
    local dropout="${13}"

    local coarse_len=$((pred_len / 4))
    local model_id="${base_id}_${config_id}_${data_name}_${seq_len}_${pred_len}_${ma_type}"
    local log_file="${log_dir}/${model_id}.log"

    if is_done "${pred_len}" "${config_id}"; then
        echo "Skip completed: pred_len=${pred_len} config=${config_id}"
        return 0
    fi

    echo "================================================================"
    echo "ETTh1 refined lr=1e-4 | pred_len=${pred_len} | config=${config_id} | batch=${batch}"
    echo "obj=${train_objective} mse_w=${mse_weight} vali=${vali_objective} con=(${con1},${con2},${cont}) rel=${rel} pcal=${pcal} d=${dmodel} drop=${dropout}"
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
        --des 'Refine' \
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

    echo "$(date '+%F %T'),${pred_len},${config_id},${batch},${lr},${train_objective},${mse_weight},${vali_objective},${con1},${con2},${cont},${rel},${pcal},${corr},${dmodel},${dropout},${mse},${mae},${log_file}" >> "${summary}"
    echo "Recorded summary: pred_len=${pred_len} config=${config_id} mse=${mse} mae=${mae}"
}

run_pred() {
    local pred_len="$1"

    run_one "${pred_len}" r01_mix30_d96_drop05      mix 3.0 mae 0.03 0.03 0.06 -2.5 -3.0 -2.2 96 0.05
    run_one "${pred_len}" r02_mix25_d96_drop05      mix 2.5 mae 0.025 0.025 0.05 -2.75 -3.5 -2.5 96 0.05
    run_one "${pred_len}" r03_mix20_d96_drop05      mix 2.0 mae 0.02 0.02 0.04 -3.0 -4.0 -3.0 96 0.05
    run_one "${pred_len}" r04_mix15_d96_drop05      mix 1.5 mae 0.015 0.015 0.03 -3.5 -5.0 -4.0 96 0.05
    run_one "${pred_len}" r05_mae_weak_d96_drop05   mae 0.5 mse 0.01 0.01 0.02 -4.0 -6.0 -5.0 96 0.05
    run_one "${pred_len}" r06_mix30_d64_drop05      mix 3.0 mae 0.03 0.03 0.06 -2.5 -3.0 -2.2 64 0.05
    run_one "${pred_len}" r07_mix20_d64_drop05      mix 2.0 mae 0.02 0.02 0.04 -3.0 -4.0 -3.0 64 0.05
    run_one "${pred_len}" r08_mae_weak_d64_drop05   mae 0.5 mse 0.01 0.01 0.02 -4.0 -6.0 -5.0 64 0.05
}

for pred_len in ${PRED_LIST:-96 192 336 720}; do
    run_pred "${pred_len}"
done

python - <<'PY'
import csv
from pathlib import Path

summary = Path("logs/HARP_FDN_optimized_search_ETTh1_refined_lr1e4/summary.csv")
rows = list(csv.DictReader(summary.open()))
valid = [r for r in rows if r["mse"] != "NA"]
print("\nBest by pred_len:")
for pred in ["96", "192", "336", "720"]:
    candidates = [r for r in valid if r["pred_len"] == pred]
    if not candidates:
        print(f"pred_len={pred}: no valid results")
        continue
    best_mse = min(candidates, key=lambda r: float(r["mse"]))
    best_mae = min(candidates, key=lambda r: float(r["mae"]))
    print(
        f"pred_len={pred}: best_mse={best_mse['config_id']} mse={best_mse['mse']} mae={best_mse['mae']} | "
        f"best_mae={best_mae['config_id']} mse={best_mae['mse']} mae={best_mae['mae']}"
    )
PY
