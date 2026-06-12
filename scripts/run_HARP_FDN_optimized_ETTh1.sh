#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN

mkdir -p logs/HARP_FDN_optimized

model_name=HARP_FDN_optimized
base_id=HARP_FDN_optimized

ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

root_path="./dataset/ETT-small/"
data_path="ETTh1.csv"
data_name="ETTh1"
enc_in=7

run_mix40() {
    local pred_len="$1"
    local batch="$2"
    local coarse_len=$((pred_len / 4))

    echo "====================================="
    echo "ETTh1 | pred_len=${pred_len} | optimized MIX40 | batch=${batch}"
    echo "====================================="

    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${base_id}_MIX40_ETTh1_${seq_len}_${pred_len}_${ma_type}" \
        --model "${model_name}" \
        --data "${data_name}" \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in}" \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 100 \
        --patience 10 \
        --batch_size "${batch}" \
        --learning_rate 0.0001 \
        --lradj 'sigmoid' \
        --train_objective mix \
        --mse_loss_weight 4.0 \
        --vali_objective mae \
        --use_vali_ratio 0 \
        --con_cls_1 0.05 \
        --con_cls_2 0.05 \
        --con_time 0.1 \
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
        --relative_strength_init -2.5 \
        --relative_residual_scale_init 0.5 \
        --transport_var_weight 0.05 \
        --posterior_calib_strength_init -3.0 \
        --use_horizon_segment 1 \
        --use_residual_correction 1 \
        --use_spectral_transport 1 \
        --correction_strength_init -2.2 \
        --use_dynamic_support 1 \
        --mixer_d_model 64 \
        --mixer_layers 2 \
        --mixer_dropout 0.1 \
        --use_checkpoint 1 \
        --light_head 1 \
        2>&1 | tee "logs/HARP_FDN_optimized/${base_id}_MIX40_ETTh1_${seq_len}_${pred_len}.log"
}

run_720() {
    echo "====================================="
    echo "ETTh1 | pred_len=720 | optimized WEAKCON_STABLE | batch=1024"
    echo "====================================="

    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${base_id}_WEAKCON_STABLE_ETTh1_${seq_len}_720_${ma_type}" \
        --model "${model_name}" \
        --data "${data_name}" \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len 720 \
        --enc_in "${enc_in}" \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 100 \
        --patience 10 \
        --batch_size 1024 \
        --learning_rate 0.0001 \
        --lradj 'sigmoid' \
        --train_objective mae \
        --mse_loss_weight 0.5 \
        --vali_objective mse \
        --use_vali_ratio 0 \
        --con_cls_1 0.01 \
        --con_cls_2 0.01 \
        --con_time 0.02 \
        --ma_type "${ma_type}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --patch_len 16 \
        --stride 8 \
        --padding_patch end \
        --support_temperature 1.0 \
        --support_path idx2.xlsx \
        --num_support 25 \
        --coarse_len 180 \
        --use_relative_decode 1 \
        --use_distribution_transport 1 \
        --use_uncertainty_fusion 1 \
        --use_posterior_calib 1 \
        --relative_strength_init -4.0 \
        --relative_residual_scale_init 0.5 \
        --transport_var_weight 0.05 \
        --posterior_calib_strength_init -6.0 \
        --use_horizon_segment 1 \
        --use_residual_correction 1 \
        --use_spectral_transport 1 \
        --correction_strength_init -5.0 \
        --use_dynamic_support 1 \
        --mixer_d_model 64 \
        --mixer_layers 2 \
        --mixer_dropout 0.1 \
        --use_checkpoint 1 \
        --light_head 1 \
        2>&1 | tee "logs/HARP_FDN_optimized/${base_id}_WEAKCON_STABLE_ETTh1_${seq_len}_720.log"
}

run_mix40 96 1024
run_mix40 192 1024
run_mix40 336 1024
run_720
