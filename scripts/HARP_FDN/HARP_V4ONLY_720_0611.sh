#!/usr/bin/env bash
set -u
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_PCX
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_720() {
  local tag="$1"
  local lr="$2"
  local obj="$3"
  local vali="$4"
  local msew="$5"
  local c1="$6"
  local c2="$7"
  local ct="$8"
  local rel="$9"
  local post="${10}"
  local use_rel="${11}"
  local cond="${12}"
  local rs="${13}"
  local hcm="${14}"
  local strength="${15}"
  local log="logs/HARP_FDN_PCX/${tag}.log"
  echo "===== $(date '+%F %T') START ${tag} =====" | tee "$log"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ${tag} \
    --model HARP_FDN_PCX_V4 \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --enc_in 7 \
    --des Exp \
    --itr 1 \
    --train_epochs 100 \
    --patience 10 \
    --batch_size 1024 \
    --learning_rate ${lr} \
    --lradj sigmoid \
    --train_objective ${obj} \
    --mse_loss_weight ${msew} \
    --vali_objective ${vali} \
    --use_vali_ratio 0 \
    --con_cls_1 ${c1} \
    --con_cls_2 ${c2} \
    --con_time ${ct} \
    --ma_type ema \
    --alpha 0.3 \
    --beta 0.3 \
    --support_temperature 1.0 \
    --support_path idx2.xlsx \
    --num_support 25 \
    --use_relative_decode ${use_rel} \
    --use_distribution_transport 1 \
    --use_uncertainty_fusion 1 \
    --use_posterior_calib 1 \
    --relative_strength_init=${rel} \
    --relative_residual_scale_init 0.5 \
    --transport_var_weight 0.0 \
    --posterior_calib_strength_init=${post} \
    --residual_support_scale 0.2 \
    --horizon_gate_strength 1.0 \
    --horizon_gate_bias_init 0.0 \
    --use_residual_support ${rs} \
    --use_conditional_residual ${cond} \
    --conditional_residual_strength_init=-8.0 \
    --use_conditional_residual_logits 0 \
    --use_direct_adapter 0 \
    --use_trend_anchor 0 \
    --use_horizon_affine 0 \
    --fusion_uncertainty_weight 0.975 \
    --use_horizon_candidate_mixer ${hcm} \
    --candidate_mixer_modes last \
    --candidate_mixer_strength_init=${strength} \
    --candidate_mixer_max_gate 0.04 \
    --candidate_mixer_use_uncertainty 0 \
    --candidate_mixer_apply_in_train 0 \
    2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') END ${tag} =====" | tee -a "$log"
}

# V4-only 720 long-horizon configurations.
run_720 V4ONLY720_A_mae_valmse_weakcon_rel 0.00010 mae mse 0.5 0.01 0.01 0.02 -4.0 -6.0 1 1 1 0 -5.0
run_720 V4ONLY720_B_mse_valmse_interlike 0.00010 mse mse 1.0 0.01 0.01 0.02 -2.5 -6.0 0 0 0 0 -5.0
run_720 V4ONLY720_C_mix05_valmse_interlike 0.00010 mix mse 0.5 0.01 0.01 0.02 -2.5 -6.0 0 0 0 0 -5.0
run_720 V4ONLY720_D_mae_valmae_interlike 0.00010 mae mae 0.5 0.01 0.01 0.02 -2.5 -6.0 0 0 0 0 -5.0
run_720 V4ONLY720_E_mix1_valmse_relWeak 0.00015 mix mse 1.0 0.01 0.01 0.02 -6.0 -8.0 1 1 1 0 -5.0
