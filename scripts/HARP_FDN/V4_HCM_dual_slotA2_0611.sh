#!/usr/bin/env bash
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_PCX
export CUDA_VISIBLE_DEVICES=0

run_cfg() {
  local tag="$1"
  local lr="$2"
  local msew="$3"
  local fuw="$4"
  local cls="$5"
  local strength="$6"
  local log="logs/HARP_FDN_PCX/${tag}_ETTh1_96_96.log"
  echo "===== $(date '+%F %T') START ${tag} lr=${lr} msew=${msew} fuw=${fuw} cls=${cls} strength=${strength} =====" | tee "$log"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ${tag}_ETTh1_96_96_ema \
    --model HARP_FDN_PCX_V4 \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --des Exp \
    --itr 1 \
    --train_epochs 100 \
    --patience 10 \
    --batch_size 1024 \
    --learning_rate ${lr} \
    --lradj sigmoid \
    --train_objective mix \
    --mse_loss_weight ${msew} \
    --vali_objective mae \
    --use_vali_ratio 0 \
    --con_cls_1 ${cls} \
    --con_cls_2 ${cls} \
    --con_time 0.1 \
    --ma_type ema \
    --alpha 0.3 \
    --beta 0.3 \
    --support_temperature 1.0 \
    --support_path idx2.xlsx \
    --num_support 25 \
    --use_relative_decode 1 \
    --use_distribution_transport 1 \
    --use_uncertainty_fusion 1 \
    --use_posterior_calib 1 \
    --relative_strength_init -8.0 \
    --relative_residual_scale_init 0.5 \
    --transport_var_weight 0.0 \
    --posterior_calib_strength_init -8.0 \
    --residual_support_scale 0.2 \
    --horizon_gate_strength 1.0 \
    --horizon_gate_bias_init 0.0 \
    --use_residual_support 1 \
    --use_conditional_residual 1 \
    --conditional_residual_strength_init -8.0 \
    --use_conditional_residual_logits 0 \
    --use_direct_adapter 0 \
    --use_trend_anchor 0 \
    --use_horizon_affine 0 \
    --fusion_uncertainty_weight ${fuw} \
    --use_horizon_candidate_mixer 1 \
    --candidate_mixer_modes last \
    --candidate_mixer_strength_init ${strength} \
    --candidate_mixer_max_gate 0.08 \
    --candidate_mixer_use_uncertainty 0 \
    --candidate_mixer_apply_in_train 0 2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') END ${tag} =====" | tee -a "$log"
}

run_cfg V4_HCM_A2_lr91_mix9_s355_fuw0975_cls095 0.00091 9.0 0.975 0.095 -3.55
run_cfg V4_HCM_A2_lr925_mix9_s355_fuw0975_cls095 0.000925 9.0 0.975 0.095 -3.55
run_cfg V4_HCM_A2_lr93_mix9_s355_fuw0975_cls095 0.00093 9.0 0.975 0.095 -3.55
run_cfg V4_HCM_A2_lr94_mix9_s355_fuw0975_cls095 0.00094 9.0 0.975 0.095 -3.55
