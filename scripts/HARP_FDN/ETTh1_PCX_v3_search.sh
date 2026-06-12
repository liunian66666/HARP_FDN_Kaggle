#!/usr/bin/env bash
set -uo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN
mkdir -p logs/HARP_FDN_PCX_v3

model_name=HARP_FDN_PCX_v3
base_id=HARP_FDN_PCX_v3_ETTh1
dataset=ETTh1
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96
preds="${PREDS:-96}"

run_one() {
  local pred="$1"
  local lr="$2"
  local msew="$3"
  local rel="$4"
  local residual_support="$5"
  local use_rel="$6"
  local use_cond="$7"
  local tag="$8"
  shift 8
  local extra_args=("$@")

  local model_id="${base_id}_${tag}_${seq_len}_${pred}_${ma_type}"
  local log="logs/HARP_FDN_PCX_v3/${model_id}.log"

  echo "model=${model_name} | dataset=${dataset} | pred_len=${pred} | tag=${tag}"
  python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${dataset}.csv \
    --model_id ${model_id} --model ${model_name} --data ${dataset} --features M \
    --seq_len ${seq_len} --label_len 48 --pred_len ${pred} --enc_in 7 --des Exp --itr 1 \
    --train_epochs 100 --patience 10 --batch_size 1024 --learning_rate ${lr} --lradj sigmoid \
    --loss mse --train_objective mix --mse_loss_weight ${msew} --vali_objective mae --use_vali_ratio 0 \
    --con_cls_1 0.05 --con_cls_2 0.05 --con_time 0.1 --ma_type ${ma_type} --alpha ${alpha} --beta ${beta} \
    --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --use_relative_decode ${use_rel} --use_distribution_transport 1 --use_uncertainty_fusion 1 --use_posterior_calib 1 \
    --relative_strength_init ${rel} --relative_residual_scale_init 0.5 --transport_var_weight 0.0 \
    --posterior_kl_weight 0.0 --residual_support_scale ${residual_support} --horizon_gate_strength 1.0 --horizon_gate_bias_init 0.0 \
    --use_conditional_residual 1 --conditional_residual_hidden_ratio 2 --conditional_residual_strength_init -8.0 \
    --use_conditional_residual_logits 0 --conditional_residual_logit_hidden_ratio 4 --conditional_residual_logit_strength_init -8.0 \
    --posterior_calib_strength_init -8.0 --use_horizon_segment 1 --use_residual_correction 1 --use_spectral_transport 1 \
    --num_horizon_segments 4 --transport_spectral_weight 0.03 --spectral_low_ratio 0.5 --correction_strength_init -2.2 \
    --correction_dropout 0.05 --use_dynamic_support 1 --dynamic_support_segments 4 --support_shift_limit 0.35 --support_scale_limit 0.35 \
    --mixer_d_model 96 --mixer_layers 2 --mixer_dropout 0.1 --anchor_fusion_strength_init -4.0 --prototype_gate_bias_init -8.0 \
    --use_residual_support 1 --use_v3_conditional_correction ${use_cond} --v3_conditional_hidden 28 \
    --v3_conditional_dropout 0.05 --v3_correction_modulation_scale_init 0.1 \
    "${extra_args[@]}" \
    2>&1 | tee "${log}"
}

for pred in ${preds}; do
  case "${pred}" in
    96)
      run_one 96 0.00088 8.0 -8.0 0.2 1 0 "lr88_mix8_valcandidate_mafine" \
        --use_val_candidate_blend 1 --val_candidate_blend_modes last,ma2,ma3,ma4,ma5,ma6,ma7,ma8 \
        --val_naive_blend_min 0.02 --val_naive_blend_max 0.07 --val_naive_blend_steps 251
      ;;
    192)
      run_one 192 0.0005 12.0 -8.0 0.2 1 1 "lr5e4_mix12_v3cond"
      ;;
    336)
      run_one 336 0.0005 12.0 -8.0 0.2 1 1 "lr5e4_mix12_v3cond"
      ;;
    720)
      run_one 720 0.0001 2.0 -2.5 0.2 0 1 "lr1e4_mix2_interlike_v3cond"
      ;;
    *)
      echo "Unsupported pred_len: ${pred}" >&2
      exit 2
      ;;
  esac
done
