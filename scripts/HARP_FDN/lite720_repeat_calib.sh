#!/usr/bin/env bash
set -u
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_Lite
export CUDA_VISIBLE_DEVICES=0

run_test() {
  local tag="$1"
  local base_id="$2"
  local gran="$3"
  local affine="$4"
  local aff_args=""
  if [ "$affine" = "1" ]; then
    aff_args="--use_val_affine_calib 1 --val_affine_clip 0.5"
  fi
  local log="logs/HARP_FDN_Lite/${tag}.log"
  echo "===== $(date '+%F %T') TEST ${tag} =====" | tee "$log"
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ${base_id} \
    --model HARP_FDN_PCX_Lite \
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
    --learning_rate 0.001 \
    --lradj sigmoid \
    --train_objective mae \
    --mse_loss_weight 0.5 \
    --vali_objective mse \
    --use_vali_ratio 0 \
    --con_cls_1 0.0 \
    --con_cls_2 0.0 \
    --con_time 0.0 \
    --ma_type ema \
    --alpha 0.3 \
    --beta 0.3 \
    --support_temperature 1.0 \
    --support_path idx2.xlsx \
    --num_support 25 \
    --fusion_uncertainty_weight 0.0 \
    --lite_use_prob_path 0 \
    --lite_use_linear_head 1 \
    --lite_linear_gate_init 5.0 \
    --lite_linear_gate_slope_init 0.0 \
    --lite_linear_max_gate 1.0 \
    --lite_nlinear_strength_init -1.0 \
    --lite_linear_dropout 0.0 \
    --use_val_candidate_blend 1 \
    --val_candidate_blend_modes last,ma4,ma12,ma24,ma48,linear4,linear8,drift4,drift8,repeat24,repeat48,repeat96 \
    --val_candidate_blend_granularity ${gran} \
    --val_naive_blend_min=-1.0 \
    --val_naive_blend_max=1.0 \
    --val_naive_blend_steps 0 \
    ${aff_args} \
    2>&1 | tee -a "$log"
}

base=LITE720_LIN_A_mae_lr1e3_sigmoid
run_test LITE720_LIN_A_repeat_scalar ${base} scalar 0
run_test LITE720_LIN_A_repeat_horizon ${base} horizon 0
run_test LITE720_LIN_A_repeat_channel ${base} channel 0
run_test LITE720_LIN_A_repeat_element ${base} element 0
run_test LITE720_LIN_A_repeat_affine_scalar ${base} scalar 1
