#!/usr/bin/env bash
set -u
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_Lite
export CUDA_VISIBLE_DEVICES=0

run_one() {
  local tag="$1" lr="$2" obj="$3" vali="$4" msew="$5" nlin="$6" lradj="$7"
  local log="logs/HARP_FDN_Lite/${tag}.log"
  echo "===== $(date '+%F %T') START ${tag} =====" | tee "$log"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ${tag} --model HARP_FDN_PCX_Lite --data ETTh1 --features M \
    --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --des Exp --itr 1 \
    --train_epochs 100 --patience 10 --batch_size 1024 --learning_rate ${lr} --lradj ${lradj} \
    --train_objective ${obj} --mse_loss_weight ${msew} --vali_objective ${vali} --use_vali_ratio 0 \
    --con_cls_1 0.0 --con_cls_2 0.0 --con_time 0.0 \
    --ma_type ema --alpha 0.3 --beta 0.3 --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --fusion_uncertainty_weight 0.0 \
    --lite_use_prob_path 0 --lite_use_linear_head 1 \
    --lite_linear_gate_init 5.0 --lite_linear_gate_slope_init 0.0 --lite_linear_max_gate 1.0 \
    --lite_nlinear_strength_init ${nlin} --lite_linear_dropout 0.0 \
    2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') END ${tag} =====" | tee -a "$log"
}

run_one LITE720_LIN_A_mae_lr1e3_sigmoid 0.00100 mae mse 0.5 -1.0 sigmoid
run_one LITE720_LIN_B_mix05_lr1e3_sigmoid 0.00100 mix mse 0.5 -1.0 sigmoid
run_one LITE720_LIN_C_mae_lr5e4_constant 0.00050 mae mse 0.5 -1.0 constant
run_one LITE720_LIN_D_mix1_lr5e4_constant 0.00050 mix mse 1.0 -1.0 constant
