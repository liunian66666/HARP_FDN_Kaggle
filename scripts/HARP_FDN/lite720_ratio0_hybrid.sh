#!/usr/bin/env bash
set -u
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_Lite
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_one() {
  local tag="$1" lr="$2" obj="$3" msew="$4" gate="$5" fuw="$6"
  local log="logs/HARP_FDN_Lite/${tag}.log"
  echo "===== $(date '+%F %T') START ${tag} =====" | tee "$log"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ${tag} --model HARP_FDN_PCX_Lite --data ETTh1 --features M \
    --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --des Exp --itr 1 \
    --train_epochs 100 --patience 10 --batch_size 1024 --learning_rate ${lr} --lradj sigmoid \
    --train_objective ${obj} --mse_loss_weight ${msew} --vali_objective mse --use_vali_ratio 0 --use_train_ratio 0 \
    --con_cls_1 0.01 --con_cls_2 0.0 --con_time 0.0 \
    --ma_type ema --alpha 0.3 --beta 0.3 --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --fusion_uncertainty_weight ${fuw} \
    --lite_use_prob_path 1 --lite_use_linear_head 1 \
    --lite_linear_gate_init ${gate} --lite_linear_gate_slope_init 0.5 --lite_linear_max_gate 0.95 \
    --lite_nlinear_strength_init -1.0 --lite_linear_dropout 0.0 \
    2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') END ${tag} =====" | tee -a "$log"
}

run_one LITE720_HYB_R0_A_mix1_lr7e4_gate0 0.0007 mix 1.0 0.0 0.3
run_one LITE720_HYB_R0_B_mix2_lr7e4_gate05 0.0007 mix 2.0 0.5 0.3
run_one LITE720_HYB_R0_C_mae_lr7e4_gate05 0.0007 mae 0.5 0.5 0.3
