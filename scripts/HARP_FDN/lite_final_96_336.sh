#!/usr/bin/env bash
set -u
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN || exit 1
mkdir -p logs/HARP_FDN_Lite
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_hybrid() {
  local pred="$1"
  local lr="$2"
  local msew="$3"
  local cls="$4"
  local gate="$5"
  local tag="LITE_FINAL_HYB_ETTh1_96_${pred}"
  local log="logs/HARP_FDN_Lite/${tag}.log"
  echo "===== $(date '+%F %T') START ${tag} =====" | tee "$log"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ${tag} --model HARP_FDN_PCX_Lite --data ETTh1 --features M \
    --seq_len 96 --label_len 48 --pred_len ${pred} --enc_in 7 --des Exp --itr 1 \
    --train_epochs 100 --patience 10 --batch_size 1024 --learning_rate ${lr} --lradj sigmoid \
    --train_objective mix --mse_loss_weight ${msew} --vali_objective mae --use_vali_ratio 0 --use_train_ratio 1 \
    --con_cls_1 ${cls} --con_cls_2 0.0 --con_time 0.0 \
    --ma_type ema --alpha 0.3 --beta 0.3 --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --fusion_uncertainty_weight 0.5 \
    --lite_use_prob_path 1 --lite_use_linear_head 1 \
    --lite_linear_gate_init ${gate} --lite_linear_gate_slope_init 0.3 --lite_linear_max_gate 0.75 \
    --lite_nlinear_strength_init -1.0 --lite_linear_dropout 0.0 \
    2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') END ${tag} =====" | tee -a "$log"
}

run_hybrid 96 0.00094 9.0 0.095 -3.2
run_hybrid 192 0.00094 9.0 0.095 -2.4
run_hybrid 336 0.00070 9.0 0.095 -1.8
