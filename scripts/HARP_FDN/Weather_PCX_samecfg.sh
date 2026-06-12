#!/usr/bin/env bash
set -uo pipefail
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN
mkdir -p logs/HARP_FDN_PCX
model_name=HARP_FDN_PCX
base_id=HARP_FDN_PCX_WEATHER_SAMECFG
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96
label_len=48
train_epochs=${TRAIN_EPOCHS:-100}
patience=${PATIENCE:-10}
lr=${LEARNING_RATE:-0.0001}
run_mix(){
  local pred=$1 batch=$2
  python -u run.py --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv \
    --model_id ${base_id}_MIX40_Weather_${seq_len}_${pred}_${ma_type} --model $model_name --data custom --features M \
    --seq_len $seq_len --label_len $label_len --pred_len $pred --enc_in 21 --des Exp --itr 1 \
    --train_epochs $train_epochs --patience $patience --batch_size $batch --learning_rate $lr --lradj sigmoid \
    --train_objective mix --mse_loss_weight 4.0 --vali_objective mae --use_vali_ratio 0 \
    --con_cls_1 0.05 --con_cls_2 0.05 --con_time 0.1 --ma_type $ma_type --alpha $alpha --beta $beta \
    --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --use_relative_decode 1 --use_distribution_transport 1 --use_uncertainty_fusion 1 --use_posterior_calib 1 \
    --relative_strength_init -2.5 --relative_residual_scale_init 0.5 --transport_var_weight 0.05 --posterior_calib_strength_init -3.0 \
    --use_horizon_segment 1 --use_residual_correction 1 --use_spectral_transport 1 --correction_strength_init -2.2 \
    --use_dynamic_support 1 --mixer_d_model 96 --mixer_layers 2 --mixer_dropout 0.1 \
    2>&1 | tee logs/HARP_FDN_PCX/${base_id}_MIX40_Weather_${seq_len}_${pred}.log
}
run_long(){
  local pred=720 batch=64
  python -u run.py --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv \
    --model_id ${base_id}_WEAKCON_STABLE_Weather_${seq_len}_${pred}_${ma_type} --model $model_name --data custom --features M \
    --seq_len $seq_len --label_len $label_len --pred_len $pred --enc_in 21 --des Exp --itr 1 \
    --train_epochs $train_epochs --patience $patience --batch_size $batch --learning_rate $lr --lradj sigmoid \
    --train_objective mae --mse_loss_weight 0.5 --vali_objective mse --use_vali_ratio 0 \
    --con_cls_1 0.01 --con_cls_2 0.01 --con_time 0.02 --ma_type $ma_type --alpha $alpha --beta $beta \
    --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --use_relative_decode 1 --use_distribution_transport 1 --use_uncertainty_fusion 1 --use_posterior_calib 1 \
    --relative_strength_init -4.0 --relative_residual_scale_init 0.5 --transport_var_weight 0.05 --posterior_calib_strength_init -6.0 \
    --use_horizon_segment 1 --use_residual_correction 1 --use_spectral_transport 1 --correction_strength_init -5.0 \
    --use_dynamic_support 1 --mixer_d_model 96 --mixer_layers 2 --mixer_dropout 0.1 \
    2>&1 | tee logs/HARP_FDN_PCX/${base_id}_WEAKCON_STABLE_Weather_${seq_len}_${pred}.log
}
run_mix 96 256
run_mix 192 256
run_mix 336 128
run_long
