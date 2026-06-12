#!/usr/bin/env bash
set -uo pipefail
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN
mkdir -p logs/HARP_FDN_PCX
model_name=HARP_FDN_PCX
base_id=HARP_FDN_PCX_ETT_SAMECFG
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96
# ETTh2 ETTm1 ETTm2
for dataset in ETTh1 ; do
  for pred in 96 192 336 720; do
    if [ "$pred" = 720 ]; then obj=mae; msew=0.5; vali=mse; c1=0.01; c2=0.01; ct=0.02; rel=-4.0; post=-6.0; corr=-5.0; cfg=WEAKCON_STABLE; batch=128; else obj=mix; msew=4.0; vali=mae; c1=0.05; c2=0.05; ct=0.1; rel=-2.5; post=-3.0; corr=-2.2; cfg=MIX40; batch=1024; fi
    python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${dataset}.csv \
      --model_id ${base_id}_${cfg}_${dataset}_${seq_len}_${pred}_${ma_type} --model $model_name --data $dataset --features M \
      --seq_len $seq_len --label_len 48 --pred_len $pred --enc_in 7 --des Exp --itr 1 --train_epochs 100 --patience 10 \
      --batch_size $batch --learning_rate 0.0001 --lradj sigmoid --train_objective $obj --mse_loss_weight $msew --vali_objective $vali --use_vali_ratio 0 \
      --con_cls_1 $c1 --con_cls_2 $c2 --con_time $ct --ma_type $ma_type --alpha $alpha --beta $beta \
      --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 --use_relative_decode 1 --use_distribution_transport 1 \
      --use_uncertainty_fusion 1 --use_posterior_calib 1 --relative_strength_init $rel --relative_residual_scale_init 0.5 \
      --transport_var_weight 0.05 --posterior_calib_strength_init $post --use_horizon_segment 1 --use_residual_correction 1 \
      --use_spectral_transport 1 --correction_strength_init $corr --use_dynamic_support 1 --mixer_d_model 96 --mixer_layers 2 --mixer_dropout 0.1 \
      2>&1 | tee logs/HARP_FDN_PCX/${base_id}_${cfg}_${dataset}_${seq_len}_${pred}.log
  done
done
