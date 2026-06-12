#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -uo pipefail
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN
mkdir -p logs/HARP_FDN_PCX
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
model_name=HARP_FDN_PCX
base_id=HARP_FDN_PCX_4090D_QUEUE
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96
label_len=48
train_epochs=${TRAIN_EPOCHS:-100}
patience=${PATIENCE:-10}
lr=${LEARNING_RATE:-0.0001}

run_one(){
  local tag=$1 data=$2 root=$3 file=$4 enc=$5 pred=$6 batch=$7 profile=$8
  local cfg obj msew vali c1 c2 ct rel post corr
  if [ "$profile" = long ]; then
    cfg=WEAKCON_STABLE; obj=mae; msew=0.5; vali=mse; c1=0.01; c2=0.01; ct=0.02; rel=-4.0; post=-6.0; corr=-5.0
  else
    cfg=MIX40; obj=mix; msew=4.0; vali=mae; c1=0.05; c2=0.05; ct=0.1; rel=-2.5; post=-3.0; corr=-2.2
  fi
  local log=logs/HARP_FDN_PCX/${base_id}_${cfg}_${tag}_${seq_len}_${pred}_b${batch}.log
  echo "===== $tag pred=$pred cfg=$cfg batch=$batch ====="
  python -u run.py \
    --is_training 1 --root_path "$root" --data_path "$file" \
    --model_id ${base_id}_${cfg}_${tag}_${seq_len}_${pred}_b${batch}_${ma_type} \
    --model $model_name --data $data --features M --seq_len $seq_len --label_len $label_len --pred_len $pred --enc_in $enc \
    --des Exp --itr 1 --train_epochs $train_epochs --patience $patience --batch_size $batch --learning_rate $lr --lradj sigmoid \
    --train_objective $obj --mse_loss_weight $msew --vali_objective $vali --use_vali_ratio 0 \
    --con_cls_1 $c1 --con_cls_2 $c2 --con_time $ct --ma_type $ma_type --alpha $alpha --beta $beta \
    --support_temperature 1.0 --support_path idx2.xlsx --num_support 25 \
    --use_relative_decode 1 --use_distribution_transport 1 --use_uncertainty_fusion 1 --use_posterior_calib 1 \
    --relative_strength_init $rel --relative_residual_scale_init 0.5 --transport_var_weight 0.05 --posterior_calib_strength_init $post \
    --use_horizon_segment 1 --use_residual_correction 1 --use_spectral_transport 1 --correction_strength_init $corr \
    --use_dynamic_support 1 --mixer_d_model 96 --mixer_layers 2 --mixer_dropout 0.1 2>&1 | tee "$log"
  return ${PIPESTATUS[0]}
}

retry(){
  local tag=$1 data=$2 root=$3 file=$4 enc=$5 pred=$6 profile=$7; shift 7
  for b in "$@"; do
    run_one "$tag" "$data" "$root" "$file" "$enc" "$pred" "$b" "$profile" && return 0
    echo "retry $tag pred=$pred after failed batch=$b"
    sleep 10
  done
  echo "FAILED_ALL $tag pred=$pred"
  return 1
}

retry Solar Solar ./dataset/Solar/ solar_AL.txt 137 96 mix 96 64 32 16
retry Solar Solar ./dataset/Solar/ solar_AL.txt 137 192 mix 96 64 32 16
retry Solar Solar ./dataset/Solar/ solar_AL.txt 137 336 mix 64 48 32 16
retry Solar Solar ./dataset/Solar/ solar_AL.txt 137 720 long 48 32 16 8
retry ECL custom ./dataset/electricity/ electricity.csv 321 96 mix 96 64 32 16
retry ECL custom ./dataset/electricity/ electricity.csv 321 192 mix 96 64 32 16
retry ECL custom ./dataset/electricity/ electricity.csv 321 336 mix 64 48 32 16
retry ECL custom ./dataset/electricity/ electricity.csv 321 720 long 48 32 16 8
retry Traffic custom ./dataset/traffic/ traffic.csv 862 96 mix 24 16 8 4 2
retry Traffic custom ./dataset/traffic/ traffic.csv 862 192 mix 24 16 8 4 2
retry Traffic custom ./dataset/traffic/ traffic.csv 862 336 mix 16 8 4 2
retry Traffic custom ./dataset/traffic/ traffic.csv 862 720 long 8 4 2
