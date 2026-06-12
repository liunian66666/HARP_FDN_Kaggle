# #!/usr/bin/env bash
# set -euo pipefail

# # cd /home/DM24/workspace/Time_Series_Forecasting/InterPDN/interPDN
# mkdir -p logs/HARP_FDN_PCX

# model_name=HARP_FDN_PCX_V2
# base_id=HARP_FDN_PCX_V2
# ma_type=ema
# alpha=0.3
# beta=0.3
# seq_len=96

# run_mix40() {
#   local pred_len=$1
#   local batch=$2

#   echo "====================================="
#   echo "ETTh1 pred_len=${pred_len} | MIX40 | use_vali_ratio=0 | batch=${batch}"
#   echo "====================================="

#   python -u run_patch0608.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ${base_id}_MIX40_ETTh1_${seq_len}_${pred_len}_${ma_type} \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 100 \
#     --patience 10 \
#     --batch_size $batch \
#     --learning_rate 0.0001 \
#     --lradj 'sigmoid' \
#     --train_objective mix \
#     --mse_loss_weight 4.0 \
#     --vali_objective mae \
#     --use_vali_ratio 0 \
#     --con_cls_1 0.05 \
#     --con_cls_2 0.05 \
#     --con_time 0.1 \
#     --ma_type $ma_type \
#     --alpha $alpha \
#     --beta $beta \
#     --support_temperature 1.0 \
#     --support_path idx2.xlsx \
#     --num_support 25 \
#     --use_relative_decode 1 \
#     --use_distribution_transport 1 \
#     --use_uncertainty_fusion 1 \
#     --use_posterior_calib 1 \
#     --relative_strength_init -2.5 \
#     --relative_residual_scale_init 0.5 \
#     --transport_var_weight 0.05 \
#     --posterior_calib_strength_init -3.0 \
#     --use_horizon_segment 1 \
#     --use_residual_correction 1 \
#     --use_spectral_transport 1 \
#     --correction_strength_init -2.2 \
#     --use_dynamic_support 1 \
#     --mixer_d_model 96 \
#     --mixer_layers 2 \
#     --mixer_dropout 0.1 \
#     2>&1 | tee logs/HARP_FDN_PCX/${base_id}_MIX40_ETTh1_${seq_len}_${pred_len}.log
# }

# run_720() {
#   echo "====================================="
#   echo "ETTh1 pred_len=720 | current best | use_vali_ratio=0 | batch=128"
#   echo "====================================="

#   python -u run_patch0608.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ${base_id}_WEAKCON_STABLE_ETTh1_${seq_len}_720_${ma_type} \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len $seq_len \
#     --pred_len 720 \
#     --enc_in 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 100 \
#     --patience 10 \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --lradj 'sigmoid' \
#     --train_objective mae \
#     --mse_loss_weight 0.5 \
#     --vali_objective mse \
#     --use_vali_ratio 0 \
#     --con_cls_1 0.01 \
#     --con_cls_2 0.01 \
#     --con_time 0.02 \
#     --ma_type $ma_type \
#     --alpha $alpha \
#     --beta $beta \
#     --support_temperature 1.0 \
#     --support_path idx2.xlsx \
#     --num_support 25 \
#     --use_relative_decode 1 \
#     --use_distribution_transport 1 \
#     --use_uncertainty_fusion 1 \
#     --use_posterior_calib 1 \
#     --relative_strength_init -4.0 \
#     --relative_residual_scale_init 0.5 \
#     --transport_var_weight 0.05 \
#     --posterior_calib_strength_init -6.0 \
#     --use_horizon_segment 1 \
#     --use_residual_correction 1 \
#     --use_spectral_transport 1 \
#     --correction_strength_init -5.0 \
#     --use_dynamic_support 1 \
#     --mixer_d_model 96 \
#     --mixer_layers 2 \
#     --mixer_dropout 0.1 \
#     2>&1 | tee logs/HARP_FDN_PCX/${base_id}_WEAKCON_STABLE_ETTh1_${seq_len}_720.log
# }

# run_mix40 96 1024
# run_mix40 192 1024
# run_mix40 336 512
# run_720


#!/usr/bin/env bash
set -euo pipefail

# 固定全局参数
mkdir -p logs/HARP_FDN_PCX

model_name=HARP_FDN_PCX_V2
base_id=HARP_FDN_PCX_V2
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# ===================== 通用函数定义 =====================
# 用法: run_mix40 数据集名 pred_len batch_size
# 数据集支持: ETTh1 ETTh2 ETTm1 ETTm2
run_mix40() {
    local data_name="$1"
    local pred_len="$2"
    local batch="$3"

    # ETT 系列数据集统一配置
    case "${data_name}" in
        ETTh1|ETTh2|ETTm1|ETTm2)
            root_path="./dataset/ETT-small/"
            data_path="${data_name}.csv"
            enc_in=7
            ;;
        *)
            echo "Error: 不支持数据集 ${data_name}"
            return 1
            ;;
    esac

    echo "====================================="
    echo "数据集: ${data_name} | pred_len=${pred_len} | MIX40 | use_vali_ratio=0 | batch=${batch}"
    echo "====================================="

    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${base_id}_MIX40_${data_name}_${seq_len}_${pred_len}_${ma_type}" \
        --model "${model_name}" \
        --data "${data_name}" \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in}" \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 100 \
        --patience 10 \
        --batch_size "${batch}" \
        --learning_rate 0.0001 \
        --lradj 'sigmoid' \
        --train_objective mix \
        --mse_loss_weight 4.0 \
        --vali_objective mae \
        --use_vali_ratio 0 \
        --con_cls_1 0.05 \
        --con_cls_2 0.05 \
        --con_time 0.1 \
        --ma_type "${ma_type}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --support_temperature 1.0 \
        --support_path idx2.xlsx \
        --num_support 25 \
        --use_relative_decode 1 \
        --use_distribution_transport 1 \
        --use_uncertainty_fusion 1 \
        --use_posterior_calib 1 \
        --relative_strength_init -2.5 \
        --relative_residual_scale_init 0.5 \
        --transport_var_weight 0.05 \
        --posterior_calib_strength_init -3.0 \
        --use_horizon_segment 1 \
        --use_residual_correction 1 \
        --use_spectral_transport 1 \
        --correction_strength_init -2.2 \
        --use_dynamic_support 1 \
        --mixer_d_model 96 \
        --mixer_layers 2 \
        --mixer_dropout 0.1 \
        2>&1 | tee "logs/HARP_FDN_PCX/${base_id}_MIX40_${data_name}_${seq_len}_${pred_len}.log"
}

# 用法: run_720 数据集名
# 固定 pred_len=720，使用另一套专属超参,batch在4090D上最多用512
run_720() {
    local data_name="$1"

    case "${data_name}" in
        ETTh1|ETTh2|ETTm1|ETTm2)
            root_path="./dataset/ETT-small/"
            data_path="${data_name}.csv"
            enc_in=7
            ;;
        *)
            echo "Error: 不支持数据集 ${data_name}"
            return 1
            ;;
    esac

    echo "====================================="
    echo "数据集: ${data_name} | pred_len=720 | current best | use_vali_ratio=0 | batch=128"
    echo "====================================="

    python -u run_patch0608.py \
        --is_training 1 \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${base_id}_WEAKCON_STABLE_${data_name}_${seq_len}_720_${ma_type}" \
        --model "${model_name}" \
        --data "${data_name}" \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len 720 \
        --enc_in "${enc_in}" \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 100 \
        --patience 10 \
        --batch_size 1024 \
        --learning_rate 0.0001 \
        --lradj 'sigmoid' \
        --train_objective mae \
        --mse_loss_weight 0.5 \
        --vali_objective mse \
        --use_vali_ratio 0 \
        --con_cls_1 0.01 \
        --con_cls_2 0.01 \
        --con_time 0.02 \
        --ma_type "${ma_type}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --support_temperature 1.0 \
        --support_path idx2.xlsx \
        --num_support 25 \
        --use_relative_decode 1 \
        --use_distribution_transport 1 \
        --use_uncertainty_fusion 1 \
        --use_posterior_calib 1 \
        --relative_strength_init -4.0 \
        --relative_residual_scale_init 0.5 \
        --transport_var_weight 0.05 \
        --posterior_calib_strength_init -6.0 \
        --use_horizon_segment 1 \
        --use_residual_correction 1 \
        --use_spectral_transport 1 \
        --correction_strength_init -5.0 \
        --use_dynamic_support 1 \
        --mixer_d_model 96 \
        --mixer_layers 2 \
        --mixer_dropout 0.1 \
        2>&1 | tee "logs/HARP_FDN_PCX/${base_id}_WEAKCON_STABLE_${data_name}_${seq_len}_720.log"
}

# ===================== 执行区域（按需开启/注释）=====================
# # 示例：单数据集 ETTh1 全套运行
# run_mix40 ETTh1 96 1024
# run_mix40 ETTh1 192 1024
# run_mix40 ETTh1 336 512
# run_720 ETTh1

# 如需跑其他 ETT 数据集，直接追加下面行即可，例如：
# run_mix40 ETTh2 96 1024
# run_mix40 ETTh2 192 1024
# run_mix40 ETTh2 336 512
# run_720 ETTh2

# run_mix40 ETTm1 96 1024
# run_mix40 ETTm1 192 1024
# run_mix40 ETTm1 336 512
# run_720 ETTm1

# run_mix40 ETTm2 96 1024
# run_mix40 ETTm2 192 1024
# run_mix40 ETTm2 336 512
run_720 ETTm2