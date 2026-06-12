import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021

random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='interPDN')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, default='ETTh1_96_ema', help='model id')
parser.add_argument('--model', type=str, default='interPDN',
                    help='model name, options: [interPDN, DLinear]')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# data loader
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./Checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

# Patching
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

# Moving Average
parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
parser.add_argument('--beta', type=float, default=0.3, help='beta')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='sigmoid', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--train_objective', type=str, default='mae', help='mae, mse, or mix')
parser.add_argument('--vali_objective', type=str, default='mae', help='mae or mse')
parser.add_argument('--early_stop_metric', type=str, default='loss', help='loss or raw_mse')
parser.add_argument('--mse_loss_weight', type=float, default=0.5, help='MSE weight when train_objective=mix')
parser.add_argument('--use_val_affine_calib', type=int, default=0, help='fit affine output calibration on validation set before testing')
parser.add_argument('--val_affine_clip', type=float, default=0.3, help='max absolute deviation from scale=1 for validation affine calibration')
parser.add_argument('--val_affine_ridge', type=float, default=1e-6, help='ridge term for validation affine calibration')
parser.add_argument('--use_val_naive_blend', type=int, default=0, help='fit a validation-set blend with repeated last value before testing')
parser.add_argument('--val_naive_blend_min', type=float, default=-0.2)
parser.add_argument('--val_naive_blend_max', type=float, default=0.3)
parser.add_argument('--val_naive_blend_steps', type=int, default=101)
parser.add_argument('--use_val_candidate_blend', type=int, default=0, help='select a validation-set blend from simple extrapolation candidates before testing')
parser.add_argument('--val_candidate_blend_modes', type=str, default='last,ma4,ma12,linear4,linear8,drift4,drift8,repeat24,repeat48,repeat96')
parser.add_argument('--val_candidate_blend_granularity', type=str, default='scalar')
parser.add_argument('--use_vali_ratio', type=int, default=1, help='apply horizon ratio weighting during validation')
parser.add_argument('--use_train_ratio', type=int, default=1, help='apply arctangent horizon ratio during training')
parser.add_argument('--support_temperature', type=float, default=1.0)
parser.add_argument('--support_path', type=str, default='idx2.xlsx')
parser.add_argument('--num_support', type=int, default=25)
parser.add_argument('--use_relative_decode', type=int, default=1)
parser.add_argument('--use_distribution_transport', type=int, default=1)
parser.add_argument('--use_uncertainty_fusion', type=int, default=1)
parser.add_argument('--use_posterior_calib', type=int, default=1)
parser.add_argument('--relative_strength_init', type=float, default=-2.5)
parser.add_argument('--relative_residual_scale_init', type=float, default=0.5)
parser.add_argument('--transport_var_weight', type=float, default=0.05)
parser.add_argument('--posterior_kl_weight', type=float, default=0.0)
parser.add_argument('--residual_support_scale', type=float, default=0.35)
parser.add_argument('--horizon_gate_strength', type=float, default=1.0)
parser.add_argument('--horizon_gate_bias_init', type=float, default=0.0)
parser.add_argument('--use_conditional_residual', type=int, default=1)
parser.add_argument('--conditional_residual_hidden_ratio', type=int, default=2)
parser.add_argument('--conditional_residual_strength_init', type=float, default=-8.0)
parser.add_argument('--use_conditional_residual_logits', type=int, default=0)
parser.add_argument('--conditional_residual_logit_hidden_ratio', type=int, default=4)
parser.add_argument('--conditional_residual_logit_strength_init', type=float, default=-8.0)
parser.add_argument('--posterior_calib_strength_init', type=float, default=-3.0)
parser.add_argument('--use_horizon_segment', type=int, default=1)
parser.add_argument('--use_residual_correction', type=int, default=1)
parser.add_argument('--use_spectral_transport', type=int, default=1)
parser.add_argument('--num_horizon_segments', type=int, default=4)
parser.add_argument('--transport_spectral_weight', type=float, default=0.03)
parser.add_argument('--spectral_low_ratio', type=float, default=0.5)
parser.add_argument('--correction_strength_init', type=float, default=-2.2)
parser.add_argument('--correction_dropout', type=float, default=0.05)
parser.add_argument('--use_dynamic_support', type=int, default=1)
parser.add_argument('--dynamic_support_segments', type=int, default=4)
parser.add_argument('--support_shift_limit', type=float, default=0.35)
parser.add_argument('--support_scale_limit', type=float, default=0.35)
parser.add_argument('--mixer_d_model', type=int, default=96)
parser.add_argument('--mixer_layers', type=int, default=2)
parser.add_argument('--mixer_dropout', type=float, default=0.1)
parser.add_argument('--anchor_fusion_strength_init', type=float, default=-4.0)
parser.add_argument('--prototype_gate_bias_init', type=float, default=-8.0)
parser.add_argument('--use_residual_support', type=int, default=1)
parser.add_argument('--use_direct_adapter', type=int, default=0)
parser.add_argument('--use_trend_anchor', type=int, default=0)
parser.add_argument('--use_horizon_affine', type=int, default=0)
parser.add_argument('--horizon_affine_strength_init', type=float, default=0.0)
parser.add_argument('--use_horizon_candidate_mixer', type=int, default=0)
parser.add_argument('--candidate_mixer_modes', type=str, default='last,ma4,drift4,linear4')
parser.add_argument('--candidate_mixer_strength_init', type=float, default=-3.1)
parser.add_argument('--candidate_mixer_max_gate', type=float, default=0.12)
parser.add_argument('--candidate_mixer_use_uncertainty', type=int, default=0)
parser.add_argument('--candidate_mixer_apply_in_train', type=int, default=1)
parser.add_argument('--trend_anchor_strength_init', type=float, default=-6.0)
parser.add_argument('--trend_anchor_window', type=int, default=6)
parser.add_argument('--fusion_uncertainty_weight', type=float, default=0.0)
parser.add_argument('--direct_adapter_strength_init', type=float, default=-4.0)
parser.add_argument('--direct_adapter_dropout', type=float, default=0.05)
parser.add_argument('--use_v3_conditional_correction', type=int, default=1)
parser.add_argument('--v3_conditional_hidden', type=int, default=28)
parser.add_argument('--v3_conditional_dropout', type=float, default=0.05)
parser.add_argument('--v3_correction_modulation_scale_init', type=float, default=0.1)
# parser.add_argument('--warmup_epochs',type=int,default = 0)

# Loss weight

# HARP_FDN_PCX_Lite options
parser.add_argument('--lite_use_prob_path', type=int, default=1)
parser.add_argument('--lite_use_linear_head', type=int, default=1)
parser.add_argument('--lite_linear_gate_init', type=float, default=999.0)
parser.add_argument('--lite_linear_gate_slope_init', type=float, default=0.0)
parser.add_argument('--lite_linear_max_gate', type=float, default=0.85)
parser.add_argument('--lite_linear_dropout', type=float, default=0.0)
parser.add_argument('--lite_nlinear_strength_init', type=float, default=-1.5)

parser.add_argument('--con_cls_1', type=float, default='0.05', help='Consistency cnstraint for interleaved dual branches at normal scale')
parser.add_argument('--con_cls_2', type=float, default='0.05', help='Consistency cnstraint for interleaved dual branches at coaser scale')
parser.add_argument('--con_time', type=float, default='0.1', help='Cross-scale consistency constraint')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                        args.model,
                                                        args.data,
                                                        args.features,
                                                        args.seq_len,
                                                        args.label_len,
                                                        args.pred_len,
                                                        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
