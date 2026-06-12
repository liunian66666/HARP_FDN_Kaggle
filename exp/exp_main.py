from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import interPDN, DLinear, HARP_FDN_PCX, HARP_FDN_PCX_V2, HARP_FDN_PCX_v3, HARP_FDN_PCX_V4, HARP_FDN_PCX_Lite
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.con_loss_coe_cls_1 = args.con_cls_1
        self.con_loss_coe_cls_2 = args.con_cls_2
        self.con_loss_coe_time= args.con_time

    def _build_model(self):
        model_dict = {
            'interPDN': interPDN,
            'DLinear': DLinear,
            'HARP_FDN_PCX': HARP_FDN_PCX,
            'HARP_FDN_PCX_V2': HARP_FDN_PCX_V2,
            'HARP_FDN_PCX_v3': HARP_FDN_PCX_v3,
            'HARP_FDN_PCX_V4': HARP_FDN_PCX_V4,
            'HARP_FDN_PCX_Lite': HARP_FDN_PCX_Lite,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def _resume_checkpoint_path(self, path):
        return os.path.join(path, 'resume_checkpoint.pth')

    def _checkpoint_dir(self, setting):
        return os.path.join(self.args.checkpoints, self.args.model, setting)

    def _torch_load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _save_resume_checkpoint(self, path, epoch, model_optim, early_stopping):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'early_stopping_state': {
                'counter': early_stopping.counter,
                'best_score': early_stopping.best_score,
                'early_stop': early_stopping.early_stop,
                'val_loss_min': early_stopping.val_loss_min,
            },
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, self._resume_checkpoint_path(path))

    def _load_resume_checkpoint(self, path, model_optim, early_stopping):
        resume_path = self._resume_checkpoint_path(path)
        if not os.path.exists(resume_path):
            return 0

        checkpoint = self._torch_load(resume_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])

        early_state = checkpoint.get('early_stopping_state', {})
        early_stopping.counter = early_state.get('counter', early_stopping.counter)
        early_stopping.best_score = early_state.get('best_score', early_stopping.best_score)
        early_stopping.early_stop = early_state.get('early_stop', early_stopping.early_stop)
        early_stopping.val_loss_min = early_state.get('val_loss_min', early_stopping.val_loss_min)

        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])

        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        print('Resuming training from {} at epoch {}'.format(resume_path, start_epoch + 1))
        return start_epoch

    def _objective_loss(self, pred, true, mse_criterion, mae_criterion, objective=None):
        objective = (objective or getattr(self.args, 'train_objective', 'mae')).lower()
        if objective == 'mse':
            return mse_criterion(pred, true)
        if objective == 'mix':
            return mae_criterion(pred, true) + float(getattr(self.args, 'mse_loss_weight', 0.5)) * mse_criterion(pred, true)
        return mae_criterion(pred, true)

    def vali(self, vali_data, vali_loader, criterion, is_test = True):
        total_loss = []
        total_raw_mse = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs, con_loss_cls_1, con_loss_cls_2, con_loss_time= self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # if train, use ratio to scale the prediction
                if (not is_test) and int(getattr(self.args, 'use_vali_ratio', 1)) == 1:
                    # CARD loss with weight decay
                    # self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])

                    # Arctangent loss with weight decay
                    self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio, device=outputs.device, dtype=outputs.dtype).unsqueeze(-1)

                    pred = outputs *self.ratio
                    true = batch_y *self.ratio
                else:
                    pred = outputs#.detach().cpu()
                    true = batch_y#.detach().cpu()

                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                loss += self.con_loss_coe_cls_1*con_loss_cls_1 + self.con_loss_coe_cls_2*con_loss_cls_2 + con_loss_time*self.con_loss_coe_time

                total_loss.append(loss.item())
                total_raw_mse.append(nn.functional.mse_loss(outputs, batch_y).item())
        total_loss = np.average(total_loss)
        total_raw_mse = np.average(total_raw_mse)
        self.model.train()
        if getattr(self.args, 'early_stop_metric', 'loss').lower() == 'raw_mse':
            return total_raw_mse
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self._checkpoint_dir(setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()
        start_epoch = self._load_resume_checkpoint(path, model_optim, early_stopping)
        if start_epoch >= self.args.train_epochs:
            print('Resume checkpoint already reached train_epochs; loading best checkpoint.')

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # train_time = 0 # For computational cost analysis

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                # temp = time.time() # For computational cost analysis
                outputs, con_loss_cls_1, con_loss_cls_2, con_loss_time = self.model(batch_x)
                # train_time += time.time() - temp # For computational cost analysis
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # CARD loss with weight decay
                # self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])

                if int(getattr(self.args, 'use_train_ratio', 1)) == 1:
                    # Arctangent loss with weight decay. Good for short horizons, optional for long horizons.
                    self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio, device=outputs.device, dtype=outputs.dtype).unsqueeze(-1)
                    loss_outputs = outputs * self.ratio
                    loss_batch_y = batch_y * self.ratio
                else:
                    loss_outputs = outputs
                    loss_batch_y = batch_y

                loss = self._objective_loss(loss_outputs, loss_batch_y, mse_criterion, mae_criterion)

                # downsampling
                y_down = batch_y.reshape(batch_y.shape[0], batch_y.shape[1] // 4, 4, batch_y.shape[2])
                y_down = y_down.mean(dim=2) 

                # loss_2 = mae_criterion(output_2, y_down)
                loss += self.con_loss_coe_cls_1*con_loss_cls_1 + self.con_loss_coe_cls_2*con_loss_cls_2 + con_loss_time*self.con_loss_coe_time

                # loss = criterion(outputs, batch_y) # For MSE criterion

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            # train_times.append(train_time/len(train_loader)) # For computational cost analysis
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion) # For MSE criterion
            # test_loss = self.vali(test_data, test_loader, criterion) # For MSE criterion
            vali_criterion = mae_criterion if getattr(self.args, 'vali_objective', 'mae').lower() == 'mae' else mse_criterion
            vali_loss = self.vali(vali_data, vali_loader, vali_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # adjust_learning_rate_new(model_optim, epoch + 1, self.args)

            self._save_resume_checkpoint(path, epoch, model_optim, early_stopping)

            # print('Alpha:', self.model.decomp.ma.alpha) # Print the learned alpha
            # print('Beta:', self.model.decomp.ma.beta)   # Print the learned beta

        # print("Training time: {}".format(np.sum(train_times)/len(train_times))) # For computational cost analysis
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(self._torch_load(best_model_path))
        resume_path = self._resume_checkpoint_path(path)
        if os.path.exists(resume_path):
            os.remove(resume_path)

        return self.model

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
        
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     # test_time = 0 # For computational cost analysis
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             # temp = time.time() # For computational cost analysis
    #             outputs, _, _, _ = self.model(batch_x)
    #             # test_time += time.time() - temp # For computational cost analysis

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()

    #             pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
    #             true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

    #             preds.append(pred)
    #             trues.append(true)

    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pdf = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pdf, os.path.join(folder_path, str(i) + '.pdf'))
            
    #     # print("Inference time: {}".format(test_time/len(test_loader))) # For computational cost analysis
    #     preds = np.array(preds)
    #     trues = np.array(trues)

    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    #     mae, mse = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #     return
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(self._torch_load(os.path.join(self._checkpoint_dir(setting), 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 新建 results 总目录
        res_dir = "./results"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # 按模型名定义单独日志文件
        model_name = self.args.model
        res_file = os.path.join(res_dir, f"{model_name}.txt")

        affine_scale = None
        affine_bias = None
        naive_blend = None
        blend_mode = 'last'

        def build_blend_baseline(batch_x, f_dim, mode):
            history = batch_x[:, :, f_dim:]
            pred_len = self.args.pred_len
            mode = mode.strip()
            if mode == 'last':
                return history[:, -1:, :].repeat(1, pred_len, 1)
            if mode.startswith('ma'):
                window = max(1, min(int(mode[2:]), history.shape[1]))
                return history[:, -window:, :].mean(dim=1, keepdim=True).repeat(1, pred_len, 1)
            if mode.startswith('linear'):
                window = max(2, min(int(mode[6:]), history.shape[1]))
                slope = (history[:, -1:, :] - history[:, -window:-window + 1, :]) / float(window - 1)
                steps = torch.arange(1, pred_len + 1, device=history.device, dtype=history.dtype).view(1, -1, 1)
                return history[:, -1:, :] + steps * slope
            if mode.startswith('drift'):
                window = max(2, min(int(mode[5:]), history.shape[1]))
                diffs = history[:, -window + 1:, :] - history[:, -window:-1, :]
                slope = diffs.mean(dim=1, keepdim=True)
                steps = torch.arange(1, pred_len + 1, device=history.device, dtype=history.dtype).view(1, -1, 1)
                return history[:, -1:, :] + steps * slope
            if mode.startswith('repeat'):
                period = max(1, min(int(mode[6:]), history.shape[1]))
                pattern = history[:, -period:, :]
                reps = int(np.ceil(pred_len / float(period)))
                return pattern.repeat(1, reps, 1)[:, :pred_len, :]
            if mode.startswith('seasonal'):
                period = max(1, min(int(mode[8:]), history.shape[1]))
                pattern = history[:, -period:, :]
                reps = int(np.ceil(pred_len / float(period)))
                return pattern.repeat(1, reps, 1)[:, :pred_len, :]
            raise ValueError('Unsupported val_candidate_blend mode: {}'.format(mode))

        if int(getattr(self.args, 'use_val_affine_calib', 0)) == 1:
            _, calib_loader = self._get_data(flag='val')
            calib_preds = []
            calib_trues = []
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in calib_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs, _, _, _ = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    calib_preds.append(outputs.detach().cpu().numpy())
                    calib_trues.append(batch_y.detach().cpu().numpy())
            calib_preds = np.concatenate(calib_preds, axis=0)
            calib_trues = np.concatenate(calib_trues, axis=0)
            pred_mean = calib_preds.mean(axis=0, keepdims=True)
            true_mean = calib_trues.mean(axis=0, keepdims=True)
            centered_pred = calib_preds - pred_mean
            centered_true = calib_trues - true_mean
            ridge = float(getattr(self.args, 'val_affine_ridge', 1e-6))
            scale = (centered_pred * centered_true).mean(axis=0, keepdims=True)
            scale = scale / ((centered_pred ** 2).mean(axis=0, keepdims=True) + ridge)
            clip = float(getattr(self.args, 'val_affine_clip', 0.3))
            scale = np.clip(scale, 1.0 - clip, 1.0 + clip)
            bias = true_mean - scale * pred_mean
            affine_scale = scale
            affine_bias = bias
            print('Validation affine calibration enabled: scale_mean={:.6f}, bias_mean={:.6f}'.format(
                float(scale.mean()), float(bias.mean())))
        if int(getattr(self.args, 'use_val_naive_blend', 0)) == 1 or int(getattr(self.args, 'use_val_candidate_blend', 0)) == 1:
            _, blend_loader = self._get_data(flag='val')
            blend_preds = []
            blend_trues = []
            blend_candidates = {}
            if int(getattr(self.args, 'use_val_candidate_blend', 0)) == 1:
                blend_modes = [m.strip() for m in getattr(self.args, 'val_candidate_blend_modes', '').split(',') if m.strip()]
            else:
                blend_modes = ['last']
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in blend_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs, _, _, _ = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    blend_preds.append(outputs.detach().cpu().numpy())
                    blend_trues.append(batch_y.detach().cpu().numpy())
                    for mode in blend_modes:
                        baseline = build_blend_baseline(batch_x, f_dim, mode)
                        blend_candidates.setdefault(mode, []).append(baseline.detach().cpu().numpy())
            blend_preds = np.concatenate(blend_preds, axis=0)
            blend_trues = np.concatenate(blend_trues, axis=0)
            blend_candidates = {
                mode: np.concatenate(values, axis=0)
                for mode, values in blend_candidates.items()
            }
            lo = float(getattr(self.args, 'val_naive_blend_min', -0.2))
            hi = float(getattr(self.args, 'val_naive_blend_max', 0.3))
            steps = int(getattr(self.args, 'val_naive_blend_steps', 101))
            best_loss = None
            best_lam = 0.0
            best_mode = blend_modes[0]
            granularity = getattr(self.args, 'val_candidate_blend_granularity', 'scalar').lower()
            target_delta = blend_trues - blend_preds
            for mode, baseline in blend_candidates.items():
                diff = baseline - blend_preds
                if granularity in ('element', 'horizon', 'channel'):
                    if granularity == 'horizon':
                        reduce_axes = (0, 2)
                        keepdims = False
                        num = np.mean(diff * target_delta, axis=reduce_axes)
                        den = np.mean(diff ** 2, axis=reduce_axes) + 1e-12
                        lam = np.clip(num / den, lo, hi).reshape(1, -1, 1)
                    elif granularity == 'channel':
                        reduce_axes = (0, 1)
                        num = np.mean(diff * target_delta, axis=reduce_axes)
                        den = np.mean(diff ** 2, axis=reduce_axes) + 1e-12
                        lam = np.clip(num / den, lo, hi).reshape(1, 1, -1)
                    else:
                        num = np.mean(diff * target_delta, axis=0, keepdims=True)
                        den = np.mean(diff ** 2, axis=0, keepdims=True) + 1e-12
                        lam = np.clip(num / den, lo, hi)
                    candidate = blend_preds + lam * diff
                    loss = np.mean((candidate - blend_trues) ** 2)
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_lam = lam.astype(np.float32)
                        best_mode = mode
                else:
                    num = float(np.mean(diff * target_delta))
                    den = float(np.mean(diff ** 2)) + 1e-12
                    lam = float(np.clip(num / den, lo, hi))
                    if steps > 1:
                        radius = (hi - lo) / max(steps - 1, 1)
                        grid_lo = max(lo, lam - radius)
                        grid_hi = min(hi, lam + radius)
                        candidates = np.linspace(grid_lo, grid_hi, min(21, steps))
                    else:
                        candidates = [lam]
                    for lam_i in candidates:
                        candidate = blend_preds + float(lam_i) * diff
                        loss = np.mean((candidate - blend_trues) ** 2)
                        if best_loss is None or loss < best_loss:
                            best_loss = loss
                            best_lam = float(lam_i)
                            best_mode = mode
            naive_blend = best_lam
            blend_mode = best_mode
            if isinstance(naive_blend, np.ndarray):
                print('Validation candidate blend enabled: mode={}, lambda_mean={:.6f}, lambda_min={:.6f}, lambda_max={:.6f}, val_mse={:.6f}'.format(
                    blend_mode, float(naive_blend.mean()), float(naive_blend.min()), float(naive_blend.max()), float(best_loss)))
            else:
                print('Validation candidate blend enabled: mode={}, lambda={:.6f}, val_mse={:.6f}'.format(
                    blend_mode, float(naive_blend), float(best_loss)))

        self.model.eval()
        naive_preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs, _, _, _ = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if naive_blend is not None:
                    naive = build_blend_baseline(batch_x, f_dim, blend_mode).detach().cpu().numpy()
                    naive_preds.append(naive)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pdf = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pdf, os.path.join(folder_path, str(i) + '.pdf'))
                
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if affine_scale is not None:
            preds = preds * affine_scale + affine_bias
        if naive_blend is not None:
            naive_preds = np.array(naive_preds)
            naive_preds = naive_preds.reshape(-1, naive_preds.shape[-2], naive_preds.shape[-1])
            preds = preds + naive_blend * (naive_preds - preds)

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 写入对应模型的独立文件
        with open(res_file, 'a', encoding='utf-8') as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"mse:{mse}, mae:{mae}\n")
            f.write("-" * 50 + "\n")

        return
