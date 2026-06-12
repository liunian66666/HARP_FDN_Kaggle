from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import interPDN, DLinear, HARP_FDN_MSX_patch0608,       HARP_FDN_PCX_patch0608,HARP_FDN_PCX_V2,HARP_FDN_optimized
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, value in model.state_dict().items():
            self.shadow[name] = value.detach().clone()

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, value in model_state.items():
            value = value.detach()
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def store(self, model):
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.con_loss_coe_cls_1 = args.con_cls_1
        self.con_loss_coe_cls_2 = args.con_cls_2
        self.con_loss_coe_time = args.con_time

    def _build_model(self):
        model_dict = {
            'interPDN': interPDN,
            'DLinear': DLinear,
            'HARP_FDN_MSX_patch0608': HARP_FDN_MSX_patch0608,
            'HARP_FDN_PCX_patch0608': HARP_FDN_PCX_patch0608,
            'HARP_FDN_PCX_V2': HARP_FDN_PCX_V2,
            'HARP_FDN_optimized': HARP_FDN_optimized
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
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

    def _save_resume_checkpoint(self, path, epoch, model_optim, early_stopping, model_ema=None):
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
        if model_ema is not None:
            checkpoint['model_ema_state_dict'] = model_ema.shadow
            checkpoint['model_ema_decay'] = model_ema.decay
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, self._resume_checkpoint_path(path))

    def _restore_rng_state(self, checkpoint):
        if 'torch_rng_state' in checkpoint:
            torch_rng_state = checkpoint['torch_rng_state']
            if not isinstance(torch_rng_state, torch.Tensor):
                raise TypeError('torch_rng_state must be a torch.Tensor, got {}'.format(type(torch_rng_state)))
            torch.set_rng_state(torch_rng_state.detach().cpu().to(torch.uint8))

        if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
            cuda_rng_state_all = checkpoint['cuda_rng_state_all']
            if not isinstance(cuda_rng_state_all, (list, tuple)):
                raise TypeError(
                    'cuda_rng_state_all must be a list or tuple, got {}'.format(type(cuda_rng_state_all))
                )
            cuda_rng_state_all = [
                state.detach().cpu().to(torch.uint8) if isinstance(state, torch.Tensor) else state
                for state in cuda_rng_state_all
            ]
            torch.cuda.set_rng_state_all(cuda_rng_state_all)

        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])

    def _load_resume_checkpoint(self, path, model_optim, early_stopping, model_ema_decay):
        resume_path = self._resume_checkpoint_path(path)
        if not os.path.exists(resume_path):
            message = 'Resume requested, but no checkpoint found at {}'.format(resume_path)
            print(message, flush=True)
            raise FileNotFoundError(message)

        try:
            checkpoint = self._torch_load(resume_path, map_location=self.device)
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError('resume checkpoint missing keys: {}'.format(', '.join(missing_keys)))

            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])

            early_state = checkpoint.get('early_stopping_state', {})
            early_stopping.counter = early_state.get('counter', early_stopping.counter)
            early_stopping.best_score = early_state.get('best_score', early_stopping.best_score)
            early_stopping.early_stop = early_state.get('early_stop', early_stopping.early_stop)
            early_stopping.val_loss_min = early_state.get('val_loss_min', early_stopping.val_loss_min)

            model_ema = None
            if 'model_ema_state_dict' in checkpoint:
                model_ema = ModelEMA(
                    self.model,
                    decay=float(checkpoint.get('model_ema_decay', model_ema_decay))
                )
                model_ema.shadow = checkpoint['model_ema_state_dict']

            self._restore_rng_state(checkpoint)
        except Exception as exc:
            print(
                'Failed to resume from checkpoint: {}\nReason: {}'.format(resume_path, repr(exc)),
                flush=True
            )
            raise

        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        print('Resuming training from {} at epoch {}'.format(resume_path, start_epoch + 1), flush=True)
        return start_epoch, model_ema

    def _main_forecast_loss(self, outputs, batch_y, mse_criterion, mae_criterion):
        objective = getattr(self.args, 'train_objective', 'mae').lower()

        if objective == 'mse':
            return mse_criterion(outputs, batch_y)
        if objective == 'mix':
            mse_weight = float(getattr(self.args, 'mse_loss_weight', 0.5))
            return mae_criterion(outputs, batch_y) + mse_weight * mse_criterion(outputs, batch_y)
        return mae_criterion(outputs, batch_y)

    def _validation_criterion(self, mse_criterion, mae_criterion):
        objective = getattr(self.args, 'vali_objective', 'mae').lower()
        if objective == 'mse':
            return mse_criterion
        return mae_criterion

    def _get_loss_ratio(self, dtype):
        ratio = np.array(
            [-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)]
        )
        ratio = torch.tensor(ratio, dtype=dtype, device=self.device).unsqueeze(-1)
        return ratio

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        phase = 'Test' if is_test else 'Vali'

        vali_bar = tqdm(
            vali_loader,
            desc=phase,
            leave=True,
            dynamic_ncols=True,
            disable=False,
            mininterval=0.1,
            miniters=1
        )

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs, con_loss_cls_1, con_loss_cls_2, con_loss_time = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                use_vali_ratio = bool(getattr(self.args, 'use_vali_ratio', 1))
                if not is_test and use_vali_ratio:
                    ratio = self._get_loss_ratio(outputs.dtype)
                    pred = outputs * ratio
                    true = batch_y * ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)
                loss += (
                    self.con_loss_coe_cls_1 * con_loss_cls_1
                    + self.con_loss_coe_cls_2 * con_loss_cls_2
                    + self.con_loss_coe_time * con_loss_time
                )

                total_loss.append(loss.item())
                vali_bar.set_postfix(loss=f'{loss.item():.7f}')

        total_loss = np.average(total_loss)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse = metric(preds, trues)

        print(
            '{} Loss: {:.7f} | MSE: {:.7f} MAE: {:.7f}'.format(
                phase, total_loss, mse, mae
            ),
            flush=True
        )
        self.model.train()
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
        use_model_ema = bool(getattr(self.args, 'use_model_ema', 0))
        model_ema_decay = float(getattr(self.args, 'model_ema_decay', 0.995))
        model_ema_start_epoch = int(getattr(self.args, 'model_ema_start_epoch', 1))
        model_ema = None
        if use_model_ema:
            print(
                'Using model EMA for validation/checkpointing | decay={:.6f} start_epoch={}'.format(
                    model_ema_decay, model_ema_start_epoch
                ),
                flush=True
            )

        resume = bool(getattr(self.args, 'resume', 0))
        if resume:
            start_epoch, resumed_model_ema = self._load_resume_checkpoint(
                path, model_optim, early_stopping, model_ema_decay
            )
        else:
            start_epoch, resumed_model_ema = 0, None
            resume_path = self._resume_checkpoint_path(path)
            if os.path.exists(resume_path):
                print(
                    'Resume disabled; ignoring existing checkpoint at {}'.format(resume_path),
                    flush=True
                )
        if use_model_ema:
            model_ema = resumed_model_ema
        if start_epoch >= self.args.train_epochs:
            print('Resume checkpoint already reached train_epochs; loading best checkpoint.', flush=True)

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            train_bar = tqdm(
                train_loader,
                desc=f'Epoch {epoch + 1}/{self.args.train_epochs}',
                leave=True,
                dynamic_ncols=True,
                disable=False,
                mininterval=0.1,
                miniters=1
            )

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_bar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs, con_loss_cls_1, con_loss_cls_2, con_loss_time = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                ratio = self._get_loss_ratio(outputs.dtype)
                outputs = outputs * ratio
                batch_y = batch_y * ratio

                loss = self._main_forecast_loss(outputs, batch_y, mse_criterion, mae_criterion)
                loss += (
                    self.con_loss_coe_cls_1 * con_loss_cls_1
                    + self.con_loss_coe_cls_2 * con_loss_cls_2
                    + self.con_loss_coe_time * con_loss_time
                )

                train_loss.append(loss.item())
                train_bar.set_postfix(loss=f'{loss.item():.7f}')

                if (i + 1) % 100 == 0:
                    print(
                        '\titers: {0}, epoch: {1} | loss: {2:.7f}'.format(
                            i + 1, epoch + 1, loss.item()
                        ),
                        flush=True
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), flush=True)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                if use_model_ema and (epoch + 1) >= model_ema_start_epoch:
                    if model_ema is None:
                        model_ema = ModelEMA(self.model, decay=model_ema_decay)
                        print(
                            'Initialized model EMA at epoch {} step {}'.format(epoch + 1, i + 1),
                            flush=True
                        )
                    else:
                        model_ema.update(self.model)

            print('Epoch: {} cost time: {}'.format(epoch + 1, time.time() - epoch_time), flush=True)

            train_loss = np.average(train_loss)
            vali_criterion = self._validation_criterion(mse_criterion, mae_criterion)
            if model_ema is not None:
                model_ema.store(self.model)
                model_ema.copy_to(self.model)
            vali_loss = self.vali(vali_data, vali_loader, vali_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print(
                'Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}'.format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                ),
                flush=True
            )

            early_stopping(vali_loss, self.model, path)
            if model_ema is not None:
                model_ema.restore(self.model)

            if early_stopping.early_stop:
                print('Early stopping', flush=True)
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            self._save_resume_checkpoint(path, epoch, model_optim, early_stopping, model_ema)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(self._torch_load(best_model_path))
        resume_path = self._resume_checkpoint_path(path)
        if os.path.exists(resume_path):
            os.remove(resume_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            tqdm.write('loading model')
            self.model.load_state_dict(
                self._torch_load(os.path.join(self._checkpoint_dir(setting), 'checkpoint.pth'))
            )

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        test_bar = tqdm(
            test_loader,
            desc='Testing',
            leave=True,
            dynamic_ncols=True,
            disable=False,
            mininterval=0.1,
            miniters=1
        )

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

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

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        tqdm.write('test shape: {} {}'.format(preds.shape, trues.shape))

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        tqdm.write('test shape: {} {}'.format(preds.shape, trues.shape))

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae), flush=True)
        
        res_dir = "./results"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # 按模型名定义单独日志文件
        model_name = self.args.model
        res_file = os.path.join(res_dir, f"{model_name}.txt")
        with open(res_file, 'a', encoding='utf-8') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')

        return mae, mse
