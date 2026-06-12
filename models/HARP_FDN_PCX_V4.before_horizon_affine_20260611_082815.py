import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN


class WeakResidualCalibrator(nn.Module):
    def __init__(self, channels, strength_init=-8.0):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y, uncertainty):
        # y: [B, L, C], uncertainty: [B, L, C]. Only high-uncertainty positions receive calibration.
        unc = uncertainty.detach()
        centered = torch.relu(unc - unc.mean(dim=1, keepdim=True))
        denom = centered.amax(dim=1, keepdim=True).clamp_min(1e-6)
        gate = torch.sigmoid(self.strength) * (centered / denom).clamp(0.0, 1.0)
        delta = self.net(y.transpose(1, 2)).transpose(1, 2)
        return y + gate * delta


class DirectResidualAdapter(nn.Module):
    def __init__(self, seq_len, pred_len, dropout=0.05, strength_init=-4.0):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))
        self.proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, history, output, uncertainty=None):
        last = history[:, -1:, :]
        centered = history - last
        delta = self.proj(centered.transpose(1, 2)).transpose(1, 2)
        delta = self.dropout(delta)
        gate = torch.sigmoid(self.strength)
        if uncertainty is not None:
            gate = gate * (0.75 + 0.25 * uncertainty.detach().clamp(0.0, 1.0))
        return output + gate * delta


class LocalTrendAnchor(nn.Module):
    def __init__(self, pred_len, channels, window=6, strength_init=-6.0):
        super().__init__()
        self.pred_len = pred_len
        self.window = int(window)
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))
        self.channel_gate = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, history, output, uncertainty=None):
        # history/output are in normalized space. The anchor extrapolates only the last local slope.
        steps = torch.arange(1, output.size(1) + 1, device=output.device, dtype=output.dtype).view(1, -1, 1)
        window = min(max(self.window, 1), history.size(1) - 1)
        slope = (history[:, -1:, :] - history[:, -1 - window:-window, :]) / float(window)
        anchor = history[:, -1:, :] + steps * slope
        gate = torch.sigmoid(self.strength) * torch.sigmoid(self.channel_gate)
        if uncertainty is not None:
            gate = gate * (0.75 + 0.25 * uncertainty.detach().clamp(0.0, 1.0))
        return output + gate * (anchor - output)


class ConditionalResidualCoupler(nn.Module):
    def __init__(self, channels, hidden_ratio=2, strength_init=-8.0):
        super().__init__()
        hidden = max(channels * hidden_ratio, 8)
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))
        self.net = nn.Sequential(
            nn.Conv1d(channels * 4, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels * 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, residual, proto, fine_abs, uncertainty):
        feat = torch.cat([
            residual,
            proto,
            fine_abs - proto,
            uncertainty.detach(),
        ], dim=-1).transpose(1, 2)
        shift, log_scale = self.net(feat).transpose(1, 2).chunk(2, dim=-1)
        strength = torch.sigmoid(self.strength)
        scale = 1.0 + strength * torch.tanh(log_scale)
        return residual * scale + strength * shift


class ConditionalResidualLogitCoupler(nn.Module):
    def __init__(self, channels, num_support, hidden_ratio=4, strength_init=-8.0):
        super().__init__()
        self.channels = channels
        self.num_support = num_support
        hidden = max(channels * hidden_ratio, 16)
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))
        self.net = nn.Sequential(
            nn.Conv1d(channels * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels * num_support, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, logits, proto, fine_abs, uncertainty):
        b, total, c = logits.shape
        length = total // self.num_support
        base = logits.reshape(b, length, self.num_support, c)
        feat = torch.cat([
            proto,
            fine_abs - proto,
            uncertainty.detach(),
        ], dim=-1).transpose(1, 2)
        bias = self.net(feat).transpose(1, 2).reshape(b, length, self.num_support, c)
        return (base + torch.sigmoid(self.strength) * bias).reshape(b, total, c)


class Model(nn.Module):
    """
    HARP_FDN_PCX_V4: short-horizon optimized prototype-residual model.

    It keeps the original InterFDN decoding path as the dominant path, then adds
    weakly initialized prototype residual decoding and posterior-aware calibration.
    At initialization it is intentionally close to InterFDN, so the model should
    not collapse while we optimize modules one by one.
    """
    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        self.pred_len = pred_len
        self.c_in = c_in
        self.revin = configs.revin
        self.ma_type = configs.ma_type
        self.num_support = int(getattr(configs, 'num_support', 25))
        self.support_temperature = float(getattr(configs, 'support_temperature', 1.0))
        self.use_relative_decode = int(getattr(configs, 'use_relative_decode', 1))
        self.use_posterior_calib = int(getattr(configs, 'use_posterior_calib', 1))
        self.use_residual_support = int(getattr(configs, 'use_residual_support', 1))
        self.transport_var_weight = float(getattr(configs, 'transport_var_weight', 0.0))
        self.posterior_kl_weight = float(getattr(configs, 'posterior_kl_weight', 0.0))
        self.horizon_gate_strength = float(getattr(configs, 'horizon_gate_strength', 1.0))
        self.use_conditional_residual = int(getattr(configs, 'use_conditional_residual', 1))
        self.use_conditional_residual_logits = int(getattr(configs, 'use_conditional_residual_logits', 0))
        self.use_direct_adapter = int(getattr(configs, 'use_direct_adapter', 0))
        self.use_trend_anchor = int(getattr(configs, 'use_trend_anchor', 0))
        self.fusion_uncertainty_weight = float(getattr(configs, 'fusion_uncertainty_weight', 0.0))

        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)

        self.net_1 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')
        self.net_2 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')
        self.net_3 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'grain')
        self.net_4 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'grain')

        df = pd.read_excel(getattr(configs, 'support_path', 'idx2.xlsx'), usecols=[1, 2], header=None, skiprows=1, nrows=self.num_support)
        idx_1 = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(self.num_support, 1)
        idx_2 = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(self.num_support, 1)
        self.register_buffer('idx_1', idx_1)
        self.register_buffer('idx_2', idx_2)
        res_scale = float(getattr(configs, 'residual_support_scale', 0.35))
        self.register_buffer('res_idx_1', (idx_1 - idx_1.mean(dim=0, keepdim=True)) * res_scale)
        self.register_buffer('res_idx_2', (idx_2 - idx_2.mean(dim=0, keepdim=True)) * res_scale)

        self.relative_gate = nn.Parameter(torch.tensor(float(getattr(configs, 'relative_strength_init', -8.0))))
        self.residual_scale = nn.Parameter(torch.tensor(float(getattr(configs, 'relative_residual_scale_init', 0.5))))
        self.prototype_gate = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.prototype_gate[-1].weight)
        nn.init.constant_(self.prototype_gate[-1].bias, float(getattr(configs, 'prototype_gate_bias_init', -8.0)))
        self.horizon_gate = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.horizon_gate[-1].weight)
        nn.init.constant_(self.horizon_gate[-1].bias, float(getattr(configs, 'horizon_gate_bias_init', 0.0)))
        self.residual_coupler = ConditionalResidualCoupler(
            c_in,
            hidden_ratio=int(getattr(configs, 'conditional_residual_hidden_ratio', 2)),
            strength_init=float(getattr(configs, 'conditional_residual_strength_init', -8.0)),
        )
        self.residual_logit_coupler = ConditionalResidualLogitCoupler(
            c_in,
            self.num_support,
            hidden_ratio=int(getattr(configs, 'conditional_residual_logit_hidden_ratio', 4)),
            strength_init=float(getattr(configs, 'conditional_residual_logit_strength_init', -8.0)),
        )
        self.direct_adapter = None
        if self.use_direct_adapter:
            self.direct_adapter = DirectResidualAdapter(
                seq_len,
                pred_len,
                dropout=float(getattr(configs, 'direct_adapter_dropout', 0.05)),
                strength_init=float(getattr(configs, 'direct_adapter_strength_init', -4.0)),
            )
        self.trend_anchor = None
        if self.use_trend_anchor:
            self.trend_anchor = LocalTrendAnchor(
                pred_len,
                c_in,
                window=int(getattr(configs, 'trend_anchor_window', 6)),
                strength_init=float(getattr(configs, 'trend_anchor_strength_init', -6.0)),
            )
        self.calibrator = WeakResidualCalibrator(c_in, getattr(configs, 'posterior_calib_strength_init', -8.0))

    def _decode(self, logits, support):
        b, total, c = logits.shape
        length = total // self.num_support
        logits = logits.reshape(b, length, self.num_support, c)
        prob = torch.softmax(logits / max(self.support_temperature, 1e-4), dim=2)
        mean = torch.einsum('blec,ei->blic', prob, support).squeeze(2)
        support_values = support.view(1, 1, self.num_support, 1)
        var = (prob * (support_values - mean.unsqueeze(2)).pow(2)).sum(dim=2)
        return mean, var, prob

    def _branch_confidence(self, prob, var):
        conf = prob.max(dim=2).values
        if self.fusion_uncertainty_weight > 0:
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=2)
            entropy = entropy / torch.log(torch.tensor(float(self.num_support), device=prob.device, dtype=prob.dtype))
            var_scale = var.detach().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            var_score = var.sqrt() / var_scale
            weight = self.fusion_uncertainty_weight
            conf = conf * torch.exp(-weight * entropy) * torch.exp(-0.5 * weight * var_score)
        return conf

    def _confidence_fuse(self, mean_a, var_a, prob_a, mean_b, var_b, prob_b):
        if self.fusion_uncertainty_weight <= 0:
            conf_a = prob_a.max(dim=2).values.permute(0, 2, 1)
            conf_b = prob_b.max(dim=2).values.permute(0, 2, 1)
            mask = (conf_a / (conf_a + conf_b + 1e-6)).permute(0, 2, 1)
            mean = mean_a * mask + mean_b * (1.0 - mask)
            var = var_a * mask + var_b * (1.0 - mask)
            return mean, var, mask

        conf_a = self._branch_confidence(prob_a, var_a)
        conf_b = self._branch_confidence(prob_b, var_b)
        mask = conf_a / (conf_a + conf_b + 1e-6)
        mean = mean_a * mask + mean_b * (1.0 - mask)
        var = mask.pow(2) * var_a + (1.0 - mask).pow(2) * var_b
        var = var + mask * (mean_a - mean).pow(2) + (1.0 - mask) * (mean_b - mean).pow(2)
        return mean, var, mask

    def _posterior_fuse(self, prob_a, prob_b, mask):
        mask = mask.unsqueeze(2)
        return prob_a * mask + prob_b * (1.0 - mask)

    def _horizon_gate(self, length, device, dtype):
        pos = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype).view(1, length, 1)
        feats = torch.cat([
            pos,
            pos.pow(2),
            torch.sin(pos * torch.pi),
            torch.cos(pos * torch.pi),
        ], dim=-1)
        gate = torch.sigmoid(self.horizon_gate(feats))
        return 1.0 + self.horizon_gate_strength * (gate - 0.5)

    def _kl_to_coarse(self, fine_prob, coarse_prob):
        fine_down = fine_prob[:, :coarse_prob.size(1) * 4, :, :]
        fine_down = fine_down.reshape(fine_prob.size(0), coarse_prob.size(1), 4, self.num_support, fine_prob.size(3)).mean(dim=2)
        fine_down = fine_down.clamp_min(1e-6)
        coarse_prob = coarse_prob.clamp_min(1e-6)
        return F.kl_div(fine_down.log(), coarse_prob, reduction='batchmean') / max(coarse_prob.size(1) * coarse_prob.size(2), 1)

    def _condition_residual_logits(self, logits, proto, fine_abs, uncertainty):
        if not self.use_conditional_residual_logits:
            return logits
        return self.residual_logit_coupler(logits, proto, fine_abs, uncertainty)

    def _prototype_residual(self, fine_abs, fine_res, grain, uncertainty, x_norm, proto_var=None):
        if not self.use_relative_decode:
            return fine_abs
        proto = F.interpolate(grain.transpose(1, 2), size=fine_abs.size(1), mode='linear', align_corners=False).transpose(1, 2)
        if proto_var is None:
            proto_unc = uncertainty.mean(dim=1)
        else:
            proto_unc = F.interpolate(proto_var.transpose(1, 2), size=fine_abs.size(1), mode='linear', align_corners=False).transpose(1, 2).mean(dim=1)
        hist = torch.stack([
            x_norm.mean(dim=1),
            x_norm.std(dim=1, unbiased=False),
            x_norm[:, -1, :],
            x_norm[:, -1, :] - x_norm[:, 0, :],
            uncertainty.mean(dim=1),
            proto_unc,
        ], dim=-1)
        channel_gate = torch.sigmoid(self.prototype_gate(hist)).permute(0, 2, 1)
        horizon_gate = self._horizon_gate(fine_abs.size(1), fine_abs.device, fine_abs.dtype)
        global_gate = torch.sigmoid(self.relative_gate)
        gate = global_gate * channel_gate * horizon_gate * uncertainty.detach().clamp(0.0, 1.0)
        gate = gate.clamp(0.0, 1.0)
        residual = torch.tanh(self.residual_scale) * fine_res
        if self.use_conditional_residual:
            residual = self.residual_coupler(residual, proto, fine_abs, uncertainty)
        relative = proto + residual
        return fine_abs * (1.0 - gate) + relative * gate

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x_norm = x

        if self.ma_type == 'reg':
            seasonal_init, trend_init = x, x
        else:
            seasonal_init, trend_init = self.decomp(x)

        x_1 = self.net_1(seasonal_init, trend_init)
        x_2 = self.net_2(seasonal_init, trend_init)
        x_3 = self.net_3(seasonal_init, trend_init)
        x_4 = self.net_4(seasonal_init, trend_init)

        mean_1, var_1, prob_1 = self._decode(x_1, self.idx_1)
        mean_2, var_2, prob_2 = self._decode(x_2, self.idx_2)
        output_fine, var_fine, mask_fine = self._confidence_fuse(mean_1, var_1, prob_1, mean_2, var_2, prob_2)
        fine_prob = self._posterior_fuse(prob_1, prob_2, mask_fine)

        mean_3, var_3, prob_3 = self._decode(x_3, self.idx_1)
        mean_4, var_4, prob_4 = self._decode(x_4, self.idx_2)
        output_grain, var_grain, mask_grain = self._confidence_fuse(mean_3, var_3, prob_3, mean_4, var_4, prob_4)
        grain_prob = self._posterior_fuse(prob_3, prob_4, mask_grain)

        uncertainty = (var_fine / (var_fine.detach().mean(dim=1, keepdim=True) + 1e-6)).sigmoid()
        proto_for_residual = F.interpolate(
            output_grain.transpose(1, 2),
            size=output_fine.size(1),
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)
        if self.use_residual_support:
            cond_x_1 = self._condition_residual_logits(x_1, proto_for_residual, output_fine, uncertainty)
            cond_x_2 = self._condition_residual_logits(x_2, proto_for_residual, output_fine, uncertainty)
            res_1, _, _ = self._decode(cond_x_1, self.res_idx_1)
            res_2, _, _ = self._decode(cond_x_2, self.res_idx_2)
            residual_fine = res_1 * mask_fine + res_2 * (1.0 - mask_fine)
        else:
            residual_fine = output_fine

        grain_len = output_grain.size(1)
        output_down_fine = output_fine[:, :grain_len * 4, :].reshape(output_fine.size(0), grain_len, 4, output_fine.size(2)).mean(dim=2)
        var_down_fine = var_fine[:, :grain_len * 4, :].reshape(var_fine.size(0), grain_len, 4, var_fine.size(2)).mean(dim=2)

        con_loss_time = F.mse_loss(output_grain, output_down_fine)
        if self.transport_var_weight > 0:
            con_loss_time = con_loss_time + self.transport_var_weight * F.mse_loss(var_grain, var_down_fine)
        if self.posterior_kl_weight > 0:
            con_loss_time = con_loss_time + self.posterior_kl_weight * self._kl_to_coarse(fine_prob, grain_prob)
        con_loss_cls_1 = F.mse_loss(mean_2, mean_1)
        con_loss_cls_2 = F.mse_loss(mean_4, mean_3)

        output = self._prototype_residual(output_fine, residual_fine, output_grain, uncertainty, x_norm, var_grain)
        if self.use_trend_anchor:
            output = self.trend_anchor(x_norm, output, uncertainty)
        if self.use_direct_adapter:
            output = self.direct_adapter(x_norm, output, uncertainty)
        if self.use_posterior_calib:
            output = self.calibrator(output, uncertainty)

        if self.revin:
            output = self.revin_layer(output, 'denorm')
        return output, con_loss_cls_1, con_loss_cls_2, con_loss_time
