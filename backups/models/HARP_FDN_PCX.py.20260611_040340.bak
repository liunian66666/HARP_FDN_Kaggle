# 似乎是原来丢失的那版
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN


class Model(nn.Module):
    """
    HARP-FDN v2: Hierarchical Anchor-Relative Probability Forecasting Network.

    Compared with HARP-FDN v1, v2 keeps the same single-model backbone and
    strengthens the probabilistic head with three structural changes:

    1. Horizon-segmented anchor-relative decoding.
    2. Cross-scale mean/variance/spectrum distribution transport.
    3. Low-degree NLinear-style calibration after probabilistic decoding.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        self.revin = configs.revin
        self.revin_layer = RevIN(self.c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)

        self.net_1 = Network(self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'fine')
        self.net_2 = Network(self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'fine')
        self.net_3 = Network(self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'grain')
        self.net_4 = Network(self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'grain')

        self.num_support = int(getattr(configs, 'num_support', 25))
        support_path = getattr(configs, 'support_path', 'idx2.xlsx')
        df = pd.read_excel(
            support_path,
            usecols=[1, 2],
            header=None,
            skiprows=1,
            nrows=self.num_support
        )

        idx_1 = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(self.num_support, 1)
        idx_2 = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(self.num_support, 1)
        self.register_buffer('idx_1', idx_1)
        self.register_buffer('idx_2', idx_2)
        self.register_buffer('res_idx_1', idx_1 - idx_1.mean(dim=0, keepdim=True))
        self.register_buffer('res_idx_2', idx_2 - idx_2.mean(dim=0, keepdim=True))

        self.support_temperature = float(getattr(configs, 'support_temperature', 1.0))
        self.use_relative_decode = bool(getattr(configs, 'use_relative_decode', 1))
        self.use_distribution_transport = bool(getattr(configs, 'use_distribution_transport', 1))
        self.use_uncertainty_fusion = bool(getattr(configs, 'use_uncertainty_fusion', 1))
        self.use_horizon_segment = bool(getattr(configs, 'use_horizon_segment', 1))
        self.use_residual_correction = bool(getattr(configs, 'use_residual_correction', 1))
        self.use_spectral_transport = bool(getattr(configs, 'use_spectral_transport', 1))

        self.num_horizon_segments = int(getattr(configs, 'num_horizon_segments', 4))
        self.transport_var_weight = float(getattr(configs, 'transport_var_weight', 0.05))
        self.transport_spectral_weight = float(getattr(configs, 'transport_spectral_weight', 0.03))
        self.spectral_low_ratio = float(getattr(configs, 'spectral_low_ratio', 0.5))
        self.anchor_cert_temperature = float(getattr(configs, 'anchor_cert_temperature', 1.0))

        relative_strength_init = float(getattr(configs, 'relative_strength_init', -2.5))
        residual_scale_init = float(getattr(configs, 'relative_residual_scale_init', 0.5))
        uncertainty_weight_init = float(getattr(configs, 'uncertainty_fusion_weight_init', 0.25))
        correction_strength_init = float(getattr(configs, 'correction_strength_init', -2.2))

        self.relative_strength = nn.Parameter(torch.tensor(relative_strength_init, dtype=torch.float32))
        self.residual_scale_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(residual_scale_init), dtype=torch.float32)
        )
        self.uncertainty_weight_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(uncertainty_weight_init), dtype=torch.float32)
        )

        self.relative_segment_bias = nn.Parameter(torch.zeros(self.num_horizon_segments))
        self.correction_strength = nn.Parameter(torch.tensor(correction_strength_init, dtype=torch.float32))
        self.correction_segment_bias = nn.Parameter(torch.zeros(self.num_horizon_segments))

        self.correction_head = nn.Linear(self.seq_len, self.pred_len)
        self.correction_norm = nn.LayerNorm(self.pred_len)
        self.correction_dropout = nn.Dropout(float(getattr(configs, 'correction_dropout', 0.05)))

        self._init_correction()

    @staticmethod
    def _inverse_softplus(value):
        value = max(float(value), 1e-4)
        return math.log(math.exp(value) - 1.0)

    def _init_correction(self):
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _reshape_logits(self, logits):
        B, TN, C = logits.shape
        T = TN // self.num_support
        return logits.reshape(B, T, self.num_support, C)

    def _segment_gate(self, length, base_logit, segment_bias, dtype, device):
        if not self.use_horizon_segment:
            return torch.sigmoid(base_logit).to(dtype=dtype, device=device).view(1, 1, 1)

        segment = torch.arange(length, device=device)
        segment = torch.div(segment * self.num_horizon_segments, length, rounding_mode='floor')
        segment = segment.clamp_max(self.num_horizon_segments - 1)

        logits = base_logit + segment_bias.to(device=device)[segment]
        return torch.sigmoid(logits).to(dtype=dtype).view(1, length, 1)

    def _moment_decode(self, logits, support):
        logits = self._reshape_logits(logits)
        prob = F.softmax(logits / self.support_temperature, dim=2)
        support = support.to(dtype=logits.dtype, device=logits.device)
        mean = torch.einsum('btnc,ni->btic', prob, support).squeeze(2)
        second = torch.einsum('btnc,ni->btic', prob, support.pow(2)).squeeze(2)
        var = (second - mean.pow(2)).clamp_min(1e-6)
        return mean, prob, var

    def _relative_decode(self, prob, residual_support, anchor, anchor_var, absolute_output):
        residual_support = residual_support.to(dtype=prob.dtype, device=prob.device)
        residual = torch.einsum('btnc,ni->btic', prob, residual_support).squeeze(2)

        residual_scale = F.softplus(self.residual_scale_raw)
        relative_output = anchor + residual_scale * residual

        if not self.use_relative_decode:
            gate = torch.zeros_like(absolute_output)
            return absolute_output, gate

        base_gate = self._segment_gate(
            absolute_output.size(1),
            self.relative_strength,
            self.relative_segment_bias,
            absolute_output.dtype,
            absolute_output.device
        )
        anchor_cert = torch.exp(
            -anchor_var.detach() / max(self.anchor_cert_temperature, 1e-4)
        )
        gate = base_gate * anchor_cert
        output = absolute_output + gate * (relative_output - absolute_output)
        return output, gate

    def _upsample(self, x, target_len):
        if x.size(1) == target_len:
            return x
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.permute(0, 2, 1)

    def _branch_confidence(self, prob, var):
        conf = prob.max(dim=2)[0]
        if self.use_uncertainty_fusion:
            entropy = -(prob * (prob.clamp_min(1e-8).log())).sum(dim=2)
            entropy = entropy / math.log(self.num_support)
            weight = F.softplus(self.uncertainty_weight_raw)
            conf = conf * torch.exp(-weight * entropy) * torch.exp(-weight * var.sqrt())
        return conf

    def _fuse_pair(self, output_a, output_b, prob_a, prob_b, var_a, var_b):
        conf_a = self._branch_confidence(prob_a, var_a)
        conf_b = self._branch_confidence(prob_b, var_b)
        mask = conf_a / (conf_a + conf_b + 1e-8)
        output = output_a * mask + output_b * (1.0 - mask)

        var = mask.pow(2) * var_a + (1.0 - mask).pow(2) * var_b
        var = var + mask * (output_a - output).pow(2)
        var = var + (1.0 - mask) * (output_b - output).pow(2)
        return output, var, mask

    def _spectral_transport_loss(self, transported_mean, output_grain):
        if not self.use_spectral_transport:
            return transported_mean.new_tensor(0.0)

        fine_spec = torch.fft.rfft(transported_mean.float(), dim=1).abs()
        coarse_spec = torch.fft.rfft(output_grain.float(), dim=1).abs()

        keep = max(2, int(fine_spec.size(1) * self.spectral_low_ratio))
        fine_spec = torch.log1p(fine_spec[:, :keep])
        coarse_spec = torch.log1p(coarse_spec[:, :keep])

        weight = torch.linspace(1.0, 0.35, keep, device=fine_spec.device).view(1, keep, 1)
        return F.smooth_l1_loss(fine_spec * weight, coarse_spec * weight)

    def _distribution_transport_loss(self, output_fine, var_fine, output_grain, var_grain):
        B, T, C = output_fine.shape
        group = T // output_grain.size(1)

        fine_group = output_fine.reshape(B, output_grain.size(1), group, C)
        var_group = var_fine.reshape(B, output_grain.size(1), group, C)

        transported_mean = fine_group.mean(dim=2)
        temporal_spread = fine_group.var(dim=2, unbiased=False)
        transported_var = var_group.mean(dim=2) / group + temporal_spread

        mean_loss = F.mse_loss(transported_mean, output_grain)
        if not self.use_distribution_transport:
            return mean_loss

        var_loss = F.smooth_l1_loss(
            torch.log1p(transported_var.clamp_min(1e-6)),
            torch.log1p(var_grain.clamp_min(1e-6))
        )
        spectral_loss = self._spectral_transport_loss(transported_mean, output_grain)
        return mean_loss + self.transport_var_weight * var_loss + self.transport_spectral_weight * spectral_loss

    def _residual_correction(self, x, output, var):
        if not self.use_residual_correction:
            return output, output.new_tensor(0.0)

        last_value = x[:, -1:, :]
        centered = x - last_value

        delta = self.correction_head(centered.permute(0, 2, 1))
        delta = self.correction_norm(delta)
        delta = self.correction_dropout(delta)
        linear_calibration = last_value + delta.permute(0, 2, 1)

        base_gate = self._segment_gate(
            output.size(1),
            self.correction_strength,
            self.correction_segment_bias,
            output.dtype,
            output.device
        )
        uncertainty_gate = 0.5 + 0.5 * torch.tanh(var.detach().sqrt())
        gate = base_gate * uncertainty_gate

        correction = linear_calibration - output
        refined = output + gate * correction
        smooth_loss = (linear_calibration[:, 1:, :] - linear_calibration[:, :-1, :]).abs().mean()
        return refined, smooth_loss

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

        coarse_1, prob_3, var_3 = self._moment_decode(x_3, self.idx_1)
        coarse_2, prob_4, var_4 = self._moment_decode(x_4, self.idx_2)
        output_grain, var_grain, _ = self._fuse_pair(coarse_1, coarse_2, prob_3, prob_4, var_3, var_4)

        anchor_fine = self._upsample(output_grain, self.pred_len)
        anchor_var_fine = self._upsample(var_grain, self.pred_len)

        abs_1, prob_1, abs_var_1 = self._moment_decode(x_1, self.idx_1)
        abs_2, prob_2, abs_var_2 = self._moment_decode(x_2, self.idx_2)

        out_1, gate_1 = self._relative_decode(prob_1, self.res_idx_1, anchor_fine, anchor_var_fine, abs_1)
        out_2, gate_2 = self._relative_decode(prob_2, self.res_idx_2, anchor_fine, anchor_var_fine, abs_2)

        residual_scale = F.softplus(self.residual_scale_raw)
        rel_var_1 = residual_scale.pow(2) * abs_var_1
        rel_var_2 = residual_scale.pow(2) * abs_var_2
        var_1 = (1.0 - gate_1).pow(2) * abs_var_1 + gate_1.pow(2) * rel_var_1
        var_2 = (1.0 - gate_2).pow(2) * abs_var_2 + gate_2.pow(2) * rel_var_2

        output_fine, var_fine, _ = self._fuse_pair(out_1, out_2, prob_1, prob_2, var_1, var_2)
        output_fine, correction_smooth_loss = self._residual_correction(x_norm, output_fine, var_fine)

        con_loss_time = self._distribution_transport_loss(
            output_fine, var_fine, output_grain, var_grain
        )
        con_loss_time = con_loss_time + 0.01 * correction_smooth_loss
        con_loss_cls_1 = F.mse_loss(out_2, out_1)
        con_loss_cls_2 = F.mse_loss(coarse_2, coarse_1)

        if self.revin:
            output_fine = self.revin_layer(output_fine, 'denorm')

        return output_fine, con_loss_cls_1, con_loss_cls_2, con_loss_time
