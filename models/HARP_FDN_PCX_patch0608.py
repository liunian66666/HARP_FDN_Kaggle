import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.decomp import DECOMP
from layers.revin import RevIN


class MixerBlock(nn.Module):
    def __init__(self, patch_num, d_model, dropout):
        super(MixerBlock, self).__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(patch_num, patch_num * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_num * 2, patch_num),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B*C, P, D]
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class PatchMixerStream(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 d_model, e_layers, dropout):
        super(PatchMixerStream, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.patch_pos = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        self.blocks = nn.ModuleList([
            MixerBlock(self.patch_num, d_model, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.horizon_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.patch_num * d_model, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len),
        )

        nn.init.trunc_normal_(self.patch_pos, std=0.02)

    def forward(self, x):
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)

        patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patch = self.patch_embed(patch) + self.patch_pos
        for block in self.blocks:
            patch = block(patch)
        patch = self.norm(patch)

        pred = self.horizon_head(patch)
        summary = patch.mean(dim=1)
        return pred, summary


class ProbabilityMixerNetwork(nn.Module):
    """
    A channel-independent multi-scale patch mixer used to generate support logits.

    It keeps the original interPDN decomposition interface: seasonal and trend
    inputs are encoded separately. The seasonal stream is represented by two
    patch resolutions: a short stream for local deformation and the original
    stream for periodic fragments. Their mixture is input-conditioned before
    support logits are generated.
    """

    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, scale,
                 d_model=96, e_layers=2, dropout=0.1, num_support=25):
        super(ProbabilityMixerNetwork, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.scale = scale
        self.num_support = num_support

        short_patch_len = max(4, patch_len // 2)
        short_stride = max(1, stride // 2)

        self.short_stream = PatchMixerStream(
            seq_len, pred_len, short_patch_len, short_stride, padding_patch,
            d_model, e_layers, dropout
        )
        self.long_stream = PatchMixerStream(
            seq_len, pred_len, patch_len, stride, padding_patch,
            d_model, e_layers, dropout
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, pred_len),
            nn.Sigmoid(),
        )

        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.trend_gate = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Sigmoid(),
        )

        out_len = pred_len * num_support if scale == 'fine' else int(pred_len * 6.25)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(pred_len * 3),
            nn.Linear(pred_len * 3, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, out_len),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        s_center = s - s[:, -1:].detach()
        t_center = t - t[:, -1:].detach()

        short_pred, short_summary = self.short_stream(s_center)
        long_pred, long_summary = self.long_stream(s_center)
        scale_gate = self.scale_gate(torch.cat([short_summary, long_summary], dim=1))
        patch_pred = scale_gate * short_pred + (1.0 - scale_gate) * long_pred

        seasonal_pred = self.seasonal_linear(s_center)
        trend_base = self.trend_linear(t_center)
        trend_gate = self.trend_gate(t_center)
        trend_pred = trend_gate * trend_base + (1.0 - trend_gate) * t[:, -1:]

        features = torch.cat([patch_pred, seasonal_pred, trend_pred], dim=1)
        logits = self.logit_head(features)
        logits = torch.reshape(logits, (B, C, logits.size(-1))).permute(0, 2, 1)
        return logits


class Model(nn.Module):
    """
    HARP-FDN-PCX.

    This version keeps the multi-scale support-logit generator from MSX and
    adds posterior-gated residual calibration. The calibration path is weak and
    uncertainty-aware: it is mainly used to correct support quantization bias
    when the decoded posterior itself is uncertain.
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

        self.num_support = int(getattr(configs, 'num_support', 25))
        mixer_d_model = int(getattr(configs, 'mixer_d_model', 96))
        mixer_layers = int(getattr(configs, 'mixer_layers', 2))
        mixer_dropout = float(getattr(configs, 'mixer_dropout', 0.1))

        self.net_1 = ProbabilityMixerNetwork(
            self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'fine',
            d_model=mixer_d_model, e_layers=mixer_layers, dropout=mixer_dropout,
            num_support=self.num_support
        )
        self.net_2 = ProbabilityMixerNetwork(
            self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'fine',
            d_model=mixer_d_model, e_layers=mixer_layers, dropout=mixer_dropout,
            num_support=self.num_support
        )
        self.net_3 = ProbabilityMixerNetwork(
            self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'grain',
            d_model=mixer_d_model, e_layers=mixer_layers, dropout=mixer_dropout,
            num_support=self.num_support
        )
        self.net_4 = ProbabilityMixerNetwork(
            self.seq_len, self.pred_len, patch_len, stride, padding_patch, 'grain',
            d_model=mixer_d_model, e_layers=mixer_layers, dropout=mixer_dropout,
            num_support=self.num_support
        )

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
        self.use_posterior_calib = bool(getattr(configs, 'use_posterior_calib', 1))

        self.transport_var_weight = float(getattr(configs, 'transport_var_weight', 0.05))
        self.anchor_cert_temperature = float(getattr(configs, 'anchor_cert_temperature', 1.0))

        relative_strength_init = float(getattr(configs, 'relative_strength_init', -2.5))
        residual_scale_init = float(getattr(configs, 'relative_residual_scale_init', 0.5))
        uncertainty_weight_init = float(getattr(configs, 'uncertainty_fusion_weight_init', 0.25))
        posterior_calib_strength_init = float(getattr(configs, 'posterior_calib_strength_init', -3.0))

        self.relative_strength = nn.Parameter(torch.tensor(relative_strength_init, dtype=torch.float32))
        self.residual_scale_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(residual_scale_init), dtype=torch.float32)
        )
        self.uncertainty_weight_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(uncertainty_weight_init), dtype=torch.float32)
        )
        self.posterior_calib_strength = nn.Parameter(
            torch.tensor(posterior_calib_strength_init, dtype=torch.float32)
        )
        self.posterior_calib = nn.Linear(self.seq_len, self.pred_len)
        nn.init.zeros_(self.posterior_calib.weight)
        nn.init.zeros_(self.posterior_calib.bias)

    @staticmethod
    def _inverse_softplus(value):
        value = max(float(value), 1e-4)
        return math.log(math.exp(value) - 1.0)

    def _reshape_logits(self, logits):
        B, TN, C = logits.shape
        T = TN // self.num_support
        return logits.reshape(B, T, self.num_support, C)

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

        anchor_cert = torch.exp(
            -anchor_var.detach() / max(self.anchor_cert_temperature, 1e-4)
        )
        gate = torch.sigmoid(self.relative_strength) * anchor_cert
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
        return mean_loss + self.transport_var_weight * var_loss

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
        if self.use_posterior_calib:
            output_fine = self._posterior_calibrate(output_fine, var_fine, x_norm)

        con_loss_time = self._distribution_transport_loss(
            output_fine, var_fine, output_grain, var_grain
        )
        con_loss_cls_1 = F.mse_loss(out_2, out_1)
        con_loss_cls_2 = F.mse_loss(coarse_2, coarse_1)

        if self.revin:
            output_fine = self.revin_layer(output_fine, 'denorm')

        return output_fine, con_loss_cls_1, con_loss_cls_2, con_loss_time

    def _posterior_calibrate(self, output, var, x_norm):
        src = x_norm.permute(0, 2, 1)
        src = src - src[:, :, -1:].detach()
        residual = self.posterior_calib(src).permute(0, 2, 1)

        sigma = var.detach().sqrt()
        sigma_ref = sigma.mean(dim=1, keepdim=True).clamp_min(1e-6)
        uncertainty_gate = torch.sigmoid(sigma / sigma_ref - 1.0)
        strength = torch.sigmoid(self.posterior_calib_strength)
        return output + strength * uncertainty_gate * residual
