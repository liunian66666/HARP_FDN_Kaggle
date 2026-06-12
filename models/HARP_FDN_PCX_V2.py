# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd

# from layers.decomp import DECOMP
# from layers.network import Network
# from layers.revin import RevIN


# class WeakResidualCalibrator(nn.Module):
#     def __init__(self, channels, strength_init=-8.0):
#         super().__init__()
#         self.strength = nn.Parameter(torch.tensor(float(strength_init)))
#         self.net = nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
#             nn.GELU(),
#             nn.Conv1d(channels, channels, kernel_size=1),
#         )
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, y, uncertainty):
#         # y: [B, L, C], uncertainty: [B, L, C]. Only high-uncertainty positions receive calibration.
#         unc = uncertainty.detach()
#         centered = torch.relu(unc - unc.mean(dim=1, keepdim=True))
#         denom = centered.amax(dim=1, keepdim=True).clamp_min(1e-6)
#         gate = torch.sigmoid(self.strength) * (centered / denom).clamp(0.0, 1.0)
#         delta = self.net(y.transpose(1, 2)).transpose(1, 2)
#         return y + gate * delta


# class ConditionalResidualCoupler(nn.Module):
#     def __init__(self, channels, hidden_ratio=2, strength_init=-8.0):
#         super().__init__()
#         hidden = max(channels * hidden_ratio, 8)
#         self.strength = nn.Parameter(torch.tensor(float(strength_init)))
#         self.net = nn.Sequential(
#             nn.Conv1d(channels * 4, hidden, kernel_size=1),
#             nn.GELU(),
#             nn.Conv1d(hidden, channels * 2, kernel_size=1),
#         )
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, residual, proto, fine_abs, uncertainty):
#         feat = torch.cat([
#             residual,
#             proto,
#             fine_abs - proto,
#             uncertainty.detach(),
#         ], dim=-1).transpose(1, 2)
#         shift, log_scale = self.net(feat).transpose(1, 2).chunk(2, dim=-1)
#         strength = torch.sigmoid(self.strength)
#         scale = 1.0 + strength * torch.tanh(log_scale)
#         return residual * scale + strength * shift


# class ConditionalResidualLogitCoupler(nn.Module):
#     def __init__(self, channels, num_support, hidden_ratio=4, strength_init=-8.0):
#         super().__init__()
#         self.channels = channels
#         self.num_support = num_support
#         hidden = max(channels * hidden_ratio, 16)
#         self.strength = nn.Parameter(torch.tensor(float(strength_init)))
#         self.net = nn.Sequential(
#             nn.Conv1d(channels * 3, hidden, kernel_size=1),
#             nn.GELU(),
#             nn.Conv1d(hidden, channels * num_support, kernel_size=1),
#         )
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, logits, proto, fine_abs, uncertainty):
#         b, total, c = logits.shape
#         length = total // self.num_support
#         base = logits.reshape(b, length, self.num_support, c)
#         feat = torch.cat([
#             proto,
#             fine_abs - proto,
#             uncertainty.detach(),
#         ], dim=-1).transpose(1, 2)
#         bias = self.net(feat).transpose(1, 2).reshape(b, length, self.num_support, c)
#         return (base + torch.sigmoid(self.strength) * bias).reshape(b, total, c)


# class Model(nn.Module):
#     """
#     HARP_FDN_PCX_V2: conservative reconstruction.

#     It keeps the original InterFDN decoding path as the dominant path, then adds
#     weakly initialized prototype residual decoding and posterior-aware calibration.
#     At initialization it is intentionally close to InterFDN, so the model should
#     not collapse while we optimize modules one by one.
#     """
#     def __init__(self, configs):
#         super().__init__()
#         seq_len = configs.seq_len
#         pred_len = configs.pred_len
#         c_in = configs.enc_in
#         patch_len = configs.patch_len
#         stride = configs.stride
#         padding_patch = configs.padding_patch

#         self.pred_len = pred_len
#         self.c_in = c_in
#         self.revin = configs.revin
#         self.ma_type = configs.ma_type
#         self.num_support = int(getattr(configs, 'num_support', 25))
#         self.support_temperature = float(getattr(configs, 'support_temperature', 1.0))
#         self.use_relative_decode = int(getattr(configs, 'use_relative_decode', 1))
#         self.use_posterior_calib = int(getattr(configs, 'use_posterior_calib', 1))
#         self.use_residual_support = int(getattr(configs, 'use_residual_support', 1))
#         self.transport_var_weight = float(getattr(configs, 'transport_var_weight', 0.0))
#         self.posterior_kl_weight = float(getattr(configs, 'posterior_kl_weight', 0.0))
#         self.horizon_gate_strength = float(getattr(configs, 'horizon_gate_strength', 1.0))
#         self.use_conditional_residual = int(getattr(configs, 'use_conditional_residual', 1))
#         self.use_conditional_residual_logits = int(getattr(configs, 'use_conditional_residual_logits', 1))

#         self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
#         self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)

#         self.net_1 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')
#         self.net_2 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')
#         self.net_3 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'grain')
#         self.net_4 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'grain')

#         df = pd.read_excel(getattr(configs, 'support_path', 'idx2.xlsx'), usecols=[1, 2], header=None, skiprows=1, nrows=self.num_support)
#         idx_1 = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(self.num_support, 1)
#         idx_2 = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(self.num_support, 1)
#         self.register_buffer('idx_1', idx_1)
#         self.register_buffer('idx_2', idx_2)
#         res_scale = float(getattr(configs, 'residual_support_scale', 0.35))
#         self.register_buffer('res_idx_1', (idx_1 - idx_1.mean(dim=0, keepdim=True)) * res_scale)
#         self.register_buffer('res_idx_2', (idx_2 - idx_2.mean(dim=0, keepdim=True)) * res_scale)

#         self.relative_gate = nn.Parameter(torch.tensor(float(getattr(configs, 'relative_strength_init', -8.0))))
#         self.residual_scale = nn.Parameter(torch.tensor(float(getattr(configs, 'relative_residual_scale_init', 0.5))))
#         self.prototype_gate = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
#         nn.init.zeros_(self.prototype_gate[-1].weight)
#         nn.init.constant_(self.prototype_gate[-1].bias, float(getattr(configs, 'prototype_gate_bias_init', -8.0)))
#         self.horizon_gate = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
#         nn.init.zeros_(self.horizon_gate[-1].weight)
#         nn.init.constant_(self.horizon_gate[-1].bias, float(getattr(configs, 'horizon_gate_bias_init', 0.0)))
#         self.residual_coupler = ConditionalResidualCoupler(
#             c_in,
#             hidden_ratio=int(getattr(configs, 'conditional_residual_hidden_ratio', 2)),
#             strength_init=float(getattr(configs, 'conditional_residual_strength_init', -8.0)),
#         )
#         self.residual_logit_coupler = ConditionalResidualLogitCoupler(
#             c_in,
#             self.num_support,
#             hidden_ratio=int(getattr(configs, 'conditional_residual_logit_hidden_ratio', 4)),
#             strength_init=float(getattr(configs, 'conditional_residual_logit_strength_init', -8.0)),
#         )
#         self.calibrator = WeakResidualCalibrator(c_in, getattr(configs, 'posterior_calib_strength_init', -8.0))

#     def _decode(self, logits, support):
#         b, total, c = logits.shape
#         length = total // self.num_support
#         logits = logits.reshape(b, length, self.num_support, c)
#         prob = torch.softmax(logits / max(self.support_temperature, 1e-4), dim=2)
#         mean = torch.einsum('blec,ei->blic', prob, support).squeeze(2)
#         support_values = support.view(1, 1, self.num_support, 1)
#         var = (prob * (support_values - mean.unsqueeze(2)).pow(2)).sum(dim=2)
#         return mean, var, prob

#     def _confidence_fuse(self, mean_a, var_a, prob_a, mean_b, var_b, prob_b):
#         conf_a = prob_a.max(dim=2).values.permute(0, 2, 1)
#         conf_b = prob_b.max(dim=2).values.permute(0, 2, 1)
#         mask = (conf_a / (conf_a + conf_b + 1e-6)).permute(0, 2, 1)
#         mean = mean_a * mask + mean_b * (1.0 - mask)
#         var = var_a * mask + var_b * (1.0 - mask)
#         return mean, var, mask

#     def _posterior_fuse(self, prob_a, prob_b, mask):
#         mask = mask.unsqueeze(2)
#         return prob_a * mask + prob_b * (1.0 - mask)

#     def _horizon_gate(self, length, device, dtype):
#         pos = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype).view(1, length, 1)
#         feats = torch.cat([
#             pos,
#             pos.pow(2),
#             torch.sin(pos * torch.pi),
#             torch.cos(pos * torch.pi),
#         ], dim=-1)
#         gate = torch.sigmoid(self.horizon_gate(feats))
#         return 1.0 + self.horizon_gate_strength * (gate - 0.5)

#     def _kl_to_coarse(self, fine_prob, coarse_prob):
#         fine_down = fine_prob[:, :coarse_prob.size(1) * 4, :, :]
#         fine_down = fine_down.reshape(fine_prob.size(0), coarse_prob.size(1), 4, self.num_support, fine_prob.size(3)).mean(dim=2)
#         fine_down = fine_down.clamp_min(1e-6)
#         coarse_prob = coarse_prob.clamp_min(1e-6)
#         return F.kl_div(fine_down.log(), coarse_prob, reduction='batchmean') / max(coarse_prob.size(1) * coarse_prob.size(2), 1)

#     def _condition_residual_logits(self, logits, proto, fine_abs, uncertainty):
#         if not self.use_conditional_residual_logits:
#             return logits
#         return self.residual_logit_coupler(logits, proto, fine_abs, uncertainty)

#     def _prototype_residual(self, fine_abs, fine_res, grain, uncertainty, x_norm, proto_var=None):
#         if not self.use_relative_decode:
#             return fine_abs
#         proto = F.interpolate(grain.transpose(1, 2), size=fine_abs.size(1), mode='linear', align_corners=False).transpose(1, 2)
#         if proto_var is None:
#             proto_unc = uncertainty.mean(dim=1)
#         else:
#             proto_unc = F.interpolate(proto_var.transpose(1, 2), size=fine_abs.size(1), mode='linear', align_corners=False).transpose(1, 2).mean(dim=1)
#         hist = torch.stack([
#             x_norm.mean(dim=1),
#             x_norm.std(dim=1, unbiased=False),
#             x_norm[:, -1, :],
#             x_norm[:, -1, :] - x_norm[:, 0, :],
#             uncertainty.mean(dim=1),
#             proto_unc,
#         ], dim=-1)
#         channel_gate = torch.sigmoid(self.prototype_gate(hist)).permute(0, 2, 1)
#         horizon_gate = self._horizon_gate(fine_abs.size(1), fine_abs.device, fine_abs.dtype)
#         global_gate = torch.sigmoid(self.relative_gate)
#         gate = global_gate * channel_gate * horizon_gate * uncertainty.detach().clamp(0.0, 1.0)
#         gate = gate.clamp(0.0, 1.0)
#         residual = torch.tanh(self.residual_scale) * fine_res
#         if self.use_conditional_residual:
#             residual = self.residual_coupler(residual, proto, fine_abs, uncertainty)
#         relative = proto + residual
#         return fine_abs * (1.0 - gate) + relative * gate

#     def forward(self, x):
#         if self.revin:
#             x = self.revin_layer(x, 'norm')
#         x_norm = x

#         if self.ma_type == 'reg':
#             seasonal_init, trend_init = x, x
#         else:
#             seasonal_init, trend_init = self.decomp(x)

#         x_1 = self.net_1(seasonal_init, trend_init)
#         x_2 = self.net_2(seasonal_init, trend_init)
#         x_3 = self.net_3(seasonal_init, trend_init)
#         x_4 = self.net_4(seasonal_init, trend_init)

#         mean_1, var_1, prob_1 = self._decode(x_1, self.idx_1)
#         mean_2, var_2, prob_2 = self._decode(x_2, self.idx_2)
#         output_fine, var_fine, mask_fine = self._confidence_fuse(mean_1, var_1, prob_1, mean_2, var_2, prob_2)
#         fine_prob = self._posterior_fuse(prob_1, prob_2, mask_fine)

#         mean_3, var_3, prob_3 = self._decode(x_3, self.idx_1)
#         mean_4, var_4, prob_4 = self._decode(x_4, self.idx_2)
#         output_grain, var_grain, mask_grain = self._confidence_fuse(mean_3, var_3, prob_3, mean_4, var_4, prob_4)
#         grain_prob = self._posterior_fuse(prob_3, prob_4, mask_grain)

#         uncertainty = (var_fine / (var_fine.detach().mean(dim=1, keepdim=True) + 1e-6)).sigmoid()
#         proto_for_residual = F.interpolate(
#             output_grain.transpose(1, 2),
#             size=output_fine.size(1),
#             mode='linear',
#             align_corners=False,
#         ).transpose(1, 2)
#         if self.use_residual_support:
#             cond_x_1 = self._condition_residual_logits(x_1, proto_for_residual, output_fine, uncertainty)
#             cond_x_2 = self._condition_residual_logits(x_2, proto_for_residual, output_fine, uncertainty)
#             res_1, _, _ = self._decode(cond_x_1, self.res_idx_1)
#             res_2, _, _ = self._decode(cond_x_2, self.res_idx_2)
#             residual_fine = res_1 * mask_fine + res_2 * (1.0 - mask_fine)
#         else:
#             residual_fine = output_fine

#         grain_len = output_grain.size(1)
#         output_down_fine = output_fine[:, :grain_len * 4, :].reshape(output_fine.size(0), grain_len, 4, output_fine.size(2)).mean(dim=2)
#         var_down_fine = var_fine[:, :grain_len * 4, :].reshape(var_fine.size(0), grain_len, 4, var_fine.size(2)).mean(dim=2)

#         con_loss_time = F.mse_loss(output_grain, output_down_fine)
#         if self.transport_var_weight > 0:
#             con_loss_time = con_loss_time + self.transport_var_weight * F.mse_loss(var_grain, var_down_fine)
#         if self.posterior_kl_weight > 0:
#             con_loss_time = con_loss_time + self.posterior_kl_weight * self._kl_to_coarse(fine_prob, grain_prob)
#         con_loss_cls_1 = F.mse_loss(mean_2, mean_1)
#         con_loss_cls_2 = F.mse_loss(mean_4, mean_3)

#         output = self._prototype_residual(output_fine, residual_fine, output_grain, uncertainty, x_norm, var_grain)
#         if self.use_posterior_calib:
#             output = self.calibrator(output, uncertainty)

#         if self.revin:
#             output = self.revin_layer(output, 'denorm')
#         return output, con_loss_cls_1, con_loss_cls_2, con_loss_time


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
