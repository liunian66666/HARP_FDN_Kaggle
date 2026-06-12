import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN


class LiteLinearHead(nn.Module):
    """Small long-horizon linear head in normalized space."""

    def __init__(self, seq_len, pred_len, dropout=0.0, nlinear_strength_init=-1.5):
        super().__init__()
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        self.direct = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(float(dropout))
        self.nlinear_strength = nn.Parameter(torch.tensor(float(nlinear_strength_init)))
        for layer in (self.seasonal, self.trend, self.direct):
            nn.init.constant_(layer.weight, 1.0 / float(seq_len))
            nn.init.zeros_(layer.bias)

    def forward(self, x, seasonal, trend):
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        decomp_out = self.seasonal(seasonal) + self.trend(trend)
        decomp_out = decomp_out.transpose(1, 2)
        last = x[:, -1:, :]
        centered = (x - last).transpose(1, 2)
        nlinear_out = self.direct(centered).transpose(1, 2) + last
        gate = torch.sigmoid(self.nlinear_strength)
        out = decomp_out * (1.0 - gate) + nlinear_out * gate
        return self.dropout(out)


class HorizonBlendGate(nn.Module):
    """Cheap horizon/channel gate for probability-vs-linear fusion."""

    def __init__(self, pred_len, channels, gate_init=-2.0, slope_init=0.0, max_gate=0.75):
        super().__init__()
        self.pred_len = int(pred_len)
        self.max_gate = float(max_gate)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.slope = nn.Parameter(torch.tensor(float(slope_init)))
        self.horizon_bias = nn.Parameter(torch.zeros(1, self.pred_len, 1))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, length, device, dtype):
        pos = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype).view(1, length, 1)
        logit = self.gate_logit + self.slope * (pos - 0.5)
        logit = logit + self.horizon_bias[:, :length, :] + self.channel_bias
        return torch.sigmoid(logit).clamp(0.0, self.max_gate)


class Model(nn.Module):
    """
    HARP_FDN_PCX_Lite: two-branch probabilistic support + lightweight linear long-horizon head.
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
        self.fusion_uncertainty_weight = float(getattr(configs, 'fusion_uncertainty_weight', 0.0))
        self.use_prob_path = int(getattr(configs, 'lite_use_prob_path', 1))
        self.use_linear_head = int(getattr(configs, 'lite_use_linear_head', 1))

        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)

        self.net_1 = None
        self.net_2 = None
        if self.use_prob_path:
            self.net_1 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')
            self.net_2 = Network(seq_len, pred_len, patch_len, stride, padding_patch, 'fine')

        df = pd.read_excel(getattr(configs, 'support_path', 'idx2.xlsx'), usecols=[1, 2], header=None, skiprows=1, nrows=self.num_support)
        idx_1 = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(self.num_support, 1)
        idx_2 = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(self.num_support, 1)
        self.register_buffer('idx_1', idx_1)
        self.register_buffer('idx_2', idx_2)

        default_gate = -3.2 if pred_len <= 96 else (-2.4 if pred_len <= 192 else (-1.8 if pred_len <= 336 else -1.0))
        gate_raw = float(getattr(configs, 'lite_linear_gate_init', 999.0))
        gate_init = default_gate if gate_raw == 999.0 else gate_raw
        self.linear_head = LiteLinearHead(
            seq_len,
            pred_len,
            dropout=float(getattr(configs, 'lite_linear_dropout', 0.0)),
            nlinear_strength_init=float(getattr(configs, 'lite_nlinear_strength_init', -1.5)),
        )
        self.blend_gate = HorizonBlendGate(
            pred_len,
            c_in,
            gate_init=gate_init,
            slope_init=float(getattr(configs, 'lite_linear_gate_slope_init', 0.0)),
            max_gate=float(getattr(configs, 'lite_linear_max_gate', 0.85)),
        )

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
            w = self.fusion_uncertainty_weight
            conf = conf * torch.exp(-w * entropy) * torch.exp(-0.5 * w * var_score)
        return conf

    def _confidence_fuse(self, mean_a, var_a, prob_a, mean_b, var_b, prob_b):
        conf_a = self._branch_confidence(prob_a, var_a)
        conf_b = self._branch_confidence(prob_b, var_b)
        mask = conf_a / (conf_a + conf_b + 1e-6)
        mean = mean_a * mask + mean_b * (1.0 - mask)
        var = mask.pow(2) * var_a + (1.0 - mask).pow(2) * var_b
        var = var + mask * (mean_a - mean).pow(2) + (1.0 - mask) * (mean_b - mean).pow(2)
        return mean, var

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x_norm = x

        if self.ma_type == 'reg':
            seasonal_init, trend_init = x, x
        else:
            seasonal_init, trend_init = self.decomp(x)

        linear_out = self.linear_head(x_norm, seasonal_init, trend_init)

        if self.use_prob_path:
            x_1 = self.net_1(seasonal_init, trend_init)
            x_2 = self.net_2(seasonal_init, trend_init)
            mean_1, var_1, prob_1 = self._decode(x_1, self.idx_1)
            mean_2, var_2, prob_2 = self._decode(x_2, self.idx_2)
            prob_out, _ = self._confidence_fuse(mean_1, var_1, prob_1, mean_2, var_2, prob_2)
            con_loss_cls_1 = F.mse_loss(mean_2, mean_1)
        else:
            prob_out = linear_out
            con_loss_cls_1 = linear_out.sum() * 0.0

        if self.use_linear_head:
            gate = self.blend_gate(prob_out.size(1), prob_out.device, prob_out.dtype)
            output = prob_out * (1.0 - gate) + linear_out * gate
        else:
            output = prob_out

        zero = output.sum() * 0.0
        if self.revin:
            output = self.revin_layer(output, 'denorm')
        return output, con_loss_cls_1, zero, zero
