"""
Zero-Gradient Negative-Budget Contraction via Closed-Form Subgraph Bridges (Section 4.5.3).

Implements Moment-Matched Foreground-Weighted Ridge (`MM-FW-Ridge`) closed-form
linear projection `P_br* = diag(sigma_Y / sigma_Yhat) Y^T Omega X (X^T Omega X + lam_br I)^{-1}`
in `R^{d_out x d_in}` plus algebraic `Conv2d + BatchNorm2d` inference folding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .zero_grad_guard import enforce_no_gradients


def fit_mm_fw_ridge_bridge(
    x_feat: torch.Tensor,
    y_target: torch.Tensor,
    fg_weights: torch.Tensor | None = None,
    lam_rel: float = 1e-2,
    with_bias: bool = True,
    moment_match_max_gain: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fits closed-form Moment-Matched Foreground-Weighted Ridge (`MM-FW-Ridge`) bridge:
      - `x_feat`:   [N, d_in] bypassed upstream activations
      - `y_target`: [N, d_out] frozen downstream expected feature activations
      - `fg_weights`: optional [N] foreground-active spatial/query weights (Omega)
    Returns `(W_bridge, b_bridge)` with `W_bridge` of shape `[d_out, d_in]`.
    """
    enforce_no_gradients()
    with torch.no_grad():
        xd = x_feat.detach().double()
        yd = y_target.detach().double()
        n_tok, d_in = xd.shape
        d_out = yd.shape[1]

        if fg_weights is not None:
            wd = fg_weights.detach().double().reshape(n_tok, 1).clamp(min=1e-6)
        else:
            wd = torch.ones((n_tok, 1), device=xd.device, dtype=torch.float64)

        if with_bias:
            ones = torch.ones((n_tok, 1), device=xd.device, dtype=torch.float64)
            x_aug = torch.cat([xd, ones], dim=1)
        else:
            x_aug = xd

        x_weighted = x_aug * wd
        xtwx = x_aug.t() @ x_weighted
        xtwy = x_weighted.t() @ yd

        diag_mean = float(torch.diagonal(xtwx)[:d_in].mean().item())
        lam = max(1e-6, float(lam_rel) * max(1e-6, diag_mean))
        reg = torch.zeros(xtwx.shape[0], device=xd.device, dtype=torch.float64)
        reg[:d_in] = lam
        if with_bias:
            reg[d_in] = 1e-8 * max(1.0, diag_mean)
        a_mat = xtwx + torch.diag(reg)

        chol, info = torch.linalg.cholesky_ex(a_mat)
        if int(info.item()) == 0:
            theta = torch.cholesky_solve(xtwy, chol)
        else:
            theta = torch.linalg.lstsq(a_mat, xtwy).solution

        w_br = theta[:d_in, :].t().contiguous()
        b_br = theta[d_in, :].contiguous() if with_bias else torch.zeros(d_out, device=xd.device, dtype=torch.float64)

        if moment_match_max_gain > 1.0:
            y_hat = xd @ w_br.t() + (b_br.unsqueeze(0) if with_bias else 0.0)
            w_sum = wd.sum().clamp(min=1e-8)
            if with_bias:
                mu_y = (yd * wd).sum(dim=0) / w_sum
                mu_h = (y_hat * wd).sum(dim=0) / w_sum
                var_y = ((yd - mu_y) ** 2 * wd).sum(dim=0) / w_sum
                var_h = ((y_hat - mu_h) ** 2 * wd).sum(dim=0) / w_sum
                gain = torch.sqrt(var_y.clamp(min=1e-10) / var_h.clamp(min=1e-10)).clamp(1.0, float(moment_match_max_gain))
                w_br = w_br * gain.unsqueeze(1)
                b_br = mu_y + gain * (b_br - mu_h)
            else:
                rms_y = torch.sqrt((yd ** 2 * wd).sum(dim=0) / w_sum)
                rms_h = torch.sqrt((y_hat ** 2 * wd).sum(dim=0) / w_sum)
                gain = (rms_y.clamp(min=1e-10) / rms_h.clamp(min=1e-10)).clamp(1.0, float(moment_match_max_gain))
                w_br = w_br * gain.unsqueeze(1)

        return w_br.float().cpu().contiguous(), b_br.float().cpu().contiguous()


def fuse_conv_and_bn_tensors(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraically fuses `Conv2d + BatchNorm2d` into `(W_fused, b_fused)` in FP32."""
    enforce_no_gradients()
    with torch.no_grad():
        w = conv.weight.detach().float()
        b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
        rm = bn.running_mean.detach().float()
        rv = bn.running_var.detach().float()
        gm = bn.weight.detach().float()
        bt = bn.bias.detach().float()
        scale = gm * torch.rsqrt(rv + float(bn.eps))
        w_f = w * scale.view(-1, 1, 1, 1)
        b_f = (b - rm) * scale + bt
        return w_f, b_f
