"""
Exact Function-Preserving Inverse Reparameterization (E -> Solve -> C) Operators for IR-ISWSP.

Covers all 8 exact zero-gradient operator families in the IR-ISWSP paper:
  1. ExpandedDCTConv2d (2D-DCT frequency-band IR for mask boundary convolutions)
  2. ExpandedAdaBN2d & ExpandedAdaLayerNorm (moment-decoupled normalization IR)
  3. ExpandedBilinearMaskIR (Theorem 1: algebraically exact (3K+1) bilinear mask lift & fold)
  4. solve_two_stage_full_matrix_wq (Algorithm 1: Two-Stage Full-Matrix W_q Dual-Projection Gauss-Newton)
  5. ExpandedStemWhitenConv2d (exact interior-pixel Conv2d / Conv2d+BN affine whitening for label-free OOD)
  6. apply_zero_image_corrupted_head_repair (0-image 2D-DCT checkerboard + MAD channel-norm Winsorization)
  7. ExpandedPCAProcrustesLinear & solve_stiefel_procrustes (Proposition 1 Haar-O(D) invariant readout & transport)
  8. solve_mm_fw_ridge (closed-form Moment-Matched Foreground-Weighted Ridge contraction bridge)
"""

from __future__ import annotations

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zero_grad_guard import enforce_no_gradients

enforce_no_gradients()


def _dct_matrix_1d(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    c = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    c[0] *= 1.0 / math.sqrt(n)
    if n > 1:
        c[1:] *= math.sqrt(2.0 / n)
    return c


class ExpandedAdaBN2d(nn.Module):
    """Exact expand-edit-collapse Empirical-Bayes BatchNorm2d IR operator."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.bn = copy.deepcopy(bn).eval()
        c = bn.num_features
        self.register_buffer("delta_scale", torch.ones(c))
        self.register_buffer("delta_shift", torch.zeros(c))

    def set_empirical_bayes_moments(self, mu_new: torch.Tensor, var_new: torch.Tensor, eta: float = 0.25) -> None:
        mu0 = self.bn.running_mean.float()
        var0 = self.bn.running_var.float()
        eps = float(self.bn.eps)
        mu_mix = (1.0 - eta) * mu0 + eta * mu_new.float().to(mu0.device)
        var_mix = (1.0 - eta) * var0 + eta * var_new.float().to(var0.device).clamp(min=1e-6)
        s0 = torch.rsqrt(var0 + eps)
        s1 = torch.rsqrt(var_mix + eps)
        self.delta_scale.copy_(s1 / s0)
        self.delta_shift.copy_((mu0 - mu_mix) * s1 * self.bn.weight.float())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        return y * self.delta_scale.view(1, -1, 1, 1) + self.delta_shift.view(1, -1, 1, 1)

    @torch.no_grad()
    def collapse(self) -> nn.BatchNorm2d:
        out = copy.deepcopy(self.bn).eval()
        out.weight.copy_(out.weight * self.delta_scale)
        out.bias.copy_(out.bias * self.delta_scale + self.delta_shift)
        return out


class ExpandedAdaLayerNorm(nn.Module):
    """Exact expand-edit-collapse LayerNorm IR operator."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.ln = copy.deepcopy(ln).eval()
        d = int(ln.normalized_shape[0])
        self.register_buffer("delta_scale", torch.ones(d))
        self.register_buffer("delta_shift", torch.zeros(d))

    def set_empirical_bayes_moments(self, mu_new: torch.Tensor, var_new: torch.Tensor, eta: float = 0.20) -> None:
        dev = self.ln.weight.device
        s = (1.0 - eta) + eta * torch.rsqrt(var_new.float().to(dev).clamp(min=1e-4))
        self.delta_scale.copy_(s.clamp(0.75, 1.35))
        self.delta_shift.copy_((-eta * mu_new.float().to(dev)) * self.ln.weight.float())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x) * self.delta_scale + self.delta_shift

    @torch.no_grad()
    def collapse(self) -> nn.LayerNorm:
        out = copy.deepcopy(self.ln).eval()
        out.weight.copy_(out.weight * self.delta_scale)
        out.bias.copy_(out.bias * self.delta_scale + self.delta_shift)
        return out


class ExpandedDCTConv2d(nn.Module):
    """Exact 2D-DCT frequency-band Inverse Reparameterization for KxK Conv2d layers."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = copy.deepcopy(conv).eval()
        kh, kw = conv.kernel_size
        dev, dt = conv.weight.device, torch.float64
        self.register_buffer("dh", _dct_matrix_1d(kh, dev, dt))
        self.register_buffer("dw", _dct_matrix_1d(kw, dev, dt))
        self.register_buffer("band_gain", torch.ones((kh, kw), device=dev, dtype=dt))

    def set_frequency_band_scales(self, g_dc: float = 1.0, g_mid: float = 1.0, g_hi: float = 1.0) -> None:
        kh, kw = self.band_gain.shape
        g = torch.ones_like(self.band_gain)
        for u in range(kh):
            for v in range(kw):
                r = u + v
                if r == 0:
                    g[u, v] = g_dc
                elif r <= max(1, (kh + kw) // 2 - 1):
                    g[u, v] = g_mid
                else:
                    g[u, v] = g_hi
        self.band_gain.copy_(g)

    @torch.no_grad()
    def _effective_weight(self) -> torch.Tensor:
        w = self.conv.weight.double()
        dh = self.dh.double()
        dw = self.dw.double()
        bg = self.band_gain.double()
        w_dct = torch.einsum("ab,oibc,dc->oiad", dh, w, dw)
        w_mod = w_dct * bg.view(1, 1, *bg.shape)
        w_rec = torch.einsum("ab,oiad,dc->oibc", dh, w_mod, dw)
        return w_rec.to(self.conv.weight.dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self._effective_weight(),
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    @torch.no_grad()
    def collapse(self) -> nn.Conv2d:
        out = copy.deepcopy(self.conv).eval()
        out.weight.copy_(self._effective_weight())
        return out


class ExpandedBilinearMaskIR(nn.Module):
    """
    Theorem 1: Algebraically Exact Bilinear Mask-Factorization IR (`IR-ISWSP` Core):
        M_i(x,y) = c_i^T P(x,y) + b_0,
        where c_i = W_q h_i + b_q  (instance mask coefficients, R^K)
          and P(x,y) = W_p * F(x,y) + b_p  (spatial prototype basis, R^{K x H x W}).

    Lifts into the (3K+1) decoupled bilinear gauge-and-offset subspace (ds, dd, de, db0):
        c_i* = diag(1 + ds) c_i + dd
        P*   = P + de / (1 + ds)
        b0*  = b0 + db0 - dot(dd, de / (1 + ds))
    with exact zero quadratic cross-term error and 0% extra deployment parameters / FLOPs.
    """

    def __init__(
        self,
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_p: torch.Tensor,
        b_p: torch.Tensor,
        b0: torch.Tensor,
    ):
        super().__init__()
        k = int(w_q.shape[0])
        self.register_buffer("w_q", w_q.detach().float().clone())
        self.register_buffer("b_q", b_q.detach().float().clone())
        self.register_buffer("w_p", w_p.detach().float().clone())
        self.register_buffer("b_p", b_p.detach().float().clone())
        self.register_buffer("b0", b0.detach().float().clone())
        self.register_buffer("ds", torch.zeros(k))
        self.register_buffer("dd", torch.zeros(k))
        self.register_buffer("de", torch.zeros(k))
        self.register_buffer("db0", torch.zeros(1))

    @torch.no_grad()
    def set_bilinear_update(self, ds: torch.Tensor, dd: torch.Tensor, de: torch.Tensor, db0: float) -> None:
        self.ds.copy_(ds.float().clamp(-0.35, 0.35))
        self.dd.copy_(dd.float())
        self.de.copy_(de.float())
        self.db0.fill_(float(db0))

    @torch.no_grad()
    def forward_expanded(self, h_q: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        """Evaluates the linearized lifted expansion directly for exact equivalence verification."""
        c = F.linear(h_q.float(), self.w_q, self.b_q)
        p = F.conv2d(f_p.float(), self.w_p, self.b_p)
        m0 = torch.einsum("bqk,bkhw->bqhw", c, p) + self.b0.view(1, 1, 1, 1)
        dm_s = torch.einsum("bqk,k,bkhw->bqhw", c, self.ds, p)
        dm_d = torch.einsum("k,bkhw->bhw", self.dd, p).unsqueeze(1)
        dm_e = torch.einsum("bqk,k->bq", c, self.de).unsqueeze(-1).unsqueeze(-1)
        return m0 + dm_s + dm_d + dm_e + self.db0.view(1, 1, 1, 1)

    @torch.no_grad()
    def collapse_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algebraically exact collapse back into native (W_q*, b_q*, W_p*, b_p*, b0*) tensors."""
        scale = 1.0 + self.ds
        w_q_star = scale.unsqueeze(1) * self.w_q
        b_q_star = scale * self.b_q + self.dd
        w_p_star = self.w_p.clone()
        delta_bp = self.de / scale
        b_p_star = self.b_p + delta_bp
        cross_cancel = torch.dot(self.dd, delta_bp)
        b0_star = self.b0 + self.db0.view_as(self.b0) - cross_cancel
        return w_q_star, b_q_star, w_p_star, b_p_star, b0_star

    @torch.no_grad()
    def forward_collapsed(self, h_q: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        wq, bq, wp, bp, b0 = self.collapse_weights()
        c = F.linear(h_q.float(), wq, bq)
        p = F.conv2d(f_p.float(), wp, bp)
        return torch.einsum("bqk,bkhw->bqhw", c, p) + b0.view(1, 1, 1, 1)


@torch.no_grad()
def solve_two_stage_full_matrix_wq(
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    h_feats: torch.Tensor,
    proto_crops: torch.Tensor,
    gt_masks: torch.Tensor,
    boundary_masks: torch.Tensor | None = None,
    beta_bnd: float = 2.5,
    lam_c: float = 10.0,
    lam_h: float = 40.0,
    eta_w: float = 0.08,
    b0: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1: Two-Stage Full-Matrix W_q Dual-Projection Gauss-Newton (0 gradients).

    Args:
      w_q: [K, D_q] native mask coefficient projection matrix.
      b_q: [K] native mask coefficient bias.
      h_feats: [N, D_q] matched instance pre-projection query features.
      proto_crops: [N, K, S] flattened spatial prototypes inside each instance RoI.
      gt_masks: [N, S] binary ground-truth masks in {0, 1}.
      boundary_masks: optional [N, S] binary boundary indicator masks.
      beta_bnd: boundary curvature up-weighting factor (default 2.5).
      lam_c: Stage-1 per-instance KxK ridge damping.
      lam_h: Stage-2 (D_q+1)x(D_q+1) multi-output ridge damping.
      eta_w: trust-region step size.
      b0: scalar global mask bias.

    Returns:
      (w_q_star, b_q_star) folded directly into native (W_q, b_q) with 0% extra overhead.
    """
    enforce_no_gradients()
    dev = w_q.device
    n_inst, d_q = h_feats.shape
    k = int(w_q.shape[0])
    h_h = torch.zeros((d_q + 1, d_q + 1), device=dev, dtype=torch.float64)
    g_h = torch.zeros((d_q + 1, k), device=dev, dtype=torch.float64)
    eye_k = torch.eye(k, device=dev, dtype=torch.float64)

    c_all = F.linear(h_feats.float(), w_q.float(), b_q.float())  # [N, K]
    for i in range(n_inst):
        p_i = proto_crops[i].double()  # [K, S]
        y_i = gt_masks[i].double()  # [S]
        logits_i = c_all[i].double() @ p_i + float(b0)
        probs_i = torch.sigmoid(logits_i)
        if boundary_masks is not None:
            omega = 1.0 + float(beta_bnd) * boundary_masks[i].double()
        else:
            omega = torch.ones_like(y_i)
        r_i = omega * (y_i - probs_i)
        w_i = omega * probs_i * (1.0 - probs_i)

        # Stage 1: exact KxK per-instance Gauss-Newton linear system in FP64
        a_i = (p_i * w_i.unsqueeze(0)) @ p_i.t() + float(lam_c) * eye_k
        rhs_i = p_i @ r_i
        dc_i = torch.linalg.solve(a_i, rhs_i)

        # Stage 2 accumulation onto augmented feature basis [h_i; 1]
        phi_i = torch.cat([h_feats[i].double(), torch.ones(1, device=dev, dtype=torch.float64)], dim=0)
        h_h += torch.outer(phi_i, phi_i)
        g_h += torch.outer(phi_i, dc_i)

    eye_h = torch.eye(d_q + 1, device=dev, dtype=torch.float64)
    delta_theta = torch.linalg.solve(h_h + float(lam_h) * eye_h, g_h)  # [D_q+1, K]
    dw_q = delta_theta[:-1, :].t().to(dtype=w_q.dtype)
    db_q = delta_theta[-1, :].to(dtype=b_q.dtype)
    return w_q + float(eta_w) * dw_q, b_q + float(eta_w) * db_q


class ExpandedStemWhitenConv2d(nn.Module):
    """
    Exact Label-Free Stem Conv2d Affine Whitening (`IR-StemWhiten`, ΔParams=0%).

    Folds damped per-channel whitening x_hat_c = a_c * x_c + b_c directly into a 3-channel
    input stem Conv2d (and optional paired BatchNorm2d running_mean when conv.bias is None).
    """

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d | None = None):
        super().__init__()
        if conv.in_channels != 3:
            raise ValueError(f"Expected 3-channel RGB stem Conv2d, got in_channels={conv.in_channels}")
        self.conv = copy.deepcopy(conv).eval()
        self.bn = copy.deepcopy(bn).eval() if bn is not None else None
        self.register_buffer("a_c", torch.ones(3))
        self.register_buffer("b_c", torch.zeros(3))

    @torch.no_grad()
    def set_shift_moments(
        self,
        mu_shift: torch.Tensor,
        std_shift: torch.Tensor,
        mu_ref: torch.Tensor,
        std_ref: torch.Tensor,
        alpha: float = 0.85,
    ) -> None:
        ratio = std_ref.double() / std_shift.double().clamp(min=1e-4)
        a = (1.0 - float(alpha)) + float(alpha) * ratio
        b = float(alpha) * (mu_ref.double() - ratio * mu_shift.double())
        self.a_c.copy_(a.to(self.a_c.device, dtype=self.a_c.dtype))
        self.b_c.copy_(b.to(self.b_c.device, dtype=self.b_c.dtype))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = x * self.a_c.view(1, 3, 1, 1) + self.b_c.view(1, 3, 1, 1)
        y = self.conv(x_hat)
        return self.bn(y) if self.bn is not None else y

    @torch.no_grad()
    def collapse(self) -> Tuple[nn.Conv2d, nn.BatchNorm2d | None]:
        out_conv = copy.deepcopy(self.conv).eval()
        out_bn = copy.deepcopy(self.bn).eval() if self.bn is not None else None
        w = out_conv.weight.double()  # [C_out, 3, Kh, Kw]
        a = self.a_c.double()
        b = self.b_c.double()
        w_star = w * a.view(1, 3, 1, 1)
        delta_b = (w.sum(dim=(2, 3)) * b.view(1, 3)).sum(dim=1)  # [C_out]
        out_conv.weight.copy_(w_star.to(out_conv.weight.dtype))
        if out_conv.bias is not None:
            out_conv.bias.copy_((out_conv.bias.double() + delta_b).to(out_conv.bias.dtype))
        elif out_bn is not None and out_bn.running_mean is not None:
            out_bn.running_mean.copy_((out_bn.running_mean.double() - delta_b).to(out_bn.running_mean.dtype))
        else:
            out_conv.bias = nn.Parameter(delta_b.to(out_conv.weight.dtype), requires_grad=False)
        return out_conv, out_bn


@torch.no_grad()
def apply_zero_image_corrupted_head_repair(
    weight: torch.Tensor,
    mad_thresh: float = 2.5,
    high_freq_damp: float = 0.35,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    Zero-Image Corrupted Prototype Mask-Head Repair (0 images, 0 gradients, ΔParams=0%):
      1. Suppresses anomalous high-frequency 2D-DCT checkerboard energy on KxK spatial kernels (K >= 3).
      2. Winsorizes outlier channel-norm spikes exceeding `median + mad_thresh * MAD`.
    """
    enforce_no_gradients()
    w = weight.detach().double().clone()
    n_channels = int(w.shape[0])
    repaired_dct = 0
    if w.ndim == 4 and w.shape[2] >= 3 and w.shape[3] >= 3:
        kh, kw = int(w.shape[2]), int(w.shape[3])
        dh = _dct_matrix_1d(kh, w.device, torch.float64)
        dw = _dct_matrix_1d(kw, w.device, torch.float64)
        w_dct = torch.einsum("ab,oibc,dc->oiad", dh, w, dw)
        corner_energy = w_dct[:, :, kh - 1, kw - 1].abs().mean(dim=1)
        med_c = corner_energy.median()
        mad_c = (corner_energy - med_c).abs().median().clamp(min=1e-6)
        bad_freq = corner_energy > (med_c + mad_thresh * mad_c)
        if bad_freq.any():
            w_dct[bad_freq, :, kh - 1, kw - 1] *= float(high_freq_damp)
            repaired_dct = int(bad_freq.sum().item())
        w = torch.einsum("ab,oiad,dc->oibc", dh, w_dct, dw)

    ch_norms = w.reshape(n_channels, -1).norm(dim=1)
    med_n = ch_norms.median()
    mad_n = (ch_norms - med_n).abs().median().clamp(min=1e-6)
    cap = med_n + float(mad_thresh) * mad_n
    outlier_mask = ch_norms > cap
    n_clipped = int(outlier_mask.sum().item())
    if n_clipped > 0:
        scale = torch.where(outlier_mask, cap / ch_norms.clamp(min=1e-8), torch.ones_like(ch_norms))
        view_shape = [n_channels] + [1] * (w.ndim - 1)
        w = w * scale.view(*view_shape)

    audit = {
        "channels_total": float(n_channels),
        "channels_freq_repaired": float(repaired_dct),
        "channels_norm_winsorized": float(n_clipped),
    }
    return w.to(dtype=weight.dtype), audit


class ExpandedPCAProcrustesLinear(nn.Module):
    """Exact whitened PCA-Procrustes linear readout lift (Proposition 1 Haar-O(D) invariant)."""

    def __init__(self, w0: torch.Tensor, b0: torch.Tensor, p_whiten: torch.Tensor):
        super().__init__()
        self.register_buffer("w0", w0.detach().float().clone())
        self.register_buffer("b0", b0.detach().float().clone())
        self.register_buffer("p_whiten", p_whiten.detach().float().clone())
        self.register_buffer("dw_z", torch.zeros((w0.shape[0], p_whiten.shape[0])))
        self.register_buffer("db", torch.zeros_like(b0.float()))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x.float(), self.p_whiten)
        return F.linear(x.float(), self.w0, self.b0) + F.linear(z, self.dw_z, self.db)

    @torch.no_grad()
    def collapse_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w_star = self.w0 + self.dw_z @ self.p_whiten
        b_star = self.b0 + self.db
        return w_star, b_star


@torch.no_grad()
def solve_stiefel_procrustes(m_src: torch.Tensor, m_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Closed-form orthogonal Stiefel Procrustes transport Q* = argmin_{Q in O(D)} ||M_src Q - M_tgt||_F."""
    a = m_src.double()
    b = m_tgt.double()
    u, _, vh = torch.linalg.svd(a.t() @ b, full_matrices=False)
    q = (u @ vh).float()
    aligned = m_src.float() @ q
    cos_sim = float(F.cosine_similarity(aligned.reshape(1, -1), m_tgt.float().reshape(1, -1)).item())
    return q, aligned, cos_sim


@torch.no_grad()
def solve_mm_fw_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    fg_weights: torch.Tensor | None = None,
    lam_rel: float = 0.015,
    moment_gain: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form Moment-Matched Foreground-Weighted Ridge (`MM-FW-Ridge`) subgraph bridge
    for negative latency-budget contraction (0 gradients).
    """
    xf = x.double()
    yf = y.double()
    n, d_in = xf.shape
    if fg_weights is None:
        w = torch.ones((n, 1), device=xf.device, dtype=torch.float64)
    else:
        w = fg_weights.double().view(-1, 1).clamp(min=1e-4)
    w = w / w.sum().clamp(min=1e-12)

    mu_x = (w * xf).sum(dim=0, keepdim=True)
    mu_y = (w * yf).sum(dim=0, keepdim=True)
    xc = xf - mu_x
    yc = yf - mu_y

    s_xx = (w * xc).t() @ xc
    s_xy = (w * xc).t() @ yc
    tr = float(torch.trace(s_xx).item())
    lam = max(1e-6, lam_rel * tr / max(1, d_in))
    a_mat = s_xx + lam * torch.eye(d_in, device=xf.device, dtype=torch.float64)
    w_t = torch.linalg.solve(a_mat, s_xy)
    if moment_gain != 1.0:
        w_t = w_t * float(moment_gain)
    b = (mu_y - mu_x @ w_t).squeeze(0)
    return w_t.t().float(), b.float()
