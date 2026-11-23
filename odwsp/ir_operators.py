"""
Exact Function-Preserving Inverse Reparameterization Operators (Section 4.3).

Implements the four decoupled statistical lifting (E) and algebraic collapse (C)
modules of IR-ODWSP:
  1. `ExpandedAdaBN2d` / `ExpandedAdaLayerNorm` (`IR-AdaNorm`: moment decoupling)
  2. `ExpandedDCTConv2d` (`IR-DCT`: 2D orthonormal spatial-frequency decoupling)
  3. `ExpandedPCAProcrustesLinear` (`IR-PCA + Procrustes`: covariance & auxiliary subspace lift)
  4. `ExpandedBoxAffineLinear` (`IR-BoxAffine`: 4x4 bounding-box distribution decoupling)
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .zero_grad_guard import enforce_no_gradients


class ExpandedAdaBN2d(nn.Module):
    """Moment-decoupled BatchNorm2d wrapper with exact algebraic collapse."""

    def __init__(self, orig_bn: nn.BatchNorm2d) -> None:
        super().__init__()
        enforce_no_gradients()
        self.orig_bn = copy.deepcopy(orig_bn).eval()
        self.orig_bn.requires_grad_(False)
        c = orig_bn.num_features
        dev, dt = orig_bn.weight.device, orig_bn.weight.dtype
        self.register_buffer("mu_delta", torch.zeros(c, device=dev, dtype=dt))
        self.register_buffer("sigma_delta", torch.ones(c, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.orig_bn(x)
        return (z - self.mu_delta.view(1, -1, 1, 1)) / self.sigma_delta.view(1, -1, 1, 1)

    def set_empirical_bayes_moments(self, feat_mean: torch.Tensor, feat_var: torch.Tensor, eta: float) -> None:
        with torch.no_grad():
            self.mu_delta.copy_((eta * feat_mean).to(self.mu_delta.device, self.mu_delta.dtype))
            var_shrunk = (1.0 - eta) + eta * feat_var.clamp(0.25, 4.0)
            self.sigma_delta.copy_(torch.sqrt(var_shrunk).to(self.sigma_delta.device, self.sigma_delta.dtype))

    def collapse(self) -> nn.BatchNorm2d:
        bn_out = copy.deepcopy(self.orig_bn).eval()
        with torch.no_grad():
            inv_s = 1.0 / self.sigma_delta.double().clamp(min=1e-8)
            w_new = bn_out.weight.double() * inv_s
            b_new = (bn_out.bias.double() - self.mu_delta.double()) * inv_s
            bn_out.weight.data.copy_(w_new.to(bn_out.weight.dtype))
            bn_out.bias.data.copy_(b_new.to(bn_out.bias.dtype))
        bn_out.requires_grad_(False)
        return bn_out


class ExpandedAdaLayerNorm(nn.Module):
    """Moment-decoupled LayerNorm wrapper with exact algebraic collapse."""

    def __init__(self, orig_ln: nn.LayerNorm) -> None:
        super().__init__()
        enforce_no_gradients()
        self.orig_ln = copy.deepcopy(orig_ln).eval()
        self.orig_ln.requires_grad_(False)
        dev, dt = orig_ln.weight.device, orig_ln.weight.dtype
        self.register_buffer("mu_delta", torch.zeros_like(orig_ln.bias, device=dev, dtype=dt))
        self.register_buffer("sigma_delta", torch.ones_like(orig_ln.weight, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.orig_ln(x) - self.mu_delta) / self.sigma_delta

    def set_empirical_bayes_moments(self, feat_mean: torch.Tensor, feat_var: torch.Tensor, eta: float) -> None:
        with torch.no_grad():
            self.mu_delta.copy_((eta * feat_mean).to(self.mu_delta.device, self.mu_delta.dtype))
            var_shrunk = (1.0 - eta) + eta * feat_var.clamp(0.25, 4.0)
            self.sigma_delta.copy_(torch.sqrt(var_shrunk).to(self.sigma_delta.device, self.sigma_delta.dtype))

    def collapse(self) -> nn.LayerNorm:
        ln_out = copy.deepcopy(self.orig_ln).eval()
        with torch.no_grad():
            inv_s = 1.0 / self.sigma_delta.double().clamp(min=1e-8)
            ln_out.weight.data.copy_((ln_out.weight.double() * inv_s).to(ln_out.weight.dtype))
            ln_out.bias.data.copy_(((ln_out.bias.double() - self.mu_delta.double()) * inv_s).to(ln_out.bias.dtype))
        ln_out.requires_grad_(False)
        return ln_out


def build_2d_dct_basis(k: int, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Returns orthonormal 2D-DCT basis of shape [k*k, k, k]."""
    dct_1d = torch.zeros((k, k), device=device, dtype=dtype)
    for u in range(k):
        alpha = math.sqrt(1.0 / k) if u == 0 else math.sqrt(2.0 / k)
        for x in range(k):
            dct_1d[u, x] = alpha * math.cos(math.pi * (2 * x + 1) * u / (2.0 * k))
    basis = []
    for u in range(k):
        for v in range(k):
            basis.append(torch.outer(dct_1d[u], dct_1d[v]))
    return torch.stack(basis, dim=0)


class ExpandedDCTConv2d(nn.Module):
    """Decomposes a KxK Conv2d onto K^2 orthonormal 2D-DCT spatial-frequency bands."""

    def __init__(self, orig_conv: nn.Conv2d) -> None:
        super().__init__()
        enforce_no_gradients()
        if orig_conv.kernel_size[0] != orig_conv.kernel_size[1]:
            raise ValueError("ExpandedDCTConv2d requires a square kernel.")
        self.orig_conv = copy.deepcopy(orig_conv).eval()
        self.orig_conv.requires_grad_(False)
        k = orig_conv.kernel_size[0]
        dev = orig_conv.weight.device
        basis = build_2d_dct_basis(k, dev, torch.float64)
        w64 = orig_conv.weight.detach().double()
        coeffs = torch.einsum("oihw,bhw->boi", w64, basis)
        self.register_buffer("basis", basis.float())
        self.register_buffer("coeffs", coeffs.float())
        self.register_buffer("band_scales", torch.ones(k * k, device=dev, dtype=torch.float32))

    def set_frequency_band_scales(self, dc_scale: float = 1.0, mid_scale: float = 1.0, high_scale: float = 1.0) -> None:
        k = int(round(math.sqrt(self.band_scales.numel())))
        with torch.no_grad():
            for u in range(k):
                for v in range(k):
                    idx = u * k + v
                    if u == 0 and v == 0:
                        self.band_scales[idx] = dc_scale
                    elif u + v >= 2 * (k - 1) - 1:
                        self.band_scales[idx] = high_scale
                    else:
                        self.band_scales[idx] = mid_scale

    def _effective_weight(self) -> torch.Tensor:
        scaled = self.coeffs.double() * self.band_scales.double().view(-1, 1, 1)
        w = torch.einsum("boi,bhw->oihw", scaled, self.basis.double())
        return w.to(self.orig_conv.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self._effective_weight(),
            self.orig_conv.bias,
            stride=self.orig_conv.stride,
            padding=self.orig_conv.padding,
            dilation=self.orig_conv.dilation,
            groups=self.orig_conv.groups,
        )

    def collapse(self) -> nn.Conv2d:
        out = copy.deepcopy(self.orig_conv).eval()
        with torch.no_grad():
            out.weight.data.copy_(self._effective_weight())
        out.requires_grad_(False)
        return out


class ExpandedPCAProcrustesLinear(nn.Module):
    """
    Covariance-whitened principal subspace + Stiefel-Procrustes auxiliary readout lift (Eq. 12-13):
      z(h) = (W_core h + b) + W_pca (P_whiten h) + B_proc (Q_proc h)
    Initialized at W_pca = 0, B_proc = 0 (exact zero pre-edit discrepancy), collapsing into (W*, b*).
    """

    def __init__(
        self,
        w0: torch.Tensor,
        b0: torch.Tensor,
        p_whiten: torch.Tensor,
        q_proc: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        enforce_no_gradients()
        out_c, d_in = w0.shape
        dev = w0.device
        self.register_buffer("w_core", w0.detach().float().clone())
        self.register_buffer("b_vec", b0.detach().float().clone())
        self.register_buffer("p_whiten", p_whiten.detach().float().clone())
        self.register_buffer("w_pca", torch.zeros((out_c, p_whiten.shape[0]), device=dev, dtype=torch.float32))
        if q_proc is not None and q_proc.numel() > 0:
            self.register_buffer("q_proc", q_proc.detach().float().clone())
            self.register_buffer("b_proc", torch.zeros((out_c, q_proc.shape[0]), device=dev, dtype=torch.float32))
        else:
            self.register_buffer("q_proc", torch.zeros((0, d_in), device=dev, dtype=torch.float32))
            self.register_buffer("b_proc", torch.zeros((out_c, 0), device=dev, dtype=torch.float32))

    def jacobian_collapse(self) -> torch.Tensor:
        """Returns the linear collapse operator J_C of shape [d_in, d_in + r_pca + r_proc]."""
        d_in = self.w_core.shape[1]
        blocks = [torch.eye(d_in, device=self.w_core.device, dtype=torch.float64), self.p_whiten.double().t()]
        if self.q_proc.shape[0] > 0:
            blocks.append(self.q_proc.double().t())
        return torch.cat(blocks, dim=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hf = h.float()
        z = F.linear(hf, self.w_core, self.b_vec) + F.linear(F.linear(hf, self.p_whiten), self.w_pca)
        if self.q_proc.shape[0] > 0:
            z = z + F.linear(F.linear(hf, self.q_proc), self.b_proc)
        return z.to(h.dtype)

    def collapse_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            w_col = self.w_core.double() + self.w_pca.double() @ self.p_whiten.double()
            if self.q_proc.shape[0] > 0:
                w_col = w_col + self.b_proc.double() @ self.q_proc.double()
            return w_col.float(), self.b_vec.clone()


class ExpandedBoxAffineLinear(nn.Module):
    """4x4 post-affine bounding-box distribution adapter collapsing into (W_box*, b_box*)."""

    def __init__(self, w_box: torch.Tensor, b_box: torch.Tensor) -> None:
        super().__init__()
        enforce_no_gradients()
        out_d = w_box.shape[0]
        dev = w_box.device
        self.register_buffer("w_core", w_box.detach().float().clone())
        self.register_buffer("b_core", b_box.detach().float().clone())
        self.register_buffer("m_dist", torch.eye(out_d, device=dev, dtype=torch.float32))
        self.register_buffer("d_dist", torch.zeros(out_d, device=dev, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.linear(h.float(), self.w_core, self.b_core)
        return F.linear(z, self.m_dist, self.d_dist).to(h.dtype)

    def collapse_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            w_new = (self.m_dist.double() @ self.w_core.double()).float()
            b_new = (self.m_dist.double() @ self.b_core.double() + self.d_dist.double()).float()
            return w_new, b_new


def build_whitening_projector_from_covariance(
    cov: torch.Tensor,
    rank_r: int = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Computes P_whiten = (Lambda_r + eps I)^{1/2} U_r^T from empirical covariance or static Gram."""
    enforce_no_gradients()
    with torch.no_grad():
        sym = 0.5 * (cov.double() + cov.double().t())
        evals, evecs = torch.linalg.eigh(sym)
        r = min(rank_r, sym.shape[0])
        top_vals = evals[-r:].clamp(min=0.0)
        top_vecs = evecs[:, -r:]
        med = float(top_vals.median().clamp(min=1e-6).item())
        scales = torch.sqrt(top_vals / med + eps)
        p_whiten = (top_vecs * scales.unsqueeze(0)).t().contiguous()
        return p_whiten.float()
