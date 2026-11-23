"""
Single-Model Dormant Basis Extraction & Cross-Model Stiefel Orthogonal Procrustes Alignment (Section 4.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zero_grad_guard import enforce_no_gradients


def solve_stiefel_procrustes(m_src: torch.Tensor, m_tgt: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Solves Q* = argmin_{Q^T Q = I} ||M_src Q - M_tgt||_F = U V^T via SVD(M_src^T M_tgt).
    Returns (Q*, raw_cosine, aligned_cosine).
    """
    enforce_no_gradients()
    with torch.no_grad():
        a = F.normalize(m_src.double(), dim=1)
        b = F.normalize(m_tgt.double(), dim=1)
        d_min = min(a.shape[1], b.shape[1])
        raw_cos = float((F.normalize(a[:, :d_min], dim=1) * F.normalize(b[:, :d_min], dim=1)).sum(dim=1).mean().item())
        u, _, vh = torch.linalg.svd(a.t() @ b, full_matrices=False)
        q_star = u @ vh
        a_rot = F.normalize(a @ q_star, dim=1)
        aligned_cos = float((a_rot * b).sum(dim=1).mean().item())
        return q_star.float(), round(raw_cos, 4), round(aligned_cos, 4)


def extract_orthonormal_auxiliary_basis(
    aux_weights: list[torch.Tensor],
    target_w: torch.Tensor,
    max_rank: int = 48,
) -> torch.Tensor:
    """
    Aligns one or more auxiliary readout matrices (dormant o2m / early decoder heads
    or ontology-mapped cross-checkpoint heads) to `target_w` via Stiefel Procrustes and
    returns an orthonormal row basis Q_proc of shape [r_proc, d_in].
    """
    enforce_no_gradients()
    if not aux_weights:
        return torch.zeros((0, target_w.shape[1]), device=target_w.device, dtype=torch.float32)
    aligned_rows = []
    with torch.no_grad():
        for w_aux in aux_weights:
            if w_aux.ndim > 2:
                w_aux = w_aux.reshape(w_aux.shape[0], -1)
            q_rot, _, _ = solve_stiefel_procrustes(w_aux.to(target_w.device), target_w)
            aligned_rows.append((w_aux.to(target_w.device).double() @ q_rot.double()))
        stacked = torch.cat(aligned_rows, dim=0)
        _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
        r = min(max_rank, vh.shape[0])
        return vh[:r].float().contiguous()


def align_and_interpolate_trunk_parameters(
    model: nn.Module,
    sd_aux: dict[str, torch.Tensor],
    sd_target: dict[str, torch.Tensor],
    alpha: float = 0.85,
    mode: str = "procrustes",
    skip_substrings: tuple[str, ...] = ("score_head", "class_embed", "cv3", "norm", "bn", "embed", "pos"),
) -> dict[str, float]:
    """
    Controlled symmetry-group trunk interpolation over `model.named_parameters()`:
      - `mode == "raw_euclidean"`: Group {I} naive WiSE-FT `(1 - alpha) * W_aux + alpha * W_target`
      - `mode == "diag_norm"`:     Group Diag+(d) row-RMS matched interpolation without rotation
      - `mode == "procrustes"`:    Group O(d) Stiefel Orthogonal Procrustes + row-RMS variance preservation
    """
    enforce_no_gradients()
    device = next(model.parameters()).device
    raw_cos_vals, aligned_cos_vals = [], []
    edited = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in sd_aux or sd_aux[name].shape != param.shape or param.ndim < 2:
                continue
            if any(tok in name for tok in skip_substrings):
                continue
            w0 = sd_aux[name].to(device).float()
            w1 = sd_target[name].to(device).float()
            if not (torch.isfinite(w0).all() and torch.isfinite(w1).all()):
                continue
            out_c = w1.shape[0]
            m0 = w0.reshape(out_c, -1)
            m1 = w1.reshape(out_c, -1)

            q_rot, r_cos, a_cos = solve_stiefel_procrustes(m0, m1)
            raw_cos_vals.append(r_cos)
            aligned_cos_vals.append(a_cos)

            if mode == "raw_euclidean":
                m_new = (1.0 - alpha) * m0 + alpha * m1
            elif mode == "diag_norm":
                r1 = m1.norm(dim=1, keepdim=True).clamp(min=1e-6)
                r0 = m0.norm(dim=1, keepdim=True).clamp(min=1e-6)
                m_blend = (1.0 - alpha) * (m0 * (r1 / r0)) + alpha * m1
                m_new = m_blend * (r1 / m_blend.norm(dim=1, keepdim=True).clamp(min=1e-6))
            elif mode == "procrustes":
                r1 = m1.norm(dim=1, keepdim=True).clamp(min=1e-6)
                if m0.shape[0] <= m0.shape[1]:
                    u, _, vh = torch.linalg.svd((m1.double() @ m0.double().t()), full_matrices=False)
                    m0_rot = ((u @ vh) @ m0.double()).float()
                else:
                    m0_rot = m0 @ q_rot
                m0_rot = m0_rot * (r1 / m0_rot.norm(dim=1, keepdim=True).clamp(min=1e-6))
                m_blend = (1.0 - alpha) * m0_rot + alpha * m1
                m_new = m_blend * (r1 / m_blend.norm(dim=1, keepdim=True).clamp(min=1e-6))
            else:
                raise ValueError(f"Unsupported symmetry mode: {mode}")

            param.copy_(torch.nan_to_num(m_new, nan=0.0, posinf=1e4, neginf=-1e4).reshape_as(w1).to(param.dtype))
            edited += 1

    return {
        "edited_layers": float(edited),
        "mean_raw_cosine": round(sum(raw_cos_vals) / max(len(raw_cos_vals), 1), 4),
        "mean_procrustes_cosine": round(sum(aligned_cos_vals) / max(len(aligned_cos_vals), 1), 4),
    }
