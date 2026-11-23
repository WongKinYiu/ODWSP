#!/usr/bin/env python3
"""
Zero-Gradient IR-ODWSP Weight-Space Calibration CLI (`calibrate.py`, Algorithm 1).

Executes Function-Preserving Inverse Reparameterization (E), Task-Aligned
Winner-Exclusive Set-Prediction Normal Equations, Cross-Basin Stiefel-Procrustes
Alignment, and Exact Algebraic Re-Collapse (C) with zero backpropagation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from odwsp import (
    ExpandedPCAProcrustesLinear,
    IRODWSPConfig,
    SufficientStatsAccumulator,
    build_whitening_projector_from_covariance,
    enforce_no_gradients,
    extract_orthonormal_auxiliary_basis,
    max_ieee_fp32_diff,
    safe_load_checkpoint,
    solve_lifted_ir_odwsp_readout,
)


def calibrate_readout_subspace(
    w_cls: torch.Tensor,
    b_cls: torch.Tensor,
    stats: SufficientStatsAccumulator,
    aux_heads: list[torch.Tensor] | None = None,
    cfg: IRODWSPConfig | None = None,
    pca_rank: int = 64,
    supercategory_groups: list[list[int]] | None = None,
    class_counts: torch.Tensor | None = None,
) -> dict[str, object]:
    """Runs Algorithm 1 (Stages I-IV) on a detector readout and verifies collapse equivalence."""
    enforce_no_gradients()
    cfg = cfg or IRODWSPConfig()
    cov = stats.cov_h / max(stats.n_tokens, 1) if stats.n_tokens > 0 else (w_cls.double().t() @ w_cls.double())
    p_whiten = build_whitening_projector_from_covariance(cov, rank_r=pca_rank)
    q_proc = extract_orthonormal_auxiliary_basis(aux_heads or [], w_cls)

    lifted = ExpandedPCAProcrustesLinear(w_cls, b_cls, p_whiten=p_whiten, q_proc=q_proc)
    probe = torch.randn(16, w_cls.shape[1], device=w_cls.device, dtype=torch.float32)
    pre_diff = max_ieee_fp32_diff(
        lambda: lifted(probe),
        lambda: torch.nn.functional.linear(probe, w_cls.float(), b_cls.float()),
    )

    w_star, b_star = solve_lifted_ir_odwsp_readout(
        lifted,
        stats.h_mat,
        stats.g_vec,
        lam_reg=cfg.lam_reg,
        eta_head=cfg.eta_head,
        supercategory_groups=supercategory_groups,
        class_counts=class_counts,
        js_kappa0=cfg.js_kappa0,
    )
    post_diff = max_ieee_fp32_diff(
        lambda: lifted(probe),
        lambda: torch.nn.functional.linear(probe, w_star, b_star),
    )
    return {
        "w_star": w_star,
        "b_star": b_star,
        "pre_collapse_max_diff": pre_diff,
        "post_collapse_max_diff": post_diff,
        "delta_params": 0,
        "delta_flops": 0,
    }


def main() -> None:
    enforce_no_gradients()
    parser = argparse.ArgumentParser(description="IR-ODWSP Zero-Gradient Calibration CLI (Algorithm 1)")
    parser.add_argument("--stats-ckpt", type=Path, default=None, help="Optional cached sufficient-statistics dict (.pt)")
    parser.add_argument("--eta-head", type=float, default=0.20)
    parser.add_argument("--lam-reg", type=float, default=2800.0)
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path for collapsed weights (.pt)")
    args = parser.parse_args()

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    dev = torch.device(dev_str)
    cfg = IRODWSPConfig(eta_head=args.eta_head, lam_reg=args.lam_reg)

    if args.stats_ckpt is not None:
        pkg = safe_load_checkpoint(args.stats_ckpt, map_location=dev)
        w0, b0 = pkg["w_cls"].to(dev), pkg["b_cls"].to(dev)
        acc = SufficientStatsAccumulator(w0.shape[1], w0.shape[0], dev)
        acc.h_mat.copy_(pkg["h_mat"].to(dev))
        acc.g_vec.copy_(pkg["g_vec"].to(dev))
        acc.cov_h.copy_(pkg["cov_h"].to(dev))
        acc.n_tokens = int(pkg.get("n_tokens", 1))
        aux_list = [pkg["w_aux"].to(dev)] if "w_aux" in pkg else []
    else:
        torch.manual_seed(0)
        d_feat, n_cls = 128, 80
        w0 = torch.randn(n_cls, d_feat, device=dev) * 0.08
        b0 = torch.full((n_cls,), -4.2, device=dev)
        aux_list = [w0 @ torch.linalg.qr(torch.randn(d_feat, d_feat, device=dev))[0] + 0.01 * torch.randn_like(w0)]
        acc = SufficientStatsAccumulator(d_feat, n_cls, dev)
        feats = torch.randn(256, d_feat, device=dev)
        logits = torch.nn.functional.linear(feats, w0, b0)
        p_boxes = torch.rand(256, 4, device=dev) * 200.0
        p_boxes[:, 2:] += p_boxes[:, :2] + 24.0
        gt_boxes = p_boxes[:12].clone() + 2.0
        gt_cls = torch.arange(12, device=dev) % n_cls
        acc.update(feats, logits, p_boxes, gt_boxes, gt_cls, iou_gamma=cfg.iou_gamma)

    res = calibrate_readout_subspace(w0, b0, acc, aux_heads=aux_list, cfg=cfg, pca_rank=args.pca_rank)
    summary = {
        "zero_gradients_enforced": not torch.is_grad_enabled(),
        "pre_collapse_max_diff": res["pre_collapse_max_diff"],
        "post_collapse_max_diff": res["post_collapse_max_diff"],
        "weight_l2_update_norm": round(float((res["w_star"] - w0).norm().item()), 6),
        "delta_params_pct": 0.0,
        "delta_flops_pct": 0.0,
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        out_p = args.out.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": "0.1.0", "w_cls": res["w_star"].cpu(), "b_cls": res["b_star"].cpu()}, out_p)


if __name__ == "__main__":
    main()
