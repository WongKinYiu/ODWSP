#!/usr/bin/env python3
"""
Zero-Gradient IR-ISWSP Weight-Space Calibration & Repair CLI (`calibrate_iswsp.py`).

Supports strictly gradient-free (`0` backpropagation) calibration and verification modes:
  1. `--mode zero_data`   : Apply fixed zero-image `IR-DCT-MaskBoundary` to a segmenter checkpoint.
  2. `--mode repair`      : Zero-image 2D-DCT + MAD Winsorization repair of corrupted prototype heads.
  3. `--mode bilinear`    : Verify Theorem 1 exact (3K+1) Bilinear Mask-IR lift & algebraic fold.
  4. `--mode full_wq`     : Verify Algorithm 1 Two-Stage Full-Matrix W_q Dual-Projection Gauss-Newton.
  5. `--mode stem_whiten` : Verify exact zero-parameter label-free Night+Fog Stem Conv2d whitening fold.
  6. `--mode bridge`      : Closed-form `MM-FW-Ridge` prototype contraction bridge calibration.

Examples:
  python calibrate_iswsp.py --mode zero_data --model ecseg_s --pretrained ./ecseg-s-iswsp.pt --edgecrafter-root ./EdgeCrafter --out ./runs/calibrate/ecseg_s_dct.pt
  python calibrate_iswsp.py --mode bilinear
  python calibrate_iswsp.py --mode full_wq
  python calibrate_iswsp.py --mode stem_whiten
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from iswsp import (
    ExpandedBilinearMaskIR,
    ExpandedDCTConv2d,
    ExpandedStemWhitenConv2d,
    ISWSP_PAPER_MODELS,
    apply_zero_image_corrupted_head_repair,
    enforce_no_gradients,
    instantiate_segmenter,
    max_ieee_fp32_diff,
    normalize_iswsp_model_key,
    safe_load_checkpoint,
    solve_mm_fw_ridge,
    solve_two_stage_full_matrix_wq,
)


def _sha256_state_dict(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        if t.is_floating_point():
            h.update(t.float().numpy().tobytes())
        else:
            h.update(t.to(torch.int64).numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="Zero-Gradient IR-ISWSP Calibration, Repair & Verification CLI")
    ap.add_argument(
        "--mode",
        type=str,
        default="bilinear",
        choices=["zero_data", "repair", "bilinear", "full_wq", "stem_whiten", "bridge"],
        help="Zero-gradient IR-ISWSP operation mode",
    )
    ap.add_argument("--model", type=str, default=None, help="ISWSP model key (required for zero_data / repair)")
    ap.add_argument("--pretrained", type=Path, default=None, help="Path to clean weights_only=True checkpoint (.pt)")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (for ecseg_*)")
    ap.add_argument("--g-hi", type=float, default=1.04, help="High-frequency 2D-DCT boundary gain (zero_data mode)")
    ap.add_argument("--mad-thresh", type=float, default=2.5, help="MAD outlier threshold (repair mode)")
    ap.add_argument("--lam-rel", type=float, default=0.015, help="Relative ridge damping (bridge mode)")
    ap.add_argument("--device", type=str, default="cpu", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--out", type=Path, default=None, help="Optional output path to save calibrated checkpoint (.pt)")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device)
    torch.manual_seed(42)

    if args.mode == "bilinear":
        bsz, q_cnt, d_q, c_p, k, h, w = 2, 16, 64, 32, 32, 16, 16
        op = ExpandedBilinearMaskIR(
            torch.randn(k, d_q, device=dev) * 0.1,
            torch.randn(k, device=dev) * 0.05,
            torch.randn(k, c_p, 1, 1, device=dev) * 0.1,
            torch.randn(k, device=dev) * 0.05,
            torch.randn(1, device=dev) * 0.1,
        )
        op.set_bilinear_update(
            ds=torch.randn(k, device=dev) * 0.08,
            dd=torch.randn(k, device=dev) * 0.05,
            de=torch.randn(k, device=dev) * 0.05,
            db0=0.07,
        )
        h_q = torch.randn(bsz, q_cnt, d_q, device=dev)
        f_p = torch.randn(bsz, c_p, h, w, device=dev)
        diff = max_ieee_fp32_diff(lambda: op.forward_expanded(h_q, f_p), lambda: op.forward_collapsed(h_q, f_p))
        print(json.dumps({"mode": "bilinear", "theorem1_max_fp32_diff": diff, "delta_params": 0, "delta_flops": 0}, indent=2))
        return

    if args.mode == "full_wq":
        k, d_q, n_inst, s_px = 32, 48, 24, 64
        w_q = torch.randn(k, d_q, device=dev) * 0.1
        b_q = torch.zeros(k, device=dev)
        h_f = torch.randn(n_inst, d_q, device=dev)
        p_c = torch.randn(n_inst, k, s_px, device=dev)
        y_gt = (torch.rand(n_inst, s_px, device=dev) > 0.5).float()
        bnd = (torch.rand(n_inst, s_px, device=dev) > 0.85).float()
        wq_star, bq_star = solve_two_stage_full_matrix_wq(w_q, b_q, h_f, p_c, y_gt, boundary_masks=bnd)
        print(json.dumps({
            "mode": "full_wq",
            "shape_wq": list(wq_star.shape),
            "delta_wq_fro": round(float((wq_star - w_q).norm().item()), 6),
            "delta_bq_norm": round(float((bq_star - b_q).norm().item()), 6),
            "delta_params": 0,
        }, indent=2))
        return

    if args.mode == "stem_whiten":
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=0, bias=False).to(dev).eval()
        bn = nn.BatchNorm2d(16).to(dev).eval()
        op = ExpandedStemWhitenConv2d(conv, bn)
        op.set_shift_moments(
            mu_shift=torch.tensor([0.25, 0.24, 0.23], device=dev),
            std_shift=torch.tensor([0.11, 0.11, 0.10], device=dev),
            mu_ref=torch.tensor([0.48, 0.46, 0.41], device=dev),
            std_ref=torch.tensor([0.23, 0.22, 0.22], device=dev),
            alpha=0.85,
        )
        c_col, bn_col = op.collapse()
        x = torch.rand(2, 3, 16, 16, device=dev)
        diff = max_ieee_fp32_diff(lambda: op(x), lambda: bn_col(c_col(x)))
        print(json.dumps({"mode": "stem_whiten", "max_interior_fp32_diff": diff, "delta_params": 0}, indent=2))
        return

    if args.mode == "bridge":
        x = torch.randn(256, 32, device=dev)
        w_true = torch.randn(64, 32, device=dev) * 0.2
        y = x @ w_true.t() + 0.02 * torch.randn(256, 64, device=dev)
        w_br, b_br = solve_mm_fw_ridge(x, y, fg_weights=torch.rand(256, device=dev) + 0.2, lam_rel=args.lam_rel)
        rel_err = float(((x @ w_br.t() + b_br - y).norm() / y.norm()).item())
        report = {"mode": "bridge", "bridge_shape": list(w_br.shape), "relative_recon_error": round(rel_err, 6)}
        if args.out is not None:
            out_p = args.out.resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"version": "0.1.1", "weight": w_br.cpu(), "bias": b_br.cpu(), "lam_rel": args.lam_rel}, out_p)
            report["saved_to"] = str(out_p)
        print(json.dumps(report, indent=2))
        return

    if args.model is None or args.pretrained is None:
        raise ValueError(f"`--mode {args.mode}` requires `--model <key>` and `--pretrained <path.pt>`.")

    mkey = normalize_iswsp_model_key(args.model)
    spec = ISWSP_PAPER_MODELS[mkey]
    pkg = safe_load_checkpoint(args.pretrained, map_location="cpu")
    sd = pkg["state_dict"] if "state_dict" in pkg else pkg["model"]
    model, _ = instantiate_segmenter(spec, sd, dev, edgecrafter_root=args.edgecrafter_root, arch_yaml=pkg.get("arch_yaml"))

    edited_layers: list[str] = []
    if args.mode == "zero_data":
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Conv2d) and mod.kernel_size[0] >= 3 and any(tok in name.lower() for tok in ("proto", "mask", "seg")):
                op = ExpandedDCTConv2d(mod)
                op.set_frequency_band_scales(g_dc=1.0, g_mid=1.0, g_hi=args.g_hi)
                mod.weight.copy_(op.collapse().weight)
                edited_layers.append(name)
    elif args.mode == "repair":
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Conv2d) and any(tok in name.lower() for tok in ("proto", "mask", "seg")):
                w_rep, aud = apply_zero_image_corrupted_head_repair(mod.weight, mad_thresh=args.mad_thresh)
                if aud["channels_freq_repaired"] > 0 or aud["channels_norm_winsorized"] > 0:
                    mod.weight.copy_(w_rep)
                    edited_layers.append(name)

    sd_out = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    digest = _sha256_state_dict(sd_out)
    report = {
        "mode": args.mode,
        "model": mkey,
        "edited_layers": edited_layers,
        "num_edited_layers": len(edited_layers),
        "state_dict_sha256": digest,
        "delta_params": 0,
    }
    if args.out is not None:
        out_p = args.out.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": "0.1.1", "model_key": mkey, "mode": args.mode, "sha256": digest, "state_dict": sd_out}, out_p)
        report["saved_to"] = str(out_p)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
