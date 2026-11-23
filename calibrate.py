#!/usr/bin/env python3
"""
Zero-Gradient IR-ODWSP Weight-Space Calibration CLI (`calibrate.py`, Algorithm 1 & Sec. 4.5.3).

Supports three zero-gradient Weight-Space Programming calibration workflows (`0` backpropagation):
  1. `--mode detector`: End-to-end detector calibration on a COCO-formatted split (`train2017` / custom dataset).
     Runs Function-Preserving Inverse Reparameterization (E), Winner-Exclusive Set-Prediction Normal Equations,
     optional Cross-Basin Stiefel-Procrustes Alignment, and Exact Algebraic Re-Collapse (C) into a `1.00x` `.pt` checkpoint.
  2. `--mode bridge`  : Closed-form Moment-Matched Foreground-Weighted Cholesky Ridge (`MM-FW-Ridge`) subgraph bridge
     calibration for negative latency budgets (`beta < 0%`).
  3. `--mode readout` : Standalone readout calibration from cached sufficient statistics (`--stats-ckpt`) or verification probe.

Examples:
  # 1) End-to-end zero-gradient WSP calibration on a detector + COCO-format calibration images
  python calibrate.py --mode detector --model yolo26s --pretrained ./yolo26s-odwsp.pt \
      --coco-root /coco --split train2017 --max-images 2400 --device 0 --out ./runs/calibrate/yolo26s-custom-odwsp.pt

  # 2) Closed-form MM-FW-Ridge subgraph bridge calibration for negative budgets (beta < 0%)
  python calibrate.py --mode bridge --lam-rel 0.015 --moment-gain 1.85 --device 0 --out ./runs/calibrate/mm_fw_bridge.pt

  # 3) Standalone Algorithm 1 readout calibration & exact collapse verification
  python calibrate.py --mode readout --eta-head 0.20 --lam-reg 2800.0 --pca-rank 64 --device 0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from odwsp import (
    COCO80_TO_91,
    ExpandedPCAProcrustesLinear,
    IRODWSPConfig,
    PAPER_MODELS,
    SufficientStatsAccumulator,
    align_and_interpolate_trunk_parameters,
    build_whitening_projector_from_covariance,
    enforce_no_gradients,
    extract_orthonormal_auxiliary_basis,
    fit_mm_fw_ridge_bridge,
    max_ieee_fp32_diff,
    safe_load_checkpoint,
    solve_lifted_ir_odwsp_readout,
)
from odwsp.models import instantiate_subspace_model, make_yolo26_arch_cfg, normalize_model_key

COCO91_TO_80 = {cid: idx for idx, cid in enumerate(COCO80_TO_91)}


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
    """Runs Algorithm 1 (Stages I-IV) on a detector readout and verifies IEEE-754 FP32 collapse equivalence."""
    enforce_no_gradients()
    cfg = cfg or IRODWSPConfig()
    cov = stats.cov_h / max(stats.n_tokens, 1) if stats.n_tokens > 0 else (w_cls.double().t() @ w_cls.double())
    p_whiten = build_whitening_projector_from_covariance(cov, rank_r=min(pca_rank, w_cls.shape[1]))
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


def _load_raw_state_dict(ckpt_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pkg = safe_load_checkpoint(ckpt_path, map_location="cpu")
    if "state_dict" in pkg and isinstance(pkg["state_dict"], dict):
        sd = pkg["state_dict"]
    elif "model" in pkg and isinstance(pkg["model"], dict):
        sd = pkg["model"]
    elif all(isinstance(v, torch.Tensor) for v in pkg.values()):
        sd = pkg  # type: ignore[assignment]
    else:
        raise ValueError(f"Checkpoint {ckpt_path.name} must contain a pure `state_dict` tensor mapping.")
    meta = {k: v for k, v in pkg.items() if k not in ("state_dict", "model")}
    return sd, meta


def calibrate_detector_on_coco(
    model_key: str,
    pretrained_path: Path,
    coco_root: Path,
    split: str = "train2017",
    ann_file: Path | None = None,
    aux_pretrained_path: Path | None = None,
    edgecrafter_root: Path | None = None,
    max_images: int = 2400,
    cfg: IRODWSPConfig | None = None,
    pca_rank: int = 64,
    trunk_alpha: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    End-to-end zero-gradient IR-ODWSP calibration directly on a pretrained YOLO26, EdgeCrafter,
    or RF-DETR detector over `max_images` labeled calibration images (`0` gradients everywhere).
    """
    enforce_no_gradients()
    from pycocotools.coco import COCO

    cfg = cfg or IRODWSPConfig()
    key = normalize_model_key(model_key)
    spec = PAPER_MODELS[key]
    sd_main, meta_in = _load_raw_state_dict(pretrained_path)
    sd_aux: dict[str, torch.Tensor] | None = None
    if aux_pretrained_path is not None:
        sd_aux, _ = _load_raw_state_dict(aux_pretrained_path)
        if trunk_alpha > 0.0:
            sd_main = align_and_interpolate_trunk_parameters(sd_main, sd_aux, alpha=trunk_alpha, use_procrustes=True)

    arch_y = meta_in.get("arch_yaml") if isinstance(meta_in.get("arch_yaml"), dict) else (make_yolo26_arch_cfg(spec.scale) if spec.family == "yolo26" else None)
    model, _ = instantiate_subspace_model(spec, copy.deepcopy(sd_main), device=device, edgecrafter_root=edgecrafter_root, arch_yaml=arch_y)

    ann_path = ann_file.resolve() if ann_file is not None else (coco_root.resolve() / "annotations" / f"instances_{split}.json")
    img_dir = coco_root.resolve() / split
    coco = COCO(str(ann_path))
    img_ids = sorted(coco.getImgIds())[:max_images]
    res = spec.resolution

    if spec.family == "yolo26":
        det = model.model[-1]
        accs = [SufficientStatsAccumulator(int(det.one2one_cv3[l][2].in_channels), 80, device) for l in range(3)]
        cls_counts = torch.zeros(80, device=device, dtype=torch.float64)

        for img_id in img_ids:
            info = coco.loadImgs([img_id])[0]
            im = cv2.imread(str(img_dir / info["file_name"]), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h0, w0 = im.shape[:2]
            r = res / max(h0, w0)
            w, h = (min(math.ceil(w0 * r), res), min(math.ceil(h0 * r), res)) if r != 1.0 else (w0, h0)
            if r != 1.0:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            r_lb = min(res / h, res / w, 1.0)
            unpad = (int(round(w * r_lb)), int(round(h * r_lb)))
            if (w, h) != unpad:
                im = cv2.resize(im, unpad, interpolation=cv2.INTER_LINEAR)
            dw, dh = (res - unpad[0]) / 2.0, (res - unpad[1]) / 2.0
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im_pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            t = torch.from_numpy(np.ascontiguousarray(im_pad[..., ::-1].transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(device)

            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
            gt_b_l, gt_c_l = [], []
            sc_tot = r * r_lb
            for a in anns:
                if a["category_id"] not in COCO91_TO_80 or a["bbox"][2] <= 1.0 or a["bbox"][3] <= 1.0:
                    continue
                bx, by, bw, bh = a["bbox"]
                gt_b_l.append([bx * sc_tot + left, by * sc_tot + top, (bx + bw) * sc_tot + left, (by + bh) * sc_tot + top])
                c80 = COCO91_TO_80[a["category_id"]]
                gt_c_l.append(c80)
                cls_counts[c80] += 1.0
            gt_b = torch.tensor(gt_b_l, device=device, dtype=torch.float32) if gt_b_l else torch.zeros((0, 4), device=device)
            gt_c = torch.tensor(gt_c_l, device=device, dtype=torch.long) if gt_c_l else torch.zeros((0,), device=device, dtype=torch.long)

            saved: dict[int, torch.Tensor] = {}
            cur = t
            for i in range(23):
                blk = model.model[i]
                if i >= 11 and blk.f != -1:
                    cur = [cur if j == -1 else saved[j] for j in blk.f]
                cur = blk(cur)
                if i in (4, 6, 10, 13, 16, 19, 22):
                    saved[i] = cur
            nf = [saved[16], saved[19], saved[22]]
            _, pr = det(nf)
            dbox_all = det._get_decode_boxes(pr["one2one"])[0].permute(1, 0)
            off = 0
            for l_idx in range(3):
                fl = nf[l_idx]
                h_pre = det.one2one_cv3[l_idx][1](det.one2one_cv3[l_idx][0](fl))
                z_l = det.one2one_cv3[l_idx][2](h_pre)
                n_l = int(h_pre.shape[2] * h_pre.shape[3])
                h_tok = h_pre[0].permute(1, 2, 0).reshape(n_l, -1)
                z_tok = z_l[0].permute(1, 2, 0).reshape(n_l, 80)
                b_tok = dbox_all[off : off + n_l]
                accs[l_idx].update(h_tok, z_tok, b_tok, gt_b, gt_c, iou_gamma=cfg.iou_gamma)
                off += n_l

        sd_out = copy.deepcopy(sd_main)
        max_post_diff = 0.0
        l2_norms = []
        for l_idx in range(3):
            k_w = f"model.23.one2one_cv3.{l_idx}.2.weight"
            k_b = f"model.23.one2one_cv3.{l_idx}.2.bias"
            w0 = sd_out[k_w].squeeze(-1).squeeze(-1).to(device).float()
            b0 = sd_out[k_b].to(device).float()
            aux_h = []
            k_o2m = f"model.23.cv3.{l_idx}.2.weight"
            if k_o2m in sd_out:
                aux_h.append(sd_out[k_o2m].squeeze(-1).squeeze(-1).to(device).float())
            if sd_aux is not None and k_w in sd_aux:
                aux_h.append(sd_aux[k_w].squeeze(-1).squeeze(-1).to(device).float())
            cal = calibrate_readout_subspace(w0, b0, accs[l_idx], aux_heads=aux_h, cfg=cfg, pca_rank=pca_rank, class_counts=cls_counts)
            w_new, b_new = cal["w_star"], cal["b_star"]
            sd_out[k_w] = w_new.unsqueeze(-1).unsqueeze(-1).cpu().to(sd_out[k_w].dtype)
            sd_out[k_b] = b_new.cpu().to(sd_out[k_b].dtype)
            max_post_diff = max(max_post_diff, float(cal["post_collapse_max_diff"]))
            l2_norms.append(float((w_new - w0).norm().item()))

        ckpt_out = {
            "version": "0.1.0",
            "model_key": key,
            "family": spec.family,
            "arch_yaml": arch_y,
            "ingraph_cfg": meta_in.get("ingraph_cfg", {"beta_box": 0.30, "gamma_dn": 0.08, "gamma_up": 0.20, "tau_iou": 0.55, "tau_dom": 0.35, "tau_t": 1.0}),
            "state_dict": sd_out,
        }
        report = {
            "mode": "detector",
            "model_key": key,
            "images_calibrated": len(img_ids),
            "zero_gradients_enforced": not torch.is_grad_enabled(),
            "post_collapse_max_diff": max_post_diff,
            "per_scale_l2_update_norm": [round(v, 6) for v in l2_norms],
            "delta_params_pct": 0.0,
            "delta_flops_pct": 0.0,
        }
        return ckpt_out, report

    import torchvision.transforms.v2.functional as F_v2
    from torchvision.transforms import InterpolationMode

    if spec.family == "rfdetr":
        w0 = sd_main["class_embed.weight"].to(device).float()
        b0 = sd_main["class_embed.bias"].to(device).float()
        n_c, d_f = int(w0.shape[0]), int(w0.shape[1])
        acc = SufficientStatsAccumulator(d_f, n_c, device)
        captured: list[torch.Tensor] = []
        hook = model.class_embed.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach()))

        for img_id in img_ids:
            info = coco.loadImgs([img_id])[0]
            p_im = img_dir / info["file_name"]
            if not p_im.is_file():
                continue
            with Image.open(p_im) as pil_im:
                rgb = pil_im.convert("RGB")
            im_r = F_v2.resize(rgb, [res, res], interpolation=InterpolationMode.BILINEAR, antialias=True)
            t = F_v2.normalize(F_v2.pil_to_tensor(im_r).float().div_(255.0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
            captured.clear()
            out = model(t)
            h_last = captured[-1]
            if h_last.ndim == 4:
                h_last = h_last[-1]
            h_tok = h_last[0]
            z_tok = out["pred_logits"][0]
            pb = out["pred_boxes"][0]
            boxes_xyxy = torch.stack([pb[:, 0] - 0.5 * pb[:, 2], pb[:, 1] - 0.5 * pb[:, 3], pb[:, 0] + 0.5 * pb[:, 2], pb[:, 1] + 0.5 * pb[:, 3]], dim=-1) * float(res)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
            ow, oh = rgb.size
            gt_b_l, gt_c_l = [], []
            for a in anns:
                if a["category_id"] >= n_c or a["bbox"][2] <= 1.0 or a["bbox"][3] <= 1.0:
                    continue
                bx, by, bw, bh = a["bbox"]
                gt_b_l.append([bx / ow * res, by / oh * res, (bx + bw) / ow * res, (by + bh) / oh * res])
                gt_c_l.append(int(a["category_id"]))
            gt_b = torch.tensor(gt_b_l, device=device, dtype=torch.float32) if gt_b_l else torch.zeros((0, 4), device=device)
            gt_c = torch.tensor(gt_c_l, device=device, dtype=torch.long) if gt_c_l else torch.zeros((0,), device=device, dtype=torch.long)
            acc.update(h_tok, z_tok, boxes_xyxy, gt_b, gt_c, iou_gamma=cfg.iou_gamma)
        hook.remove()

        aux_h = [sd_main[f"transformer.enc_out_class_embed.{g}.weight"].to(device).float() for g in range(min(4, 13)) if f"transformer.enc_out_class_embed.{g}.weight" in sd_main]
        cal = calibrate_readout_subspace(w0, b0, acc, aux_heads=aux_h, cfg=cfg, pca_rank=pca_rank)
        sd_out = copy.deepcopy(sd_main)
        sd_out["class_embed.weight"] = cal["w_star"].cpu().to(sd_out["class_embed.weight"].dtype)
        sd_out["class_embed.bias"] = cal["b_star"].cpu().to(sd_out["class_embed.bias"].dtype)
        return (
            {"version": "0.1.0", "model_key": key, "family": spec.family, "state_dict": sd_out},
            {
                "mode": "detector",
                "model_key": key,
                "images_calibrated": len(img_ids),
                "zero_gradients_enforced": not torch.is_grad_enabled(),
                "post_collapse_max_diff": cal["post_collapse_max_diff"],
                "weight_l2_update_norm": round(float((cal["w_star"] - w0).norm().item()), 6),
                "delta_params_pct": 0.0,
                "delta_flops_pct": 0.0,
            },
        )

    # EdgeCrafter final decoder readout calibration
    eval_idx = int(model.decoder.decoder.eval_idx)
    k_w = f"decoder.dec_score_head.{eval_idx}.weight"
    k_b = f"decoder.dec_score_head.{eval_idx}.bias"
    w0 = sd_main[k_w].to(device).float()
    b0 = sd_main[k_b].to(device).float()
    acc_ec = SufficientStatsAccumulator(int(w0.shape[1]), 80, device)
    cap_ec: list[torch.Tensor] = []
    hook_ec = model.decoder.dec_score_head[eval_idx].register_forward_pre_hook(lambda _m, inp: cap_ec.append(inp[0].detach()))

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        p_im = img_dir / info["file_name"]
        if not p_im.is_file():
            continue
        with Image.open(p_im) as pil_im:
            rgb = pil_im.convert("RGB")
        im_r = F_v2.resize(rgb, [res, res], interpolation=InterpolationMode.BILINEAR, antialias=True)
        t = F_v2.normalize(F_v2.pil_to_tensor(im_r).float().div_(255.0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
        cap_ec.clear()
        out = model(t)
        h_tok = cap_ec[-1][0]
        z_tok = out["pred_logits"][0]
        pb = out["pred_boxes"][0]
        boxes_xyxy = torch.stack([pb[:, 0] - 0.5 * pb[:, 2], pb[:, 1] - 0.5 * pb[:, 3], pb[:, 0] + 0.5 * pb[:, 2], pb[:, 1] + 0.5 * pb[:, 3]], dim=-1) * float(res)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
        ow, oh = rgb.size
        gt_b_l, gt_c_l = [], []
        for a in anns:
            if a["category_id"] not in COCO91_TO_80 or a["bbox"][2] <= 1.0 or a["bbox"][3] <= 1.0:
                continue
            bx, by, bw, bh = a["bbox"]
            gt_b_l.append([bx / ow * res, by / oh * res, (bx + bw) / ow * res, (by + bh) / oh * res])
            gt_c_l.append(COCO91_TO_80[a["category_id"]])
        gt_b = torch.tensor(gt_b_l, device=device, dtype=torch.float32) if gt_b_l else torch.zeros((0, 4), device=device)
        gt_c = torch.tensor(gt_c_l, device=device, dtype=torch.long) if gt_c_l else torch.zeros((0,), device=device, dtype=torch.long)
        acc_ec.update(h_tok, z_tok, boxes_xyxy, gt_b, gt_c, iou_gamma=cfg.iou_gamma)
    hook_ec.remove()

    aux_ec = [sd_main["decoder.enc_score_head.weight"].to(device).float()]
    if sd_aux is not None and k_w in sd_aux:
        aux_ec.append(sd_aux[k_w].to(device).float())
    cal_ec = calibrate_readout_subspace(w0, b0, acc_ec, aux_heads=aux_ec, cfg=cfg, pca_rank=pca_rank)
    sd_out = copy.deepcopy(sd_main)
    sd_out[k_w] = cal_ec["w_star"].cpu().to(sd_out[k_w].dtype)
    sd_out[k_b] = cal_ec["b_star"].cpu().to(sd_out[k_b].dtype)
    return (
        {"version": "0.1.0", "model_key": key, "family": spec.family, "state_dict": sd_out},
        {
            "mode": "detector",
            "model_key": key,
            "images_calibrated": len(img_ids),
            "zero_gradients_enforced": not torch.is_grad_enabled(),
            "post_collapse_max_diff": cal_ec["post_collapse_max_diff"],
            "weight_l2_update_norm": round(float((cal_ec["w_star"] - w0).norm().item()), 6),
            "delta_params_pct": 0.0,
            "delta_flops_pct": 0.0,
        },
    )


def main() -> None:
    enforce_no_gradients()
    parser = argparse.ArgumentParser(description="IR-ODWSP Zero-Gradient Calibration CLI (Algorithm 1 & MM-FW-Ridge)")
    parser.add_argument("--mode", type=str, default=None, choices=["detector", "bridge", "readout"], help="Calibration workflow mode")
    parser.add_argument("--model", type=str, default=None, help="Detector model key for --mode detector (e.g., yolo26s, rfdetr_m, ecdet_s)")
    parser.add_argument("--pretrained", type=Path, default=None, help="Pretrained state_dict checkpoint (.pt) for --mode detector")
    parser.add_argument("--aux-pretrained", type=Path, default=None, help="Optional auxiliary basin checkpoint (e.g., O365) for Procrustes alignment")
    parser.add_argument("--trunk-alpha", type=float, default=0.0, help="Stiefel-Procrustes trunk interpolation coefficient in [0, 1]")
    parser.add_argument("--coco-root", type=Path, default=None, help="MS COCO or custom COCO-format dataset root for --mode detector")
    parser.add_argument("--split", type=str, default="train2017", help="Calibration image folder name under --coco-root")
    parser.add_argument("--ann-file", type=Path, default=None, help="Optional explicit COCO annotation JSON path")
    parser.add_argument("--max-images", type=int, default=2400, help="Maximum calibration images (disjoint from test split)")
    parser.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (for ecdet_*)")
    parser.add_argument("--stats-ckpt", type=Path, default=None, help="Cached sufficient-statistics dict (.pt) for --mode readout")
    parser.add_argument("--bridge-feats", type=Path, default=None, help="Optional activation pair dict {x_feat, y_target, fg_weights} (.pt) for --mode bridge")
    parser.add_argument("--eta-head", type=float, default=0.20)
    parser.add_argument("--lam-reg", type=float, default=2800.0)
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--lam-rel", type=float, default=0.015, help="Relative ridge regularization for --mode bridge")
    parser.add_argument("--moment-gain", type=float, default=1.85, help="Max foreground variance-restoration gain for --mode bridge")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    parser.add_argument("--out", type=Path, default=None, help="Output path for calibrated checkpoint or bridge (.pt)")
    args = parser.parse_args()

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    dev = torch.device(dev_str)
    mode = args.mode or ("detector" if args.pretrained is not None else "readout")
    cfg = IRODWSPConfig(eta_head=args.eta_head, lam_reg=args.lam_reg)

    if mode == "detector":
        if args.model is None or args.pretrained is None or args.coco_root is None:
            raise ValueError("`--mode detector` requires `--model`, `--pretrained`, and `--coco-root`.")
        ckpt_out, summary = calibrate_detector_on_coco(
            model_key=args.model,
            pretrained_path=args.pretrained,
            coco_root=args.coco_root,
            split=args.split,
            ann_file=args.ann_file,
            aux_pretrained_path=args.aux_pretrained,
            edgecrafter_root=args.edgecrafter_root,
            max_images=args.max_images,
            cfg=cfg,
            pca_rank=args.pca_rank,
            trunk_alpha=args.trunk_alpha,
            device=dev,
        )
        print(json.dumps(summary, indent=2))
        if args.out is not None:
            out_p = args.out.resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_out, out_p)
        return

    if mode == "bridge":
        if args.bridge_feats is not None:
            pkg_b = safe_load_checkpoint(args.bridge_feats, map_location=dev)
            x_f = pkg_b["x_feat"].to(dev)
            y_t = pkg_b["y_target"].to(dev)
            fg_w = pkg_b["fg_weights"].to(dev) if "fg_weights" in pkg_b else None
        else:
            torch.manual_seed(0)
            x_f = torch.randn(1024, 128, device=dev)
            w_true = torch.randn(256, 128, device=dev) * 0.15
            y_t = x_f @ w_true.t() + 0.05 * torch.randn(1024, 256, device=dev)
            fg_w = 1.0 + 10.0 * torch.rand(1024, device=dev)
        w_br, b_br = fit_mm_fw_ridge_bridge(
            x_f, y_t, fg_weights=fg_w, lam_rel=args.lam_rel, with_bias=True, moment_match_max_gain=args.moment_gain
        )
        summary_b = {
            "mode": "bridge",
            "zero_gradients_enforced": not torch.is_grad_enabled(),
            "bridge_shape": list(w_br.shape),
            "lam_rel": args.lam_rel,
            "moment_match_max_gain": args.moment_gain,
        }
        print(json.dumps(summary_b, indent=2))
        if args.out is not None:
            out_p = args.out.resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"version": "0.1.0", "W_bridge": w_br, "b_bridge": b_br}, out_p)
        return

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
    summary_r = {
        "mode": "readout",
        "zero_gradients_enforced": not torch.is_grad_enabled(),
        "pre_collapse_max_diff": res["pre_collapse_max_diff"],
        "post_collapse_max_diff": res["post_collapse_max_diff"],
        "weight_l2_update_norm": round(float((res["w_star"] - w0).norm().item()), 6),
        "delta_params_pct": 0.0,
        "delta_flops_pct": 0.0,
    }
    print(json.dumps(summary_r, indent=2))
    if args.out is not None:
        out_p = args.out.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": "0.1.0", "w_cls": res["w_star"].cpu(), "b_cls": res["b_star"].cpu()}, out_p)


if __name__ == "__main__":
    main()
