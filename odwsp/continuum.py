"""
Bidirectional Latency-Budget-Aware Weight-Space Programming Engine (`continuum.py`).

Supports the entire zero-gradient deployment continuum across all 17 pretrained detectors:
  - Negative budgets (`beta in [-100%, -10%]`): Closed-form `MM-FW-Ridge` contracted subgraphs (`0` external NMS).
  - Zero-overhead (`beta = 0%`, `1.00x`): Collapsed `IR-ODWSP` readout (`0%` extra params, `0%` extra FLOPs).
  - Positive budgets (`beta in [+10%, +100%]`): Static-Graph (`sg`), Dynamic (`dyn`), and Unified (`auto`) WSP graphs.
  - Unconstrained ceiling (`beta = full` / `plus`): Multi-subspace consensus (`IR-ODWSP+`).
  - Live target-GPU solving (`solve_budget_on_target_gpu`): Measures `T_0` and `T(g)` on your GPU and selects `g*(beta)`.

All checkpoint and archive loaders strictly enforce `torch.load(..., weights_only=True)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contraction import fuse_conv_and_bn_tensors
from .models import (
    COCO80_NAMES,
    COCO80_TO_91,
    COCO91_TO_NAME,
    PAPER_MODELS,
    DetectorSpec,
    LoadedDetector,
    compute_sha256,
    instantiate_subspace_model,
    load_odwsp_detector,
    make_yolo26_arch_cfg,
    normalize_model_key,
    safe_load_checkpoint,
)
from .routing import benchmark_cuda_callable, capture_static_cudagraph
from .zero_grad_guard import enforce_no_gradients

_CONTINUUM_JSON = Path(__file__).resolve().parent / "continuum_table.json"
_TABLE_CACHE: dict[str, Any] | None = None


def get_continuum_table() -> dict[str, Any]:
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        _TABLE_CACHE = json.loads(_CONTINUUM_JSON.read_text(encoding="utf-8"))
    return _TABLE_CACHE


def normalize_budget_cli_argv(argv: list[str] | None = None) -> list[str]:
    """Rewrites `['--budget', '-20%']` to `['--budget=-20%']` so Python `argparse` accepts negative percentages cleanly."""
    import sys

    src = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(src):
        if src[i] == "--budget" and (i + 1) < len(src) and src[i + 1].startswith("-"):
            out.append(f"--budget={src[i + 1]}")
            i += 2
        else:
            out.append(src[i])
            i += 1
    return out


def parse_budget_spec(budget: str | float | int) -> tuple[str, float | None, str]:
    """
    Parses a user budget specification (`"-50%"`, `"-10%"`, `"0%"`, `"+10%"`, `"+50%"`, `"+100%"`, `"full"`, or float)
    into `(regime, numeric_pct, canonical_tier_key)`.
    For intermediate budgets (e.g., `"-25%"` or `"+35%"`), maps conservatively to the tightest tier `<= beta`.
    """
    if isinstance(budget, (int, float)):
        raw = f"{float(budget):+.1f}%"
    else:
        raw = str(budget).strip().lower()

    if raw in ("full", "plus", "ir-odwsp+", "unconstrained", "max", "inf"):
        return ("unconstrained", None, "full")
    if raw in ("0", "0%", "+0%", "-0%", "1x", "1.0x", "1.00x"):
        return ("zero", 0.0, "0%")

    cleaned = raw.rstrip("%").strip()
    val = float(cleaned)
    if abs(val) < 1e-6:
        return ("zero", 0.0, "0%")
    if val < 0.0:
        if val <= -100.0:
            return ("negative", val, "-100%")
        snapped = int(math.floor(val / 10.0) * 10)
        snapped = max(-100, min(-10, snapped))
        return ("negative", val, f"{snapped}%")

    snapped_pos = int(math.floor(val / 10.0) * 10)
    if snapped_pos <= 0:
        return ("zero", val, "0%")
    snapped_pos = min(100, snapped_pos)
    return ("positive", val, f"+{snapped_pos}%")


def resolve_budget_config(
    model_key: str,
    budget: str | float | int = "0%",
    paradigm: str = "auto",
    batch_size: int = 1,
    cudagraph: bool = True,
) -> dict[str, Any]:
    """Resolves the pre-solved candidate configuration and reference metrics for `(model_key, budget, paradigm)`."""
    key = normalize_model_key(model_key)
    par = paradigm.strip().lower()
    if par not in ("auto", "dyn", "sg"):
        raise ValueError(f"Unknown paradigm '{paradigm}'. Choose from: auto, dyn, sg.")
    tbl = get_continuum_table()["models"][key]
    regime, num_pct, tier_key = parse_budget_spec(budget)

    if regime == "zero":
        z = dict(tbl["zero_budget"])
        return {
            "model_key": key,
            "regime": "zero",
            "requested_budget": str(budget),
            "resolved_tier": "0%",
            "paradigm": par,
            "cand_id": z["cand_id"],
            "assets_v0_1_0": tbl["assets_v0_1_0"],
            "paper_reference": z,
        }

    if regime == "unconstrained":
        u = dict(tbl["unconstrained"])
        return {
            "model_key": key,
            "regime": "unconstrained",
            "requested_budget": str(budget),
            "resolved_tier": "full",
            "paradigm": par,
            "cand_id": u["cand_id"],
            "assets_v0_1_0": tbl["assets_v0_1_0"],
            "paper_reference": u,
        }

    if regime == "negative":
        lat_mode = f"{'cg' if cudagraph else 'pt'}_b{batch_size}"
        if lat_mode not in tbl["negative"]:
            lat_mode = "cg_b1"
        n_cfg = dict(tbl["negative"][lat_mode][tier_key])
        return {
            "model_key": key,
            "regime": "negative",
            "requested_budget": str(budget),
            "numeric_budget_pct": num_pct,
            "resolved_tier": tier_key,
            "latency_mode": lat_mode,
            "paradigm": "contraction",
            "cand_id": n_cfg["cand_id"],
            "assets_v0_1_0": tbl["assets_v0_1_0"],
            "paper_reference": n_cfg,
        }

    lat_mode_p = "pt_b1" if not cudagraph else (f"cg_b{batch_size}" if f"cg_b{batch_size}" in tbl["positive"][par] else "cg_b1")
    p_cfg = dict(tbl["positive"][par][lat_mode_p][tier_key])
    return {
        "model_key": key,
        "regime": "positive",
        "requested_budget": str(budget),
        "numeric_budget_pct": num_pct,
        "resolved_tier": tier_key,
        "latency_mode": lat_mode_p,
        "paradigm": par,
        "cand_id": p_cfg["cand_id"],
        "assets_v0_1_0": tbl["assets_v0_1_0"],
        "paper_reference": p_cfg,
    }


def _safe_load_from_path_or_tar(
    archive_or_pt: Path,
    member_suffix: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Loads a `.pt` tensor dictionary (`weights_only=True`) either directly from disk or from a `.tar.gz` release archive."""
    resolved = Path(archive_or_pt).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint or archive not found: {resolved}")
    if expected_sha256 is not None:
        actual = compute_sha256(resolved)
        if actual.lower() != expected_sha256.strip().lower():
            raise ValueError(f"SHA-256 mismatch for {resolved.name}: expected {expected_sha256}, got {actual}")

    if resolved.is_dir():
        matches = list(resolved.rglob(f"*{member_suffix}"))
        if not matches:
            raise FileNotFoundError(f"No file matching '*{member_suffix}' inside directory {resolved}")
        return safe_load_checkpoint(matches[0], map_location="cpu")

    if resolved.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if resolved.name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(resolved, mode) as tf:
            for m in tf.getmembers():
                if m.isfile() and (m.name == member_suffix or m.name.endswith("/" + member_suffix) or m.name.endswith(member_suffix)):
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        break
                    raw = fobj.read()
                    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    if not isinstance(obj, dict):
                        raise TypeError(f"Expected dict in {m.name}, got {type(obj).__name__}")
                    return obj
        raise FileNotFoundError(f"Member '{member_suffix}' not found inside archive {resolved.name}")

    return safe_load_checkpoint(resolved, map_location="cpu")


# ==============================================================================
# SHARED TENSOR / GEOMETRY HELPERS (`0` GRADIENTS, `0` EXTERNAL NMS)
# ==============================================================================
def _trt_safe_inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    xc = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(xc / (1.0 - xc))


def trt_safe_iou_1to1_cxcywh(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    x1_1, y1_1 = b1[..., 0] - 0.5 * b1[..., 2], b1[..., 1] - 0.5 * b1[..., 3]
    x2_1, y2_1 = b1[..., 0] + 0.5 * b1[..., 2], b1[..., 1] + 0.5 * b1[..., 3]
    x1_2, y1_2 = b2[..., 0] - 0.5 * b2[..., 2], b2[..., 1] - 0.5 * b2[..., 3]
    x2_2, y2_2 = b2[..., 0] + 0.5 * b2[..., 2], b2[..., 1] + 0.5 * b2[..., 3]
    inter = (torch.minimum(x2_1, x2_2) - torch.maximum(x1_1, x1_2)).clamp(min=0.0) * (
        torch.minimum(y2_1, y2_2) - torch.maximum(y1_1, y1_2)
    ).clamp(min=0.0)
    a1 = (b1[..., 2].clamp(min=0.0) * b1[..., 3].clamp(min=0.0)).clamp(min=1e-5)
    a2 = (b2[..., 2].clamp(min=0.0) * b2[..., 3].clamp(min=0.0)).clamp(min=1e-5)
    return inter / (a1 + a2 - inter + 1e-5)


def trt_safe_iou_pairwise_cxcywh(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    b1e = b1.unsqueeze(2)
    b2e = b2.unsqueeze(1)
    x1_1, y1_1 = b1e[..., 0] - 0.5 * b1e[..., 2], b1e[..., 1] - 0.5 * b1e[..., 3]
    x2_1, y2_1 = b1e[..., 0] + 0.5 * b1e[..., 2], b1e[..., 1] + 0.5 * b1e[..., 3]
    x1_2, y1_2 = b2e[..., 0] - 0.5 * b2e[..., 2], b2e[..., 1] - 0.5 * b2e[..., 3]
    x2_2, y2_2 = b2e[..., 0] + 0.5 * b2e[..., 2], b2e[..., 1] + 0.5 * b2e[..., 3]
    inter = (torch.minimum(x2_1, x2_2) - torch.maximum(x1_1, x1_2)).clamp(min=0.0) * (
        torch.minimum(y2_1, y2_2) - torch.maximum(y1_1, y1_2)
    ).clamp(min=0.0)
    a1 = (b1e[..., 2].clamp(min=0.0) * b1e[..., 3].clamp(min=0.0)).clamp(min=1e-5)
    a2 = (b2e[..., 2].clamp(min=0.0) * b2e[..., 3].clamp(min=0.0)).clamp(min=1e-5)
    return inter / (a1 + a2 - inter + 1e-5)


def detr_topk300_preds6(probs: torch.Tensor, boxes_cxcywh: torch.Tensor, nc: int) -> torch.Tensor:
    B = int(probs.shape[0])
    scores, index = torch.topk(probs.reshape(B, -1), 300, dim=-1)
    q_idx = index // nc
    labels_f = (index - q_idx * nc).to(probs.dtype)
    sel = torch.gather(boxes_cxcywh, 1, q_idx.unsqueeze(-1).expand(-1, -1, 4))
    cx, cy, bw, bh = sel[..., 0], sel[..., 1], sel[..., 2], sel[..., 3]
    boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=-1)
    return torch.cat([boxes_xyxy, scores.unsqueeze(-1), labels_f.unsqueeze(-1)], dim=-1)


def yolo_multilabel_topk300(boxes_xyxy: torch.Tensor, probs_nc: torch.Tensor, max_det: int = 300) -> torch.Tensor:
    B, N_a, nc = [int(d) for d in probs_nc.shape]
    k_anc = min(N_a, max_det)
    _, ori_idx = torch.topk(probs_nc.max(dim=-1).values, k_anc, dim=-1)
    probs_sub = torch.gather(probs_nc, 1, ori_idx.unsqueeze(-1).expand(-1, -1, nc))
    scores, flat_idx = torch.topk(probs_sub.reshape(B, k_anc * nc), max_det, dim=-1)
    cls_f = (flat_idx % nc).to(probs_nc.dtype)
    anc_sel = torch.gather(ori_idx, 1, flat_idx // nc)
    bx_sel = torch.gather(boxes_xyxy, 1, anc_sel.unsqueeze(-1).expand(-1, -1, 4))
    return torch.cat([bx_sel, scores.unsqueeze(-1), cls_f.unsqueeze(-1)], dim=-1)


def fuse_yolo_conv_bn_inplace(root: nn.Module) -> nn.Module:
    for sub in root.modules():
        if hasattr(sub, "conv") and hasattr(sub, "bn") and isinstance(sub.conv, nn.Conv2d) and isinstance(sub.bn, nn.BatchNorm2d):
            wf, bf = fuse_conv_and_bn_tensors(sub.conv, sub.bn)
            new_c = nn.Conv2d(
                sub.conv.in_channels,
                sub.conv.out_channels,
                kernel_size=sub.conv.kernel_size,
                stride=sub.conv.stride,
                padding=sub.conv.padding,
                dilation=sub.conv.dilation,
                groups=sub.conv.groups,
                bias=True,
            ).to(device=wf.device, dtype=sub.conv.weight.dtype)
            new_c.weight.data.copy_(wf.to(new_c.weight.dtype))
            new_c.bias.data.copy_(bf.to(new_c.bias.dtype))
            sub.conv = new_c
            sub.bn = nn.Identity()
            if hasattr(sub, "forward_fuse"):
                sub.forward = sub.forward_fuse
    return root


class NullZeroBudgetModule(nn.Module):
    """Exact `-100%` null-budget endpoint (`0` compute, empty `[B, 300, 6]` tensor)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((int(x.shape[0]), 300, 6), device=x.device, dtype=x.dtype)


# ==============================================================================
# 1. FAMILY A: YOLO26 NEGATIVE & POSITIVE WSP GRAPH MODULES
# ==============================================================================
YOLO_NEG_SPECS: dict[str, dict[str, Any]] = {
    "n00": {"arch": "d0",       "res": 640, "wsp": True},
    "n01": {"arch": "o2o_full", "res": 640, "wsp": True},
    "n02": {"arch": "o2o_full", "res": 576, "wsp": True},
    "n03": {"arch": "o2o_full", "res": 512, "wsp": True},
    "n04": {"arch": "o2o_full", "res": 448, "wsp": True},
    "n05": {"arch": "o2o_full", "res": 384, "wsp": True},
    "n06": {"arch": "o2o_full", "res": 320, "wsp": True},
    "n07": {"arch": "o2o_full", "res": 256, "wsp": True},
    "n08": {"arch": "o2o_full", "res": 192, "wsp": True},
    "n09": {"arch": "pan_byp",  "res": 448, "wsp": True},
    "n10": {"arch": "pan_byp",  "res": 384, "wsp": True},
    "n11": {"arch": "pan_byp",  "res": 320, "wsp": True},
    "n12": {"arch": "pan_byp",  "res": 256, "wsp": True},
    "n13": {"arch": "bb6_s16",  "res": 352, "wsp": True},
    "n14": {"arch": "bb6_s16",  "res": 256, "wsp": True},
    "n15": {"arch": "bb6_s16",  "res": 192, "wsp": True},
    "n16": {"arch": "stem5",    "res": 192, "wsp": True},
    "n17": {"arch": "stem3",    "res": 128, "wsp": True},
    "n18": {"arch": "o2o_full", "res": 160, "wsp": True},
    "n19": {"arch": "pan_byp",  "res": 224, "wsp": True},
    "n20": {"arch": "pan_byp",  "res": 192, "wsp": True},
    "n21": {"arch": "pan_byp",  "res": 160, "wsp": True},
}

YOLO_DYN_TO_SG_EQUIV = {
    "dyn_y39_fov352_gw": "y39_fov352_gw",
    "dyn_y39_fl384_gw": "y39_fl384_gw",
    "dyn_y39_fl448_gw": "y39_fl448_gw",
    "dyn_y39_fl512_gw": "y39_fl512_gw",
}


class YOLONegGraphModule(nn.Module):
    """Negative-Budget `MM-FW-Ridge` Contracted Graph Module for `YOLO26` (`[B, 300, 6]` output)."""

    def __init__(self, basins: dict[str, Any], bridges: dict[str, Any], cid: str, eval_resize: bool = True) -> None:
        super().__init__()
        enforce_no_gradients()
        self.cid = cid
        sp = YOLO_NEG_SPECS[cid]
        self.arch = sp["arch"]
        self.res = int(sp["res"])
        self.wsp = bool(sp["wsp"])
        self.eval_resize = eval_resize

        if self.arch == "d0":
            m0 = copy.deepcopy(basins["m_d0"]).eval()
            m0.end2end = True
            m0.model[-1].end2end = True
            self.m_d0 = m0
            return

        src_m = copy.deepcopy(basins["m_d0"]).eval()
        fuse_yolo_conv_bn_inplace(src_m)
        det_src = src_m.model[23]

        if self.arch == "o2o_full":
            self.blocks = nn.ModuleList([src_m.model[i] for i in range(23)])
            self.o2o_cv2 = det_src.one2one_cv2
            self.o2o_cv3 = det_src.one2one_cv3
            if self.wsp and self.res <= 448 and "o2o_cal_cv3" in bridges:
                alpha = 0.22 * min(1.0, (640.0 - float(self.res)) / 320.0)
                for l_idx in range(3):
                    w_cal, b_cal = bridges["o2o_cal_cv3"][l_idx]
                    w0 = self.o2o_cv3[l_idx][2].weight.data.float().cpu()
                    b0 = self.o2o_cv3[l_idx][2].bias.data.float().cpu()
                    self.o2o_cv3[l_idx][2].weight.data.copy_((1.0 - alpha) * w0 + alpha * w_cal)
                    self.o2o_cv3[l_idx][2].bias.data.copy_((1.0 - alpha) * b0 + alpha * b_cal)

        elif self.arch == "pan_byp":
            self.blocks_0_9 = nn.ModuleList([src_m.model[i] for i in range(10)])
            c9, c10 = int(bridges["W_psa"].shape[1]), int(bridges["W_psa"].shape[0])
            self.psa_fold = nn.Conv2d(c9, c10, 1, bias=True)
            self.up11 = src_m.model[11]
            self.c13 = src_m.model[13]
            self.up14 = src_m.model[14]
            self.c16 = src_m.model[16]
            c13_d, c19_d = int(bridges["W_pan19"].shape[1]), int(bridges["W_pan19"].shape[0])
            c22_d = int(bridges["W_pan22"].shape[0])
            self.pan19_br = nn.Conv2d(c13_d, c19_d, 1, bias=True)
            self.pan22_br = nn.Conv2d(c10, c22_d, 1, bias=True)
            self.o2o_cv2 = det_src.one2one_cv2
            self.o2o_cv3 = det_src.one2one_cv3
            self.psa_fold.weight.data.copy_(bridges["W_psa"].view(c10, c9, 1, 1))
            self.psa_fold.bias.data.copy_(bridges["b_psa"])
            self.pan19_br.weight.data.copy_(bridges["W_pan19"].view(c19_d, c13_d, 1, 1))
            self.pan19_br.bias.data.copy_(bridges["b_pan19"])
            self.pan22_br.weight.data.copy_(bridges["W_pan22"].view(c22_d, c10, 1, 1))
            self.pan22_br.bias.data.copy_(bridges["b_pan22"])
            for l_idx in (1, 2):
                wb, bb = bridges["pan_cal_cv2"][l_idx]
                wc, bc = bridges["pan_cal_cv3"][l_idx]
                self.o2o_cv2[l_idx][2].weight.data.copy_(0.60 * self.o2o_cv2[l_idx][2].weight.data.float().cpu() + 0.40 * wb)
                self.o2o_cv2[l_idx][2].bias.data.copy_(0.60 * self.o2o_cv2[l_idx][2].bias.data.float().cpu() + 0.40 * bb)
                self.o2o_cv3[l_idx][2].weight.data.copy_(0.60 * self.o2o_cv3[l_idx][2].weight.data.float().cpu() + 0.40 * wc)
                self.o2o_cv3[l_idx][2].bias.data.copy_(0.60 * self.o2o_cv3[l_idx][2].bias.data.float().cpu() + 0.40 * bc)

        elif self.arch == "bb6_s16":
            self.blocks_0_6 = nn.ModuleList([src_m.model[i] for i in range(7)])
            c6_d, c19_d = int(bridges["W_bb6_19"].shape[1]), int(bridges["W_bb6_19"].shape[0])
            self.bb6_br = nn.Conv2d(c6_d, c19_d, 1, bias=True)
            self.cv2_s16 = det_src.one2one_cv2[1]
            self.cv3_s16 = det_src.one2one_cv3[1]
            self.bb6_br.weight.data.copy_(bridges["W_bb6_19"].view(c19_d, c6_d, 1, 1))
            self.bb6_br.bias.data.copy_(bridges["b_bb6_19"])
            wb, bb = bridges["bb6_cal_cv2"]
            wc, bc = bridges["bb6_cal_cv3"]
            self.cv2_s16[2].weight.data.copy_(wb); self.cv2_s16[2].bias.data.copy_(bb)
            self.cv3_s16[2].weight.data.copy_(wc); self.cv3_s16[2].bias.data.copy_(bc)

        elif self.arch == "stem5":
            self.c0 = src_m.model[0]
            self.c1 = src_m.model[1]
            w3, b3 = bridges["w3_fold"], bridges["b3_orig"]
            w5, b5 = bridges["w5_fold"], bridges["b5_orig"]
            ws5, bs5 = bridges["W_s5_19"], bridges["b_s5_19"]
            self.c3_fold = nn.Conv2d(w3.shape[1], w3.shape[0], 3, stride=2, padding=1, bias=True)
            self.c3_fold.weight.data.copy_(w3); self.c3_fold.bias.data.copy_(b3)
            self.c5_fold = nn.Conv2d(w5.shape[1], w5.shape[0], 3, stride=2, padding=1, bias=True)
            self.c5_fold.weight.data.copy_(w5); self.c5_fold.bias.data.copy_(b5)
            self.s5_br = nn.Conv2d(ws5.shape[1], ws5.shape[0], 1, bias=True)
            self.s5_br.weight.data.copy_(ws5.view(ws5.shape[0], ws5.shape[1], 1, 1)); self.s5_br.bias.data.copy_(bs5)
            self.cv2_s16 = det_src.one2one_cv2[1]
            self.cv3_s16 = det_src.one2one_cv3[1]
            wb, bb = bridges["s5_cal_cv2"]
            wc, bc = bridges["s5_cal_cv3"]
            self.cv2_s16[2].weight.data.copy_(wb); self.cv2_s16[2].bias.data.copy_(bb)
            self.cv3_s16[2].weight.data.copy_(wc); self.cv3_s16[2].bias.data.copy_(bc)

        elif self.arch == "stem3":
            wc0, bc0 = bridges["w_stem3_c0"], bridges["b_s4_0"]
            wc1, bc1 = bridges["w_stem3_c1"], bridges["b_s3_1"]
            ws5, bs5 = bridges["W_s5_19"], bridges["b_s5_19"]
            self.s0 = nn.Conv2d(3, wc0.shape[0], 4, stride=4, padding=0, bias=True)
            self.s0.weight.data.copy_(wc0); self.s0.bias.data.copy_(bc0)
            self.s1 = nn.Conv2d(wc1.shape[1], wc1.shape[0], 4, stride=4, padding=0, bias=True)
            self.s1.weight.data.copy_(wc1); self.s1.bias.data.copy_(bc1)
            self.s3_br = nn.Conv2d(ws5.shape[1], ws5.shape[0], 1, bias=True)
            self.s3_br.weight.data.copy_(ws5.view(ws5.shape[0], ws5.shape[1], 1, 1)); self.s3_br.bias.data.copy_(bs5)
            self.cv2_s16 = det_src.one2one_cv2[1]
            self.cv3_s16 = det_src.one2one_cv3[1]
            wb, bb = bridges["s3_cal_cv2"]
            wc, bc = bridges["s3_cal_cv3"]
            self.cv2_s16[2].weight.data.copy_(wb); self.cv2_s16[2].bias.data.copy_(bb)
            self.cv3_s16[2].weight.data.copy_(wc); self.cv3_s16[2].bias.data.copy_(bc)

    def _decode_levels(self, boxes_list: list[torch.Tensor], logits_list: list[torch.Tensor], strides: tuple[float, ...], sx_back: float, sy_back: float) -> torch.Tensor:
        B = int(boxes_list[0].shape[0])
        bx_all, pr_all = [], []
        gcache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = getattr(self, "_neg_grid_cache", {})
        for b_lvl, z_lvl, st in zip(boxes_list, logits_list, strides):
            _, _, h, w = [int(d) for d in b_lvl.shape]
            gkey = (h, w, b_lvl.device, b_lvl.dtype)
            if gkey not in gcache:
                gy = (torch.arange(h, device=b_lvl.device, dtype=b_lvl.dtype) + 0.5).view(1, h, 1).expand(1, h, w)
                gx = (torch.arange(w, device=b_lvl.device, dtype=b_lvl.dtype) + 0.5).view(1, 1, w).expand(1, h, w)
                gcache[gkey] = (gx.reshape(1, -1), gy.reshape(1, -1))
                setattr(self, "_neg_grid_cache", gcache)
            ax, ay = gcache[gkey]
            bf = b_lvl.reshape(B, 4, h * w)
            x1 = (ax - bf[:, 0]) * (st * sx_back)
            y1 = (ay - bf[:, 1]) * (st * sy_back)
            x2 = (ax + bf[:, 2]) * (st * sx_back)
            y2 = (ay + bf[:, 3]) * (st * sy_back)
            bx_all.append(torch.stack([x1, y1, x2, y2], dim=-1))
            pr_all.append(z_lvl.reshape(B, 80, h * w).transpose(1, 2).sigmoid())
        boxes_cat = bx_all[0] if len(bx_all) == 1 else torch.cat(bx_all, dim=1)
        probs_cat = pr_all[0] if len(pr_all) == 1 else torch.cat(pr_all, dim=1)
        return yolo_multilabel_topk300(boxes_cat, probs_cat, 300)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "d0":
            self.m_d0.end2end = True
            self.m_d0.model[-1].end2end = True
            return self.m_d0(x)[0]

        H_in, W_in = int(x.shape[2]), int(x.shape[3])
        if self.eval_resize and (H_in != self.res or W_in != self.res):
            scale = float(self.res) / float(max(H_in, W_in))
            H_r = max(32, int(round(H_in * scale / 32.0)) * 32)
            W_r = max(32, int(round(W_in * scale / 32.0)) * 32)
            xr = F.interpolate(x, size=(H_r, W_r), mode="bilinear", align_corners=False)
            sx_back, sy_back = float(W_in) / float(W_r), float(H_in) / float(H_r)
        else:
            xr = x
            sx_back = sy_back = 1.0

        if self.arch == "o2o_full":
            saved: dict[int, torch.Tensor] = {}
            cur = xr
            for i, blk in enumerate(self.blocks):
                if i >= 11 and blk.f != -1:
                    cur = [cur if j == -1 else saved[j] for j in blk.f]
                cur = blk(cur)
                if i in (4, 6, 10, 13, 16, 19, 22):
                    saved[i] = cur
            f3 = [saved[16], saved[19], saved[22]]
            return self._decode_levels([self.o2o_cv2[l](f3[l]) for l in range(3)], [self.o2o_cv3[l](f3[l]) for l in range(3)], (8.0, 16.0, 32.0), sx_back, sy_back)

        if self.arch == "pan_byp":
            cur = xr
            f4 = f6 = cur
            for i, blk in enumerate(self.blocks_0_9):
                cur = blk(cur)
                if i == 4:
                    f4 = cur
                elif i == 6:
                    f6 = cur
            f10_h = self.psa_fold(cur)
            f13 = self.c13(torch.cat([self.up11(f10_h), f6], dim=1))
            f16 = self.c16(torch.cat([self.up14(f13), f4], dim=1))
            f19_h = self.pan19_br(torch.cat([F.avg_pool2d(f16, 2, 2), f13], dim=1))
            f22_h = self.pan22_br(f10_h)
            f3 = [f16, f19_h, f22_h]
            return self._decode_levels([self.o2o_cv2[l](f3[l]) for l in range(3)], [self.o2o_cv3[l](f3[l]) for l in range(3)], (8.0, 16.0, 32.0), sx_back, sy_back)

        if self.arch == "bb6_s16":
            cur = xr
            for blk in self.blocks_0_6:
                cur = blk(cur)
            f19_h = self.bb6_br(cur)
            return self._decode_levels([self.cv2_s16(f19_h)], [self.cv3_s16(f19_h)], (16.0,), sx_back, sy_back)

        if self.arch == "stem5":
            f19_h = self.s5_br(F.silu(self.c5_fold(F.silu(self.c3_fold(self.c1(self.c0(xr)))))))
            return self._decode_levels([self.cv2_s16(f19_h)], [self.cv3_s16(f19_h)], (16.0,), sx_back, sy_back)

        f19_h = self.s3_br(F.silu(self.s1(F.silu(self.s0(xr)))))
        return self._decode_levels([self.cv2_s16(f19_h)], [self.cv3_s16(f19_h)], (16.0,), sx_back, sy_back)


def _fuse_yolo_cb_single(m_cb: nn.Module) -> nn.Conv2d:
    c = copy.deepcopy(m_cb).eval()
    if hasattr(c, "conv") and hasattr(c, "bn") and isinstance(c.bn, nn.BatchNorm2d):
        wf, bf = fuse_conv_and_bn_tensors(c.conv, c.bn)
        nc = nn.Conv2d(c.conv.in_channels, c.conv.out_channels, c.conv.kernel_size, c.conv.stride, c.conv.padding, c.conv.dilation, c.conv.groups, bias=True)
        nc.weight.data.copy_(wf)
        nc.bias.data.copy_(bf)
        return nc.to(c.conv.weight.device)
    return c.conv if hasattr(c, "conv") else c  # type: ignore[return-value]


class FusedStiefelDetectHeads(nn.Module):
    """Horizontally fuses `K` `(cv2, cv3)` Detect branches into a single Conv2d stack per feature scale."""

    def __init__(self, specs: list[tuple[tuple[nn.ModuleList, nn.ModuleList], str, float]]) -> None:
        super().__init__()
        K = len(specs)
        self.K = K
        o_idx = [j for j, s in enumerate(specs) if s[1] == "o2o"]
        m_idx = [j for j, s in enumerate(specs) if s[1] == "o2m"]
        self.has_m = len(m_idx) > 0

        wo = [specs[j][2] for j in o_idx]
        so = sum(wo)
        self.wo = [w / so for w in wo]
        wm = [specs[j][2] for j in m_idx] if self.has_m else []
        sm = sum(wm) if self.has_m else 1.0
        self.wm = [w / sm for w in wm]

        b_tups = [s[0] for s in specs]
        self.no_box = int(b_tups[0][0][0][2].out_channels)
        self.nc = int(b_tups[0][1][0][2].out_channels)
        self.b_l0, self.b_l1, self.b_l2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.c_dw0, self.c_pw0, self.c_dw1, self.c_pw1, self.c_l2 = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        )
        self.act = nn.SiLU()
        dev_w = b_tups[0][0][0][2].weight.device

        def _cat_b(fs: list[nn.Conv2d]) -> torch.Tensor:
            return torch.cat([f.bias if f.bias is not None else torch.zeros(f.out_channels, device=f.weight.device, dtype=f.weight.dtype) for f in fs], dim=0)

        with torch.no_grad():
            for i in range(3):
                f0s = [_fuse_yolo_cb_single(bt[0][i][0]) for bt in b_tups]
                f1s = [_fuse_yolo_cb_single(bt[0][i][1]) for bt in b_tups]
                f2s = [_fuse_yolo_cb_single(bt[0][i][2]) for bt in b_tups]
                cin, c2 = f0s[0].in_channels, f0s[0].out_channels
                l0 = nn.Conv2d(cin, K * c2, 3, padding=1, bias=True).to(dev_w)
                l0.weight.copy_(torch.cat([f.weight for f in f0s], dim=0)); l0.bias.copy_(_cat_b(f0s))
                l1 = nn.Conv2d(K * c2, K * c2, 3, padding=1, groups=K, bias=True).to(dev_w)
                l1.weight.copy_(torch.cat([f.weight for f in f1s], dim=0)); l1.bias.copy_(_cat_b(f1s))
                l2 = nn.Conv2d(K * c2, K * self.no_box, 1, groups=K, bias=True).to(dev_w)
                l2.weight.copy_(torch.cat([f.weight for f in f2s], dim=0)); l2.bias.copy_(_cat_b(f2s))
                self.b_l0.append(l0); self.b_l1.append(l1); self.b_l2.append(l2)

                dw0s = [_fuse_yolo_cb_single(bt[1][i][0][0]) for bt in b_tups]
                pw0s = [_fuse_yolo_cb_single(bt[1][i][0][1]) for bt in b_tups]
                dw1s = [_fuse_yolo_cb_single(bt[1][i][1][0]) for bt in b_tups]
                pw1s = [_fuse_yolo_cb_single(bt[1][i][1][1]) for bt in b_tups]
                cl2s = [_fuse_yolo_cb_single(bt[1][i][2]) for bt in b_tups]
                c3 = pw0s[0].out_channels
                dw0 = nn.Conv2d(K * cin, K * cin, 3, padding=1, groups=K * cin, bias=True).to(dev_w)
                dw0.weight.copy_(torch.cat([f.weight for f in dw0s], dim=0)); dw0.bias.copy_(_cat_b(dw0s))
                pw0 = nn.Conv2d(K * cin, K * c3, 1, groups=K, bias=True).to(dev_w)
                pw0.weight.copy_(torch.cat([f.weight for f in pw0s], dim=0)); pw0.bias.copy_(_cat_b(pw0s))
                dw1 = nn.Conv2d(K * c3, K * c3, 3, padding=1, groups=K * c3, bias=True).to(dev_w)
                dw1.weight.copy_(torch.cat([f.weight for f in dw1s], dim=0)); dw1.bias.copy_(_cat_b(dw1s))
                pw1 = nn.Conv2d(K * c3, K * c3, 1, groups=K, bias=True).to(dev_w)
                pw1.weight.copy_(torch.cat([f.weight for f in pw1s], dim=0)); pw1.bias.copy_(_cat_b(pw1s))
                cl2 = nn.Conv2d(K * c3, K * self.nc, 1, groups=K, bias=True).to(dev_w)
                cl2.weight.copy_(torch.cat([f.weight for f in cl2s], dim=0)); cl2.bias.copy_(_cat_b(cl2s))
                self.c_dw0.append(dw0); self.c_pw0.append(pw0)
                self.c_dw1.append(dw1); self.c_pw1.append(pw1); self.c_l2.append(cl2)

        mat_o = torch.zeros((K,), dtype=torch.float32)
        for j, w in zip(o_idx, self.wo):
            mat_o[j] = w
        self.register_buffer("w_o", mat_o.view(1, K, 1, 1), persistent=False)
        if self.has_m:
            mat_m = torch.zeros((K,), dtype=torch.float32)
            for j, w in zip(m_idx, self.wm):
                mat_m[j] = w
            self.register_buffer("w_m", mat_m.view(1, K, 1, 1), persistent=False)

    def forward(self, nf_levels: list[torch.Tensor]):
        B = int(nf_levels[0].shape[0])
        bo_l, bm_l, so_l, sm_l = [], [], [], []
        wo = self.w_o.to(nf_levels[0].dtype)
        wm = self.w_m.to(nf_levels[0].dtype) if self.has_m else None
        for i in range(3):
            fi = nf_levels[i]
            hb = self.act(self.b_l1[i](self.act(self.b_l0[i](fi))))
            bx_k = self.b_l2[i](hb).reshape(B, self.K, self.no_box, -1)
            bo_l.append((bx_k * wo).sum(dim=1))
            if wm is not None:
                bm_l.append((bx_k * wm).sum(dim=1))
            fi_k = fi.repeat(1, self.K, 1, 1)
            hc = self.act(self.c_pw1[i](self.act(self.c_dw1[i](self.act(self.c_pw0[i](self.act(self.c_dw0[i](fi_k))))))))
            sc_k = self.c_l2[i](hc).reshape(B, self.K, self.nc, -1).sigmoid()
            so_l.append((sc_k * wo).sum(dim=1))
            if wm is not None:
                sm_l.append((sc_k * wm).sum(dim=1))
        bo, so = torch.cat(bo_l, dim=-1), torch.cat(so_l, dim=-1)
        if not self.has_m:
            return bo, so, None, None
        return bo, so, torch.cat(bm_l, dim=-1), torch.cat(sm_l, dim=-1)


class FusedStiefelRawHeads(nn.Module):
    def __init__(self, specs: list[tuple[tuple[nn.ModuleList, nn.ModuleList], str, float]], raw_logits: bool = True) -> None:
        super().__init__()
        self.base = FusedStiefelDetectHeads(specs)
        self.raw_logits = bool(raw_logits and len(specs) == 2)

    def forward(self, nf3: list[torch.Tensor]):
        if not self.raw_logits:
            return self.base(nf3)
        b = self.base
        B = int(nf3[0].shape[0])
        bo_l, bm_l, zo_l, zm_l = [], [], [], []
        for i in range(3):
            fi = nf3[i]
            hb = b.act(b.b_l1[i](b.act(b.b_l0[i](fi))))
            bx_k = b.b_l2[i](hb).reshape(B, 2, b.no_box, -1)
            bo_l.append(bx_k[:, 0]); bm_l.append(bx_k[:, 1])
            fi_k = fi.repeat(1, 2, 1, 1)
            hc = b.act(b.c_pw1[i](b.act(b.c_dw1[i](b.act(b.c_pw0[i](b.act(b.c_dw0[i](fi_k))))))))
            zk = b.c_l2[i](hc).reshape(B, 2, b.nc, -1)
            zo_l.append(zk[:, 0]); zm_l.append(zk[:, 1])
        return torch.cat(bo_l, dim=-1), torch.cat(zo_l, dim=-1), torch.cat(bm_l, dim=-1), torch.cat(zm_l, dim=-1)


def _cg_safe_yolo_decode_boxes(det: nn.Module, boxes: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
    """Bit-identical to `Detect._get_decode_boxes` while caching per-shape `(anchors, strides)` for CUDAGraph safety."""
    cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = getattr(det, "_cg_anchor_cache", {})
    key = (tuple(int(f.shape[2]) for f in feats), tuple(int(f.shape[3]) for f in feats), boxes.device, boxes.dtype)
    if key not in cache:
        ax_l, ay_l, st_l = [], [], []
        strides_vec = det.stride.detach().float().tolist() if hasattr(det, "stride") else [8.0, 16.0, 32.0]
        for f, st in zip(feats, strides_vec):
            h, w = int(f.shape[2]), int(f.shape[3])
            sy = torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5
            sx = torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            ax_l.append(gx.reshape(-1))
            ay_l.append(gy.reshape(-1))
            st_l.append(torch.full((h * w,), float(st), device=boxes.device, dtype=boxes.dtype))
        anc = torch.stack([torch.cat(ax_l, dim=0), torch.cat(ay_l, dim=0)], dim=0).unsqueeze(0)
        stv = torch.cat(st_l, dim=0).view(1, 1, -1)
        cache[key] = (anc, stv)
        setattr(det, "_cg_anchor_cache", cache)
    anc_u, st_u = cache[key]
    b_dfl = det.dfl(boxes)
    lt, rb = b_dfl.chunk(2, 1)
    return torch.cat((anc_u - lt, anc_u + rb), dim=1) * st_u


def _yolo_bb_neck_feats(mod: nn.Module, x: torch.Tensor) -> dict[int, torch.Tensor]:
    saved: dict[int, torch.Tensor] = {}
    cur = x
    for i in range(23):
        blk = mod.model[i]
        if i >= 11 and blk.f != -1:
            cur = [cur if j == -1 else saved[j] for j in blk.f]
        cur = blk(cur)
        if i in (4, 6, 10, 13, 16, 19, 22):
            saved[i] = cur
    return saved


def _yolo_shbb_neck(m_shbb: nn.Module, f4: torch.Tensor, f6: torch.Tensor, f10: torch.Tensor) -> list[torch.Tensor]:
    s2 = {4: f4, 6: f6, 10: f10}
    c2 = f10
    for i in range(11, 23):
        b2 = m_shbb.model[i]
        if b2.f != -1:
            c2 = [c2 if j == -1 else s2[j] for j in b2.f]
        c2 = b2(c2)
        if i in (13, 16, 19, 22):
            s2[i] = c2
    return [s2[16], s2[19], s2[22]]


def _yolo_gridwarp_consensus_pt(
    dbox_pri: torch.Tensor, z_pri_o: torch.Tensor, z_pri_m: torch.Tensor,
    dbox_aux: torch.Tensor | None, z_aux_o: torch.Tensor | None, z_aux_m: torch.Tensor | None,
    H_pri: int, W_pri: int, H_aux: int, W_aux: int,
    aux_mode: int, inputs_are_prob: int,
    gamma_cls: float, w_warp: float, eta_box: float, alpha_cor: float, dup_decay: float,
    tril64: torch.Tensor,
) -> torch.Tensor:
    B = int(dbox_pri.shape[0])
    po = z_pri_o.clamp(min=1e-7) if inputs_are_prob else torch.sigmoid(z_pri_o)
    pm = z_pri_m.clamp(min=1e-7) if inputs_are_prob else torch.sigmoid(z_pri_m)

    if aux_mode != 0 and z_aux_m is not None and w_warp > 0.0:
        pma = z_aux_m.clamp(min=1e-7) if inputs_are_prob else torch.sigmoid(z_aux_m)
        warps, masks, off_a = [], [], 0
        for st in (8, 16, 32):
            ha, wa = H_aux // st, W_aux // st
            hp, wp = H_pri // st, W_pri // st
            na = ha * wa
            mp = pma[:, :, off_a : off_a + na].reshape(B, 80, ha, wa)
            if aux_mode == 2:
                mp = torch.flip(mp, dims=[-1])
                warps.append(F.interpolate(mp, size=(hp, wp), mode="bilinear", align_corners=False).reshape(B, 80, hp * wp))
                masks.append(torch.ones((1, 1, hp * wp), device=pm.device, dtype=pm.dtype))
            elif aux_mode == 1:
                warps.append((F.interpolate(mp, size=(hp, wp), mode="bilinear", align_corners=False) if (ha != hp or wa != wp) else mp).reshape(B, 80, hp * wp))
                masks.append(torch.ones((1, 1, hp * wp), device=pm.device, dtype=pm.dtype))
            else:
                frac = 0.125 if aux_mode == 3 else 0.10
                px_x = int(round(W_pri * frac / 32.0)) * 32
                py_y = int(round(H_pri * frac / 32.0)) * 32
                lp, tp = px_x // st, py_y // st
                ih, iw = hp - 2 * tp, wp - 2 * lp
                sl = F.interpolate(mp, size=(ih, iw), mode="bilinear", align_corners=False)
                warps.append(F.pad(sl, (lp, lp, tp, tp), value=0.0).reshape(B, 80, hp * wp))
                m_in = torch.ones((1, 1, max(1, ih - 2), max(1, iw - 2)), device=pm.device, dtype=pm.dtype)
                masks.append(F.pad(m_in, (lp + 1, lp + 1, tp + 1, tp + 1), value=0.0).reshape(1, 1, hp * wp))
            off_a += na
        wm_t = w_warp * torch.cat(masks, dim=2)
        pm = (pm + wm_t * torch.cat(warps, dim=2)) / (1.0 + wm_t)

    sc_p = (po.clamp(min=1e-7) ** (1.0 - gamma_cls)) * (pm.clamp(min=1e-7) ** gamma_cls)
    t2_s, t2_c = sc_p.permute(0, 2, 1).topk(2, dim=-1)
    N_pri = int(dbox_pri.shape[2])
    flat_s = t2_s.reshape(B, 2 * N_pri)
    flat_c = t2_c.reshape(B, 2 * N_pri)
    top_s, top_i = flat_s.topk(300, dim=1)
    anc_i = top_i // 2
    db_t = dbox_pri.permute(0, 2, 1)
    top_b = torch.gather(db_t, 1, anc_i.unsqueeze(-1).expand(-1, -1, 4))
    top_c = torch.gather(flat_c, 1, top_i).to(top_b.dtype)
    y_pri = torch.cat([top_b, top_s.unsqueeze(-1), top_c.unsqueeze(-1)], dim=-1)

    y_top, y_tail = y_pri[:, :64], y_pri[:, 64:]
    wb, ws, wc = y_top[:, :, :4], y_top[:, :, 4], y_top[:, :, 5].long()

    if aux_mode != 0 and dbox_aux is not None and z_aux_o is not None and z_aux_m is not None:
        poa = z_aux_o.clamp(min=1e-7) if inputs_are_prob else torch.sigmoid(z_aux_o)
        pma2 = z_aux_m.clamp(min=1e-7) if inputs_are_prob else torch.sigmoid(z_aux_m)
        sca = ((poa.clamp(min=1e-7) ** (1.0 - gamma_cls)) * (pma2.clamp(min=1e-7) ** gamma_cls)).permute(0, 2, 1)
        dba = dbox_aux.permute(0, 2, 1)
        Ws, Hs, Wp, Hp = float(W_aux), float(H_aux), float(W_pri), float(H_pri)
        if aux_mode == 1:
            sx, sy = Wp / Ws, Hp / Hs
            dba = torch.stack([dba[..., 0] * sx, dba[..., 1] * sy, dba[..., 2] * sx, dba[..., 3] * sy], dim=-1)
        elif aux_mode == 2:
            sx, sy = Wp / Ws, Hp / Hs
            dba = torch.stack([(Ws - dba[..., 2]) * sx, dba[..., 1] * sy, (Ws - dba[..., 0]) * sx, dba[..., 3] * sy], dim=-1)
        else:
            mg = 6.0
            keep = (dba[..., 0] > mg) & (dba[..., 2] < Ws - mg) & (dba[..., 1] > mg) & (dba[..., 3] < Hs - mg)
            sca = sca * keep.to(sca.dtype).unsqueeze(-1)
            frac = 0.125 if aux_mode == 3 else 0.10
            px_x = round((Wp * frac) / 32.0) * 32.0
            py_y = round((Hp * frac) / 32.0) * 32.0
            sx, sy = (Wp - 2.0 * px_x) / Ws, (Hp - 2.0 * py_y) / Hs
            dba = torch.stack([px_x + dba[..., 0] * sx, py_y + dba[..., 1] * sy, px_x + dba[..., 2] * sx, py_y + dba[..., 3] * sy], dim=-1)

        ma_s, ma_c = sca.max(dim=-1)
        k_aux = min(128, int(ma_s.shape[1]))
        ca_s, ca_i = ma_s.topk(k_aux, dim=1)
        ca_b = torch.gather(dba, 1, ca_i.unsqueeze(-1).expand(-1, -1, 4))
        ca_c = torch.gather(ma_c, 1, ca_i)

        x1 = torch.maximum(wb[:, :, None, 0], ca_b[:, None, :, 0])
        y1 = torch.maximum(wb[:, :, None, 1], ca_b[:, None, :, 1])
        x2 = torch.minimum(wb[:, :, None, 2], ca_b[:, None, :, 2])
        y2 = torch.minimum(wb[:, :, None, 3], ca_b[:, None, :, 3])
        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        aw = ((wb[:, :, 2] - wb[:, :, 0]).clamp(min=0.0) * (wb[:, :, 3] - wb[:, :, 1]).clamp(min=0.0)).clamp(min=1e-6)
        ac = ((ca_b[:, :, 2] - ca_b[:, :, 0]).clamp(min=0.0) * (ca_b[:, :, 3] - ca_b[:, :, 1]).clamp(min=0.0)).clamp(min=1e-6)
        iou = inter / (aw[:, :, None] + ac[:, None, :] - inter + 1e-6)
        v_cor = (ws[:, :, None] >= 0.002) & (ca_s[:, None, :] >= 0.006) & (wc[:, :, None] == ca_c[:, None, :]) & (iou >= 0.74)
        v_box = v_cor & (iou >= 0.78)
        bw = torch.where(v_box, (ca_s[:, None, :] ** 2) * (iou.clamp(min=1e-6) ** 3.25), torch.zeros_like(iou))
        bws = bw.sum(dim=2, keepdim=True)
        wb = torch.where(bws > 1e-4, (1.0 - eta_box) * wb + eta_box * (torch.bmm(bw, ca_b) / bws.clamp(min=1e-4)), wb)
        if alpha_cor > 0.0:
            ws = torch.where(ws >= 0.002, (ws * (1.0 + alpha_cor * (v_cor.to(ws.dtype).sum(dim=2).clamp(max=8.0) * 0.125))).clamp(max=0.999), ws)

    ox1 = torch.maximum(wb[:, :, None, 0], wb[:, None, :, 0])
    oy1 = torch.maximum(wb[:, :, None, 1], wb[:, None, :, 1])
    ox2 = torch.minimum(wb[:, :, None, 2], wb[:, None, :, 2])
    oy2 = torch.minimum(wb[:, :, None, 3], wb[:, None, :, 3])
    oi = (ox2 - ox1).clamp(min=0.0) * (oy2 - oy1).clamp(min=0.0)
    oa = ((wb[:, :, 2] - wb[:, :, 0]).clamp(min=0.0) * (wb[:, :, 3] - wb[:, :, 1]).clamp(min=0.0)).clamp(min=1e-6)
    oiou = oi / (oa[:, :, None] + oa[:, None, :] - oi + 1e-6)
    dup = tril64.to(wb.device) & (wc[:, :, None] == wc[:, None, :]) & (oiou >= 0.68) & (ws[:, None, :] >= 0.002)
    ws_f = torch.where(dup.float().sum(dim=2) > 0.5, ws * dup_decay, ws)
    return torch.cat([torch.cat([wb, ws_f.unsqueeze(-1), y_top[:, :, 5:6]], dim=-1), y_tail], dim=1)


class YOLOCoDesignModule(nn.Module):
    """Positive & Unconstrained Multi-Subspace WSP Graph Module for `YOLO26` (`[B, 300, 6]` output)."""

    def __init__(self, basins: dict[str, Any], cand_id: str) -> None:
        super().__init__()
        enforce_no_gradients()
        self.raw_cid = cand_id
        self.cand_id = YOLO_DYN_TO_SG_EQUIV.get(cand_id, cand_id)
        self.gamma_cls = float(basins.get("gamma_cls", 0.48))
        self.eta_box = float(basins.get("eta_box", 0.58))
        self.dup_decay = float(basins.get("dup_decay", 0.03))
        self.register_buffer("_tril64", torch.tril(torch.ones((64, 64), dtype=torch.bool), diagonal=-1).unsqueeze(0), persistent=False)

        cid = self.cand_id
        if cid == "d0":
            m0 = copy.deepcopy(basins["m_d0"]).eval()
            m0.end2end = True
            m0.model[-1].end2end = True
            self.m_d0 = m0
            return

        self.m_ep = copy.deepcopy(basins["m_ext_pure"]).eval()
        self.m_hd_i = copy.deepcopy(basins["m_hd_int_pure"]).eval() if cid in ("y39_hd4_sm120", "full_wsp") else None
        self.m_shbb = copy.deepcopy(basins["m_shbb_pure"]).eval() if cid in ("y39_shbb_gw", "full_wsp") else None
        self.m_ip = copy.deepcopy(basins["m_int_pure"]).eval() if cid in ("y39_fov352_gw", "y39_fl384_gw", "y39_fl448_gw", "y39_fl512_gw", "full_wsp") else None

        def _o(d): return (d.one2one_cv2, d.one2one_cv3)
        def _m(d): return (d.cv2, d.cv3)

        de = self.m_ep.model[-1]
        if cid == "y39_hd4_sm120":
            dh = self.m_hd_i.model[-1]
            self.fhd = FusedStiefelRawHeads([(_o(de), "o2o", 0.68), (_o(dh), "o2o", 0.32), (_m(de), "o2m", 0.68), (_m(dh), "o2m", 0.32)], raw_logits=False)
            self.fhd_aux = None
        elif cid == "sg1_p_s512":
            self.fhd = FusedStiefelRawHeads([(_o(de), "o2o", 1.0), (_m(de), "o2m", 1.0)], raw_logits=True)
            self.fhd_aux = None
        elif cid == "full_wsp":
            dh = self.m_hd_i.model[-1]
            self.fhd = FusedStiefelRawHeads([(_o(de), "o2o", 0.70), (_o(dh), "o2o", 0.30), (_m(de), "o2m", 0.70), (_m(dh), "o2m", 0.30)], raw_logits=False)
            di = self.m_ip.model[-1]
            self.fhd_aux = FusedStiefelRawHeads([(_o(di), "o2o", 1.0), (_m(di), "o2m", 1.0)], raw_logits=False)
            ds = self.m_shbb.model[-1]
            self.fhd_shbb = FusedStiefelRawHeads([(_o(ds), "o2o", 1.0), (_m(ds), "o2m", 1.0)], raw_logits=False)
        else:
            self.fhd = FusedStiefelRawHeads([(_o(de), "o2o", 1.0), (_m(de), "o2m", 1.0)], raw_logits=True)
            if self.m_shbb is not None:
                ds = self.m_shbb.model[-1]
                self.fhd_aux = FusedStiefelRawHeads([(_o(ds), "o2o", 1.0), (_m(ds), "o2m", 1.0)], raw_logits=True)
            elif self.m_ip is not None:
                di = self.m_ip.model[-1]
                self.fhd_aux = FusedStiefelRawHeads([(_o(di), "o2o", 1.0), (_m(di), "o2m", 1.0)], raw_logits=True)
            else:
                self.fhd_aux = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cid = self.cand_id
        if cid == "d0":
            self.m_d0.end2end = True
            self.m_d0.model[-1].end2end = True
            return self.m_d0(x)[0]

        B, _, H, W = [int(d) for d in x.shape]
        det = self.m_ep.model[-1]

        if cid == "y39_hd4_sm120":
            sv = _yolo_bb_neck_feats(self.m_ep, x)
            nf = [sv[16], sv[19], sv[22]]
            bo, so, bm, sm = self.fhd(nf)
            db_p = _cg_safe_yolo_decode_boxes(det, 0.37 * bo + 0.63 * bm, nf)
            return _yolo_gridwarp_consensus_pt(db_p, so, sm, None, None, None, H, W, H, W, 0, 1, self.gamma_cls, 0.0, self.eta_box, 0.0, self.dup_decay, self._tril64)

        if cid == "sg1_p_s512":
            sv0 = _yolo_bb_neck_feats(self.m_ep, x)
            nf0 = [sv0[16], sv0[19], sv0[22]]
            bo0, zo0, bm0, zm0 = self.fhd(nf0)
            db0 = _cg_safe_yolo_decode_boxes(det, 0.37 * bo0 + 0.63 * bm0, nf0)
            Hs, Ws = max(64, int(round(H * 0.80 / 32.0)) * 32), max(64, int(round(W * 0.80 / 32.0)) * 32)
            svs = _yolo_bb_neck_feats(self.m_ep, F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False))
            nfs = [svs[16], svs[19], svs[22]]
            boa, zoa, bma, zma = self.fhd(nfs)
            dba = _cg_safe_yolo_decode_boxes(det, 0.37 * boa + 0.63 * bma, nfs)
            return _yolo_gridwarp_consensus_pt(db0, zo0, zm0, dba, zoa, zma, H, W, Hs, Ws, 1, 0, self.gamma_cls, 0.30, self.eta_box, 0.065, self.dup_decay, self._tril64)

        if cid == "full_wsp":
            x2 = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            sv2 = _yolo_bb_neck_feats(self.m_ep, x2)
            nf2 = [sv2[16], sv2[19], sv2[22]]
            bo2, so2, bm2, sm2 = self.fhd(nf2)
            db2 = _cg_safe_yolo_decode_boxes(det, 0.37 * bo2 + 0.63 * bm2, nf2)
            nfs_sh = _yolo_shbb_neck(self.m_shbb, sv2[4][:B], sv2[6][:B], sv2[10][:B])
            bos, sos, bms, sms = self.fhd_shbb(nfs_sh)
            so_p = 0.80 * so2[:B] + 0.20 * sos
            sm_p = 0.80 * sm2[:B] + 0.20 * sms
            db_p = 0.82 * db2[:B] + 0.18 * _cg_safe_yolo_decode_boxes(self.m_shbb.model[-1], 0.37 * bos + 0.63 * bms, nfs_sh)
            Hh, Wh = max(64, int(round(H * 1.25 / 32.0)) * 32), max(64, int(round(W * 1.25 / 32.0)) * 32)
            svh = _yolo_bb_neck_feats(self.m_ip, F.interpolate(x, size=(Hh, Wh), mode="bilinear", align_corners=False))
            nfh = [svh[16], svh[19], svh[22]]
            boh, soh, bmh, smh = self.fhd_aux(nfh)
            dbh = _cg_safe_yolo_decode_boxes(self.m_ip.model[-1], 0.37 * boh + 0.63 * bmh, nfh)
            y_fl = _yolo_gridwarp_consensus_pt(db_p, so_p, sm_p, db2[B:], so2[B:], sm2[B:], H, W, H, W, 2, 1, self.gamma_cls, 0.32, self.eta_box, 0.068, self.dup_decay, self._tril64)
            y_hi = _yolo_gridwarp_consensus_pt(db_p, so_p, sm_p, dbh, soh, smh, H, W, Hh, Wh, 1, 1, self.gamma_cls, 0.28, self.eta_box, 0.068, self.dup_decay, self._tril64)
            return torch.cat([0.5 * (y_fl[:, :, :4] + y_hi[:, :, :4]), 0.5 * (y_fl[:, :, 4:5] + y_hi[:, :, 4:5]), y_fl[:, :, 5:6]], dim=-1)

        if cid in ("y39_2b512_gw", "y39_2b544_gw", "y39_2b640_gw"):
            smul_2b = 0.80 if cid == "y39_2b512_gw" else (0.85 if cid == "y39_2b544_gw" else 1.0)
            Hs = H if smul_2b == 1.0 else max(64, int(round(H * smul_2b / 32.0)) * 32)
            Ws = W if smul_2b == 1.0 else max(64, int(round(W * smul_2b / 32.0)) * 32)
            xs = x if smul_2b == 1.0 else F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)
            x2 = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=0)
            sv2 = _yolo_bb_neck_feats(self.m_ep, x2)
            nf2 = [sv2[16], sv2[19], sv2[22]]
            bo2, zo2, bm2, zm2 = self.fhd(nf2)
            db2 = _cg_safe_yolo_decode_boxes(det, 0.37 * bo2 + 0.63 * bm2, nf2)
            y = _yolo_gridwarp_consensus_pt(db2[:B], zo2[:B], zm2[:B], db2[B:], zo2[B:], zm2[B:], Hs, Ws, Hs, Ws, 2, 0, self.gamma_cls, 0.36, self.eta_box, 0.068, self.dup_decay, self._tril64)
            if smul_2b != 1.0:
                sx_i, sy_i = float(W) / float(Ws), float(H) / float(Hs)
                y = torch.cat([y[..., 0:1] * sx_i, y[..., 1:2] * sy_i, y[..., 2:3] * sx_i, y[..., 3:4] * sy_i, y[..., 4:6]], dim=-1)
            return y

        sv = _yolo_bb_neck_feats(self.m_ep, x)
        nf = [sv[16], sv[19], sv[22]]
        bo0, zo0, bm0, zm0 = self.fhd(nf)
        db0 = _cg_safe_yolo_decode_boxes(det, 0.37 * bo0 + 0.63 * bm0, nf)

        if cid == "y39_fhd2_sm120":
            return _yolo_gridwarp_consensus_pt(db0, zo0, zm0, None, None, None, H, W, H, W, 0, 0, self.gamma_cls, 0.0, self.eta_box, 0.0, self.dup_decay, self._tril64)

        if cid == "y39_shbb_gw":
            nfs = _yolo_shbb_neck(self.m_shbb, sv[4], sv[6], sv[10])
            boa, zoa, bma, zma = self.fhd_aux(nfs)
            dba = _cg_safe_yolo_decode_boxes(self.m_shbb.model[-1], 0.37 * boa + 0.63 * bma, nfs)
            return _yolo_gridwarp_consensus_pt(db0, zo0, zm0, dba, zoa, zma, H, W, H, W, 1, 0, self.gamma_cls, 0.28, self.eta_box, 0.060, self.dup_decay, self._tril64)

        if cid == "y39_fov352_gw":
            Ha, Wa = max(64, int(round(H * 0.55 / 32.0)) * 32), max(64, int(round(W * 0.55 / 32.0)) * 32)
            px_x, py_y = int(round(W * 0.10 / 32.0)) * 32, int(round(H * 0.10 / 32.0)) * 32
            vf = F.interpolate(x[:, :, py_y : H - py_y, px_x : W - px_x], size=(Ha, Wa), mode="bilinear", align_corners=False)
            svf = _yolo_bb_neck_feats(self.m_ip, vf)
            nff = [svf[16], svf[19], svf[22]]
            boa, zoa, bma, zma = self.fhd_aux(nff)
            dba = _cg_safe_yolo_decode_boxes(self.m_ip.model[-1], 0.37 * boa + 0.63 * bma, nff)
            return _yolo_gridwarp_consensus_pt(db0, zo0, zm0, dba, zoa, zma, H, W, Ha, Wa, 4, 0, self.gamma_cls, 0.26, self.eta_box, 0.060, self.dup_decay, self._tril64)

        smul_map = {"y39_fl384_gw": (0.60, 0.30), "y39_fl448_gw": (0.70, 0.33), "y39_fl512_gw": (0.80, 0.35)}
        smul, w_w = smul_map[cid]
        Ha, Wa = max(64, int(round(H * smul / 32.0)) * 32), max(64, int(round(W * smul / 32.0)) * 32)
        xf = torch.flip(F.interpolate(x, size=(Ha, Wa), mode="bilinear", align_corners=False), dims=[-1])
        svs = _yolo_bb_neck_feats(self.m_ip, xf)
        nfs = [svs[16], svs[19], svs[22]]
        boa, zoa, bma, zma = self.fhd_aux(nfs)
        dba = _cg_safe_yolo_decode_boxes(self.m_ip.model[-1], 0.37 * boa + 0.63 * bma, nfs)
        return _yolo_gridwarp_consensus_pt(db0, zo0, zm0, dba, zoa, zma, H, W, Ha, Wa, 2, 0, self.gamma_cls, w_w, self.eta_box, 0.065, self.dup_decay, self._tril64)


# ==============================================================================
# 2. FAMILY B: EDGECRAFTER NEGATIVE & POSITIVE WSP GRAPH MODULES
# ==============================================================================
ECDET_NEG_SPECS: dict[str, dict[str, Any]] = {
    "n00": {"arch": "d0",       "res": 640, "l_dec": 4, "nq": 300, "k_vit": 12, "wsp": True},
    "n01": {"arch": "fdr_part", "res": 640, "l_dec": 3, "nq": 240, "k_vit": 12, "wsp": True},
    "n02": {"arch": "fdr_part", "res": 576, "l_dec": 3, "nq": 220, "k_vit": 12, "wsp": True},
    "n03": {"arch": "fdr_part", "res": 640, "l_dec": 2, "nq": 200, "k_vit": 12, "wsp": True},
    "n04": {"arch": "fdr_part", "res": 512, "l_dec": 2, "nq": 200, "k_vit": 12, "wsp": True},
    "n05": {"arch": "fdr_part", "res": 640, "l_dec": 1, "nq": 180, "k_vit": 12, "wsp": True},
    "n06": {"arch": "fdr_part", "res": 448, "l_dec": 2, "nq": 180, "k_vit": 12, "wsp": True},
    "n07": {"arch": "fdr_part", "res": 448, "l_dec": 1, "nq": 160, "k_vit": 12, "wsp": True},
    "n08": {"arch": "dec0",     "res": 576, "l_dec": 0, "nq": 300, "k_vit": 12, "wsp": True},
    "n09": {"arch": "dec0",     "res": 448, "l_dec": 0, "nq": 260, "k_vit": 12, "wsp": True},
    "n10": {"arch": "dec0",     "res": 384, "l_dec": 0, "nq": 220, "k_vit": 12, "wsp": True},
    "n11": {"arch": "vit_enc",  "res": 448, "l_dec": 1, "nq": 160, "k_vit": 8,  "wsp": True},
    "n12": {"arch": "vit_enc",  "res": 384, "l_dec": 1, "nq": 150, "k_vit": 6,  "wsp": True},
    "n13": {"arch": "vit_enc",  "res": 320, "l_dec": 1, "nq": 140, "k_vit": 4,  "wsp": True},
    "n14": {"arch": "vit_byp",  "res": 256, "l_dec": 0, "nq": 200, "k_vit": 2,  "wsp": True},
    "n15": {"arch": "vit_byp",  "res": 256, "l_dec": 0, "nq": 180, "k_vit": 1,  "wsp": True},
    "n16": {"arch": "vit_byp",  "res": 192, "l_dec": 0, "nq": 144, "k_vit": 0,  "wsp": True},
    "n17": {"arch": "stem_lin", "res": 128, "l_dec": 0, "nq": 64,  "k_vit": 0,  "wsp": True},
    "n18": {"arch": "fdr_part", "res": 384, "l_dec": 2, "nq": 150, "k_vit": 12, "wsp": True},
    "n19": {"arch": "fdr_part", "res": 320, "l_dec": 2, "nq": 120, "k_vit": 12, "wsp": True},
    "n20": {"arch": "fdr_part", "res": 256, "l_dec": 2, "nq": 100, "k_vit": 12, "wsp": True},
    "n21": {"arch": "fdr_part", "res": 192, "l_dec": 1, "nq": 80,  "k_vit": 12, "wsp": True},
}


def _prepare_ecdet_deploy(mod: nn.Module, h: int = 640, w: int = 640) -> None:
    dev = next(mod.parameters()).device
    mod.encoder.eval_spatial_size = [h, w]
    mod.encoder._reset_parameters()
    for idx in mod.encoder.use_encoder_idx:
        pe = getattr(mod.encoder, f"pos_embed{idx}", None)
        if isinstance(pe, torch.Tensor):
            setattr(mod.encoder, f"pos_embed{idx}", pe.to(dev))
    mod.decoder.eval_spatial_size = [h, w]
    anch, vmask = mod.decoder._generate_anchors(device=dev)
    mod.decoder.anchors = anch.to(dev).clamp(min=-10.0, max=10.0)
    mod.decoder.valid_mask = vmask.to(dev)
    for sub in list(mod.modules()):
        cls_name = type(sub).__name__
        if cls_name == "ConvNormLayer_fuse" and hasattr(sub, "norm"):
            sub.convert_to_deploy()
        elif cls_name in ("VGGBlock", "RepVggBlock") and hasattr(sub, "conv1"):
            sub.convert_to_deploy()


def _ecdet_select_queries(dec_m: nn.Module, mem0: torch.Tensor, nq: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = int(mem0.shape[0])
    anchors = dec_m.anchors.clamp(min=-10.0, max=10.0)
    valid_mask = dec_m.valid_mask
    if B > 1:
        anchors = anchors.expand(B, -1, -1)
    vm = valid_mask.to(mem0.dtype)
    mem0_v = vm * mem0
    enc_raw = dec_m.enc_score_head(mem0_v)
    enc_z = torch.where(valid_mask, enc_raw, torch.full_like(enc_raw, -1e4))
    _, topk_ind = torch.topk(enc_z.max(-1).values, min(nq, int(enc_z.shape[1])), dim=-1)
    c1 = mem0_v.gather(dim=1, index=topk_ind.unsqueeze(-1).expand(-1, -1, int(mem0.shape[-1])))
    anc_sel = anchors.gather(dim=1, index=topk_ind.unsqueeze(-1).expand(-1, -1, 4))
    r1 = dec_m.enc_bbox_head(c1) + anc_sel
    return c1, r1


def _ecdet_run_decoder(dec_mod: nn.Module, c1: torch.Tensor, r1_unact: torch.Tensor, memory: torch.Tensor, sp_hw: list, n_layers: int | None = None, fast_last_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    from engine.edgecrafter.decoder import distance2bbox, weighting_function

    inner = dec_mod.decoder
    output = c1
    output_detach = pred_corners_undetach = 0
    value = inner.value_op(memory, None, None, None, sp_hw)
    project = inner.project if hasattr(inner, "project") else weighting_function(inner.reg_max, inner.up, inner.reg_scale)
    ref_points_detach = torch.sigmoid(r1_unact)
    qpos = dec_mod.query_pos_head(ref_points_detach).clamp(min=-10.0, max=10.0)
    layers = inner.layers if n_layers is None else inner.layers[:n_layers]
    eval_idx = int(inner.eval_idx) if fast_last_only else -1

    layer_logits, layer_bboxes = [], []
    for i, layer in enumerate(layers):
        output = layer(output, ref_points_detach.unsqueeze(2), value, sp_hw, None, qpos)
        if i == 0:
            pre_bboxes = torch.sigmoid(dec_mod.pre_bbox_head(output) + _trt_safe_inverse_sigmoid(ref_points_detach))
            ref_points_initial = pre_bboxes.detach()
        pred_corners = dec_mod.dec_bbox_head[i](output + output_detach) + pred_corners_undetach
        inter_ref_bbox = distance2bbox(ref_points_initial, dec_mod.integral(pred_corners, project), inner.reg_scale)
        if fast_last_only and i == eval_idx:
            scores = inner.lqe_layers[i](dec_mod.dec_score_head[i](output), pred_corners)
            return scores, inter_ref_bbox
        scores_i = inner.lqe_layers[i](dec_mod.dec_score_head[i](output), pred_corners)
        layer_logits.append(scores_i)
        layer_bboxes.append(inter_ref_bbox)
        pred_corners_undetach = pred_corners
        ref_points_detach = inter_ref_bbox.detach()
        output_detach = output.detach()

    if len(layer_logits) == 1:
        return layer_logits[0], layer_bboxes[0]
    if len(layer_logits) >= 3:
        z_tr = 0.85 * layer_logits[-1] + 0.11 * layer_logits[-2] + 0.04 * layer_logits[-3]
        b_tr = 0.88 * layer_bboxes[-1] + 0.09 * layer_bboxes[-2] + 0.03 * layer_bboxes[-3]
    else:
        z_tr = 0.88 * layer_logits[-1] + 0.12 * layer_logits[-2]
        b_tr = 0.90 * layer_bboxes[-1] + 0.10 * layer_bboxes[-2]
    return z_tr, b_tr


def _ecdet_accum_branch(b0: torch.Tensor, pt: torch.Tensor, bt: torch.Tensor, w_p_s: float, w_b_s: float, iou_p_thr: float = 0.55, iou_b_thr: float = 0.76) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    iou = trt_safe_iou_1to1_cxcywh(b0, bt).unsqueeze(-1)
    cond_p = iou >= iou_p_thr
    cond_b = iou >= iou_b_thr
    wp = torch.where(cond_p, w_p_s * (0.91 + 0.14 * (iou.clamp(min=0.0) ** 1.15)), torch.zeros_like(iou))
    sf = (iou / 0.88).clamp(min=0.5, max=1.15) ** 1.4
    wb = torch.where(cond_b, w_b_s * sf, torch.zeros_like(iou))
    return wp, wb, cond_b.to(pt.dtype)


class ECDetNegGraphModule(nn.Module):
    """Negative-Budget `MM-FW-Ridge` Contracted Graph Module for `EdgeCrafter` (`[B, 300, 6]` normalized boxes)."""

    def __init__(self, basins: dict[str, Any], bridges: dict[str, Any], cid: str) -> None:
        super().__init__()
        enforce_no_gradients()
        self.cid = cid
        sp = ECDET_NEG_SPECS[cid]
        self.arch = sp["arch"]
        self.res = int(sp["res"])
        self.l_dec = int(sp["l_dec"])
        self.nq = int(sp["nq"])
        self.k_vit = int(sp["k_vit"])
        self.wsp = bool(sp["wsp"])
        r_ratio = float(self.res) / 640.0
        self.logit_temp = (1.0 - 0.05 * (1.0 - r_ratio)) if (self.wsp and self.res < 640) else 1.0

        if self.arch == "d0":
            m0 = copy.deepcopy(basins["m_d0"]).eval()
            _prepare_ecdet_deploy(m0, 640, 640)
            self.m_d0 = m0
            return

        src_m = copy.deepcopy(basins["m_pri"]).eval()
        _prepare_ecdet_deploy(src_m, self.res, self.res)

        if self.arch in ("fdr_part", "dec0", "vit_enc"):
            self.backbone = src_m.backbone
            self.encoder = src_m.encoder
            self.dec = src_m.decoder
            self.sp_hw = [[self.res // 8, self.res // 8], [self.res // 16, self.res // 16], [self.res // 32, self.res // 32]]
            self.register_buffer("anc_buf", src_m.decoder.anchors.detach().clone().clamp(-10.0, 10.0), persistent=False)
            self.register_buffer("vm_buf", src_m.decoder.valid_mask.detach().clone(), persistent=False)

            if self.arch == "vit_enc":
                vit = self.backbone.backbone
                self.patch_embed = vit.patch_embed
                self.register_token = nn.Parameter(vit.register_token.detach().clone(), requires_grad=False)
                self.vit_blocks = nn.ModuleList([vit.blocks[i] for i in range(self.k_vit)])
                hc = self.res // 16
                rsin, rcos = vit.rope_embed(H=hc, W=hc)
                self.register_buffer("rope_sin", rsin.detach().clone(), persistent=False)
                self.register_buffer("rope_cos", rcos.detach().clone(), persistent=False)
                Wk, bk = bridges["vit_enc_bridges"][self.k_vit]
                d_v = int(Wk.shape[0])
                self.vit_br = nn.Linear(d_v, d_v, bias=True)
                self.vit_br.weight.data.copy_(Wk); self.vit_br.bias.data.copy_(bk)

            if self.l_dec == 0:
                self.dec.enc_score_head.weight.data.copy_(bridges["W_enc_folded"])
                self.dec.enc_score_head.bias.data.copy_(bridges["b_enc_folded"])
                self.box_res_lin = nn.Linear(bridges["W_b0"].shape[1], 4, bias=True)
                self.box_res_lin.weight.data.copy_(bridges["W_b0"])
                self.box_res_lin.bias.data.copy_(bridges["b_b0"])
            else:
                self.box_res_lin = None
        else:
            vit = src_m.backbone.backbone
            self.patch_embed = vit.patch_embed
            if self.k_vit > 0:
                self.register_token = nn.Parameter(vit.register_token.detach().clone(), requires_grad=False)
                self.blocks = nn.ModuleList([vit.blocks[i] for i in range(self.k_vit)])
                hc = self.res // 16
                rsin, rcos = vit.rope_embed(H=hc, W=hc)
                self.register_buffer("rope_sin", rsin.detach().clone(), persistent=False)
                self.register_buffer("rope_cos", rcos.detach().clone(), persistent=False)
            Wh, bh = bridges["vit_byp_heads"][self.k_vit]
            self.d_vit = int(Wh.shape[1])
            self.head84 = nn.Linear(self.d_vit, 84, bias=True)
            self.head84.weight.data.copy_(Wh); self.head84.bias.data.copy_(bh)
            hc = self.res // 16
            gy, gx = torch.meshgrid(torch.arange(hc, dtype=torch.float32), torch.arange(hc, dtype=torch.float32), indexing="ij")
            xy = torch.stack([(gx + 0.5) / float(hc), (gy + 0.5) / float(hc)], dim=-1)
            anc = torch.cat([xy, torch.full_like(xy, 0.12)], dim=-1).reshape(1, hc * hc, 4).clamp(1e-3, 1.0 - 1e-3)
            self.register_buffer("anc_s16", torch.log(anc / (1.0 - anc)).clamp(-10.0, 10.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "d0":
            x640 = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False) if (x.shape[-2] != 640 or x.shape[-1] != 640) else x
            d_out = self.m_d0(x640)
            return detr_topk300_preds6(torch.sigmoid(d_out["pred_logits"]).clamp(1e-5, 1.0 - 1e-5), d_out["pred_boxes"].clamp(1e-5, 1.0 - 1e-5), 80)

        xr = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False) if (x.shape[-2] != self.res or x.shape[-1] != self.res) else x
        B = int(xr.shape[0])

        if self.arch in ("fdr_part", "dec0", "vit_enc"):
            for idx in self.encoder.use_encoder_idx:
                pe = getattr(self.encoder, f"pos_embed{idx}", None)
                if isinstance(pe, torch.Tensor) and pe.device != xr.device:
                    setattr(self.encoder, f"pos_embed{idx}", pe.to(xr.device))
            if self.arch == "vit_enc":
                x_emb = self.patch_embed(xr)
                _, D_v, Hc, Wc = [int(d) for d in x_emb.shape]
                tok = x_emb.flatten(2).transpose(1, 2)
                h_cur = torch.cat([self.register_token.expand(B, -1, -1), tok], dim=1)
                rope = (self.rope_sin, self.rope_cos)
                for blk in self.vit_blocks:
                    h_cur = blk(h_cur, rope_sincos=rope)
                fused = self.vit_br(F.layer_norm(h_cur[:, 1:], (D_v,))).transpose(1, 2).contiguous().view(B, D_v, Hc, Wc)
                p_feats = [F.interpolate(fused, size=[int(Hc * (2 ** (1 - i))), int(Wc * (2 ** (1 - i)))], mode="bilinear", align_corners=False) for i in range(3)]
                if len(self.backbone.projector) == 1:
                    p_feats[-1] = self.backbone.projector[-1](p_feats[-1])
                else:
                    p_feats = [lyr(ft) for lyr, ft in zip(self.backbone.projector, p_feats)]
                mem, _ = self.dec._get_encoder_input(self.encoder(p_feats))
            else:
                mem, _ = self.dec._get_encoder_input(self.encoder(self.backbone(xr)))

            anchors = self.anc_buf.expand(B, -1, -1) if B > 1 else self.anc_buf
            vm = self.vm_buf
            mem_v = vm.to(mem.dtype) * mem
            enc_z_raw = self.dec.enc_score_head(mem_v)
            enc_z = torch.where(vm, enc_z_raw, torch.full_like(enc_z_raw, -1e4))
            kq = min(self.nq, int(enc_z.shape[1]))
            _, topk_ind = torch.topk(enc_z.max(-1).values, kq, dim=-1)
            c1 = mem_v.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, int(mem.shape[-1])))
            anc_sel = anchors.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 4))
            if self.l_dec == 0:
                z_sel = enc_z.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 80))
                db = self.dec.enc_bbox_head(c1)
                if self.box_res_lin is not None:
                    db = 0.45 * db + 0.55 * self.box_res_lin(c1)
                return detr_topk300_preds6(torch.sigmoid(z_sel / self.logit_temp).clamp(1e-5, 1.0 - 1e-5), torch.sigmoid(db + anc_sel).clamp(1e-5, 1.0 - 1e-5), 80)

            r1 = self.dec.enc_bbox_head(c1) + anc_sel
            enc_z_sel = enc_z.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 80))
            zd, bd = _ecdet_run_decoder(self.dec, c1, r1, mem, self.sp_hw, n_layers=self.l_dec)
            zd = 0.90 * zd + 0.10 * enc_z_sel
            return detr_topk300_preds6(torch.sigmoid(zd / self.logit_temp).clamp(1e-5, 1.0 - 1e-5), bd.clamp(1e-5, 1.0 - 1e-5), 80)

        tok = self.patch_embed(xr).flatten(2).transpose(1, 2)
        if self.k_vit > 0:
            h_cur = torch.cat([self.register_token.expand(B, -1, -1), tok], dim=1)
            rope = (self.rope_sin, self.rope_cos)
            for blk in self.blocks:
                h_cur = blk(h_cur, rope_sincos=rope)
            tok = h_cur[:, 1:]
        out84 = self.head84(F.layer_norm(tok, (self.d_vit,)))
        anc = self.anc_s16.expand(B, -1, -1) if B > 1 else self.anc_s16
        z_all, db_all = out84[..., :80], out84[..., 80:]
        kq = min(self.nq, int(z_all.shape[1]))
        if kq < int(z_all.shape[1]):
            _, topk_ind = torch.topk(z_all.max(-1).values, kq, dim=-1)
            z_all = z_all.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 80))
            db_all = db_all.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 4))
            anc = anc.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, 4))
        return detr_topk300_preds6(torch.sigmoid(z_all / self.logit_temp).clamp(1e-5, 1.0 - 1e-5), torch.sigmoid(db_all + anc).clamp(1e-5, 1.0 - 1e-5), 80)


class ECDetCoDesignModule(nn.Module):
    """Positive & Unconstrained Multi-Subspace WSP Graph Module for `EdgeCrafter` (`[B, 300, 6]` output)."""

    def __init__(self, basins: dict[str, Any], cand_id: str) -> None:
        super().__init__()
        enforce_no_gradients()
        self.cand_id = cand_id
        self.sp_640 = [[80, 80], [40, 40], [20, 20]]
        self.alpha_cor = float(basins.get("alpha_cor", 0.015))

        if cand_id == "d0":
            m0 = copy.deepcopy(basins["m_d0"]).eval()
            _prepare_ecdet_deploy(m0, 640, 640)
            self.m_d0 = m0
            return

        self.m_pri = copy.deepcopy(basins["m_pri"]).eval()
        _prepare_ecdet_deploy(self.m_pri, 640, 640)
        self.dec_ie = None
        self.dec_me = None
        self.enc_i = self.dec_i = None
        self.enc_m = self.dec_m = None
        self.m_pri_aux = None

        if cand_id in ("d5", "ec39_shenc_sm120", "ec39_shbb_locsafe_sm120", "ec39_shbb4_sm120", "ec39_fl384_shenc", "ec39_fl448_shenc", "ec39_fl512_shenc", "ec39_2b640_shenc", "ec39_fl448_shbb", "ec39_fl512_shbb", "ec39_hi704_shenc", "ec39_2b640_shbb", "full_wsp"):
            m_ie = copy.deepcopy(basins["m_int_shenc"]).eval()
            _prepare_ecdet_deploy(m_ie, 640, 640)
            self.dec_ie = m_ie.decoder
        if cand_id in ("d8", "ec39_shbb4_sm120", "full_wsp"):
            m_me = copy.deepcopy(basins["m_mid_shenc"]).eval()
            _prepare_ecdet_deploy(m_me, 640, 640)
            self.dec_me = m_me.decoder
        if cand_id in ("d7", "d8", "ec39_shbb_locsafe_sm120", "ec39_shbb4_sm120", "ec39_fl448_shbb", "ec39_fl512_shbb", "ec39_2b640_shbb", "full_wsp"):
            m_ish = copy.deepcopy(basins["m_int_shbb"]).eval()
            _prepare_ecdet_deploy(m_ish, 640, 640)
            self.enc_i, self.dec_i = m_ish.encoder, m_ish.decoder
        if cand_id in ("ec39_shbb4_sm120", "full_wsp"):
            m_msh = copy.deepcopy(basins["m_mid_shbb"]).eval()
            _prepare_ecdet_deploy(m_msh, 640, 640)
            self.enc_m, self.dec_m = m_msh.encoder, m_msh.decoder

        aux_res_map = {
            "sg1_s512": 512, "ec39_fl384_2s": 384, "ec39_fl448_2s": 448, "ec39_fl512_2s": 512,
            "ec39_fl384_shenc": 384, "ec39_fl448_shenc": 448, "ec39_fl512_shenc": 512,
            "ec39_fl448_shbb": 448, "ec39_fl512_shbb": 512, "ec39_hi704_shenc": 704, "full_wsp": 704,
        }
        if cand_id in aux_res_map:
            rs = aux_res_map[cand_id]
            self.m_pri_aux = copy.deepcopy(basins["m_pri"]).eval()
            _prepare_ecdet_deploy(self.m_pri_aux, rs, rs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cid = self.cand_id
        x640 = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False) if (x.shape[-2] != 640 or x.shape[-1] != 640) else x
        B = int(x640.shape[0])

        if cid == "d0":
            d_out = self.m_d0(x640)
            return detr_topk300_preds6(torch.sigmoid(d_out["pred_logits"]).clamp(1e-5, 1.0 - 1e-5), d_out["pred_boxes"].clamp(1e-5, 1.0 - 1e-5), 80)

        dec1 = self.m_pri.decoder
        if cid == "u1":
            mem0, _ = dec1._get_encoder_input(self.m_pri.encoder(self.m_pri.backbone(x640)))
            c1, r1 = _ecdet_select_queries(dec1, mem0, 300)
            z0, b0 = _ecdet_run_decoder(dec1, c1, r1, mem0, self.sp_640, fast_last_only=True)
            return detr_topk300_preds6(torch.sigmoid(z0).clamp(1e-5, 1.0 - 1e-5), b0.clamp(1e-5, 1.0 - 1e-5), 80)

        if cid == "d2":
            mem0, _ = dec1._get_encoder_input(self.m_pri.encoder(self.m_pri.backbone(x640)))
            c1, r1 = _ecdet_select_queries(dec1, mem0, 460)
            z0, b0 = _ecdet_run_decoder(dec1, c1, r1, mem0, self.sp_640)
            return detr_topk300_preds6(torch.sigmoid(z0).clamp(1e-5, 1.0 - 1e-5), b0.clamp(1e-5, 1.0 - 1e-5), 80)

        do_2b_640 = cid in ("ec39_2b640_2s", "ec39_2b640_shenc", "ec39_2b640_shbb", "full_wsp")
        f_in = self.m_pri.backbone(torch.cat([x640, torch.flip(x640, dims=[-1])], dim=0) if do_2b_640 else x640)
        mem_in, _ = dec1._get_encoder_input(self.m_pri.encoder(f_in))
        mem0 = mem_in[:B] if do_2b_640 else mem_in
        f0 = [ft[:B] for ft in f_in] if do_2b_640 else f_in
        c1, r1 = _ecdet_select_queries(dec1, mem0, 460)

        if do_2b_640:
            r1_fl = torch.cat([-r1[..., 0:1], r1[..., 1:]], dim=-1)
            z2, b2 = _ecdet_run_decoder(dec1, torch.cat([c1, c1], dim=0), torch.cat([r1, r1_fl], dim=0), mem_in, self.sp_640)
            p0, b0 = torch.sigmoid(z2[:B]), b2[:B]
            pfl = torch.sigmoid(z2[B:])
            bfl = torch.cat([1.0 - b2[B:, :, 0:1], b2[B:, :, 1:]], dim=-1)
        else:
            z0, b0 = _ecdet_run_decoder(dec1, c1, r1, mem0, self.sp_640)
            p0 = torch.sigmoid(z0)
            pfl = bfl = None

        p_num, p_den = p0, torch.ones_like(p0[..., :1])
        b_num, b_den = b0, torch.ones_like(b0[..., :1])
        cor = torch.zeros_like(p0[..., :1])

        def _add(pv: torch.Tensor, bv: torch.Tensor, wps: float, wbs: float) -> None:
            nonlocal p_num, p_den, b_num, b_den, cor
            wp_k, wb_k, ck = _ecdet_accum_branch(b0, pv, bv, wps, wbs)
            p_num, p_den = p_num + wp_k * pv, p_den + wp_k
            b_num, b_den = b_num + wb_k * bv, b_den + wb_k
            cor = cor + ck

        if self.dec_ie is not None:
            zie, bie = _ecdet_run_decoder(self.dec_ie, c1, r1, mem0, self.sp_640)
            w_ie = (0.25, 0.25) if cid == "d5" else ((0.25, 0.20) if cid == "ec39_shenc_sm120" else (0.20, 0.14))
            _add(torch.sigmoid(zie), bie, w_ie[0], w_ie[1])
        if self.dec_me is not None:
            zme, bme = _ecdet_run_decoder(self.dec_me, c1, r1, mem0, self.sp_640)
            _add(torch.sigmoid(zme), bme, 0.20, 0.16)
        if self.enc_i is not None and self.dec_i is not None:
            mi, _ = self.dec_i._get_encoder_input(self.enc_i(f0))
            zi, bi = _ecdet_run_decoder(self.dec_i, c1, r1, mi, self.sp_640)
            w_i = (0.28, 0.28) if cid in ("d7", "d8") else (0.18, 0.08)
            _add(torch.sigmoid(zi), bi, w_i[0], w_i[1])
        if self.enc_m is not None and self.dec_m is not None:
            mm, _ = self.dec_m._get_encoder_input(self.enc_m(f0))
            zm, bm = _ecdet_run_decoder(self.dec_m, c1, r1, mm, self.sp_640)
            _add(torch.sigmoid(zm), bm, 0.20, 0.18)
        if pfl is not None and bfl is not None:
            _add(pfl, bfl, 0.37, 0.36)

        if self.m_pri_aux is not None:
            rs_fl_map = {
                "sg1_s512": (512, False, 0.26, 0.26),
                "ec39_fl384_2s": (384, True, 0.28, 0.25),
                "ec39_fl448_2s": (448, True, 0.33, 0.30),
                "ec39_fl512_2s": (512, True, 0.36, 0.34),
                "ec39_fl384_shenc": (384, True, 0.26, 0.24),
                "ec39_fl448_shenc": (448, True, 0.31, 0.29),
                "ec39_fl512_shenc": (512, True, 0.34, 0.32),
                "ec39_fl448_shbb": (448, True, 0.30, 0.28),
                "ec39_fl512_shbb": (512, True, 0.32, 0.30),
                "ec39_hi704_shenc": (704, False, 0.36, 0.36),
                "full_wsp": (704, False, 0.36, 0.36),
            }
            rs, do_fl, wps, wbs = rs_fl_map[cid]
            sp_rs = [[rs // 8, rs // 8], [rs // 16, rs // 16], [rs // 32, rs // 32]]
            xs = F.interpolate(x640, size=(rs, rs), mode="bilinear", align_corners=False)
            if do_fl:
                xs = torch.flip(xs, dims=[-1])
            for idx in self.m_pri_aux.encoder.use_encoder_idx:
                pe = getattr(self.m_pri_aux.encoder, f"pos_embed{idx}", None)
                if isinstance(pe, torch.Tensor) and pe.device != xs.device:
                    setattr(self.m_pri_aux.encoder, f"pos_embed{idx}", pe.to(xs.device))
            ms_mem, _ = self.m_pri_aux.decoder._get_encoder_input(self.m_pri_aux.encoder(self.m_pri_aux.backbone(xs)))
            r_in = torch.cat([-r1[..., 0:1], r1[..., 1:]], dim=-1) if do_fl else r1
            zs, bs_r = _ecdet_run_decoder(self.m_pri_aux.decoder, c1, r_in, ms_mem, sp_rs)
            bs = torch.cat([1.0 - bs_r[..., 0:1], bs_r[..., 1:]], dim=-1) if do_fl else bs_r
            _add(torch.sigmoid(zs), bs, wps, wbs)

        pf = ((p_num / p_den) * (1.0 + (self.alpha_cor * 0.45) * cor.clamp(max=6.0))).clamp(1e-5, 1.0 - 1e-5)
        bf = (b_num / b_den).clamp(1e-5, 1.0 - 1e-5)
        return detr_topk300_preds6(pf, bf, 80)


# ==============================================================================
# 3. FAMILY C: RF-DETR NEGATIVE & POSITIVE WSP GRAPH MODULES
# ==============================================================================
def get_rfdetr_neg_specs(nat_res: int, dec_layers: int) -> dict[str, dict[str, Any]]:
    R0, L0 = int(nat_res), int(dec_layers)
    Lm1 = max(1, L0 - 1)
    return {
        "n00": {"arch": "d0",        "res": R0,                 "l_dec": L0,  "nq": 300, "k_dino": 12, "wsp": True},
        "n01": {"arch": "dec_part",  "res": max(320, R0 - 32),  "l_dec": L0,  "nq": 250, "k_dino": 12, "wsp": True},
        "n02": {"arch": "dec_part",  "res": R0,                 "l_dec": Lm1, "nq": 240, "k_dino": 12, "wsp": True},
        "n03": {"arch": "dec_part",  "res": max(288, R0 - 64),  "l_dec": Lm1, "nq": 220, "k_dino": 12, "wsp": True},
        "n04": {"arch": "dec_part",  "res": max(256, R0 - 96),  "l_dec": Lm1, "nq": 200, "k_dino": 12, "wsp": True},
        "n05": {"arch": "dec_part",  "res": R0,                 "l_dec": 1,   "nq": 180, "k_dino": 12, "wsp": True},
        "n06": {"arch": "dec_part",  "res": max(256, R0 - 96),  "l_dec": 1,   "nq": 200, "k_dino": 12, "wsp": True},
        "n07": {"arch": "dec0",      "res": max(320, R0 - 64),  "l_dec": 0,   "nq": 300, "k_dino": 12, "wsp": True},
        "n08": {"arch": "dec0",      "res": max(256, R0 - 128), "l_dec": 0,   "nq": 260, "k_dino": 12, "wsp": True},
        "n09": {"arch": "dec0",      "res": 256,                "l_dec": 0,   "nq": 240, "k_dino": 12, "wsp": True},
        "n10": {"arch": "dino_proj", "res": max(288, R0 - 96),  "l_dec": 0,   "nq": 260, "k_dino": 9,  "wsp": True},
        "n11": {"arch": "dino_proj", "res": max(256, R0 - 128), "l_dec": 0,   "nq": 240, "k_dino": 6,  "wsp": True},
        "n12": {"arch": "dino_proj", "res": 224,                "l_dec": 0,   "nq": 196, "k_dino": 6,  "wsp": True},
        "n13": {"arch": "dino_proj", "res": 256,                "l_dec": 0,   "nq": 240, "k_dino": 3,  "wsp": True},
        "n14": {"arch": "dino_proj", "res": 192,                "l_dec": 0,   "nq": 144, "k_dino": 3,  "wsp": True},
        "n15": {"arch": "dino_dir",  "res": 192,                "l_dec": 0,   "nq": 144, "k_dino": 2,  "wsp": True},
        "n16": {"arch": "dino_dir",  "res": 160,                "l_dec": 0,   "nq": 100, "k_dino": 1,  "wsp": True},
        "n17": {"arch": "dino_stem", "res": 128,                "l_dec": 0,   "nq": 64,  "k_dino": 0,  "wsp": True},
        "n18": {"arch": "dec_part",  "res": 384 if R0 > 384 else 256, "l_dec": 1, "nq": 160, "k_dino": 12, "wsp": True},
        "n19": {"arch": "dec_part",  "res": 288 if R0 > 384 else 224, "l_dec": 1, "nq": 140, "k_dino": 12, "wsp": True},
        "n20": {"arch": "dec_part",  "res": 224 if R0 > 384 else 192, "l_dec": 1, "nq": 110, "k_dino": 12, "wsp": True},
        "n21": {"arch": "dec_part",  "res": 160,                      "l_dec": 1, "nq": 80,  "k_dino": 12, "wsp": True},
    }


def _export_rfdetr_submodules(m_cp: nn.Module) -> None:
    for sub in m_cp.modules():
        if hasattr(sub, "export") and callable(sub.export) and sub is not m_cp and hasattr(sub, "_export") and not sub._export:
            sub.export()


def _rfdetr_encode_with_bb(bb: nn.Module, tr: nn.Module, x: torch.Tensor):
    srcs, _, poss, cross_attn_srcs = bb(x)
    B = int(x.shape[0])
    src_flat, pos_flat, sp_hw = [], [], []
    for src, pos_e in zip(srcs, poss):
        h, w = int(src.shape[2]), int(src.shape[3])
        sp_hw.append((h, w))
        src_flat.append(src.flatten(2).transpose(1, 2))
        pos_flat.append(pos_e.flatten(2).transpose(1, 2))
    memory = src_flat[0].contiguous() if len(src_flat) == 1 else torch.cat(src_flat, 1)
    lvl_pos = pos_flat[0].contiguous() if len(pos_flat) == 1 else torch.cat(pos_flat, 1)
    sp_t = torch.tensor(sp_hw, device=x.device, dtype=torch.long)
    lsi = torch.cat((sp_t.new_zeros((1,)), sp_t.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.ones((B, len(srcs), 2), device=x.device, dtype=memory.dtype)
    dec_mem = memory
    if cross_attn_srcs is not None and not tr.dual_projector_kp_only:
        ca_flat = [cs.flatten(2).transpose(1, 2) for cs in cross_attn_srcs]
        dec_mem = ca_flat[0].contiguous() if len(ca_flat) == 1 else torch.cat(ca_flat, 1)
    return tr, memory, dec_mem, valid_ratios, lvl_pos, sp_t, sp_hw, lsi


def _interp_dinov2_patch_pos(pos_embed_param: torch.Tensor, h_p: int, w_p: int) -> tuple[torch.Tensor, torch.Tensor]:
    pe = pos_embed_param.detach().float()
    cls_pe = pe[:, :1, :]
    patch_pe = pe[:, 1:, :]
    gs0 = int(round(math.sqrt(int(patch_pe.shape[1]))))
    C = int(pe.shape[-1])
    grid = patch_pe.reshape(1, gs0, gs0, C).permute(0, 3, 1, 2)
    if gs0 != h_p or gs0 != w_p:
        grid = F.interpolate(grid, size=(h_p, w_p), mode="bicubic", align_corners=False)
    return cls_pe.contiguous(), grid.permute(0, 2, 3, 1).reshape(1, h_p * w_p, C).contiguous()


class RFDETRNegGraphModule(nn.Module):
    """Negative-Budget `MM-FW-Ridge` Contracted Graph Module for `RF-DETR` (`[B, 300, 6]` output)."""

    def __init__(self, basins: dict[str, Any], bridges: dict[str, Any], cid: str) -> None:
        super().__init__()
        enforce_no_gradients()
        from rfdetr.models.transformer import gen_encoder_output_proposals

        self.cid = cid
        nat_res = int(basins["nat_res"])
        base_m = basins["m_rf"]
        dec_layers = len(base_m.transformer.decoder.layers)
        sp = get_rfdetr_neg_specs(nat_res, dec_layers)[cid]
        self.arch = sp["arch"]
        self.res = int(sp["res"])
        self.l_dec = int(sp["l_dec"])
        self.nq = int(sp["nq"])
        self.k_dino = int(sp["k_dino"])
        self.wsp = bool(sp["wsp"])

        m_cp = copy.deepcopy(basins["m_rf"]).eval()
        m_cp.backbone[0].encoder.shape = (self.res, self.res)
        _export_rfdetr_submodules(m_cp)

        nc = int(m_cp.class_embed.out_features)
        self.nc = nc
        valid91 = torch.zeros(nc, dtype=torch.bool)
        for c in COCO80_TO_91:
            if c < nc:
                valid91[c] = True
        self.register_buffer("valid91", valid91, persistent=False)
        self.register_buffer("v91_flt", valid91.float().view(1, 1, nc), persistent=False)
        r_ratio = float(self.res) / float(nat_res)
        self.logit_temp = (1.0 - 0.05 * max(0.0, 1.0 - r_ratio)) if (self.wsp and self.res < nat_res) else 1.0

        if self.arch == "d0":
            self.mod = m_cp
            return

        if self.arch in ("dec_part", "dec0", "dino_proj"):
            self.mod = m_cp
            tr = self.mod.transformer
            if self.arch == "dec_part" and self.l_dec < len(tr.decoder.layers):
                tr.decoder.layers = nn.ModuleList(list(tr.decoder.layers[: self.l_dec]))
            if self.arch == "dec_part":
                tr.enc_out_class_embed[0].weight.data.copy_(bridges["W_13c"])
                tr.enc_out_class_embed[0].bias.data.copy_(bridges["b_13c"])
                self.box_res_lin = None
            else:
                tr.enc_out_class_embed[0].weight.data.copy_(bridges["W_0dec_cls"])
                tr.enc_out_class_embed[0].bias.data.copy_(bridges["b_0dec_cls"])
                self.box_res_lin = nn.Linear(256, 4, bias=True)
                self.box_res_lin.weight.data.copy_(bridges["W_db"])
                self.box_res_lin.bias.data.copy_(bridges["b_db"])
            if self.arch == "dino_proj":
                dino_bb = self.mod.backbone[0].encoder.encoder
                proj = self.mod.backbone[0].projector
                keep_stages = [f"stage{s}" for s in (3, 6, 9, 12) if s <= self.k_dino]
                dino_bb.config.out_features = keep_stages
                dino_bb.out_features = keep_stages
                dino_bb.encoder.layer = nn.ModuleList(list(dino_bb.encoder.layer[: self.k_dino]))
                wk_map = {9: bridges["w_cv1_k9"], 6: bridges["w_cv1_k6"], 3: bridges["w_cv1_k3"]}[self.k_dino]
                new_conv = nn.Conv2d(wk_map.shape[1], wk_map.shape[0], kernel_size=1, stride=1, bias=False)
                new_conv.weight.data.copy_(wk_map)
                proj.stages[0][0].cv1.conv = new_conv
                proj.stages_sampling[0] = nn.ModuleList([nn.Sequential() for _ in keep_stages])
        else:
            dino_bb = m_cp.backbone[0].encoder.encoder
            tr = m_cp.transformer
            self.bbox_reparam = bool(tr.bbox_reparam)
            hp = self.res // 16
            cls_pe, patch_pe = _interp_dinov2_patch_pos(dino_bb.embeddings.position_embeddings, hp, hp)
            self.patch_conv = copy.deepcopy(dino_bb.embeddings.patch_embeddings.projection)
            self.register_buffer("cls_pe", cls_pe, persistent=False)
            self.register_buffer("patch_pe", patch_pe, persistent=False)
            self.cls_token = nn.Parameter(dino_bb.embeddings.cls_token.detach().clone(), requires_grad=False)
            self.n_reg = int(dino_bb.num_register_tokens)
            if self.n_reg > 0:
                self.reg_tokens = nn.Parameter(dino_bb.embeddings.register_tokens.detach().clone(), requires_grad=False)
            self.layernorm = copy.deepcopy(dino_bb.layernorm)
            self.blocks = nn.ModuleList([copy.deepcopy(dino_bb.encoder.layer[i]) for i in range(self.k_dino)])
            dummy_mem = torch.zeros((1, hp * hp, 256), dtype=torch.float32)
            _, prop = gen_encoder_output_proposals(dummy_mem, None, [(hp, hp)], unsigmoid=not self.bbox_reparam)
            self.register_buffer("prop_s16", prop.clamp(-10.0, 10.0), persistent=False)
            Wh, bh = bridges["dir_heads"][self.k_dino]
            self.dir_head = nn.Linear(int(Wh.shape[1]), nc + 4, bias=True)
            self.dir_head.weight.data.copy_(Wh)
            self.dir_head.bias.data.copy_(bh)

    def _decode_rf_0dec(self, pn0: torch.Tensor, out_prop: torch.Tensor) -> torch.Tensor:
        tr = self.mod.transformer
        ec0 = tr.enc_out_class_embed[0](pn0)
        ec_masked = torch.where(self.valid91.view(1, 1, self.nc), ec0, torch.full_like(ec0, -100.0))
        kq = min(self.nq, int(pn0.shape[1]))
        if kq < int(pn0.shape[1]):
            _, topk_idx = ec_masked.max(-1)[0].topk(kq, dim=1)
            pn_s = torch.gather(pn0, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 256))
            prop_s = torch.gather(out_prop, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
            z_s = torch.gather(ec0, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        else:
            pn_s, prop_s, z_s = pn0, out_prop, ec0
        d0 = tr.enc_out_bbox_embed[0](pn_s)
        if self.box_res_lin is not None:
            d0 = 0.75 * d0 + 0.25 * self.box_res_lin(pn_s)
        boxes = torch.cat([d0[..., :2] * prop_s[..., 2:] + prop_s[..., :2], d0[..., 2:].clamp(-4.0, 4.0).exp() * prop_s[..., 2:]], dim=-1) if tr.bbox_reparam else (d0 + prop_s).sigmoid()
        probs = (z_s / self.logit_temp).sigmoid() * self.v91_flt
        return detr_topk300_preds6(probs.clamp(0.0, 1.0 - 1e-5), boxes.clamp(1e-5, 1.0 - 1e-5), self.nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from rfdetr.models.transformer import gen_encoder_output_proposals

        xr = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False) if (x.shape[-2] != self.res or x.shape[-1] != self.res) else x
        B = int(xr.shape[0])

        if self.arch == "d0":
            tr = self.mod.transformer
            pk0 = _rfdetr_encode_with_bb(self.mod.backbone, tr, xr)
            out_mem, out_prop = gen_encoder_output_proposals(pk0[1], None, pk0[6], unsigmoid=not tr.bbox_reparam)
            pn0 = tr.enc_output_norm[0](tr.enc_output[0](out_mem))
            ec0 = tr.enc_out_class_embed[0](pn0)
            nq = int(self.mod.num_queries)
            _, tidx = ec0.max(-1)[0].topk(min(nq, int(ec0.shape[-2])), dim=1)
            prop_s = torch.gather(out_prop, 1, tidx.unsqueeze(-1).expand(-1, -1, 4))
            d0 = tr.enc_out_bbox_embed[0](torch.gather(pn0, 1, tidx.unsqueeze(-1).expand(-1, -1, int(pn0.shape[-1]))))
            ref_ts = torch.cat([d0[..., :2] * prop_s[..., 2:] + prop_s[..., :2], d0[..., 2:].exp() * prop_s[..., 2:]], dim=-1) if tr.bbox_reparam else (d0 + prop_s)
            tgt = self.mod.query_feat.weight[:nq].unsqueeze(0).expand(B, -1, -1).contiguous()
            ref_e = self.mod.refpoint_embed.weight[:nq].unsqueeze(0).expand(B, -1, -1).contiguous()
            ref_in = torch.cat([ref_e[..., :2] * ref_ts[..., 2:] + ref_ts[..., :2], ref_e[..., 2:].exp() * ref_ts[..., 2:]], dim=-1) if tr.bbox_reparam else (ref_e + ref_ts)
            if tr.num_registers > 0:
                tgt = torch.cat([tgt, tr.register_tokens.unsqueeze(0).expand(B, -1, -1)], dim=1)
                ref_in = torch.cat([ref_in, tr.register_ref_points.unsqueeze(0).expand(B, -1, -1)], dim=1)
            lp = pk0[4].expand(B, -1, -1).contiguous() if int(pk0[4].shape[0]) == 1 and B > 1 else pk0[4]
            hs, ref = tr.decoder(tgt, pk0[2], memory_key_padding_mask=None, pos=lp, refpoints_unsigmoid=ref_in, level_start_index=pk0[7], spatial_shapes=pk0[5], spatial_shapes_hw=pk0[6], valid_ratios=pk0[3])[:2]
            if hs.ndim == 3:
                hs, ref = hs.unsqueeze(0), ref.unsqueeze(0)
            if tr.num_registers > 0:
                hs, ref = hs[:, :, :nq, :], ref[:, :, :nq, :]
            db = self.mod.bbox_embed(hs[-1])
            b0 = torch.cat([db[..., :2] * ref[-1][..., 2:] + ref[-1][..., :2], db[..., 2:].exp() * ref[-1][..., 2:]], dim=-1) if self.mod.bbox_reparam else (db + ref[-1]).sigmoid()
            return detr_topk300_preds6(self.mod.class_embed(hs[-1]).sigmoid() * self.v91_flt, b0.clamp(1e-5, 1.0 - 1e-5), self.nc)

        if self.arch == "dec_part":
            tr = self.mod.transformer
            pk = _rfdetr_encode_with_bb(self.mod.backbone, tr, xr)
            memory, dec_mem, valid_ratios, lvl_pos, sp_t, sp_hw, lsi = pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]
            out_mem, out_prop = gen_encoder_output_proposals(memory, None, sp_hw, unsigmoid=not tr.bbox_reparam)
            pn0 = tr.enc_output_norm[0](tr.enc_output[0](out_mem))
            ec0 = tr.enc_out_class_embed[0](pn0)
            ec_m = torch.where(self.valid91.view(1, 1, self.nc), ec0, torch.full_like(ec0, -100.0))
            kq = min(self.nq, int(ec0.shape[1]))
            _, topk_idx = ec_m.max(-1)[0].topk(kq, dim=1)
            prop_s = torch.gather(out_prop, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
            pn_s = torch.gather(pn0, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 256))
            enc_z_s = torch.gather(ec0, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.nc))
            d0 = tr.enc_out_bbox_embed[0](pn_s)
            ref_ts = torch.cat([d0[..., :2] * prop_s[..., 2:] + prop_s[..., :2], d0[..., 2:].exp() * prop_s[..., 2:]], dim=-1) if tr.bbox_reparam else (d0 + prop_s)
            tgt = self.mod.query_feat.weight[:kq].unsqueeze(0).expand(B, -1, -1).contiguous()
            ref_e = self.mod.refpoint_embed.weight[:kq].unsqueeze(0).expand(B, -1, -1).contiguous()
            ref_in = torch.cat([ref_e[..., :2] * ref_ts[..., 2:] + ref_ts[..., :2], ref_e[..., 2:].exp() * ref_ts[..., 2:]], dim=-1) if tr.bbox_reparam else (ref_e + ref_ts)
            if tr.num_registers > 0:
                tgt = torch.cat([tgt, tr.register_tokens.unsqueeze(0).expand(B, -1, -1)], dim=1)
                ref_in = torch.cat([ref_in, tr.register_ref_points.unsqueeze(0).expand(B, -1, -1)], dim=1)
            lvl_pos_u = lvl_pos.expand(B, -1, -1).contiguous() if int(lvl_pos.shape[0]) == 1 and B > 1 else lvl_pos
            hs, ref = tr.decoder(tgt, dec_mem, memory_key_padding_mask=None, pos=lvl_pos_u, refpoints_unsigmoid=ref_in, level_start_index=lsi, spatial_shapes=sp_t, spatial_shapes_hw=sp_hw, valid_ratios=valid_ratios)[:2]
            if hs.ndim == 3:
                hs, ref = hs.unsqueeze(0), ref.unsqueeze(0)
            if tr.num_registers > 0:
                hs, ref = hs[:, :, :kq, :], ref[:, :, :kq, :]
            db = self.mod.bbox_embed(hs[-1])
            b_last = torch.cat([db[..., :2] * ref[-1][..., 2:] + ref[-1][..., :2], db[..., 2:].exp() * ref[-1][..., 2:]], dim=-1) if self.mod.bbox_reparam else (db + ref[-1]).sigmoid()
            z_last = self.mod.class_embed(hs[-1])
            z_last = (0.88 * z_last + 0.06 * self.mod.class_embed(hs[-2]) + 0.06 * enc_z_s) if hs.shape[0] >= 2 else (0.92 * z_last + 0.08 * enc_z_s)
            probs = (z_last / self.logit_temp).sigmoid() * self.v91_flt
            return detr_topk300_preds6(probs.clamp(0.0, 1.0 - 1e-5), b_last.clamp(1e-5, 1.0 - 1e-5), self.nc)

        if self.arch == "dec0":
            tr = self.mod.transformer
            pk = _rfdetr_encode_with_bb(self.mod.backbone, tr, xr)
            out_mem, out_prop = gen_encoder_output_proposals(pk[1], None, pk[6], unsigmoid=not tr.bbox_reparam)
            return self._decode_rf_0dec(tr.enc_output_norm[0](tr.enc_output[0](out_mem)), out_prop)

        if self.arch == "dino_proj":
            tr = self.mod.transformer
            feats = self.mod.backbone[0].projector(list(self.mod.backbone[0].encoder(xr)))
            h_f, w_f = int(feats[0].shape[2]), int(feats[0].shape[3])
            memory = feats[0].flatten(2).transpose(1, 2).contiguous()
            out_mem, out_prop = gen_encoder_output_proposals(memory, None, [(h_f, w_f)], unsigmoid=not tr.bbox_reparam)
            return self._decode_rf_0dec(tr.enc_output_norm[0](tr.enc_output[0](out_mem)), out_prop)

        p_tok = self.patch_conv(xr).flatten(2).transpose(1, 2) + self.patch_pe
        cls_t = self.cls_token.expand(B, -1, -1) + self.cls_pe
        h_g = torch.cat([cls_t, self.reg_tokens.expand(B, -1, -1), p_tok], dim=1) if self.n_reg > 0 else torch.cat([cls_t, p_tok], dim=1)
        for blk in self.blocks:
            h_g = blk(h_g, False, False)[0]
        tok_ln = self.layernorm(h_g)[:, 1 + self.n_reg :, :]
        prop = self.prop_s16.expand(B, -1, -1) if B > 1 else self.prop_s16
        out_all = self.dir_head(tok_ln)
        z_all, d_all = out_all[..., : self.nc], out_all[..., self.nc :]
        boxes = torch.cat([d_all[..., :2] * prop[..., 2:] + prop[..., 2:], d_all[..., 2:].clamp(-4.0, 4.0).exp() * prop[..., 2:]], dim=-1) if self.bbox_reparam else (d_all + prop).sigmoid()
        probs = (z_all / self.logit_temp).sigmoid() * self.v91_flt
        return detr_topk300_preds6(probs.clamp(0.0, 1.0 - 1e-5), boxes.clamp(1e-5, 1.0 - 1e-5), self.nc)


RFDETR_POS_SPECS: dict[str, tuple[int, int, int, str]] = {
    # cid -> (dr, aux_g, do_flip, prop_mode)
    "d0":              (0,   0, 0, "1g_last"),
    "sg2_r0_1g_traj":  (0,   0, 0, "1g"),
    "u2":              (32,  0, 0, "1g"),
    "d8":              (192, 0, 0, "13g_plain"),
    "rf39_r0_13g":     (0,   0, 0, "13g"),
    "rf39_r0_13g_g2":  (0,   1, 0, "13g"),
    "rf39_r32_13g":    (32,  0, 0, "13g"),
    "rf39_r64_13g":    (64,  0, 0, "13g"),
    "rf39_r64_13g_g2": (64,  1, 0, "13g"),
    "rf39_r96_13g":    (96,  0, 0, "13g"),
    "rf39_r96_13g_g2": (96,  1, 0, "13g"),
    "rf39_r128_13g":   (128, 0, 0, "13g"),
    "rf39_r128_13g_g2":(128, 1, 0, "13g"),
    "rf39_r128_13g_g3":(128, 2, 0, "13g"),
    "rf39_r160_13g":   (160, 0, 0, "13g"),
    "rf39_r160_13g_g2":(160, 1, 0, "13g"),
    "rf39_r160_13g_g3":(160, 2, 0, "13g"),
    "rf39_P_13g":      (192, 0, 0, "13g"),
    "rf39_P_13g_g2":   (192, 1, 0, "13g"),
    "rf39_P_13g_g3":   (192, 2, 0, "13g"),
    "rf39_P32_13g":    (224, 0, 0, "13g"),
    "rf39_r0_2b_13g":  (0,   0, 1, "13g"),
    "rf39_r64_2b_13g": (64,  0, 1, "13g"),
    "full_wsp":        (192, 1, 1, "13g"),
}


class RFDETRCoDesignModule(nn.Module):
    """Positive & Unconstrained Multi-Group WSP Graph Module for `RF-DETR` (`[B, 300, 6]` output)."""

    def __init__(self, basins: dict[str, Any], cand_id: str) -> None:
        super().__init__()
        enforce_no_gradients()
        self.cand_id = cand_id
        self.nat_res = int(basins["nat_res"])
        dr, self.aux_g, self.do_flip, self.prop_mode = RFDETR_POS_SPECS[cand_id]
        self.r_main = self.nat_res + dr
        m_cp = copy.deepcopy(basins["m_rf"]).eval()
        m_cp.backbone[0].encoder.shape = (self.r_main, self.r_main)
        _export_rfdetr_submodules(m_cp)
        self.mod = m_cp
        nc = int(m_cp.class_embed.out_features)
        self.nc = nc
        valid91 = torch.zeros(nc, dtype=torch.bool)
        for c in COCO80_TO_91:
            if c < nc:
                valid91[c] = True
        self.register_buffer("valid91", valid91, persistent=False)
        self.register_buffer("v91_flt", valid91.float().view(1, 1, nc), persistent=False)

    def _select_proposals(self, memory: torch.Tensor, sp_hw: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        from rfdetr.models.transformer import gen_encoder_output_proposals

        tr = self.mod.transformer
        out_mem, out_prop = gen_encoder_output_proposals(memory, None, sp_hw, unsigmoid=not tr.bbox_reparam)
        nq = int(tr.num_queries)
        if self.prop_mode in ("1g", "1g_last"):
            pn0 = tr.enc_output_norm[0](tr.enc_output[0](out_mem))
            ec0 = tr.enc_out_class_embed[0](pn0)
            ec_m = torch.where(self.valid91.view(1, 1, self.nc), ec0, torch.full_like(ec0, -100.0)) if self.cand_id == "sg2_r0_1g_traj" else ec0
            _, tidx = ec_m.max(-1)[0].topk(min(nq, int(ec0.shape[-2])), dim=1)
            prop_s = torch.gather(out_prop, 1, tidx.unsqueeze(-1).expand(-1, -1, 4))
            ecs_s = torch.gather(ec0, 1, tidx.unsqueeze(-1).expand(-1, -1, self.nc))
            d0 = tr.enc_out_bbox_embed[0](torch.gather(pn0, 1, tidx.unsqueeze(-1).expand(-1, -1, int(pn0.shape[-1]))))
            rts = torch.cat([d0[..., :2] * prop_s[..., 2:] + prop_s[..., :2], d0[..., 2:].exp() * prop_s[..., 2:]], dim=-1) if tr.bbox_reparam else (d0 + prop_s)
            return rts, ecs_s

        G = len(tr.enc_output)
        w0 = 0.45 if self.prop_mode == "13g_plain" else 0.44
        wr = (1.0 - w0) / float(max(1, G - 1))
        beta_cor = 0.0 if self.prop_mode == "13g_plain" else 0.06
        pn_list = [tr.enc_output_norm[g](tr.enc_output[g](out_mem)) for g in range(G)]
        z0 = tr.enc_out_class_embed[0](pn_list[0])
        ec = w0 * z0
        ag = torch.sigmoid(z0) if beta_cor > 0.0 else None
        for g in range(1, G):
            zg = tr.enc_out_class_embed[g](pn_list[g])
            ec = ec + wr * zg
            if ag is not None:
                ag = ag + torch.sigmoid(zg)
        if ag is not None:
            ec = ec + beta_cor * (ag / float(G))
        ec_m = torch.where(self.valid91.view(1, 1, self.nc), ec, torch.full_like(ec, -100.0))
        _, tidx = ec_m.max(-1)[0].topk(min(nq, int(ec.shape[-2])), dim=1)
        prop_s = torch.gather(out_prop, 1, tidx.unsqueeze(-1).expand(-1, -1, 4))
        ecs_s = torch.gather(ec, 1, tidx.unsqueeze(-1).expand(-1, -1, self.nc))
        idx_c = tidx.unsqueeze(-1).expand(-1, -1, int(pn_list[0].shape[-1]))
        d_ens = w0 * tr.enc_out_bbox_embed[0](torch.gather(pn_list[0], 1, idx_c))
        for g in range(1, G):
            d_ens = d_ens + wr * tr.enc_out_bbox_embed[g](torch.gather(pn_list[g], 1, idx_c))
        rts = torch.cat([d_ens[..., :2] * prop_s[..., 2:] + prop_s[..., :2], d_ens[..., 2:].exp() * prop_s[..., 2:]], dim=-1) if tr.bbox_reparam else (d_ens + prop_s)
        return rts, ecs_s

    def _decode_traj(self, pk, tgt_kb: torch.Tensor, ref_ts_kb: torch.Tensor, ref_e_kb: torch.Tensor, ecs_kb: torch.Tensor, last_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        tr, _, dec_mem, valid_ratios, lvl_pos, sp_t, sp_hw, lsi = pk
        KB = int(tgt_kb.shape[0])
        nq = int(tgt_kb.shape[1])
        dm = dec_mem if int(dec_mem.shape[0]) == KB else dec_mem[:1].expand(KB, -1, -1).contiguous()
        vr = valid_ratios if int(valid_ratios.shape[0]) == KB else valid_ratios[:1].expand(KB, -1, -1).contiguous()
        lp = lvl_pos if int(lvl_pos.shape[0]) == KB else lvl_pos[:1].expand(KB, -1, -1).contiguous()
        ref_in = torch.cat([ref_e_kb[..., :2] * ref_ts_kb[..., 2:] + ref_ts_kb[..., :2], ref_e_kb[..., 2:].exp() * ref_ts_kb[..., 2:]], dim=-1) if tr.bbox_reparam else (ref_e_kb + ref_ts_kb)
        tgt = tgt_kb
        if tr.num_registers > 0:
            tgt = torch.cat([tgt, tr.register_tokens.unsqueeze(0).expand(KB, -1, -1)], dim=1)
            ref_in = torch.cat([ref_in, tr.register_ref_points.unsqueeze(0).expand(KB, -1, -1)], dim=1)
        hs, ref = tr.decoder(tgt, dm, memory_key_padding_mask=None, pos=lp, refpoints_unsigmoid=ref_in, level_start_index=lsi, spatial_shapes=sp_t, spatial_shapes_hw=sp_hw, valid_ratios=vr)[:2]
        if hs.ndim == 3:
            hs, ref = hs.unsqueeze(0), ref.unsqueeze(0)
        if tr.num_registers > 0:
            hs, ref = hs[:, :, :nq, :], ref[:, :, :nq, :]
        coord = torch.cat([self.mod.bbox_embed(hs)[..., :2] * ref[..., 2:] + ref[..., :2], self.mod.bbox_embed(hs)[..., 2:].exp() * ref[..., 2:]], dim=-1) if self.mod.bbox_reparam else (self.mod.bbox_embed(hs) + ref).sigmoid()
        cls_l = self.mod.class_embed(hs)
        if last_only:
            return cls_l[-1].sigmoid() * self.v91_flt, coord[-1]
        n_l = int(cls_l.shape[0])
        if n_l >= 3:
            z_tr = 0.87 * cls_l[-1] + 0.05 * cls_l[-2] + 0.02 * cls_l[-3] + 0.06 * ecs_kb
            b_tr = 0.94 * coord[-1] + 0.045 * coord[-2] + 0.015 * coord[-3]
        elif n_l >= 2:
            z_tr = 0.89 * cls_l[-1] + 0.05 * cls_l[-2] + 0.06 * ecs_kb
            b_tr = 0.95 * coord[-1] + 0.05 * coord[-2]
        else:
            z_tr = 0.94 * cls_l[-1] + 0.06 * ecs_kb
            b_tr = coord[-1]
        return z_tr.sigmoid() * self.v91_flt, b_tr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rm = self.r_main
        xm = F.interpolate(x, size=(rm, rm), mode="bilinear", align_corners=False) if (x.shape[-2] != rm or x.shape[-1] != rm) else x
        B = int(xm.shape[0])
        nq = int(self.mod.num_queries)
        tr = self.mod.transformer

        x_in = torch.cat([xm, torch.flip(xm, dims=[-1])], dim=0) if self.do_flip else xm
        pk = _rfdetr_encode_with_bb(self.mod.backbone, tr, x_in)
        rts, ecs = self._select_proposals(pk[1][:B], pk[6])
        tgt0 = self.mod.query_feat.weight[:nq].unsqueeze(0).expand(B, -1, -1).contiguous()
        ref_e0 = self.mod.refpoint_embed.weight[:nq].unsqueeze(0).expand(B, -1, -1).contiguous()

        if self.prop_mode == "1g_last":
            p0, b0 = self._decode_traj(pk, tgt0, rts, ref_e0, ecs, last_only=True)
            return detr_topk300_preds6(p0.clamp(0.0, 1.0 - 1e-5), b0.clamp(1e-5, 1.0 - 1e-5), self.nc)

        if self.do_flip:
            ref_e_fl = torch.cat([-ref_e0[..., 0:1], ref_e0[..., 1:]], dim=-1)
            rts_fl = torch.cat([1.0 - rts[..., 0:1], rts[..., 1:]], dim=-1)
            p2, b2 = self._decode_traj(pk, torch.cat([tgt0, tgt0], dim=0), torch.cat([rts, rts_fl], dim=0), torch.cat([ref_e0, ref_e_fl], dim=0), torch.cat([ecs, ecs], dim=0))
            p0, b0 = p2[:B], b2[:B]
            pfl = p2[B:]
            bfl = torch.cat([1.0 - b2[B:, :, 0:1], b2[B:, :, 1:]], dim=-1)
        else:
            p0, b0 = self._decode_traj(pk, tgt0, rts, ref_e0, ecs)
            pfl = bfl = None

        p_num, p_den = p0, torch.ones_like(p0[..., :1])
        b_num, b_den = b0, torch.ones_like(b0[..., :1])
        cor = torch.zeros_like(p0[..., :1])

        if pfl is not None and bfl is not None:
            iou = trt_safe_iou_1to1_cxcywh(b0, bfl).unsqueeze(-1)
            wp = torch.where(iou >= 0.55, torch.full_like(iou, 0.34), torch.zeros_like(iou))
            wb = torch.where(iou >= 0.80, (0.85 * 0.34) * ((iou / 0.88).clamp(0.5, 1.15) ** 2.0), torch.zeros_like(iou))
            p_num, p_den = p_num + wp * pfl, p_den + wp
            b_num, b_den = b_num + wb * bfl, b_den + wb
            cor = cor + (iou >= 0.72).to(p0.dtype)

        if self.aux_g > 0:
            pk_orig = (pk[0], pk[1][:B], pk[2][:B], pk[3][:B], pk[4][:B] if int(pk[4].shape[0]) > 1 else pk[4], pk[5], pk[6], pk[7])
            pg_l, bg_l = [], []
            max_g = max(0, (int(self.mod.query_feat.weight.shape[0]) // nq) - 1)
            for g_i in range(1, 1 + min(self.aux_g, max_g)):
                tg = self.mod.query_feat.weight[g_i * nq : (g_i + 1) * nq].unsqueeze(0).expand(B, -1, -1).contiguous()
                rg = self.mod.refpoint_embed.weight[g_i * nq : (g_i + 1) * nq].unsqueeze(0).expand(B, -1, -1).contiguous()
                pg_i, bg_i = self._decode_traj(pk_orig, tg, rts, rg, ecs)
                pg_l.append(pg_i); bg_l.append(bg_i)
            if pg_l:
                pg, bg = torch.cat(pg_l, dim=1), torch.cat(bg_l, dim=1)
                pw_iou = trt_safe_iou_pairwise_cxcywh(b0, bg)
                best_iou, best_j = pw_iou.max(dim=-1)
                pg_m = torch.gather(pg, 1, best_j.unsqueeze(-1).expand(-1, -1, self.nc))
                bg_m = torch.gather(bg, 1, best_j.unsqueeze(-1).expand(-1, -1, 4))
                bi_u = best_iou.unsqueeze(-1)
                wp_g = torch.where(bi_u >= 0.65, torch.full_like(bi_u, 0.14), torch.zeros_like(bi_u))
                wb_g = torch.where(bi_u >= 0.86, 0.04 * ((bi_u / 0.90).clamp(0.6, 1.15) ** 2.0), torch.zeros_like(bi_u))
                p_num, p_den = p_num + wp_g * pg_m, p_den + wp_g
                b_num, b_den = b_num + wb_g * bg_m, b_den + wb_g
                cor = cor + 0.45 * (bi_u >= 0.75).to(p0.dtype)

        pf = ((p_num / p_den) * (1.0 + 0.0045 * cor.clamp(max=6.0))).clamp(0.0, 1.0 - 1e-5)
        bf = (b_num / b_den).clamp(1e-5, 1.0 - 1e-5)
        return detr_topk300_preds6(pf, bf, self.nc)


# ==============================================================================
# 4. UNIFIED LOADER, IMAGE PREDICTOR, AND LIVE TARGET-GPU BUDGET SOLVER
# ==============================================================================
def _needed_subspaces_for_cand(family: str, cand_id: str) -> list[str]:
    if cand_id == "n_null":
        return []
    if family == "yolo26":
        if cand_id in ("d0", "n00") or cand_id.startswith("n"):
            return ["m_d0"]
        cid = YOLO_DYN_TO_SG_EQUIV.get(cand_id, cand_id)
        if cid == "y39_hd4_sm120":
            return ["m_ext_pure", "m_hd_int_pure"]
        if cid == "y39_shbb_gw":
            return ["m_ext_pure", "m_shbb_pure"]
        if cid in ("y39_fov352_gw", "y39_fl384_gw", "y39_fl448_gw", "y39_fl512_gw"):
            return ["m_ext_pure", "m_int_pure"]
        if cid == "full_wsp":
            return ["m_ext_pure", "m_hd_int_pure", "m_shbb_pure", "m_int_pure"]
        return ["m_ext_pure"]

    if family == "edgecrafter":
        if cand_id in ("d0", "n00"):
            return ["m_d0"]
        if cand_id.startswith("n") or cand_id in ("u1", "d2", "sg1_s512", "ec39_fl384_2s", "ec39_fl448_2s", "ec39_fl512_2s", "ec39_2b640_2s"):
            return ["m_pri"]
        if cand_id in ("d5", "ec39_shenc_sm120", "ec39_fl384_shenc", "ec39_fl448_shenc", "ec39_fl512_shenc", "ec39_2b640_shenc", "ec39_hi704_shenc"):
            return ["m_pri", "m_int_shenc"]
        if cand_id == "d7":
            return ["m_pri", "m_int_shbb"]
        if cand_id == "d8":
            return ["m_pri", "m_int_shbb", "m_mid_shenc"]
        if cand_id in ("ec39_shbb_locsafe_sm120", "ec39_fl448_shbb", "ec39_fl512_shbb", "ec39_2b640_shbb"):
            return ["m_pri", "m_int_shenc", "m_int_shbb"]
        return ["m_pri", "m_int_shenc", "m_mid_shenc", "m_int_shbb", "m_mid_shbb"]

    return ["m_rf"]


def load_basins_dict(
    spec: DetectorSpec,
    basins_path: Path,
    device: torch.device,
    needed_subspaces: list[str] | None = None,
    edgecrafter_root: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Loads a `v0.1.0` `*-odwsp-basins.pt` pack strictly via `weights_only=True` and instantiates requested subspaces."""
    pkg = safe_load_checkpoint(basins_path, map_location="cpu", expected_sha256=expected_sha256)
    subs_raw: dict[str, dict[str, torch.Tensor]] = pkg["subspaces"]
    meta: dict[str, Any] = dict(pkg.get("metadata", {}))
    out: dict[str, Any] = dict(meta)
    names = needed_subspaces if needed_subspaces is not None else list(subs_raw.keys())
    arch_y = make_yolo26_arch_cfg(spec.scale) if spec.family == "yolo26" else None
    for s_name in names:
        if s_name not in subs_raw:
            raise KeyError(f"Subspace '{s_name}' not found in {basins_path.name}; available: {sorted(subs_raw)}")
        mod, _ = instantiate_subspace_model(spec, subs_raw[s_name], device=device, edgecrafter_root=edgecrafter_root, arch_yaml=arch_y)
        out[s_name] = mod
    return out


def _instantiate_empty_subspace(spec: DetectorSpec, fuse_ecdet: bool = False, edgecrafter_root: Path | None = None) -> nn.Module:
    """Instantiates an uninitialized host architecture skeleton on CPU (`0` checkpoint files needed)."""
    if spec.family == "yolo26":
        from ultralytics.nn.tasks import DetectionModel

        m = DetectionModel(cfg=make_yolo26_arch_cfg(spec.scale), ch=3, nc=80, verbose=False)
        m.end2end = True
        m.model[-1].end2end = True
        return m.eval().requires_grad_(False)
    if spec.family == "rfdetr":
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        cls_map = {"n": RFDETRNano, "s": RFDETRSmall, "m": RFDETRMedium, "l": RFDETRLarge}
        return cls_map[spec.scale](pretrain_weights=None, num_classes=90, device="cpu").model.model.eval().requires_grad_(False)

    if edgecrafter_root is None:
        raise ValueError("EdgeCrafter models require `--edgecrafter-root /path/to/EdgeCrafter`.")
    import sys

    ec_resolved = Path(edgecrafter_root).resolve()
    if (ec_resolved / "ecdetseg").is_dir():
        ec_resolved = ec_resolved / "ecdetseg"
    if str(ec_resolved) not in sys.path:
        sys.path.insert(0, str(ec_resolved))
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import load_config as _ec_load_config

    if _ec_load_config.__defaults__ and isinstance(_ec_load_config.__defaults__[0], dict):
        _ec_load_config.__defaults__[0].clear()
    assert spec.ecdet_cfg_rel is not None
    cfg = YAMLConfig(str(ec_resolved / spec.ecdet_cfg_rel))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    m_ec = cfg.model.eval().requires_grad_(False)
    if fuse_ecdet:
        _prepare_ecdet_deploy(m_ec, 640, 640)
    return m_ec


def _infer_dummy_bridges_from_deploy_sd(spec: DetectorSpec, cid: str, sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Reconstructs conformable placeholder bridge tensors from `state_dict_fp16` shapes prior to `load_state_dict`."""
    if not (cid.startswith("n") and cid != "n00"):
        return {}
    if spec.family == "yolo26":
        br: dict[str, Any] = {}
        if "psa_fold.weight" in sd:
            br["W_psa"] = sd["psa_fold.weight"].float().squeeze(-1).squeeze(-1)
            br["b_psa"] = sd["psa_fold.bias"].float()
            br["W_pan19"] = sd["pan19_br.weight"].float().squeeze(-1).squeeze(-1)
            br["b_pan19"] = sd["pan19_br.bias"].float()
            br["W_pan22"] = sd["pan22_br.weight"].float().squeeze(-1).squeeze(-1)
            br["b_pan22"] = sd["pan22_br.bias"].float()
            br["pan_cal_cv2"] = {l: (sd[f"o2o_cv2.{l}.2.weight"].float(), sd[f"o2o_cv2.{l}.2.bias"].float()) for l in (1, 2)}
            br["pan_cal_cv3"] = {l: (sd[f"o2o_cv3.{l}.2.weight"].float(), sd[f"o2o_cv3.{l}.2.bias"].float()) for l in (1, 2)}
        if "bb6_br.weight" in sd:
            br["W_bb6_19"] = sd["bb6_br.weight"].float().squeeze(-1).squeeze(-1)
            br["b_bb6_19"] = sd["bb6_br.bias"].float()
            br["bb6_cal_cv2"] = (sd["cv2_s16.2.weight"].float(), sd["cv2_s16.2.bias"].float())
            br["bb6_cal_cv3"] = (sd["cv3_s16.2.weight"].float(), sd["cv3_s16.2.bias"].float())
        if "c3_fold.weight" in sd:
            br["w3_fold"], br["b3_orig"] = sd["c3_fold.weight"].float(), sd["c3_fold.bias"].float()
            br["w5_fold"], br["b5_orig"] = sd["c5_fold.weight"].float(), sd["c5_fold.bias"].float()
            br["W_s5_19"], br["b_s5_19"] = sd["s5_br.weight"].float().squeeze(-1).squeeze(-1), sd["s5_br.bias"].float()
            br["s5_cal_cv2"] = (sd["cv2_s16.2.weight"].float(), sd["cv2_s16.2.bias"].float())
            br["s5_cal_cv3"] = (sd["cv3_s16.2.weight"].float(), sd["cv3_s16.2.bias"].float())
        if "s0.weight" in sd:
            br["w_stem3_c0"], br["b_s4_0"] = sd["s0.weight"].float(), sd["s0.bias"].float()
            br["w_stem3_c1"], br["b_s3_1"] = sd["s1.weight"].float(), sd["s1.bias"].float()
            br["W_s5_19"], br["b_s5_19"] = sd["s3_br.weight"].float().squeeze(-1).squeeze(-1), sd["s3_br.bias"].float()
            br["s3_cal_cv2"] = (sd["cv2_s16.2.weight"].float(), sd["cv2_s16.2.bias"].float())
            br["s3_cal_cv3"] = (sd["cv3_s16.2.weight"].float(), sd["cv3_s16.2.bias"].float())
        return br
    if spec.family == "edgecrafter":
        kv = int(ECDET_NEG_SPECS[cid]["k_vit"])
        br_ec: dict[str, Any] = {}
        if "vit_br.weight" in sd:
            br_ec["vit_enc_bridges"] = {kv: (sd["vit_br.weight"].float(), sd["vit_br.bias"].float())}
        if "box_res_lin.weight" in sd:
            br_ec["W_enc_folded"] = sd["dec.enc_score_head.weight"].float()
            br_ec["b_enc_folded"] = sd["dec.enc_score_head.bias"].float()
            br_ec["W_b0"] = sd["box_res_lin.weight"].float()
            br_ec["b_b0"] = sd["box_res_lin.bias"].float()
        if "head84.weight" in sd:
            br_ec["vit_byp_heads"] = {kv: (sd["head84.weight"].float(), sd["head84.bias"].float())}
        return br_ec
    br_rf: dict[str, Any] = {}
    if "mod.transformer.enc_out_class_embed.0.weight" in sd:
        w_ec = sd["mod.transformer.enc_out_class_embed.0.weight"].float()
        b_ec = sd["mod.transformer.enc_out_class_embed.0.bias"].float()
        br_rf["W_13c"], br_rf["b_13c"] = w_ec, b_ec
        br_rf["W_0dec_cls"], br_rf["b_0dec_cls"] = w_ec, b_ec
    if "box_res_lin.weight" in sd:
        br_rf["W_db"], br_rf["b_db"] = sd["box_res_lin.weight"].float(), sd["box_res_lin.bias"].float()
    if "mod.backbone.0.projector.stages.0.0.cv1.conv.weight" in sd:
        w_cv = sd["mod.backbone.0.projector.stages.0.0.cv1.conv.weight"].float()
        br_rf["w_cv1_k9"] = br_rf["w_cv1_k6"] = br_rf["w_cv1_k3"] = w_cv
    if "dir_head.weight" in sd:
        for k_d in (0, 1, 2):
            br_rf.setdefault("dir_heads", {})[k_d] = (sd["dir_head.weight"].float(), sd["dir_head.bias"].float())
    return br_rf


def build_continuum_module(
    spec: DetectorSpec,
    cand_id: str,
    basins: dict[str, Any],
    bridges: dict[str, Any] | None = None,
) -> tuple[nn.Module, int]:
    """Instantiates the executable `[B, 300, 6]` `nn.Module` and native input resolution for `cand_id`."""
    enforce_no_gradients()
    if cand_id == "n_null":
        return NullZeroBudgetModule(), spec.resolution

    if cand_id.startswith("n"):
        if bridges is None and cand_id != "n00":
            raise ValueError(
                f"Negative-budget candidate '{cand_id}' requires `--bridges <odwsp-contraction-bridges-all17.tar.gz>`."
            )
        br = bridges or {}
        if spec.family == "yolo26":
            mod = YOLONegGraphModule(basins, br, cand_id, eval_resize=True)
            return mod, int(YOLO_NEG_SPECS[cand_id]["res"])
        if spec.family == "edgecrafter":
            mod = ECDetNegGraphModule(basins, br, cand_id)
            return mod, int(ECDET_NEG_SPECS[cand_id]["res"])
        mod = RFDETRNegGraphModule(basins, br, cand_id)
        return mod, int(mod.res)

    if spec.family == "yolo26":
        return YOLOCoDesignModule(basins, cand_id), 640
    if spec.family == "edgecrafter":
        return ECDetCoDesignModule(basins, cand_id), 640
    mod_rf = RFDETRCoDesignModule(basins, cand_id)
    return mod_rf, int(mod_rf.r_main)


@dataclass
class LoadedContinuumDetector:
    """Drop-in multi-budget detector compatible with `val.py`, `detect.py`, and `deploy.py` (`0` external NMS)."""

    spec: DetectorSpec
    model: nn.Module
    device: torch.device
    input_res: int
    budget_info: dict[str, Any]
    metadata: dict[str, Any]

    def predict_image(self, img_path: Path, conf_thr: float = 0.25) -> list[dict[str, Any]]:
        enforce_no_gradients()
        if isinstance(self.model, NullZeroBudgetModule):
            return []

        param_dtype = next((p.dtype for p in self.model.parameters()), torch.float32)
        if self.spec.family == "yolo26":
            res = 640
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if im is None:
                with Image.open(img_path) as pil_im:
                    im = np.array(pil_im.convert("RGB"), copy=True)[..., ::-1]
            h0, w0 = im.shape[:2]
            r = res / max(h0, w0)
            if r != 1.0:
                w, h = min(math.ceil(w0 * r), res), min(math.ceil(h0 * r), res)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                h, w = h0, w0
            r_lb = min(res / h, res / w, 1.0)
            new_unpad = int(round(w * r_lb)), int(round(h * r_lb))
            dw, dh = (res - new_unpad[0]) / 2.0, (res - new_unpad[1]) / 2.0
            if (w, h) != new_unpad:
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im_pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            im_rgb = im_pad[..., ::-1]
            t = torch.from_numpy(np.ascontiguousarray(im_rgb.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(device=self.device, dtype=param_dtype)
            with torch.no_grad():
                out = self.model(t)
                y = (out[0] if isinstance(out, tuple) else out)[0].detach().float().cpu().numpy()
            scale = r * r_lb
            dets: list[dict[str, Any]] = []
            for row in y:
                sc = float(row[4])
                if sc < conf_thr:
                    continue
                c_idx = int(round(float(row[5])))
                if not (0 <= c_idx < 80):
                    continue
                x1 = float(np.clip((row[0] - left) / scale, 0.0, w0))
                y1 = float(np.clip((row[1] - top) / scale, 0.0, h0))
                x2 = float(np.clip((row[2] - left) / scale, 0.0, w0))
                y2 = float(np.clip((row[3] - top) / scale, 0.0, h0))
                dets.append({"bbox_xyxy": [x1, y1, x2, y2], "score": sc, "category_id": COCO80_TO_91[c_idx], "label": COCO80_NAMES[c_idx]})
            return dets

        import torchvision.transforms.v2.functional as F_v2
        from torchvision.transforms import InterpolationMode

        with Image.open(img_path) as pil_im:
            pil_rgb = pil_im.convert("RGB")
        orig_w, orig_h = pil_rgb.size
        r_in = self.input_res
        im_r = F_v2.resize(pil_rgb, [r_in, r_in], interpolation=InterpolationMode.BILINEAR, antialias=True)
        t = F_v2.pil_to_tensor(im_r).float().div_(255.0)
        t = F_v2.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device=self.device, dtype=param_dtype)
        with torch.no_grad():
            y = self.model(t)[0].detach().float().cpu().numpy()

        dets = []
        for row in y:
            sc = float(row[4])
            if sc < conf_thr:
                continue
            raw_cls = int(round(float(row[5])))
            if self.spec.family == "edgecrafter":
                if not (0 <= raw_cls < 80):
                    continue
                cid = COCO80_TO_91[raw_cls]
                lbl = COCO80_NAMES[raw_cls]
            else:
                cid = raw_cls
                if cid not in COCO91_TO_NAME:
                    continue
                lbl = COCO91_TO_NAME[cid]
            x1 = float(np.clip(row[0] * orig_w, 0.0, orig_w))
            y1 = float(np.clip(row[1] * orig_h, 0.0, orig_h))
            x2 = float(np.clip(row[2] * orig_w, 0.0, orig_w))
            y2 = float(np.clip(row[3] * orig_h, 0.0, orig_h))
            dets.append({"bbox_xyxy": [x1, y1, x2, y2], "score": sc, "category_id": cid, "label": lbl})
        return dets


def load_continuum_detector(
    model_key: str,
    budget: str | float | int = "0%",
    paradigm: str = "auto",
    batch_size: int = 1,
    cudagraph: bool = True,
    weights_path: Path | None = None,
    basins_path: Path | None = None,
    bridges_path: Path | None = None,
    deploy_ckpt_path: Path | None = None,
    cand_override: str | None = None,
    device: torch.device | str = "cuda:0",
    edgecrafter_root: Path | None = None,
    expected_sha256: str | None = None,
) -> LoadedDetector | LoadedContinuumDetector:
    """
    Unified multi-budget WSP detector loader (`beta in [-100%, +100%]` or `full`).
      - At `budget="0%"` with `--weights`: delegates directly to `load_odwsp_detector` (`1.00x` folded checkpoint).
      - At any negative / zero / positive / `full` budget with `--basins` (and `--bridges` for `< 0%`)
        or `--deploy-ckpt`: constructs the exact zero-NMS `[B, 300, 6]` WSP inference graph (`weights_only=True`).
    """
    enforce_no_gradients()
    key = normalize_model_key(model_key)
    spec = PAPER_MODELS[key]
    dev = torch.device(device)
    b_cfg = resolve_budget_config(key, budget=budget, paradigm=paradigm, batch_size=batch_size, cudagraph=cudagraph)
    cid = cand_override or str(b_cfg["cand_id"])

    if b_cfg["regime"] == "zero" and cand_override is None and weights_path is not None and basins_path is None and deploy_ckpt_path is None:
        return load_odwsp_detector(
            model_key=key,
            weights_path=weights_path,
            device=dev,
            edgecrafter_root=edgecrafter_root,
            expected_sha256=expected_sha256,
        )

    if cid == "n_null":
        return LoadedContinuumDetector(
            spec=spec,
            model=NullZeroBudgetModule().to(dev).eval(),
            device=dev,
            input_res=spec.resolution,
            budget_info=b_cfg,
            metadata={"cand_id": "n_null", "weights_only_verified": True},
        )

    if basins_path is None and deploy_ckpt_path is None:
        assets = b_cfg["assets_v0_1_0"]
        if b_cfg["regime"] == "zero":
            raise ValueError(
                f"For `--budget 0%`, provide `--weights ./{assets['folded_1x']}` (or `--basins ./{assets['basins']}`)."
            )
        if b_cfg["regime"] == "negative":
            raise ValueError(
                f"For negative budget `{budget}`, provide `--basins ./{assets['basins']} "
                f"--bridges ./{assets['bridges_archive']}` (or `--deploy-ckpt ./{assets['deploy_archive']}`)."
            )
        raise ValueError(
            f"For budget `{budget}`, provide `--basins ./{assets['basins']}` "
            f"(or `--deploy-ckpt ./{assets['deploy_archive']}`)."
        )

    bridges_dict: dict[str, Any] | None = None
    if cid.startswith("n") and cid != "n00" and basins_path is not None:
        if bridges_path is None:
            raise ValueError(
                f"Negative-budget candidate '{cid}' (`{budget}`) requires `--bridges ./odwsp-contraction-bridges-all17.tar.gz` "
                f"(or extracted `./{b_cfg['assets_v0_1_0']['bridges_member']}`)."
            )
        raw_br = _safe_load_from_path_or_tar(bridges_path, b_cfg["assets_v0_1_0"]["bridges_member"])
        bridges_dict = raw_br["bridges"] if "bridges" in raw_br and isinstance(raw_br["bridges"], dict) else raw_br

    if basins_path is not None:
        needed = _needed_subspaces_for_cand(spec.family, "d0" if cid == "d0_gated" else cid)
        basins_dict = load_basins_dict(
            spec=spec,
            basins_path=basins_path,
            device=torch.device("cpu"),
            needed_subspaces=needed,
            edgecrafter_root=edgecrafter_root,
            expected_sha256=expected_sha256,
        )
        mod, in_res = build_continuum_module(spec, "d0" if cid == "d0_gated" else cid, basins_dict, bridges_dict)
        if deploy_ckpt_path is not None:
            dep_pkg = _safe_load_from_path_or_tar(deploy_ckpt_path, f"{key}__{cid}_deploy.pt")
            sd_fp32 = {k: v.float() for k, v in dep_pkg["state_dict_fp16"].items()}
            mod.load_state_dict(sd_fp32, strict=False)
    else:
        assert deploy_ckpt_path is not None
        dep_pkg = _safe_load_from_path_or_tar(deploy_ckpt_path, f"{key}__{cid}_deploy.pt", expected_sha256=expected_sha256)
        if "cand_id" in dep_pkg and cand_override is None and deploy_ckpt_path.suffix == ".pt":
            cid = str(dep_pkg["cand_id"])
        dep_sd: dict[str, torch.Tensor] = dep_pkg["state_dict_fp16"]
        sd_fp32 = {k: v.float() for k, v in dep_sd.items()}
        if cid == "d0_gated":
            from .models import YOLOGated1xModule

            m_empty = _instantiate_empty_subspace(spec)
            mod = YOLOGated1xModule(m_empty, {"beta_box": 0.30, "gamma_dn": 0.08, "gamma_up": 0.20, "tau_iou": 0.55, "tau_dom": 0.35, "tau_t": 1.0})
            in_res = 640
        else:
            eff_cid = "d0" if cid == "d0_gated" else cid
            needed = _needed_subspaces_for_cand(spec.family, eff_cid)
            is_small = spec.scale in ("n", "s", "m")
            skel_basins: dict[str, Any] = {
                "sc": spec.scale,
                "nat_res": spec.resolution,
                "gamma_cls": 0.48 if is_small else 0.39,
                "dup_decay": 0.03 if is_small else 0.09,
                "eta_box": 0.58 if is_small else 0.32,
                "alpha_cor": 0.012 if "o365" in spec.model_key else 0.015,
            }
            fuse_ec = any("conv_bn_fused" in k for k in sd_fp32)
            for s_nm in needed:
                skel_basins[s_nm] = _instantiate_empty_subspace(spec, fuse_ecdet=fuse_ec, edgecrafter_root=edgecrafter_root)
            dummy_br = _infer_dummy_bridges_from_deploy_sd(spec, eff_cid, sd_fp32)
            mod, in_res = build_continuum_module(spec, eff_cid, skel_basins, dummy_br)
        missing_k, _ = mod.load_state_dict(sd_fp32, strict=False)
        if missing_k:
            raise RuntimeError(f"Missing keys loading standalone deploy checkpoint `{cid}`: {missing_k[:5]}")

    mod = mod.to(dev).eval().requires_grad_(False)
    return LoadedContinuumDetector(
        spec=spec,
        model=mod,
        device=dev,
        input_res=in_res,
        budget_info=b_cfg,
        metadata={"cand_id": cid, "input_res": in_res, "weights_only_verified": True},
    )


def solve_budget_on_target_gpu(
    model_key: str,
    budget: str | float | int,
    basins_path: Path,
    bridges_path: Path | None = None,
    paradigm: str = "auto",
    batch_size: int = 1,
    fp16: bool = True,
    cudagraph: bool = True,
    device: torch.device | str = "cuda:0",
    edgecrafter_root: Path | None = None,
) -> dict[str, Any]:
    """
    Live Hardware-Adaptive Latency-Budget WSP Solver on your target GPU (`Section 4.5`):
      1. Measures baseline `1.00x` latency `T_0` (`d0`) directly on `device`.
      2. Benchmarks candidate WSP graphs `g` in the target regime (`beta < 0%` or `beta >= 0%`).
      3. Enforces `T(g) <= (1 + beta) * T_0` and selects `g*(beta) = argmax_{feasible g} AP_hold(g)`.
    """
    enforce_no_gradients()
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("`--solve-on-gpu` requires a CUDA GPU device.")

    key = normalize_model_key(model_key)
    spec = PAPER_MODELS[key]
    tbl = get_continuum_table()["models"][key]
    regime, num_pct, tier_key = parse_budget_spec(budget)
    beta_frac = 0.0 if num_pct is None else (float(num_pct) / 100.0)
    dtype = torch.float16 if fp16 else torch.float32

    def _bench_cand(mod: nn.Module, res: int) -> float:
        m_run = mod.to(dev).eval()
        m_run = m_run.half() if dtype == torch.float16 else m_run.float()
        x_b = torch.rand((batch_size, 3, res, res), device=dev, dtype=dtype)
        if cudagraph:
            g_cuda, _, _ = capture_static_cudagraph(lambda z: m_run(z), x_b, warmup_iters=4)
            ms = benchmark_cuda_callable(g_cuda.replay, dev, warmup=10, iters=40, batch_size=batch_size)
        else:
            ms = benchmark_cuda_callable(lambda: m_run(x_b), dev, warmup=8, iters=30, batch_size=batch_size)
        del m_run, x_b
        torch.cuda.empty_cache()
        return ms

    b_d0 = load_basins_dict(spec, basins_path, torch.device("cpu"), needed_subspaces=_needed_subspaces_for_cand(spec.family, "d0"), edgecrafter_root=edgecrafter_root)
    m_d0, r_d0 = build_continuum_module(spec, "d0", b_d0)
    t0_ms = _bench_cand(m_d0, r_d0)
    cap_ms = t0_ms * max(0.0, 1.0 + beta_frac) if regime != "unconstrained" else float("inf")

    if regime == "negative":
        if beta_frac <= -1.0:
            return {"model_key": key, "budget": str(budget), "t0_ms": round(t0_ms, 4), "cap_ms": 0.0, "selected_cand_id": "n_null", "measured_ms": 0.0, "rel_pct": -100.0}
        if bridges_path is None:
            raise ValueError("Negative-budget GPU solving requires `--bridges ./odwsp-contraction-bridges-all17.tar.gz`.")
        raw_br = _safe_load_from_path_or_tar(bridges_path, tbl["assets_v0_1_0"]["bridges_member"])
        br_dict = raw_br["bridges"] if "bridges" in raw_br and isinstance(raw_br["bridges"], dict) else raw_br
        b_neg = load_basins_dict(spec, basins_path, torch.device("cpu"), needed_subspaces=_needed_subspaces_for_cand(spec.family, "n01"), edgecrafter_root=edgecrafter_root)
        neg_pool = tbl["candidate_pools"]["negative"]
        cands_to_test = [c for c in sorted(neg_pool.keys()) if c not in ("n00", "n_null")]
        measured_rows = []
        for cid in cands_to_test:
            mc, rc = build_continuum_module(spec, cid, b_neg, br_dict)
            t_ms = _bench_cand(mc, rc)
            rel = (t_ms - t0_ms) / max(t0_ms, 1e-6) * 100.0
            measured_rows.append({
                "cand_id": cid,
                "input_res": rc,
                "measured_ms": round(t_ms, 4),
                "rel_pct": round(rel, 2),
                "feasible": bool(t_ms <= cap_ms * 1.002),
                "holdout_AP": neg_pool[cid]["holdout_AP"],
                "paper_val2017_AP": neg_pool[cid]["val2017_AP"],
            })
        feasible = [r for r in measured_rows if r["feasible"]]
        best = max(feasible, key=lambda r: r["holdout_AP"]) if feasible else min(measured_rows, key=lambda r: r["measured_ms"])
        return {
            "model_key": key,
            "budget": str(budget),
            "resolved_tier": tier_key,
            "t0_ms_per_img": round(t0_ms, 4),
            "budget_cap_ms_per_img": round(cap_ms, 4),
            "selected": best,
            "num_candidates_profiled": len(measured_rows),
            "num_feasible": len(feasible),
            "profiled_candidates": measured_rows,
        }

    pos_pool = tbl["candidate_pools"]["positive"]
    par = paradigm.strip().lower()
    cands_pos = sorted(pos_pool.keys())
    if par == "sg":
        cands_pos = [c for c in cands_pos if not c.startswith("dyn_")]
    elif par == "dyn" and spec.family == "yolo26":
        cands_pos = [c for c in cands_pos if c == "d0" or c.startswith("dyn_") or c in ("y39_fhd2_sm120", "y39_2b640_gw")]

    all_needed: list[str] = []
    for c in cands_pos:
        for s_nm in _needed_subspaces_for_cand(spec.family, c):
            if s_nm not in all_needed:
                all_needed.append(s_nm)
    b_pos = load_basins_dict(spec, basins_path, torch.device("cpu"), needed_subspaces=all_needed, edgecrafter_root=edgecrafter_root)

    measured_pos = []
    for cid in cands_pos:
        t_ms = t0_ms if cid == "d0" else _bench_cand(*build_continuum_module(spec, cid, b_pos))
        rel = (t_ms - t0_ms) / max(t0_ms, 1e-6) * 100.0
        measured_pos.append({
            "cand_id": cid,
            "measured_ms": round(t_ms, 4),
            "rel_pct": round(rel, 2),
            "feasible": bool(t_ms <= cap_ms * 1.002),
            "holdout_AP": pos_pool[cid]["holdout_AP"],
            "paper_val2017_AP": pos_pool[cid]["val2017_AP"],
        })
    feasible_pos = [r for r in measured_pos if r["feasible"]]
    best_pos = max(feasible_pos, key=lambda r: r["holdout_AP"]) if feasible_pos else measured_pos[0]
    return {
        "model_key": key,
        "budget": str(budget),
        "paradigm": par,
        "t0_ms_per_img": round(t0_ms, 4),
        "budget_cap_ms_per_img": round(cap_ms, 4) if math.isfinite(cap_ms) else "unconstrained",
        "selected": best_pos,
        "num_candidates_profiled": len(measured_pos),
        "num_feasible": len(feasible_pos),
        "profiled_candidates": measured_pos,
    }
