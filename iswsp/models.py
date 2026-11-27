"""
Safe (`weights_only=True`) Instance Segmenter Loaders, GPU/CPU COCO RLE Encoding,
and Unified Zero-Gradient Inference Pipeline across all 19 Official IR-ISWSP Models.

Supports:
  - YOLO26-Seg (5 scales: N, S, M, L, X — native end2end=True CNN + Proto26)
  - EdgeCrafter-Seg / ECSeg (4 Pure-COCO + 4 O365->COCO: S, M, L, X)
  - RF-DETR-Seg (6 scales: N, S, M, L, XL, 2XL — DINOv2 windowed ViT)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .consensus import apply_single_pass_box_mask_synergy
from .zero_grad_guard import enforce_no_gradients

COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

COCO91_TO_NAME = {cid: nm for cid, nm in zip(COCO80_TO_91, COCO80_NAMES)}

YOLO26_SEG_BASE_CFG: dict[str, Any] = {
    "nc": 80,
    "end2end": True,
    "reg_max": 1,
    "scales": {
        "n": [0.5, 0.25, 1024],
        "s": [0.5, 0.5, 1024],
        "m": [0.5, 1.0, 512],
        "l": [1.0, 1.0, 512],
        "x": [1.0, 1.5, 512],
    },
    "scale": "s",
    "channels": 3,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 2, "C3k2", [256, False, 0.25]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 2, "C3k2", [512, False, 0.25]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "SPPF", [1024, 5, 3, True]],
        [-1, 2, "C2PSA", [1024]],
    ],
    "head": [
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 1, "C3k2", [1024, True, 0.5, True]],
        [[16, 19, 22], 1, "Segment26", ["nc", 32, 256]],
    ],
}


def make_yolo26_seg_arch_cfg(scale: str) -> dict[str, Any]:
    cfg = copy.deepcopy(YOLO26_SEG_BASE_CFG)
    cfg["scale"] = scale.lower()
    return cfg


@dataclass(frozen=True)
class SegmenterSpec:
    model_key: str
    family: str  # "yolo26_seg" | "ecseg" | "rfdetr_seg"
    scale: str
    display_name: str
    resolution: int
    params_m: float
    ecseg_cfg_rel: str | None = None
    release_asset: str = ""


ISWSP_PAPER_MODELS: dict[str, SegmenterSpec] = {
    "yolo26n_seg": SegmenterSpec("yolo26n_seg", "yolo26_seg", "n", "YOLO26-N-Seg", 640, 3.13, None, "yolo26n-seg-iswsp.pt"),
    "yolo26s_seg": SegmenterSpec("yolo26s_seg", "yolo26_seg", "s", "YOLO26-S-Seg", 640, 11.51, None, "yolo26s-seg-iswsp.pt"),
    "yolo26m_seg": SegmenterSpec("yolo26m_seg", "yolo26_seg", "m", "YOLO26-M-Seg", 640, 27.11, None, "yolo26m-seg-iswsp.pt"),
    "yolo26l_seg": SegmenterSpec("yolo26l_seg", "yolo26_seg", "l", "YOLO26-L-Seg", 640, 31.52, None, "yolo26l-seg-iswsp.pt"),
    "yolo26x_seg": SegmenterSpec("yolo26x_seg", "yolo26_seg", "x", "YOLO26-X-Seg", 640, 70.69, None, "yolo26x-seg-iswsp.pt"),
    "ecseg_s": SegmenterSpec("ecseg_s", "ecseg", "s", "ECSeg-S", 640, 10.41, "configs/ecseg/ecseg_s.yml", "ecseg-s-iswsp.pt"),
    "ecseg_m": SegmenterSpec("ecseg_m", "ecseg", "m", "ECSeg-M", 640, 20.36, "configs/ecseg/ecseg_m.yml", "ecseg-m-iswsp.pt"),
    "ecseg_l": SegmenterSpec("ecseg_l", "ecseg", "l", "ECSeg-L", 640, 33.90, "configs/ecseg/ecseg_l.yml", "ecseg-l-iswsp.pt"),
    "ecseg_x": SegmenterSpec("ecseg_x", "ecseg", "x", "ECSeg-X", 640, 50.88, "configs/ecseg/ecseg_x.yml", "ecseg-x-iswsp.pt"),
    "ecseg_o365_s": SegmenterSpec("ecseg_o365_s", "ecseg", "s", "ECSeg-O365-S", 640, 10.41, "configs/ecseg/ecseg_s.yml", "ecseg-o365coco-s-iswsp.pt"),
    "ecseg_o365_m": SegmenterSpec("ecseg_o365_m", "ecseg", "m", "ECSeg-O365-M", 640, 20.36, "configs/ecseg/ecseg_m.yml", "ecseg-o365coco-m-iswsp.pt"),
    "ecseg_o365_l": SegmenterSpec("ecseg_o365_l", "ecseg", "l", "ECSeg-O365-L", 640, 33.90, "configs/ecseg/ecseg_l.yml", "ecseg-o365coco-l-iswsp.pt"),
    "ecseg_o365_x": SegmenterSpec("ecseg_o365_x", "ecseg", "x", "ECSeg-O365-X", 640, 50.88, "configs/ecseg/ecseg_x.yml", "ecseg-o365coco-x-iswsp.pt"),
    "rfdetr_seg_n": SegmenterSpec("rfdetr_seg_n", "rfdetr_seg", "n", "RF-DETR-Seg-N", 312, 33.58, None, "rfdetr-seg-n-iswsp.pt"),
    "rfdetr_seg_s": SegmenterSpec("rfdetr_seg_s", "rfdetr_seg", "s", "RF-DETR-Seg-S", 384, 33.71, None, "rfdetr-seg-s-iswsp.pt"),
    "rfdetr_seg_m": SegmenterSpec("rfdetr_seg_m", "rfdetr_seg", "m", "RF-DETR-Seg-M", 432, 35.69, None, "rfdetr-seg-m-iswsp.pt"),
    "rfdetr_seg_l": SegmenterSpec("rfdetr_seg_l", "rfdetr_seg", "l", "RF-DETR-Seg-L", 504, 35.87, None, "rfdetr-seg-l-iswsp.pt"),
    "rfdetr_seg_xl": SegmenterSpec("rfdetr_seg_xl", "rfdetr_seg", "xl", "RF-DETR-Seg-XL", 624, 38.11, None, "rfdetr-seg-xl-iswsp.pt"),
    "rfdetr_seg_2xl": SegmenterSpec("rfdetr_seg_2xl", "rfdetr_seg", "2xl", "RF-DETR-Seg-2XL", 768, 38.65, None, "rfdetr-seg-2xl-iswsp.pt"),
}


def normalize_iswsp_model_key(model_key: str) -> str:
    key = (
        model_key.strip()
        .lower()
        .replace("-", "_")
        .replace("ecseg_o365coco_", "ecseg_o365_")
        .replace("ecseg_o3652coco_", "ecseg_o365_")
    )
    if key not in ISWSP_PAPER_MODELS:
        raise KeyError(f"Unknown ISWSP model_key '{model_key}'. Supported: {sorted(ISWSP_PAPER_MODELS)}")
    return key


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_load_checkpoint(
    ckpt_path: str | Path,
    map_location: torch.device | str = "cpu",
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """
    Strictly loads an IR-ISWSP checkpoint with `weights_only=True` (zero arbitrary pickle
    code execution) and optional SHA-256 cryptographic digest verification.
    """
    resolved = Path(ckpt_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    if expected_sha256 is not None:
        actual = compute_sha256(resolved)
        if actual.lower() != expected_sha256.strip().lower():
            raise ValueError(
                f"SHA-256 mismatch for {resolved.name}: expected {expected_sha256}, got {actual}"
            )
    obj = torch.load(resolved, map_location=map_location, weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict checkpoint in {resolved.name}, got {type(obj).__name__}")
    return obj


def encode_binary_mask_rle(mask_bool: np.ndarray) -> dict[str, Any]:
    """Encodes a 2D boolean mask into COCO compressed RLE format."""
    from pycocotools import mask as mask_utils

    m_f = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mask_utils.encode(m_f)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def instantiate_segmenter(
    spec: SegmenterSpec,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    edgecrafter_root: Path | None = None,
    arch_yaml: dict[str, Any] | None = None,
) -> tuple[nn.Module, Any]:
    """Instantiates a bare instance segmenter architecture and loads `state_dict` safely (`weights_only=True`)."""
    enforce_no_gradients()

    if spec.family == "yolo26_seg":
        from ultralytics.nn.tasks import SegmentationModel

        cfg_dict = copy.deepcopy(arch_yaml) if isinstance(arch_yaml, dict) else make_yolo26_seg_arch_cfg(spec.scale)
        base_m = SegmentationModel(cfg=cfg_dict, ch=3, nc=80, verbose=False)
        missing, _ = base_m.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing YOLO26-Seg state_dict keys ({len(missing)}): {missing[:5]}")
        base_m.end2end = True
        if hasattr(base_m.model[-1], "end2end"):
            base_m.model[-1].end2end = True
        base_m.eval().requires_grad_(False).to(device)
        return base_m, None

    if spec.family == "rfdetr_seg":
        from rfdetr import (
            RFDETRSeg2XLarge,
            RFDETRSegLarge,
            RFDETRSegMedium,
            RFDETRSegNano,
            RFDETRSegSmall,
            RFDETRSegXLarge,
        )

        cls_map = {
            "n": RFDETRSegNano,
            "s": RFDETRSegSmall,
            "m": RFDETRSegMedium,
            "l": RFDETRSegLarge,
            "xl": RFDETRSegXLarge,
            "2xl": RFDETRSeg2XLarge,
        }
        n_cls = int(state_dict["class_embed.weight"].shape[0] - 1) if "class_embed.weight" in state_dict else 90
        wrapper = cls_map[spec.scale](pretrain_weights=None, num_classes=n_cls, device=str(device))
        model = wrapper.model.model
        model.load_state_dict(state_dict, strict=True)
        model.eval().requires_grad_(False).to(device)
        raw_post = getattr(wrapper.model, "postprocess", None)
        post = raw_post.to(device).eval() if isinstance(raw_post, nn.Module) else raw_post
        return model, post

    ec_root = edgecrafter_root
    if ec_root is None:
        raise ValueError(
            "EdgeCrafter-Seg (ecseg_*) models require `--edgecrafter-root /path/to/EdgeCrafter` "
            "(clone from https://github.com/Intellindust-AI-Lab/EdgeCrafter)."
        )
    ec_resolved = Path(ec_root).resolve()
    if (ec_resolved / "ecdetseg").is_dir():
        ec_resolved = ec_resolved / "ecdetseg"
    if str(ec_resolved) not in sys.path:
        sys.path.insert(0, str(ec_resolved))

    from engine.core import YAMLConfig
    from engine.core.yaml_utils import load_config as _ec_load_config

    if _ec_load_config.__defaults__ and isinstance(_ec_load_config.__defaults__[0], dict):
        _ec_load_config.__defaults__[0].clear()
    assert spec.ecseg_cfg_rel is not None
    cfg = YAMLConfig(str(ec_resolved / spec.ecseg_cfg_rel))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    model = cfg.model
    model.load_state_dict(state_dict, strict=True)
    post = cfg.postprocessor
    model.eval().requires_grad_(False).to(device)
    post.eval().to(device)
    return model, post


@dataclass
class LoadedSegmenter:
    spec: SegmenterSpec
    model: nn.Module
    postprocessor: Any
    device: torch.device
    metadata: dict[str, Any]
    use_s2_synergy: bool = False

    def predict_image(
        self,
        img_path: Path,
        conf_thr: float = 0.25,
        return_masks: bool = True,
        encode_rle: bool = False,
    ) -> list[dict[str, Any]]:
        """Runs zero-gradient instance segmentation on one image and returns boxes, scores, labels, and masks."""
        enforce_no_gradients()
        res = self.spec.resolution

        if self.spec.family == "yolo26_seg":
            from ultralytics.utils import ops

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
            t = torch.from_numpy(np.ascontiguousarray(im_pad[..., ::-1].transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(t)
                y_pair = out[0] if isinstance(out, tuple) else out
                preds_t = y_pair[0][0].float()  # [300, 38] -> xyxy(4), conf(1), cls(1), coeffs(32)
                proto = y_pair[1][0].float()  # [32, mh, mw]

            keep = preds_t[:, 4] >= conf_thr
            rows = preds_t[keep]
            if rows.shape[0] == 0:
                return []

            rh, rw = new_unpad[1] / float(h0), new_unpad[0] / float(w0)
            boxes = rows[:, :4].clone()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / rw).clamp(0.0, float(w0))
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / rh).clamp(0.0, float(h0))
            scores = rows[:, 4]
            c_ids = rows[:, 5].round().long()
            coeffs = rows[:, 6:]

            c_k, mh, mw = proto.shape
            raw_m = (coeffs @ proto.view(c_k, -1)).view(rows.shape[0], mh, mw)
            m_up = F.interpolate(raw_m.unsqueeze(1), size=(res, res), mode="bilinear", align_corners=False)
            m_crop = m_up[:, :, max(0, top) : max(top + 1, res - bottom), max(0, left) : max(left + 1, res - right)]
            m_orig = F.interpolate(m_crop, size=(h0, w0), mode="bilinear", align_corners=False).squeeze(1)
            masks_bool = ops.crop_mask(m_orig, boxes) > 0.0

            if self.use_s2_synergy:
                boxes, scores, _ = apply_single_pass_box_mask_synergy(boxes, scores, masks_bool)

            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            cids_np = c_ids.cpu().numpy()
            masks_np = masks_bool.cpu().numpy()
            results: list[dict[str, Any]] = []
            for idx in range(len(scores_np)):
                ci = int(cids_np[idx])
                if not (0 <= ci < 80):
                    continue
                item: dict[str, Any] = {
                    "bbox_xyxy": [float(v) for v in boxes_np[idx]],
                    "score": float(scores_np[idx]),
                    "category_id": COCO80_TO_91[ci],
                    "label": COCO80_NAMES[ci],
                }
                if return_masks:
                    item["mask"] = masks_np[idx]
                if encode_rle:
                    item["rle"] = encode_binary_mask_rle(masks_np[idx])
                results.append(item)
            return results

        import torchvision
        import torchvision.transforms.v2.functional as F_v2
        from torchvision.transforms import InterpolationMode

        with Image.open(img_path) as pil_im:
            pil_rgb = pil_im.convert("RGB")
        orig_w, orig_h = pil_rgb.size
        im_r = F_v2.resize(pil_rgb, [res, res], interpolation=InterpolationMode.BILINEAR, antialias=True)
        t = F_v2.pil_to_tensor(im_r).float().div_(255.0)
        t = F_v2.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t = t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_out = self.model(t)

        logits = raw_out["pred_logits"].float()
        boxes_cxcywh = raw_out["pred_boxes"].float()
        mask_logits = raw_out["pred_masks"].float()
        probs = torch.sigmoid(logits)
        num_cls = int(probs.shape[-1])
        num_sel = min(300 if self.spec.family == "rfdetr_seg" else 100, int(probs.shape[1]) * num_cls)
        top_scores, top_idx = torch.topk(probs.flatten(1), num_sel, dim=-1)
        top_labels = top_idx[0] % num_cls
        top_q = top_idx[0] // num_cls

        scale_fct = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=self.device, dtype=torch.float32)
        bbox_xyxy = torchvision.ops.box_convert(boxes_cxcywh[0], in_fmt="cxcywh", out_fmt="xyxy") * scale_fct

        scores = top_scores[0]
        keep = scores >= conf_thr
        if self.spec.family == "ecseg":
            keep = keep & (top_labels >= 0) & (top_labels < 80)
        else:
            valid_cids = torch.tensor(
                [int(lb.item()) in COCO91_TO_NAME for lb in top_labels],
                device=self.device,
                dtype=torch.bool,
            )
            keep = keep & valid_cids

        scores = scores[keep][:100]
        labels = top_labels[keep][:100]
        q_idx = top_q[keep][:100]
        if scores.numel() == 0:
            return []

        boxes = bbox_xyxy.index_select(0, q_idx)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(orig_w))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(orig_h))
        ml = mask_logits[0].index_select(0, q_idx)
        m_up = F.interpolate(ml.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)
        masks_bool = m_up > 0.0

        if self.use_s2_synergy and boxes.shape[0] > 0:
            boxes, scores, _ = apply_single_pass_box_mask_synergy(boxes, scores, masks_bool)

        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        masks_np = masks_bool.cpu().numpy()
        results = []
        for idx in range(len(scores_np)):
            raw_lb = int(labels_np[idx])
            cid = COCO80_TO_91[raw_lb] if self.spec.family == "ecseg" else raw_lb
            if cid not in COCO91_TO_NAME:
                continue
            item = {
                "bbox_xyxy": [float(v) for v in boxes_np[idx]],
                "score": float(scores_np[idx]),
                "category_id": cid,
                "label": COCO91_TO_NAME[cid],
            }
            if return_masks:
                item["mask"] = masks_np[idx]
            if encode_rle:
                item["rle"] = encode_binary_mask_rle(masks_np[idx])
            results.append(item)
        return results


def load_iswsp_segmenter(
    model_key: str,
    weights_path: str | Path,
    device: torch.device | str = "cuda:0",
    edgecrafter_root: Path | None = None,
    expected_sha256: str | None = None,
    use_s2_synergy: bool = False,
) -> LoadedSegmenter:
    """
    Safely loads any `v0.1.1` IR-ISWSP checkpoint (`weights_only=True`) across YOLO26-Seg,
    EdgeCrafter-Seg (ECSeg), and RF-DETR-Seg with zero untrusted pickle execution.
    """
    enforce_no_gradients()
    key = normalize_iswsp_model_key(model_key)
    spec = ISWSP_PAPER_MODELS[key]
    dev = torch.device(device)

    pkg = safe_load_checkpoint(weights_path, map_location="cpu", expected_sha256=expected_sha256)
    sd: dict[str, torch.Tensor] = pkg["state_dict"] if "state_dict" in pkg else pkg["model"]
    meta = {k: v for k, v in pkg.items() if k not in ("state_dict", "model", "arch_yaml")}

    model, post = instantiate_segmenter(
        spec=spec,
        state_dict=sd,
        device=dev,
        edgecrafter_root=edgecrafter_root,
        arch_yaml=pkg.get("arch_yaml") if isinstance(pkg.get("arch_yaml"), dict) else None,
    )
    return LoadedSegmenter(
        spec=spec,
        model=model,
        postprocessor=post,
        device=dev,
        metadata=meta,
        use_s2_synergy=use_s2_synergy,
    )
