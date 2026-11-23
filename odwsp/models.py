"""
Safe (`weights_only=True`) Detector Loaders, YOLO26 In-Graph Collision Gate,
and MS COCO / Zero-Shot COCO-O Evaluation Pipeline.
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


@dataclass(frozen=True)
class DetectorSpec:
    model_key: str
    family: str  # "yolo26" | "edgecrafter" | "rfdetr"
    scale: str
    resolution: int
    ecdet_cfg_rel: str | None = None


PAPER_MODELS: dict[str, DetectorSpec] = {
    "yolo26n": DetectorSpec("yolo26n", "yolo26", "n", 640),
    "yolo26s": DetectorSpec("yolo26s", "yolo26", "s", 640),
    "yolo26m": DetectorSpec("yolo26m", "yolo26", "m", 640),
    "yolo26l": DetectorSpec("yolo26l", "yolo26", "l", 640),
    "yolo26x": DetectorSpec("yolo26x", "yolo26", "x", 640),
    "ecdet_s": DetectorSpec("ecdet_s", "edgecrafter", "s", 640, "configs/ecdet/ecdet_s.yml"),
    "ecdet_m": DetectorSpec("ecdet_m", "edgecrafter", "m", 640, "configs/ecdet/ecdet_m.yml"),
    "ecdet_l": DetectorSpec("ecdet_l", "edgecrafter", "l", 640, "configs/ecdet/ecdet_l.yml"),
    "ecdet_x": DetectorSpec("ecdet_x", "edgecrafter", "x", 640, "configs/ecdet/ecdet_x.yml"),
    "ecdet_o365coco_s": DetectorSpec("ecdet_o365coco_s", "edgecrafter", "s", 640, "configs/ecdet/ecdet_s.yml"),
    "ecdet_o365coco_m": DetectorSpec("ecdet_o365coco_m", "edgecrafter", "m", 640, "configs/ecdet/ecdet_m.yml"),
    "ecdet_o365coco_l": DetectorSpec("ecdet_o365coco_l", "edgecrafter", "l", 640, "configs/ecdet/ecdet_l.yml"),
    "ecdet_o365coco_x": DetectorSpec("ecdet_o365coco_x", "edgecrafter", "x", 640, "configs/ecdet/ecdet_x.yml"),
    "rfdetr_n": DetectorSpec("rfdetr_n", "rfdetr", "n", 384),
    "rfdetr_s": DetectorSpec("rfdetr_s", "rfdetr", "s", 512),
    "rfdetr_m": DetectorSpec("rfdetr_m", "rfdetr", "m", 576),
    "rfdetr_l": DetectorSpec("rfdetr_l", "rfdetr", "l", 704),
}


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
    ckpt_path: Path,
    map_location: torch.device | str = "cpu",
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """
    Loads a checkpoint strictly with `weights_only=True` (zero arbitrary pickle execution)
    and optional SHA-256 cryptographic digest verification.
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


class YOLOGated1xModule(nn.Module):
    """FP16-safe `IR-ODWSP (1.00x)` wrapper with dual-head `o2m` collision gate (Eq. 8)."""

    def __init__(self, yolo_model: nn.Module, cfg: dict[str, float]) -> None:
        super().__init__()
        enforce_no_gradients()
        mc = copy.deepcopy(yolo_model).eval()
        mc.end2end = True
        mc.model[-1].end2end = True
        self.mod = mc
        self.end2end = True
        self.beta_box = float(cfg.get("beta_box", 0.0))
        self.gamma_dn = float(cfg.get("gamma_dn", 0.0))
        self.gamma_up = float(cfg.get("gamma_up", 0.0))
        self.tau_iou = float(cfg.get("tau_iou", 0.55))
        self.tau_dom = float(cfg.get("tau_dom", 0.35))
        self.tau_t = float(cfg.get("tau_t", 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        det = self.mod.model[-1]
        _, pr = self.mod(x)
        po, pm = pr["one2one"], pr["one2many"]
        dbox_o = det._get_decode_boxes(po)
        dbox_m = det._get_decode_boxes(pm)
        so_raw = po["scores"]
        so_prob = (so_raw / self.tau_t).sigmoid() if self.tau_t != 1.0 else so_raw.sigmoid()
        if self.beta_box <= 0.0 and self.gamma_dn <= 0.0 and self.gamma_up <= 0.0:
            y = torch.cat((dbox_o, so_prob), dim=1)
            return det.postprocess(y.permute(0, 2, 1))

        sm_prob = pm["scores"].sigmoid()
        inv_s = 1.0 / float(max(x.shape[-2], x.shape[-1]))
        bo_n, bm_n = dbox_o * inv_s, dbox_m * inv_s
        x1 = torch.maximum(bo_n[:, 0:1], bm_n[:, 0:1])
        y1 = torch.maximum(bo_n[:, 1:2], bm_n[:, 1:2])
        x2 = torch.minimum(bo_n[:, 2:3], bm_n[:, 2:3])
        y2 = torch.minimum(bo_n[:, 3:4], bm_n[:, 3:4])
        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        ao = ((bo_n[:, 2:3] - bo_n[:, 0:1]).clamp(min=0.0) * (bo_n[:, 3:4] - bo_n[:, 1:2]).clamp(min=1e-6))
        am = ((bm_n[:, 2:3] - bm_n[:, 0:1]).clamp(min=0.0) * (bm_n[:, 3:4] - bm_n[:, 1:2]).clamp(min=1e-6))
        iou_a = inter / (ao + am - inter + 1e-6)
        agree = ((iou_a - self.tau_iou) / max(1e-4, 1.0 - self.tau_iou)).clamp(0.0, 1.0)
        dbox = dbox_o + (self.beta_box * agree * agree) * (dbox_m - dbox_o) if self.beta_box > 0.0 else dbox_o
        if self.gamma_dn > 0.0 or self.gamma_up > 0.0:
            s_peak = so_prob.amax(dim=1, keepdim=True)
            dom_gate = torch.sigmoid((s_peak - self.tau_dom) / 0.025)
            g_up = self.gamma_up * agree * dom_gate
            g_eff = torch.where(sm_prob < so_prob, torch.full_like(sm_prob, self.gamma_dn), g_up.expand_as(sm_prob))
            sc = torch.exp((1.0 - g_eff) * torch.log(so_prob.clamp(min=1e-5)) + g_eff * torch.log(sm_prob.clamp(min=1e-5)))
        else:
            sc = so_prob
        return det.postprocess(torch.cat((dbox, sc), dim=1).permute(0, 2, 1))


@dataclass
class LoadedDetector:
    spec: DetectorSpec
    model: nn.Module
    postprocessor: Any
    device: torch.device
    metadata: dict[str, Any]

    def predict_image(self, img_path: Path, conf_thr: float = 0.25) -> list[dict[str, Any]]:
        """Runs zero-gradient single-image detection and returns `(bbox_xyxy, score, category_id, label)`."""
        enforce_no_gradients()
        res = self.spec.resolution
        if self.spec.family == "yolo26":
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
            t = torch.from_numpy(np.ascontiguousarray(im_rgb.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(t)
                y = (out[0] if isinstance(out, tuple) else out)[0].detach().cpu().numpy()
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
                cid = COCO80_TO_91[c_idx]
                dets.append({"bbox_xyxy": [x1, y1, x2, y2], "score": sc, "category_id": cid, "label": COCO80_NAMES[c_idx]})
            return dets

        import torchvision.transforms.v2.functional as F_v2
        from torchvision.transforms import InterpolationMode

        with Image.open(img_path) as pil_im:
            pil_rgb = pil_im.convert("RGB")
        orig_w, orig_h = pil_rgb.size
        im_r = F_v2.resize(pil_rgb, [res, res], interpolation=InterpolationMode.BILINEAR, antialias=True)
        t = F_v2.pil_to_tensor(im_r).float().div_(255.0)
        t = F_v2.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_out = self.model(t)
            if self.spec.family == "edgecrafter":
                orig_sz = torch.tensor([[orig_w, orig_h]], device=self.device, dtype=torch.float32)
                res_list = self.postprocessor(raw_out, orig_sz)
            else:
                tgt_sz = torch.tensor([[orig_h, orig_w]], device=self.device, dtype=torch.float32)
                res_list = self.postprocessor(raw_out, tgt_sz)

        scores = res_list[0]["scores"].detach().cpu().numpy()
        labels = res_list[0]["labels"].detach().cpu().numpy()
        boxes = res_list[0]["boxes"].detach().cpu().numpy()
        dets = []
        for sc, lb, bx in zip(scores, labels, boxes):
            if float(sc) < conf_thr:
                continue
            cid = int(lb)
            if cid not in COCO91_TO_NAME:
                continue
            dets.append({
                "bbox_xyxy": [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])],
                "score": float(sc),
                "category_id": cid,
                "label": COCO91_TO_NAME[cid],
            })
        return dets


def load_odwsp_detector(
    model_key: str,
    weights_path: Path,
    device: torch.device | str = "cuda:0",
    edgecrafter_root: Path | None = None,
    expected_sha256: str | None = None,
    use_collision_gate: bool = True,
) -> LoadedDetector:
    """
    Safely loads any `v0.1.0` IR-ODWSP checkpoint (`weights_only=True`) across YOLO26,
    EdgeCrafter, and RF-DETR without executing untrusted pickle classes.
    """
    enforce_no_gradients()
    key = model_key.strip().lower().replace("-", "_").replace("ecdet_o3652coco_", "ecdet_o365coco_")
    if key not in PAPER_MODELS:
        raise KeyError(f"Unknown model_key '{model_key}'. Supported: {sorted(PAPER_MODELS)}")
    spec = PAPER_MODELS[key]
    dev = torch.device(device)

    pkg = safe_load_checkpoint(weights_path, map_location="cpu", expected_sha256=expected_sha256)
    sd: dict[str, torch.Tensor] = pkg["state_dict"] if "state_dict" in pkg else pkg["model"]
    meta = {k: v for k, v in pkg.items() if k not in ("state_dict", "model")}

    if spec.family == "yolo26":
        from ultralytics.nn.tasks import DetectionModel

        arch_yaml = pkg.get("arch_yaml")
        if not isinstance(arch_yaml, dict):
            arch_yaml = f"yolo26{spec.scale}.yaml"
        base_m = DetectionModel(cfg=copy.deepcopy(arch_yaml), ch=3, nc=80, verbose=False)
        base_m.load_state_dict(sd, strict=True)
        base_m.end2end = True
        base_m.eval().requires_grad_(False)
        if use_collision_gate and isinstance(pkg.get("ingraph_cfg"), dict):
            model: nn.Module = YOLOGated1xModule(base_m, pkg["ingraph_cfg"]).to(dev).eval()
        else:
            model = base_m.to(dev).eval()
        return LoadedDetector(spec=spec, model=model, postprocessor=None, device=dev, metadata=meta)

    if spec.family == "rfdetr":
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        cls_map = {"n": RFDETRNano, "s": RFDETRSmall, "m": RFDETRMedium, "l": RFDETRLarge}
        n_cls = int(sd["class_embed.weight"].shape[0] - 1) if "class_embed.weight" in sd else 90
        wrapper = cls_map[spec.scale](pretrain_weights=None, num_classes=n_cls)
        model = wrapper.model.model
        model.load_state_dict(sd, strict=True)
        model.eval().requires_grad_(False).to(dev)
        post = wrapper.model.postprocessors["bbox"].to(dev).eval()
        return LoadedDetector(spec=spec, model=model, postprocessor=post, device=dev, metadata=meta)

    # EdgeCrafter (ECDet): requires official Intellindust EdgeCrafter repo path (`--edgecrafter-root`)
    ec_root = edgecrafter_root
    if ec_root is None:
        raise ValueError(
            "EdgeCrafter models require `--edgecrafter-root /path/to/EdgeCrafter/ecdetseg` "
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
    assert spec.ecdet_cfg_rel is not None
    cfg_path = ec_resolved / spec.ecdet_cfg_rel
    cfg = YAMLConfig(str(cfg_path))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    model = cfg.model
    model.load_state_dict(sd, strict=True)
    post = cfg.postprocessor
    model.eval().requires_grad_(False).to(dev)
    post.eval().to(dev)
    return LoadedDetector(spec=spec, model=model, postprocessor=post, device=dev, metadata=meta)
