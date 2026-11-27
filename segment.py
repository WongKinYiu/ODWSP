#!/usr/bin/env python3
"""
Visual Instance Segmentation Inference CLI for IR-ISWSP (`segment.py`).

Draws semi-transparent colored instance masks, crisp contours, bounding boxes, and
category labels strictly with zero backpropagation (`weights_only=True`, zero external NMS).

Examples:
  python segment.py --model yolo26x_seg --weights ./yolo26x-seg-iswsp.pt --source ./images --conf 0.25 --device 0
  python segment.py --model rfdetr_seg_l --weights ./rfdetr-seg-l-iswsp.pt --synergy-s2 --source ./images --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from iswsp import enforce_no_gradients, load_iswsp_segmenter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PALETTE_BGR = [
    (94, 197, 34),
    (246, 130, 59),
    (22, 115, 249),
    (247, 85, 168),
    (166, 184, 20),
    (8, 179, 234),
    (153, 72, 236),
    (212, 182, 6),
]


def draw_instances(img_bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
    canvas = img_bgr.astype(np.float32).copy()
    for i, pred in enumerate(predictions):
        col = PALETTE_BGR[i % len(PALETTE_BGR)]
        mask = pred.get("mask")
        if mask is not None and mask.any():
            canvas[mask] = canvas[mask] * 0.58 + np.array(col, dtype=np.float32) * 0.42
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, col, 2)

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    for i, pred in enumerate(predictions):
        col = PALETTE_BGR[i % len(PALETTE_BGR)]
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox_xyxy"]]
        label_txt = f"{pred['label']} {pred['score']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        y_top = max(y1, th + 6)
        cv2.rectangle(out, (x1, y_top - th - 6), (x1 + tw + 6, y_top), col, -1)
        cv2.putText(out, label_txt, (x1 + 3, y_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ISWSP Visual Instance Segmentation Inference CLI")
    ap.add_argument("--model", type=str, required=True, help="Model key (e.g., yolo26x_seg, ecseg_l, rfdetr_seg_l)")
    ap.add_argument("--weights", type=Path, required=True, help="Path to v0.1.1 IR-ISWSP checkpoint (.pt)")
    ap.add_argument("--synergy-s2", action="store_true", help="Enable single-pass Box <-> Mask geometric synergy (S2)")
    ap.add_argument("--source", type=Path, required=True, help="Path to input image or folder of images")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (required for ecseg_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 hex digest")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--output", type=Path, default=Path("./runs/segment"), help="Directory to save visual predictions")
    args = ap.parse_args()

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    seg = load_iswsp_segmenter(
        model_key=args.model,
        weights_path=args.weights,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
        use_s2_synergy=args.synergy_s2,
    )

    src = args.source.resolve()
    if src.is_file():
        img_files = [src]
    elif src.is_dir():
        img_files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXTS]
    else:
        raise FileNotFoundError(f"Source not found: {src}")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for img_p in img_files:
        preds = seg.predict_image(img_p, conf_thr=args.conf, return_masks=True, encode_rle=False)
        im_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if im_bgr is not None:
            vis = draw_instances(im_bgr, preds)
            cv2.imwrite(str(out_dir / img_p.name), vis)
        json_preds = [{k: v for k, v in p.items() if k != "mask"} for p in preds]
        summary.append({"image": img_p.name, "num_instances": len(json_preds), "instances": json_preds})

    (out_dir / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(img_files)} segmented image(s) to {out_dir}")


if __name__ == "__main__":
    main()
