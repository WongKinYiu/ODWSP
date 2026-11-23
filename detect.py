#!/usr/bin/env python3
"""
Multi-Budget Visual Object Detection Inference CLI (`detect.py`).

Examples:
  # 1.00x Zero-Overhead IR-ODWSP
  python detect.py --model yolo26s --weights ./yolo26s-odwsp.pt --source ./images --conf 0.25 --device 0

  # Negative Budget (-30% Contraction)
  python detect.py --model yolo26s --budget -30% --basins ./yolo26s-odwsp-basins.pt \
      --bridges ./odwsp-contraction-bridges-all17.tar.gz --source ./images --device 0

  # Positive Budget (+50% WSP)
  python detect.py --model yolo26s --budget +50% --paradigm auto --basins ./yolo26s-odwsp-basins.pt \
      --source ./images --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from odwsp import enforce_no_gradients, load_continuum_detector
from odwsp.continuum import normalize_budget_cli_argv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_detections(img_bgr, detections: list[dict]) -> object:
    canvas = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox_xyxy"]]
        label_txt = f"{d['label']} {d['score']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (34, 160, 44), 2)
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        y_top = max(y1, th + 6)
        cv2.rectangle(canvas, (x1, y_top - th - 6), (x1 + tw + 6, y_top), (34, 160, 44), -1)
        cv2.putText(canvas, label_txt, (x1 + 3, y_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ODWSP Multi-Budget Visual Inference CLI")
    ap.add_argument("--model", type=str, required=True, help="Model key (e.g., yolo26s, ecdet_s, rfdetr_m)")
    ap.add_argument("--budget", type=str, default="0%", help="Latency budget beta (e.g., -50%%, -10%%, 0%%, +10%%, +50%%, +100%%, full)")
    ap.add_argument("--paradigm", type=str, default="auto", choices=["auto", "dyn", "sg"], help="Positive-budget WSP paradigm")
    ap.add_argument("--weights", type=Path, default=None, help="Path to 1.00x ODWSP checkpoint (.pt) for --budget 0%%")
    ap.add_argument("--basins", type=Path, default=None, help="Path to multi-subspace basin pack (*-odwsp-basins.pt)")
    ap.add_argument("--bridges", type=Path, default=None, help="Path to contraction bridges (.pt or odwsp-contraction-bridges-all17.tar.gz)")
    ap.add_argument("--deploy-ckpt", type=Path, default=None, help="Optional pre-fused deploy archive/checkpoint")
    ap.add_argument("--cand-id", type=str, default=None, help="Optional explicit candidate graph ID override")
    ap.add_argument("--source", type=Path, required=True, help="Path to input image or folder of images")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (required for ecdet_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 hex digest")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--output", type=Path, default=Path("./runs/detect"), help="Directory to save visual predictions")
    args = ap.parse_args(normalize_budget_cli_argv())

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    det = load_continuum_detector(
        model_key=args.model,
        budget=args.budget,
        paradigm=args.paradigm,
        weights_path=args.weights,
        basins_path=args.basins,
        bridges_path=args.bridges,
        deploy_ckpt_path=args.deploy_ckpt,
        cand_override=args.cand_id,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
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
        preds = det.predict_image(img_p, conf_thr=args.conf)
        im_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if im_bgr is not None:
            vis = draw_detections(im_bgr, preds)
            cv2.imwrite(str(out_dir / img_p.name), vis)
        summary.append({"image": img_p.name, "budget": args.budget, "num_detections": len(preds), "detections": preds})

    (out_dir / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(img_files)} annotated prediction(s) [budget={args.budget}] to {out_dir}")


if __name__ == "__main__":
    main()
