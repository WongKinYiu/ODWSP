#!/usr/bin/env python3
"""
MS COCO val2017 & Zero-Shot COCO-O Evaluation CLI (`val.py`).

Example:
  python val.py --model yolo26s --weights ./yolo26s-odwsp.pt --coco-root ./coco --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from odwsp import COCO80_TO_91, enforce_no_gradients, load_odwsp_detector


def evaluate_coco_split(
    det,
    img_dir: Path,
    ann_json: Path,
    conf_thr: float = 0.001,
) -> dict[str, float]:
    """Evaluates a loaded detector on a COCO-formatted image directory + annotation JSON."""
    enforce_no_gradients()
    try:
        import faster_coco_eval
        faster_coco_eval.init_as_pycocotools()
    except Exception:
        pass
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ann_json))
    img_ids = sorted(coco_gt.getImgIds())
    predictions: list[dict[str, Any]] = []

    for img_id in img_ids:
        info = coco_gt.loadImgs([img_id])[0]
        img_p = img_dir / info["file_name"]
        rows = det.predict_image(img_p, conf_thr=conf_thr)
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue
            predictions.append({
                "image_id": int(img_id),
                "category_id": int(r["category_id"]),
                "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                "score": round(float(r["score"]), 5),
            })

    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = COCO80_TO_91
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = np.asarray(ev.stats, dtype=np.float64) * 100.0
    return {
        "AP": round(float(s[0]), 3),
        "AP50": round(float(s[1]), 3),
        "AP75": round(float(s[2]), 3),
        "APs": round(float(s[3]), 3),
        "APm": round(float(s[4]), 3),
        "APl": round(float(s[5]), 3),
        "AR1": round(float(s[6]), 3),
        "AR10": round(float(s[7]), 3),
        "AR100": round(float(s[8]), 3),
    }


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ODWSP MS COCO & Zero-Shot COCO-O Evaluation")
    ap.add_argument("--model", type=str, required=True, help="Model key (e.g., yolo26s, ecdet_s, rfdetr_m)")
    ap.add_argument("--weights", type=Path, required=True, help="Path to v0.1.0 ODWSP checkpoint (.pt)")
    ap.add_argument("--coco-root", type=Path, default=None, help="Path to MS COCO root (containing val2017/ & annotations/)")
    ap.add_argument("--coco-o-root", type=Path, default=None, help="Optional path to COCO-O root (6 natural shift domains)")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (required for ecdet_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 hex digest for weight verification")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for COCO evaluation")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional output path for JSON metrics summary")
    args = ap.parse_args()

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    det = load_odwsp_detector(
        model_key=args.model,
        weights_path=args.weights,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
    )

    results: dict[str, Any] = {
        "model": det.spec.model_key,
        "weights": args.weights.name,
        "checkpoint_metadata": det.metadata,
    }

    if args.coco_root is not None:
        coco_root = args.coco_root.resolve()
        results["coco_val2017"] = evaluate_coco_split(
            det,
            img_dir=coco_root / "val2017",
            ann_json=coco_root / "annotations" / "instances_val2017.json",
            conf_thr=args.conf,
        )

    if args.coco_o_root is not None:
        coco_o_root = args.coco_o_root.resolve()
        domains = ("cartoon", "handmake", "painting", "sketch", "tattoo", "weather")
        per_dom: dict[str, dict[str, float]] = {}
        for dom in domains:
            dom_dir = coco_o_root / dom
            img_d = dom_dir / "images" if (dom_dir / "images").is_dir() else dom_dir
            ann_cands = sorted(dom_dir.glob("*.json")) + sorted((coco_o_root / "annotations").glob(f"*{dom}*.json"))
            if not ann_cands:
                raise FileNotFoundError(f"Annotation JSON not found for COCO-O domain '{dom}' under {coco_o_root}")
            per_dom[dom] = evaluate_coco_split(det, img_dir=img_d, ann_json=ann_cands[0], conf_thr=args.conf)
        results["coco_o_per_domain"] = per_dom
        results["coco_o_mAP"] = round(float(np.mean([v["AP"] for v in per_dom.values()])), 3)

    print(json.dumps(results, indent=2))
    if args.save_json is not None:
        out_p = args.save_json.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
