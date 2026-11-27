#!/usr/bin/env python3
"""
MS COCO val2017 Instance Segmentation Evaluation CLI for IR-ISWSP (`val_iswsp.py`).

Evaluates BOTH Mask AP (`segm`) and Box AP (`bbox`) strictly with zero backpropagation
(`enforce_no_gradients()`), `weights_only=True`, and zero external NMS / Mask-NMS.

Examples:
  # 1.00x Zero-Overhead Full-Matrix W_q IR-ISWSP (ΔParams=0%, ΔFLOPs=0%)
  python val_iswsp.py --model yolo26x_seg --weights ./yolo26x-seg-iswsp.pt --coco-root /coco --device 0

  # Single-Pass Bidirectional Box <-> Mask Synergy (S2)
  python val_iswsp.py --model rfdetr_seg_2xl --weights ./rfdetr-seg-2xl-iswsp.pt --synergy-s2 --coco-root /coco --device 0

  # EdgeCrafter-Seg (ECSeg) evaluation
  python val_iswsp.py --model ecseg_o365_x --weights ./ecseg-o365coco-x-iswsp.pt --edgecrafter-root ./EdgeCrafter --coco-root /coco --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iswsp import (
    COCO80_TO_91,
    enforce_no_gradients,
    load_iswsp_segmenter,
    normalize_iswsp_budget_cli_argv,
)


def evaluate_coco_instseg_split(
    seg,
    img_dir: Path,
    ann_json: Path,
    conf_thr: float = 0.001,
    max_images: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluates a loaded IR-ISWSP segmenter on COCO `segm` (Mask AP) and `bbox` (Box AP)."""
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
    if max_images is not None:
        img_ids = img_ids[:max_images]

    pred_boxes: list[dict[str, Any]] = []
    pred_masks: list[dict[str, Any]] = []
    for img_id in img_ids:
        info = coco_gt.loadImgs([img_id])[0]
        img_p = img_dir / info["file_name"]
        rows = seg.predict_image(img_p, conf_thr=conf_thr, return_masks=False, encode_rle=True)
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue
            base = {
                "image_id": int(img_id),
                "category_id": int(r["category_id"]),
                "score": round(float(r["score"]), 5),
            }
            pred_boxes.append({**base, "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)]})
            if "rle" in r:
                pred_masks.append({**base, "segmentation": r["rle"]})

    def _run_eval(preds: list[dict[str, Any]], iou_type: str) -> dict[str, float]:
        if not preds:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0}
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.imgIds = img_ids
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
        }

    return {
        "mask": _run_eval(pred_masks, "segm"),
        "box": _run_eval(pred_boxes, "bbox"),
    }


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ISWSP MS COCO Instance Segmentation Evaluation CLI")
    ap.add_argument("--model", type=str, required=True, help="Model key (e.g., yolo26x_seg, ecseg_l, rfdetr_seg_2xl)")
    ap.add_argument("--weights", type=Path, required=True, help="Path to v0.1.1 IR-ISWSP checkpoint (.pt)")
    ap.add_argument("--synergy-s2", action="store_true", help="Enable single-pass Box <-> Mask geometric synergy (S2)")
    ap.add_argument("--coco-root", type=Path, required=True, help="Path to MS COCO root (containing val2017/ & annotations/)")
    ap.add_argument("--split", type=str, default="val2017", help="COCO split name (default: val2017)")
    ap.add_argument("--max-images", type=int, default=None, help="Optional image cap for quick smoke evaluation")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (required for ecseg_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 hex digest for weight verification")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for COCO evaluation")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional output path for JSON metrics summary")
    args = ap.parse_args(normalize_iswsp_budget_cli_argv())

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    seg = load_iswsp_segmenter(
        model_key=args.model,
        weights_path=args.weights,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
        use_s2_synergy=args.synergy_s2,
    )

    coco_root = args.coco_root.resolve()
    metrics = evaluate_coco_instseg_split(
        seg,
        img_dir=coco_root / args.split,
        ann_json=coco_root / "annotations" / f"instances_{args.split}.json",
        conf_thr=args.conf,
        max_images=args.max_images,
    )
    results = {
        "model": seg.spec.model_key,
        "display_name": seg.spec.display_name,
        "split": args.split,
        "synergy_s2": bool(args.synergy_s2),
        "checkpoint_metadata": seg.metadata,
        "coco_metrics": metrics,
    }
    print(json.dumps(results, indent=2))
    if args.save_json is not None:
        out_p = args.save_json.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
