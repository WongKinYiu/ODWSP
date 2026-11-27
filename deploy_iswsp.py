#!/usr/bin/env python3
"""
Latency-Budget-Aware Deployment, Continuum Inspector & Checkpoint Audit CLI for IR-ISWSP (`deploy_iswsp.py`).

Supports:
  1. `--list-budgets`: Print the complete paper-verified bidirectional continuum (`-60%..+100%`, `S2`, `S3`, `full`)
     for any of the 19 official IR-ISWSP segmenters (zero external files required).
  2. `--inspect-ckpt`: Verify `weights_only=True` safety and inspect metadata/SHA-256 of any `v0.1.1` checkpoint.
  3. `--benchmark`   : Measure isolated PyTorch FP16 or static CUDA Graph latency on your local GPU.
  4. `--export-onnx` / `--export-trt`: Export static-shape ONNX and optional TensorRT FP16 `.plan` engine.

Examples:
  python deploy_iswsp.py --model rfdetr_seg_2xl --list-budgets
  python deploy_iswsp.py --inspect-ckpt ./yolo26x-seg-iswsp.pt
  python deploy_iswsp.py --model yolo26x_seg --weights ./yolo26x-seg-iswsp.pt --fp16 --cudagraph --benchmark --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from iswsp import (
    compute_sha256,
    enforce_no_gradients,
    get_iswsp_continuum_table,
    load_iswsp_segmenter,
    normalize_iswsp_budget_cli_argv,
    normalize_iswsp_model_key,
    resolve_iswsp_budget_config,
    safe_load_checkpoint,
)
from odwsp import benchmark_cuda_callable, capture_static_cudagraph


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ISWSP Bidirectional Budget Continuum, Benchmark & Audit CLI")
    ap.add_argument("--model", type=str, default="yolo26x_seg", help="ISWSP model key (e.g., yolo26x_seg, ecseg_l, rfdetr_seg_2xl)")
    ap.add_argument("--budget", type=str, default="0%", help="Latency budget (-60%%..-10%%, 0%%, S2, +10%%, +50%%, +100%%, S3, full)")
    ap.add_argument("--paradigm", type=str, default="static", choices=["static", "dyn"], help="Positive-budget deployment paradigm")
    ap.add_argument("--weights", type=Path, default=None, help="Path to v0.1.1 IR-ISWSP checkpoint (.pt)")
    ap.add_argument("--list-budgets", action="store_true", help="Print all paper-verified bidirectional budget tiers for --model")
    ap.add_argument("--inspect-ckpt", type=Path, default=None, help="Inspect a v0.1.1 checkpoint safely (`weights_only=True`)")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (required for ecseg_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 digest")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking/export")
    ap.add_argument("--fp16", action="store_true", help="Run benchmark/export in FP16")
    ap.add_argument("--cudagraph", action="store_true", help="Benchmark inside a static CUDA Graph")
    ap.add_argument("--benchmark", action="store_true", help="Measure median GPU latency (ms/img)")
    ap.add_argument("--export-onnx", type=Path, default=None, help="Optional output path for static-shape ONNX export")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional output path for JSON summary")
    args = ap.parse_args(normalize_iswsp_budget_cli_argv())

    if args.inspect_ckpt is not None:
        ckpt_p = args.inspect_ckpt.resolve()
        pkg = safe_load_checkpoint(ckpt_p, map_location="cpu", expected_sha256=args.sha256)
        sd = pkg.get("state_dict", {})
        summary = {
            "file": ckpt_p.name,
            "file_sha256": compute_sha256(ckpt_p),
            "weights_only_safe": True,
            "version": pkg.get("version"),
            "model_key": pkg.get("model_key"),
            "num_tensors": len(sd) if isinstance(sd, dict) else 0,
            "top_level_keys": sorted(pkg.keys()),
        }
        print(json.dumps(summary, indent=2))
        return

    mkey = normalize_iswsp_model_key(args.model)
    if args.list_budgets:
        tbl = get_iswsp_continuum_table()
        tiers = tbl["supported_negative_tiers"] + ["0%", "S2"] + tbl["supported_positive_tiers"] + ["S3", "full"]
        rows = [resolve_iswsp_budget_config(mkey, budget=t, paradigm=args.paradigm) for t in tiers]
        payload = {"model": mkey, "paradigm": args.paradigm, "continuum": rows}
        print(json.dumps(payload, indent=2))
        if args.save_json is not None:
            out_p = args.save_json.resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    if args.weights is None:
        resolved = resolve_iswsp_budget_config(mkey, budget=args.budget, paradigm=args.paradigm)
        print(json.dumps(resolved, indent=2))
        return

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    seg = load_iswsp_segmenter(
        model_key=mkey,
        weights_path=args.weights,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
    )
    res = seg.spec.resolution
    dtype = torch.float16 if (args.fp16 and seg.device.type == "cuda") else torch.float32
    model = seg.model.half().eval() if dtype == torch.float16 else seg.model.float().eval()
    sample_x = torch.rand((args.batch_size, 3, res, res), device=seg.device, dtype=dtype)

    report: dict[str, object] = {
        "model": seg.spec.model_key,
        "display_name": seg.spec.display_name,
        "resolution": res,
        "batch_size": args.batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "weights_only_verified": True,
        "paper_continuum": resolve_iswsp_budget_config(mkey, budget=args.budget, paradigm=args.paradigm),
    }

    if args.export_onnx is not None:
        onnx_out = args.export_onnx.resolve()
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                model,
                (sample_x,),
                str(onnx_out),
                input_names=["images"],
                opset_version=18,
                do_constant_folding=True,
            )
        report["onnx_path"] = str(onnx_out)
        report["onnx_size_mb"] = round(onnx_out.stat().st_size / 1e6, 2)

    if args.benchmark:
        if seg.device.type != "cuda":
            raise RuntimeError("CUDA GPU required for `--benchmark`.")
        if args.cudagraph:
            g, _, _ = capture_static_cudagraph(lambda z: model(z), sample_x)
            ms = benchmark_cuda_callable(g.replay, seg.device, batch_size=args.batch_size)
            report["cudagraph_median_ms_per_img"] = round(ms, 3)
        else:
            ms = benchmark_cuda_callable(lambda: model(sample_x), seg.device, batch_size=args.batch_size)
            report["pytorch_median_ms_per_img"] = round(ms, 3)

    print(json.dumps(report, indent=2))
    if args.save_json is not None:
        out_p = args.save_json.resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
