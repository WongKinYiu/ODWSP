#!/usr/bin/env python3
"""
Latency-Budget-Aware Deployment & TensorRT / CUDAGraph Benchmark CLI (`deploy.py`).

Supports:
  1. PyTorch & CUDAGraph latency benchmarking on any `v0.1.0` checkpoint (`B=1` / `B=8`).
  2. ONNX & local Strongly-Typed FP16 TensorRT `.plan` compilation on your target GPU.
  3. Inspection of multi-subspace basin packs (`*-odwsp-basins.pt`) and closed-form
     `MM-FW-Ridge` negative-budget bridge packs (`*-bridges.pt`) via `weights_only=True`.

Examples:
  python deploy.py --model yolo26s --weights ./yolo26s-odwsp.pt --batch-size 1 --cudagraph --benchmark
  python deploy.py --model yolo26s --weights ./yolo26s-odwsp.pt --export-onnx ./yolo26s-odwsp.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from odwsp import (
    benchmark_cuda_callable,
    capture_static_cudagraph,
    enforce_no_gradients,
    load_odwsp_detector,
    safe_load_checkpoint,
)
from odwsp.plugins import load_compiled_trt_plugins


def export_onnx_and_optional_trt(
    model: torch.nn.Module,
    sample_x: torch.Tensor,
    onnx_path: Path,
    trt_plan_path: Path | None = None,
) -> dict[str, object]:
    """Exports a loaded ODWSP detector to constant-shape ONNX and optionally builds a local FP16 TensorRT engine."""
    enforce_no_gradients()
    onnx_out = onnx_path.resolve()
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_x,),
            str(onnx_out),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=18,
            do_constant_folding=True,
        )
    info: dict[str, object] = {"onnx_path": str(onnx_out), "onnx_size_mb": round(onnx_out.stat().st_size / 1e6, 2)}

    if trt_plan_path is not None:
        import tensorrt as trt

        load_compiled_trt_plugins()
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_out)):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"TensorRT ONNX parse failed: {errs}")
        config = builder.create_builder_config()
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build serialized engine.")
        plan_out = trt_plan_path.resolve()
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        plan_out.write_bytes(bytes(serialized))
        info["trt_plan_path"] = str(plan_out)
        info["trt_plan_size_mb"] = round(plan_out.stat().st_size / 1e6, 2)
    return info


def main() -> None:
    enforce_no_gradients()
    ap = argparse.ArgumentParser(description="IR-ODWSP Latency-Budget Deployment & TensorRT CLI")
    ap.add_argument("--model", type=str, default="yolo26s", help="Model key (e.g., yolo26s, ecdet_s, rfdetr_m)")
    ap.add_argument("--weights", type=Path, default=None, help="Path to v0.1.0 ODWSP checkpoint (.pt)")
    ap.add_argument("--inspect-pack", type=Path, default=None, help="Inspect a v0.1.0 basin or bridge pack safely (weights_only=True)")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (for ecdet_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 digest")
    ap.add_argument("--batch-size", type=int, default=1, choices=[1, 8])
    ap.add_argument("--fp16", action="store_true", help="Benchmark PyTorch FP16 execution")
    ap.add_argument("--cudagraph", action="store_true", help="Capture and replay inside a static CUDA Graph")
    ap.add_argument("--benchmark", action="store_true", help="Measure median per-image GPU latency (ms/img)")
    ap.add_argument("--export-onnx", type=Path, default=None, help="Export static-shape ONNX graph")
    ap.add_argument("--export-trt", type=Path, default=None, help="Compile local TensorRT FP16 .plan engine")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    args = ap.parse_args()

    if args.inspect_pack is not None:
        pack = safe_load_checkpoint(args.inspect_pack, map_location="cpu", expected_sha256=args.sha256)
        summary = {
            "file": args.inspect_pack.name,
            "weights_only_safe": True,
            "top_level_keys": sorted(pack.keys()),
        }
        if "subspace_names" in pack:
            summary["subspace_names"] = pack["subspace_names"]
        print(json.dumps(summary, indent=2))
        return

    if args.weights is None:
        raise ValueError("Provide `--weights <path.pt>` or `--inspect-pack <pack.pt>`.")

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    det = load_odwsp_detector(
        model_key=args.model,
        weights_path=args.weights,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
    )
    res = det.spec.resolution
    dtype = torch.float16 if (args.fp16 and det.device.type == "cuda") else torch.float32
    model = det.model.half().eval() if dtype == torch.float16 else det.model.float().eval()
    sample_x = torch.rand((args.batch_size, 3, res, res), device=det.device, dtype=dtype)

    out_report: dict[str, object] = {
        "model": det.spec.model_key,
        "resolution": res,
        "batch_size": args.batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "weights_only_verified": True,
    }

    if args.export_onnx is not None or args.export_trt is not None:
        onnx_dst = args.export_onnx or args.export_trt.with_suffix(".onnx")
        out_report["export"] = export_onnx_and_optional_trt(model, sample_x, onnx_dst, args.export_trt)

    if args.benchmark:
        if det.device.type != "cuda":
            raise RuntimeError("CUDA GPU required for `--benchmark`.")
        if args.cudagraph:
            g, _, _ = capture_static_cudagraph(lambda z: model(z), sample_x)
            ms = benchmark_cuda_callable(g.replay, det.device, batch_size=args.batch_size)
            out_report["cudagraph_median_ms_per_img"] = round(ms, 3)
        else:
            ms = benchmark_cuda_callable(lambda: model(sample_x), det.device, batch_size=args.batch_size)
            out_report["pytorch_median_ms_per_img"] = round(ms, 3)

    print(json.dumps(out_report, indent=2))


if __name__ == "__main__":
    main()
