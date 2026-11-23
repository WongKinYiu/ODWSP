#!/usr/bin/env python3
"""
Latency-Budget-Aware Deployment, Hardware-Adaptive Solver & TensorRT CLI (`deploy.py`).

Supports:
  1. Multi-budget WSP execution & latency benchmarking (`--budget` in `[-100%, +100%]` or `full`).
  2. Live target-GPU hardware-adaptive budget solving (`--solve-on-gpu`): measures `T_0` and `T(g)`
     directly on your GPU and selects the Pareto-optimal graph `g*(beta)` satisfying `T(g) <= (1+beta)*T_0`.
  3. Constant-shape ONNX & Strongly-Typed FP16 TensorRT `.plan` compilation at any budget.
  4. Continuum table lookup (`--list-budgets`) and safe `weights_only=True` pack inspection (`--inspect-pack`).

Examples:
  python deploy.py --model yolo26s --list-budgets
  python deploy.py --model yolo26s --budget -10% --basins ./yolo26s-odwsp-basins.pt \
      --bridges ./odwsp-contraction-bridges-all17.tar.gz --fp16 --cudagraph --benchmark --device 0
  python deploy.py --model yolo26s --budget +50% --paradigm auto --basins ./yolo26s-odwsp-basins.pt \
      --solve-on-gpu --fp16 --cudagraph --device 0
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
    get_continuum_table,
    load_continuum_detector,
    resolve_budget_config,
    safe_load_checkpoint,
    solve_budget_on_target_gpu,
)
from odwsp.continuum import normalize_budget_cli_argv
from odwsp.models import normalize_model_key
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
    ap = argparse.ArgumentParser(description="IR-ODWSP Multi-Budget Deployment, GPU Solver & TensorRT CLI")
    ap.add_argument("--model", type=str, default="yolo26s", help="Model key (e.g., yolo26s, ecdet_s, rfdetr_m)")
    ap.add_argument("--budget", type=str, default="0%", help="Latency budget beta (e.g., -50%%, -10%%, 0%%, +10%%, +50%%, +100%%, full)")
    ap.add_argument("--paradigm", type=str, default="auto", choices=["auto", "dyn", "sg"], help="Positive-budget WSP paradigm")
    ap.add_argument("--weights", type=Path, default=None, help="Path to 1.00x ODWSP checkpoint (.pt) for --budget 0%%")
    ap.add_argument("--basins", type=Path, default=None, help="Path to multi-subspace basin pack (*-odwsp-basins.pt)")
    ap.add_argument("--bridges", type=Path, default=None, help="Path to contraction bridges (.pt or odwsp-contraction-bridges-all17.tar.gz)")
    ap.add_argument("--deploy-ckpt", type=Path, default=None, help="Optional pre-fused deploy archive/checkpoint")
    ap.add_argument("--cand-id", type=str, default=None, help="Optional explicit candidate graph ID override")
    ap.add_argument("--list-budgets", action="store_true", help="Print the full bidirectional budget continuum table for --model")
    ap.add_argument("--solve-on-gpu", action="store_true", help="Profile candidate graphs live on your GPU and solve g*(beta)")
    ap.add_argument("--inspect-pack", type=Path, default=None, help="Inspect a v0.1.0 basin or bridge pack safely (weights_only=True)")
    ap.add_argument("--edgecrafter-root", type=Path, default=None, help="Path to EdgeCrafter repo (for ecdet_*)")
    ap.add_argument("--sha256", type=str, default=None, help="Optional expected SHA-256 digest")
    ap.add_argument("--batch-size", type=int, default=1, choices=[1, 8])
    ap.add_argument("--fp16", action="store_true", help="Benchmark / solve in FP16 precision")
    ap.add_argument("--cudagraph", action="store_true", help="Capture and replay inside a static CUDA Graph")
    ap.add_argument("--benchmark", action="store_true", help="Measure median per-image GPU latency (ms/img)")
    ap.add_argument("--export-onnx", type=Path, default=None, help="Export static-shape ONNX graph")
    ap.add_argument("--export-trt", type=Path, default=None, help="Compile local TensorRT FP16 .plan engine")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save deployment / solver JSON report")
    ap.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., 0) or 'cpu'")
    args = ap.parse_args(normalize_budget_cli_argv())

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

    mkey = normalize_model_key(args.model)
    if args.list_budgets:
        tbl = get_continuum_table()
        tiers = tbl["supported_negative_tiers"] + ["0%"] + tbl["supported_positive_tiers"] + ["full"]
        rows = [
            resolve_budget_config(mkey, budget=t, paradigm=args.paradigm, batch_size=args.batch_size, cudagraph=True)
            for t in tiers
        ]
        print(json.dumps({"model": mkey, "paradigm": args.paradigm, "batch_size": args.batch_size, "continuum": rows}, indent=2))
        return

    dev_str = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device

    if args.solve_on_gpu:
        if args.basins is None:
            raise ValueError("`--solve-on-gpu` requires `--basins <model>-odwsp-basins.pt`.")
        solved = solve_budget_on_target_gpu(
            model_key=mkey,
            budget=args.budget,
            basins_path=args.basins,
            bridges_path=args.bridges,
            paradigm=args.paradigm,
            batch_size=args.batch_size,
            fp16=args.fp16 or True,
            cudagraph=args.cudagraph,
            device=dev_str,
            edgecrafter_root=args.edgecrafter_root,
        )
        print(json.dumps(solved, indent=2))
        if args.save_json is not None:
            args.save_json.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.save_json.resolve().write_text(json.dumps(solved, indent=2) + "\n", encoding="utf-8")
        return

    det = load_continuum_detector(
        model_key=mkey,
        budget=args.budget,
        paradigm=args.paradigm,
        batch_size=args.batch_size,
        cudagraph=args.cudagraph or True,
        weights_path=args.weights,
        basins_path=args.basins,
        bridges_path=args.bridges,
        deploy_ckpt_path=args.deploy_ckpt,
        cand_override=args.cand_id,
        device=dev_str,
        edgecrafter_root=args.edgecrafter_root,
        expected_sha256=args.sha256,
    )
    res = getattr(det, "input_res", det.spec.resolution)
    dtype = torch.float16 if (args.fp16 and det.device.type == "cuda") else torch.float32
    model = det.model.half().eval() if dtype == torch.float16 else det.model.float().eval()
    sample_x = torch.rand((args.batch_size, 3, res, res), device=det.device, dtype=dtype)

    out_report: dict[str, object] = {
        "model": det.spec.model_key,
        "budget": args.budget,
        "paradigm": args.paradigm,
        "cand_id": det.metadata.get("cand_id", "d0_gated" if det.spec.family == "yolo26" else "d0"),
        "resolution": res,
        "batch_size": args.batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "weights_only_verified": True,
    }
    if hasattr(det, "budget_info"):
        out_report["paper_reference"] = det.budget_info.get("paper_reference")

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
    if args.save_json is not None:
        args.save_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.save_json.resolve().write_text(json.dumps(out_report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
