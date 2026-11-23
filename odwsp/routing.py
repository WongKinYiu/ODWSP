"""
Latency-Budget-Aware Dual-Stream Dynamic (`IR-ODWSP-Dyn`, Algorithm 2) and
Monolithic Single-Engine Static (`IR-ODWSP-SG`, Algorithm 3) Graph Execution.
"""

from __future__ import annotations

from typing import Callable
import torch
import torch.nn as nn

from .zero_grad_guard import enforce_no_gradients


class DualStreamDynamicForkJoin(nn.Module):
    """
    Executes primary and auxiliary WSP sub-branches concurrently across two CUDA streams
    (`s_pri`, `s_aux`) synchronized via static `cudaEvent` fork-join dependencies,
    ready for single-launch `torch.cuda.CUDAGraph` capture (Algorithm 2).
    """

    def __init__(
        self,
        primary_branch: nn.Module,
        aux_branch: nn.Module,
        consensus_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__()
        enforce_no_gradients()
        self.primary_branch = primary_branch.to(device).eval()
        self.aux_branch = aux_branch.to(device).eval()
        self.consensus_fn = consensus_fn
        self.device = device
        self.aux_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.ev_fork = torch.cuda.Event() if device.type == "cuda" else None
        self.ev_join = torch.cuda.Event() if device.type == "cuda" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.aux_stream is None or x.device.type != "cuda":
                y_pri = self.primary_branch(x)
                y_aux = self.aux_branch(x)
                return self.consensus_fn(y_pri, y_aux)

            cur_stream = torch.cuda.current_stream(self.device)
            assert self.ev_fork is not None and self.ev_join is not None
            self.ev_fork.record(cur_stream)
            y_pri = self.primary_branch(x)
            with torch.cuda.stream(self.aux_stream):
                self.aux_stream.wait_event(self.ev_fork)
                y_aux = self.aux_branch(x)
                self.ev_join.record(self.aux_stream)
            cur_stream.wait_event(self.ev_join)
            return self.consensus_fn(y_pri, y_aux)


def capture_static_cudagraph(
    fn: Callable[[torch.Tensor], torch.Tensor],
    sample_input: torch.Tensor,
    warmup_iters: int = 3,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    """
    Captures `fn(static_input) -> static_output` into a single `torch.cuda.CUDAGraph`.
    Returns `(graph, static_input, static_output)`.
    """
    enforce_no_gradients()
    if sample_input.device.type != "cuda":
        raise RuntimeError("CUDAGraph capture requires a CUDA tensor.")
    static_in = sample_input.detach().clone().contiguous()
    stream = torch.cuda.Stream(device=static_in.device)
    stream.wait_stream(torch.cuda.current_stream(static_in.device))
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(warmup_iters):
            _ = fn(static_in)
    torch.cuda.current_stream(static_in.device).wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        static_out = fn(static_in)
    return graph, static_in, static_out


def benchmark_cuda_callable(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
    batch_size: int = 1,
) -> float:
    """Returns median per-image GPU latency (ms/img) measured via CUDA events."""
    enforce_no_gradients()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(device)
        times = sorted(s.elapsed_time(e) / float(batch_size) for s, e in zip(starts, ends))
        return float(times[len(times) // 2])
