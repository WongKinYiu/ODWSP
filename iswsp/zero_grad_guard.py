"""Strict runtime zero-gradient guard and IEEE-754 FP32 equivalence verification for IR-ISWSP."""

from __future__ import annotations

from typing import Callable
import torch

_GUARD_INSTALLED = False

_MSG = (
    "Autograd / backpropagation is strictly prohibited in IR-ISWSP: "
    "all weight edits must be 100% gradient-free."
)


def _blocked_tensor_backward(*args, **kwargs):
    raise RuntimeError(_MSG)


def _blocked_autograd_backward(*args, **kwargs):
    raise RuntimeError(_MSG)


def _blocked_autograd_grad(*args, **kwargs):
    raise RuntimeError(_MSG)


def enforce_no_gradients() -> None:
    """Globally disables PyTorch autograd and installs runtime traps on backward/grad."""
    global _GUARD_INSTALLED
    torch.set_grad_enabled(False)
    if not _GUARD_INSTALLED:
        torch.Tensor.backward = _blocked_tensor_backward  # type: ignore[assignment]
        torch.autograd.backward = _blocked_autograd_backward  # type: ignore[assignment]
        torch.autograd.grad = _blocked_autograd_grad  # type: ignore[assignment]
        _GUARD_INSTALLED = True


def max_ieee_fp32_diff(fn_a: Callable[[], torch.Tensor], fn_b: Callable[[], torch.Tensor]) -> float:
    """Measures worst-case infinity-norm forward discrepancy under strict IEEE-754 FP32."""
    enforce_no_gradients()
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            return float((fn_a().float() - fn_b().float()).abs().max().item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c
