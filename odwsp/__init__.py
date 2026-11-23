"""
IR-ODWSP: Inverse-Reparameterized Weight-Space Programming for Object Detectors.
"""

from .contraction import fit_mm_fw_ridge_bridge, fuse_conv_and_bn_tensors
from .ir_operators import (
    ExpandedAdaBN2d,
    ExpandedAdaLayerNorm,
    ExpandedBoxAffineLinear,
    ExpandedDCTConv2d,
    ExpandedPCAProcrustesLinear,
    build_2d_dct_basis,
    build_whitening_projector_from_covariance,
)
from .models import (
    COCO80_NAMES,
    COCO80_TO_91,
    DetectorSpec,
    LoadedDetector,
    PAPER_MODELS,
    YOLOGated1xModule,
    compute_sha256,
    load_odwsp_detector,
    safe_load_checkpoint,
)
from .procrustes import (
    align_and_interpolate_trunk_parameters,
    extract_orthonormal_auxiliary_basis,
    solve_stiefel_procrustes,
)
from .routing import DualStreamDynamicForkJoin, benchmark_cuda_callable, capture_static_cudagraph
from .solver import (
    IRODWSPConfig,
    SufficientStatsAccumulator,
    apply_cross_object_collision_gate,
    assign_winner_exclusive_targets,
    pairwise_box_iou,
    solve_lifted_ir_odwsp_readout,
)
from .zero_grad_guard import enforce_no_gradients, max_ieee_fp32_diff

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "COCO80_NAMES",
    "COCO80_TO_91",
    "DetectorSpec",
    "DualStreamDynamicForkJoin",
    "ExpandedAdaBN2d",
    "ExpandedAdaLayerNorm",
    "ExpandedBoxAffineLinear",
    "ExpandedDCTConv2d",
    "ExpandedPCAProcrustesLinear",
    "IRODWSPConfig",
    "LoadedDetector",
    "PAPER_MODELS",
    "SufficientStatsAccumulator",
    "YOLOGated1xModule",
    "align_and_interpolate_trunk_parameters",
    "apply_cross_object_collision_gate",
    "assign_winner_exclusive_targets",
    "benchmark_cuda_callable",
    "build_2d_dct_basis",
    "build_whitening_projector_from_covariance",
    "capture_static_cudagraph",
    "compute_sha256",
    "enforce_no_gradients",
    "extract_orthonormal_auxiliary_basis",
    "fit_mm_fw_ridge_bridge",
    "fuse_conv_and_bn_tensors",
    "load_odwsp_detector",
    "max_ieee_fp32_diff",
    "pairwise_box_iou",
    "safe_load_checkpoint",
    "solve_lifted_ir_odwsp_readout",
    "solve_stiefel_procrustes",
]
