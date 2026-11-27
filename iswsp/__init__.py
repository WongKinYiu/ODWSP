"""
IR-ISWSP: Inverse-Reparameterized Weight-Space Programming for Instance Segmenters.
Added in ODWSP v0.1.1 (strictly zero-gradient, zero external NMS / Mask-NMS).
"""

from .consensus import (
    apply_single_pass_box_mask_synergy,
    fuse_detr_dual_geom_consensus_zero_nms,
    fuse_detr_soft_mask_consensus_zero_nms,
)
from .continuum import (
    get_iswsp_continuum_table,
    normalize_iswsp_budget_cli_argv,
    parse_iswsp_budget_spec,
    resolve_iswsp_budget_config,
)
from .ir_operators import (
    ExpandedAdaBN2d,
    ExpandedAdaLayerNorm,
    ExpandedBilinearMaskIR,
    ExpandedDCTConv2d,
    ExpandedPCAProcrustesLinear,
    ExpandedStemWhitenConv2d,
    apply_zero_image_corrupted_head_repair,
    solve_mm_fw_ridge,
    solve_stiefel_procrustes,
    solve_two_stage_full_matrix_wq,
)
from .models import (
    COCO80_NAMES,
    COCO80_TO_91,
    ISWSP_PAPER_MODELS,
    LoadedSegmenter,
    SegmenterSpec,
    compute_sha256,
    encode_binary_mask_rle,
    instantiate_segmenter,
    load_iswsp_segmenter,
    make_yolo26_seg_arch_cfg,
    normalize_iswsp_model_key,
    safe_load_checkpoint,
)
from .zero_grad_guard import enforce_no_gradients, max_ieee_fp32_diff

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "COCO80_NAMES",
    "COCO80_TO_91",
    "ExpandedAdaBN2d",
    "ExpandedAdaLayerNorm",
    "ExpandedBilinearMaskIR",
    "ExpandedDCTConv2d",
    "ExpandedPCAProcrustesLinear",
    "ExpandedStemWhitenConv2d",
    "ISWSP_PAPER_MODELS",
    "LoadedSegmenter",
    "SegmenterSpec",
    "apply_single_pass_box_mask_synergy",
    "apply_zero_image_corrupted_head_repair",
    "compute_sha256",
    "encode_binary_mask_rle",
    "enforce_no_gradients",
    "fuse_detr_dual_geom_consensus_zero_nms",
    "fuse_detr_soft_mask_consensus_zero_nms",
    "get_iswsp_continuum_table",
    "instantiate_segmenter",
    "load_iswsp_segmenter",
    "make_yolo26_seg_arch_cfg",
    "max_ieee_fp32_diff",
    "normalize_iswsp_budget_cli_argv",
    "normalize_iswsp_model_key",
    "parse_iswsp_budget_spec",
    "resolve_iswsp_budget_config",
    "safe_load_checkpoint",
    "solve_mm_fw_ridge",
    "solve_stiefel_procrustes",
    "solve_two_stage_full_matrix_wq",
]
