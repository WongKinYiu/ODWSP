"""
Bidirectional Box <-> Mask Geometric Synergy (`S2`) and Zero-External-NMS Strict 1-to-1
Dual-Geometry Soft-Mask Consensus (`S3`) for IR-ISWSP.

Preserves exact primary query cardinality (`0` duplicate detections, `0` external NMS,
`0` external Mask-NMS) while coupling bounding boxes and pixel-level soft mask logits.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .zero_grad_guard import enforce_no_gradients

enforce_no_gradients()


def _box_iou_xyxy(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    if b1.numel() == 0 or b2.numel() == 0:
        return torch.zeros((b1.shape[0], b2.shape[0]), device=b1.device, dtype=torch.float32)
    x1 = torch.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0.0) * (b1[:, 3] - b1[:, 1]).clamp(min=0.0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0.0) * (b2[:, 3] - b2[:, 1]).clamp(min=0.0)
    return inter / (a1[:, None] + a2[None, :] - inter).clamp(min=1e-6)


@torch.no_grad()
def apply_single_pass_box_mask_synergy(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    masks_bool: torch.Tensor,
    alpha_box: float = 0.25,
    beta_score: float = 0.12,
    tau_agree: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-Pass Bidirectional Box <-> Mask Geometric Synergy (`S2`, 0 external NMS):
      1. Extracts the tight pixel envelope b_tilde(M_i) of each predicted binary mask.
      2. Computes self-consistency agreement g_i = IoU(b_i, b_tilde(M_i)).
      3. Tightens boxes toward high-agreement mask envelopes and calibrates confidence scores.
    """
    enforce_no_gradients()
    n = int(boxes_xyxy.shape[0])
    if n == 0:
        return boxes_xyxy.clone(), scores.clone(), torch.zeros((0,), device=boxes_xyxy.device)

    h, w = int(masks_bool.shape[-2]), int(masks_bool.shape[-1])
    out_boxes = boxes_xyxy.float().clone()
    out_scores = scores.float().clone()
    agreements = torch.zeros((n,), device=boxes_xyxy.device, dtype=torch.float32)

    for i in range(n):
        m = masks_bool[i]
        ys, xs = torch.where(m)
        if ys.numel() < 4:
            continue
        tb = torch.tensor(
            [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0],
            device=boxes_xyxy.device,
            dtype=torch.float32,
        )
        g_i = float(_box_iou_xyxy(out_boxes[i : i + 1], tb.unsqueeze(0))[0, 0].item())
        agreements[i] = g_i
        if g_i >= tau_agree and alpha_box > 0.0:
            w_env = float(alpha_box) * g_i
            out_boxes[i] = (1.0 - w_env) * out_boxes[i] + w_env * tb
            out_boxes[i, [0, 2]] = out_boxes[i, [0, 2]].clamp(0.0, float(w))
            out_boxes[i, [1, 3]] = out_boxes[i, [1, 3]].clamp(0.0, float(h))
        if beta_score > 0.0:
            gain = 1.0 + float(beta_score) * (g_i - 0.85)
            out_scores[i] = (out_scores[i] * gain).clamp(0.0, 1.0)

    return out_boxes, out_scores, agreements


@torch.no_grad()
def _roi_soft_mask_iou(
    p_masks: torch.Tensor,
    a_masks: torch.Tensor,
    p_boxes_xyxy: torch.Tensor,
    a_boxes_xyxy: torch.Tensor,
    pool_size: int = 28,
) -> torch.Tensor:
    """Computes pairwise RoI-pooled soft-mask IoU between `[Qp, H, W]` and `[Qa, H, W]` logits."""
    qp, qa = int(p_masks.shape[0]), int(a_masks.shape[0])
    if qp == 0 or qa == 0:
        return torch.zeros((qp, qa), device=p_masks.device, dtype=torch.float32)
    p_prob = F.interpolate(torch.sigmoid(p_masks).unsqueeze(1), size=(pool_size, pool_size), mode="bilinear", align_corners=False).flatten(1)
    a_prob = F.interpolate(torch.sigmoid(a_masks).unsqueeze(1), size=(pool_size, pool_size), mode="bilinear", align_corners=False).flatten(1)
    inter = p_prob @ a_prob.t()
    sum_p = p_prob.sum(dim=1, keepdim=True)
    sum_a = a_prob.sum(dim=1, keepdim=True).t()
    return inter / (sum_p + sum_a - inter).clamp(min=1e-6)


@torch.no_grad()
def fuse_detr_dual_geom_consensus_zero_nms(
    primary: dict[str, torch.Tensor],
    aux_list: list[tuple[dict[str, torch.Tensor], float, bool]],
    iou_thr: float = 0.55,
    logit_scale: float = 0.45,
    box_scale: float = 0.45,
    mask_scale: float = 0.65,
    unmatched_damp: float = 0.12,
    use_dual_geom: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Strict 1-to-1 exclusive Dual-Geometry (`sqrt(IoU_box * IoU_mask)`) soft-mask consensus
    for DETR-family segmenters (ECSeg, RF-DETR-Seg) with exact cardinality conservation (`0` NMS).
    """
    p_logits = primary["pred_logits"].float()
    p_boxes = primary["pred_boxes"].float()
    p_masks = primary["pred_masks"].float()
    bsz, q_cnt, hm, wm = p_masks.shape
    p_scores, p_cls = torch.sigmoid(p_logits).max(dim=-1)

    logit_num = p_logits.clone()
    logit_den = torch.ones((bsz, q_cnt, 1), device=p_logits.device, dtype=torch.float32)
    box_num = p_boxes.clone()
    box_den = torch.ones((bsz, q_cnt, 1), device=p_boxes.device, dtype=torch.float32)
    mask_num = p_masks.clone()
    mask_den = torch.ones((bsz, q_cnt, 1, 1), device=p_masks.device, dtype=torch.float32)
    unmatched_pen = torch.zeros((bsz, q_cnt, 1), device=p_logits.device, dtype=torch.float32)

    for aux_out, w_aux, is_hf in aux_list:
        a_logits = aux_out["pred_logits"].float()
        a_boxes = aux_out["pred_boxes"].float().clone()
        a_masks = aux_out["pred_masks"].float()
        if is_hf:
            a_boxes[:, :, 0] = 1.0 - a_boxes[:, :, 0]
            a_masks = torch.flip(a_masks, dims=[-1])
        if a_masks.shape[-2:] != (hm, wm):
            a_masks = F.interpolate(a_masks, size=(hm, wm), mode="bilinear", align_corners=False)

        a_scores, a_cls = torch.sigmoid(a_logits).max(dim=-1)
        for b in range(bsz):
            pb = torchvision.ops.box_convert(p_boxes[b], in_fmt="cxcywh", out_fmt="xyxy")
            ab = torchvision.ops.box_convert(a_boxes[b], in_fmt="cxcywh", out_fmt="xyxy")
            box_ious = _box_iou_xyxy(pb, ab)
            if use_dual_geom:
                mask_ious = _roi_soft_mask_iou(p_masks[b], a_masks[b], pb, ab)
                geom_ious = torch.sqrt((box_ious * mask_ious).clamp(min=0.0))
            else:
                geom_ious = box_ious

            same_cls = p_cls[b].unsqueeze(1) == a_cls[b].unsqueeze(0)
            aff = (geom_ious * torch.sqrt(a_scores[b].clamp(min=1e-6)).unsqueeze(0) * same_cls.float()).cpu().numpy()
            ious_np = geom_ious.cpu().numpy()
            p_np = p_scores[b].cpu().numpy()
            a_np = a_scores[b].cpu().numpy()

            used_a = np.zeros(a_np.shape[0], dtype=bool)
            m_p, m_a, m_iou = [], [], []
            for pi in np.argsort(-p_np):
                if p_np[pi] < 0.01:
                    break
                row = np.where(used_a, -1.0, aff[pi]) if used_a.any() else aff[pi]
                aj = int(np.argmax(row))
                if row[aj] > 0.0 and ious_np[pi, aj] >= iou_thr and a_np[aj] >= 0.02:
                    used_a[aj] = True
                    m_p.append(int(pi))
                    m_a.append(aj)
                    m_iou.append(float(ious_np[pi, aj]))
                else:
                    unmatched_pen[b, int(pi), 0] += unmatched_damp * w_aux

            if not m_p:
                continue
            ip = torch.tensor(m_p, device=p_logits.device, dtype=torch.long)
            ia = torch.tensor(m_a, device=p_logits.device, dtype=torch.long)
            w_eff = (w_aux * torch.tensor(m_iou, device=p_logits.device, dtype=torch.float32)).clamp(0.0, 1.0)

            wl = (logit_scale * w_eff).unsqueeze(-1)
            wb = (box_scale * w_eff).unsqueeze(-1)
            wm_t = (mask_scale * w_eff).view(-1, 1, 1)
            logit_num[b, ip] += wl * a_logits[b, ia]
            logit_den[b, ip] += wl
            box_num[b, ip] += wb * a_boxes[b, ia]
            box_den[b, ip] += wb
            mask_num[b, ip] += wm_t * a_masks[b, ia]
            mask_den[b, ip] += wm_t

    n_aux = max(1, len(aux_list))
    return {
        "pred_logits": (logit_num / logit_den) - (unmatched_pen / n_aux),
        "pred_boxes": box_num / box_den,
        "pred_masks": mask_num / mask_den,
    }


@torch.no_grad()
def fuse_detr_soft_mask_consensus_zero_nms(
    primary: dict[str, torch.Tensor],
    aux_list: list[tuple[dict[str, torch.Tensor], float, bool]],
    iou_thr: float = 0.60,
    logit_scale: float = 0.45,
    box_scale: float = 0.45,
    mask_scale: float = 0.65,
    unmatched_damp: float = 0.12,
) -> dict[str, torch.Tensor]:
    """Box-anchored 1-to-1 exclusive soft-mask consensus (backward-compatible alias)."""
    return fuse_detr_dual_geom_consensus_zero_nms(
        primary=primary,
        aux_list=aux_list,
        iou_thr=iou_thr,
        logit_scale=logit_scale,
        box_scale=box_scale,
        mask_scale=mask_scale,
        unmatched_damp=unmatched_damp,
        use_dual_geom=False,
    )
