"""
Object-Detection Set-Prediction WSP Closed-Form Solver & Collision Gate (Section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .ir_operators import ExpandedPCAProcrustesLinear
from .zero_grad_guard import enforce_no_gradients


@dataclass
class IRODWSPConfig:
    eta_head: float = 0.20
    lam_reg: float = 2800.0
    iou_gamma: float = 0.85
    focal_gamma: float = 2.0
    sibling_iou_thr: float = 0.40
    js_kappa0: float = 0.25
    eta_box: float = 0.15
    lam_box: float = 500.0
    gate_gamma_dn: float = 0.22
    gate_gamma_up: float = 0.16
    gate_tau_iou: float = 0.55


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise IoU between [M, 4] and [G, 4] boxes in (x1, y1, x2, y2) format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=torch.float32)
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    return inter / (a1[:, None] + a2[None, :] - inter).clamp(min=1e-6)


def assign_winner_exclusive_targets(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_gamma: float = 0.85,
    sibling_iou_thr: float = 0.40,
    alpha_ta: float = 0.5,
    beta_ta: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, float]]]:
    """
    Computes Task-Aligned Top-1 Winner Exclusivity, Sibling-Ignore Mask `m_{i,c}`,
    and Soft Localization-Quality Targets `q_{i,c} = IoU^gamma` (Eq. 5).
    """
    enforce_no_gradients()
    m_queries, num_classes = pred_logits.shape
    probs = torch.sigmoid(pred_logits.float())
    targets_q = torch.zeros((m_queries, num_classes), device=pred_logits.device, dtype=torch.float32)
    valid_mask = torch.ones((m_queries, num_classes), device=pred_logits.device, dtype=torch.float32)
    matched_pairs: list[tuple[int, int, float]] = []

    if gt_boxes.shape[0] == 0:
        return targets_q, valid_mask, matched_pairs

    ious = pairwise_box_iou(pred_boxes.float(), gt_boxes.float())
    used_queries: set[int] = set()
    for g_idx in range(gt_boxes.shape[0]):
        c_g = int(gt_classes[g_idx].item())
        ta_metric = (probs[:, c_g].clamp(min=1e-6) ** alpha_ta) * (ious[:, g_idx].clamp(min=0.0) ** beta_ta)
        for uq in used_queries:
            ta_metric[uq] = -1.0
        win_i = int(torch.argmax(ta_metric).item())
        win_iou = float(ious[win_i, g_idx].item())
        if win_iou <= 0.05:
            continue
        used_queries.add(win_i)
        sib_mask = ious[:, g_idx] >= sibling_iou_thr
        valid_mask[sib_mask, c_g] = 0.0
        valid_mask[win_i, c_g] = 1.0
        targets_q[win_i, c_g] = win_iou ** iou_gamma
        matched_pairs.append((win_i, g_idx, win_iou))

    return targets_q, valid_mask, matched_pairs


class SufficientStatsAccumulator:
    """Streaming forward-only accumulator for per-class Focal Hessians, gradients, and covariance."""

    def __init__(self, d_feat: int, num_classes: int, device: torch.device, focal_gamma: float = 2.0) -> None:
        enforce_no_gradients()
        self.d_feat = d_feat
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.h_mat = torch.zeros((num_classes, d_feat + 1, d_feat + 1), device=device, dtype=torch.float64)
        self.g_vec = torch.zeros((num_classes, d_feat + 1), device=device, dtype=torch.float64)
        self.cov_h = torch.zeros((d_feat, d_feat), device=device, dtype=torch.float64)
        self.n_tokens = 0
        self.box_xtx = torch.zeros((5, 5), device=device, dtype=torch.float64)
        self.box_xty = torch.zeros((5, 4), device=device, dtype=torch.float64)

    def update(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        iou_gamma: float = 0.85,
        sibling_iou_thr: float = 0.40,
    ) -> None:
        with torch.no_grad():
            q_tgt, m_mask, pairs = assign_winner_exclusive_targets(
                logits, pred_boxes, gt_boxes, gt_classes, iou_gamma=iou_gamma, sibling_iou_thr=sibling_iou_thr
            )
            h64 = features.detach().double()
            ones = torch.ones((h64.shape[0], 1), device=h64.device, dtype=torch.float64)
            hbar = torch.cat([h64, ones], dim=1)
            self.cov_h += h64.t() @ h64
            self.n_tokens += h64.shape[0]

            p64 = torch.sigmoid(logits.detach().double())
            diff = q_tgt.double() - p64
            mod = diff.abs().pow(self.focal_gamma) * m_mask.double()
            curv = mod * (p64 * (1.0 - p64)).clamp(min=1e-4)
            grad = mod * diff

            self.g_vec += grad.t() @ hbar
            for c in range(self.num_classes):
                wc = curv[:, c]
                self.h_mat[c] += (hbar * wc.unsqueeze(1)).t() @ hbar

            if pairs:
                p_idx = [p[0] for p in pairs]
                g_idx = [p[1] for p in pairs]
                w_iou = torch.tensor([p[2] ** 2 for p in pairs], device=h64.device, dtype=torch.float64)
                bp = pred_boxes[p_idx].detach().double()
                bg = gt_boxes[g_idx].detach().double()
                bp_aug = torch.cat([bp, torch.ones((len(pairs), 1), device=h64.device, dtype=torch.float64)], dim=1)
                self.box_xtx += (bp_aug * w_iou.unsqueeze(1)).t() @ bp_aug
                self.box_xty += (bp_aug * w_iou.unsqueeze(1)).t() @ bg


def solve_lifted_ir_odwsp_readout(
    lifted_head: ExpandedPCAProcrustesLinear,
    h_mat: torch.Tensor,
    g_vec: torch.Tensor,
    lam_reg: float = 2800.0,
    eta_head: float = 0.20,
    supercategory_groups: list[list[int]] | None = None,
    class_counts: torch.Tensor | None = None,
    js_kappa0: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solves the closed-form normal equations inside the lifted IR subspace (Eq. 12-13)
    and collapses back to native readout weights (W*, b*) with 0% extra parameters.
    """
    enforce_no_gradients()
    with torch.no_grad():
        j_c = lifted_head.jacobian_collapse()
        d_in, d_tilde = j_c.shape
        num_classes = lifted_head.w_core.shape[0]
        dev = j_c.device

        j_aug = torch.zeros((d_in + 1, d_tilde + 1), device=dev, dtype=torch.float64)
        j_aug[:d_in, :d_tilde] = j_c
        j_aug[d_in, d_tilde] = 1.0

        reg_eye = float(lam_reg) * torch.eye(d_tilde + 1, device=dev, dtype=torch.float64)
        reg_eye[d_tilde, d_tilde] = float(lam_reg) * 0.25

        delta_theta = torch.zeros((num_classes, d_tilde + 1), device=dev, dtype=torch.float64)
        for c in range(num_classes):
            h_lifted = j_aug.t() @ h_mat[c] @ j_aug + reg_eye
            g_lifted = j_aug.t() @ g_vec[c]
            delta_theta[c] = float(eta_head) * torch.linalg.solve(h_lifted, g_lifted)

        if supercategory_groups is not None and class_counts is not None and js_kappa0 > 0.0:
            cnt = class_counts.to(dev, dtype=torch.float64).clamp(min=1.0)
            n_geom = torch.exp(torch.log(cnt).mean())
            kappa = float(js_kappa0) * (n_geom / (cnt + n_geom))
            db = delta_theta[:, d_tilde].clone()
            for grp in supercategory_groups:
                if len(grp) > 1:
                    idx = torch.tensor(grp, device=dev, dtype=torch.long)
                    w_prec = torch.sqrt(cnt[idx])
                    mu_s = (db[idx] * w_prec).sum() / w_prec.sum().clamp(min=1e-8)
                    db[idx] = (1.0 - kappa[idx]) * db[idx] + kappa[idx] * mu_s
            delta_theta[:, d_tilde] = db

        r_pca = lifted_head.p_whiten.shape[0]
        r_proc = lifted_head.q_proc.shape[0]
        lifted_head.w_core.add_(delta_theta[:, :d_in].float())
        lifted_head.w_pca.copy_(delta_theta[:, d_in : d_in + r_pca].float())
        if r_proc > 0:
            lifted_head.b_proc.copy_(delta_theta[:, d_in + r_pca : d_tilde].float())
        lifted_head.b_vec.add_(delta_theta[:, d_tilde].float())
        return lifted_head.collapse_weights()


def apply_cross_object_collision_gate(
    o2o_logits: torch.Tensor,
    aux_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gamma_dn: float = 0.22,
    gamma_up: float = 0.16,
    tau_iou: float = 0.55,
    peak_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-parameter Pre-Postprocess Anchor-IoU Collision & Peak-Dominance Gate (Eq. 8)."""
    enforce_no_gradients()
    with torch.no_grad():
        scores = torch.sigmoid(o2o_logits.float()).max(dim=-1).values
        ious = pairwise_box_iou(pred_boxes.float(), pred_boxes.float())
        higher_mask = scores.unsqueeze(0) > (scores.unsqueeze(1) + 1e-6)
        max_higher_iou = (ious * higher_mask.float()).max(dim=1).values
        isolated = (max_higher_iou < tau_iou).float().unsqueeze(-1)
        if peak_mask is not None:
            isolated = isolated * peak_mask.float().view(-1, 1)

        dz = aux_logits.float() - o2o_logits.float()
        gated = o2o_logits.float() + gamma_dn * torch.clamp(dz, max=0.0) + gamma_up * isolated * torch.clamp(dz, min=0.0)
        return gated.to(o2o_logits.dtype)
