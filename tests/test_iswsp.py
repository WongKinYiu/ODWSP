#!/usr/bin/env python3
"""
Unit tests verifying IR-ISWSP mathematical invariants, zero-gradient guards,
Theorem 1 exact bilinear collapse, Two-Stage Full-Matrix W_q Gauss-Newton,
label-free Stem Conv2d whitening, zero-image corrupted head repair,
strict `weights_only=True` safety, S2/S3 Zero-NMS consensus, and 19-model continuum.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from iswsp import (
    ExpandedAdaBN2d,
    ExpandedAdaLayerNorm,
    ExpandedBilinearMaskIR,
    ExpandedDCTConv2d,
    ExpandedPCAProcrustesLinear,
    ExpandedStemWhitenConv2d,
    ISWSP_PAPER_MODELS,
    apply_single_pass_box_mask_synergy,
    apply_zero_image_corrupted_head_repair,
    enforce_no_gradients,
    fuse_detr_dual_geom_consensus_zero_nms,
    fuse_detr_soft_mask_consensus_zero_nms,
    max_ieee_fp32_diff,
    parse_iswsp_budget_spec,
    resolve_iswsp_budget_config,
    safe_load_checkpoint,
    solve_mm_fw_ridge,
    solve_stiefel_procrustes,
    solve_two_stage_full_matrix_wq,
)


class _DummyUnsafePayload:
    pass


class TestISWSP(unittest.TestCase):
    def setUp(self) -> None:
        enforce_no_gradients()
        torch.manual_seed(42)

    def test_zero_gradient_guard(self) -> None:
        self.assertFalse(torch.is_grad_enabled())
        t = torch.randn(4)
        with self.assertRaises(RuntimeError):
            t.backward()

    def test_exact_collapse_norm_dct_and_stem_whitening(self) -> None:
        bn = nn.BatchNorm2d(16).eval()
        ebn = ExpandedAdaBN2d(bn)
        x4 = torch.randn(2, 16, 8, 8)
        self.assertLess(max_ieee_fp32_diff(lambda: ebn(x4), lambda: bn(x4)), 1e-6)
        ebn.set_empirical_bayes_moments(torch.randn(16) * 0.1, torch.ones(16) * 1.2, eta=0.3)
        bn_col = ebn.collapse()
        self.assertLess(max_ieee_fp32_diff(lambda: ebn(x4), lambda: bn_col(x4)), 1e-5)

        ln = nn.LayerNorm(32).eval()
        eln = ExpandedAdaLayerNorm(ln)
        x2 = torch.randn(8, 32)
        eln.set_empirical_bayes_moments(torch.randn(32) * 0.1, torch.ones(32) * 0.9, eta=0.25)
        ln_col = eln.collapse()
        self.assertLess(max_ieee_fp32_diff(lambda: eln(x2), lambda: ln_col(x2)), 1e-5)

        conv = nn.Conv2d(8, 12, kernel_size=3, padding=1).eval()
        edct = ExpandedDCTConv2d(conv)
        x_c = torch.randn(2, 8, 8, 8)
        self.assertLess(max_ieee_fp32_diff(lambda: edct(x_c), lambda: conv(x_c)), 1e-6)
        edct.set_frequency_band_scales(1.02, 0.98, 1.05)
        conv_col = edct.collapse()
        self.assertLess(max_ieee_fp32_diff(lambda: edct(x_c), lambda: conv_col(x_c)), 1e-5)

        # Stem Conv2d + BN exact whitening fold (interior pixels, padding=0)
        stem = nn.Conv2d(3, 12, kernel_size=3, padding=0, bias=False).eval()
        stem_bn = nn.BatchNorm2d(12).eval()
        ew = ExpandedStemWhitenConv2d(stem, stem_bn)
        ew.set_shift_moments(
            torch.tensor([0.24, 0.23, 0.22]),
            torch.tensor([0.11, 0.10, 0.10]),
            torch.tensor([0.48, 0.45, 0.41]),
            torch.tensor([0.23, 0.22, 0.22]),
            alpha=0.85,
        )
        c_col, b_col = ew.collapse()
        x_rgb = torch.rand(2, 3, 10, 10)
        self.assertLess(max_ieee_fp32_diff(lambda: ew(x_rgb), lambda: b_col(c_col(x_rgb))), 1e-5)

    def test_theorem1_exact_bilinear_mask_ir_collapse(self) -> None:
        bsz, q_cnt, d_q, c_p, k, h, w = 2, 16, 64, 32, 24, 12, 12
        op = ExpandedBilinearMaskIR(
            torch.randn(k, d_q) * 0.1,
            torch.randn(k) * 0.05,
            torch.randn(k, c_p, 1, 1) * 0.1,
            torch.randn(k) * 0.05,
            torch.randn(1) * 0.1,
        )
        op.set_bilinear_update(
            ds=torch.randn(k) * 0.08,
            dd=torch.randn(k) * 0.05,
            de=torch.randn(k) * 0.05,
            db0=0.07,
        )
        h_q = torch.randn(bsz, q_cnt, d_q)
        f_p = torch.randn(bsz, c_p, h, w)
        diff = max_ieee_fp32_diff(lambda: op.forward_expanded(h_q, f_p), lambda: op.forward_collapsed(h_q, f_p))
        self.assertLess(diff, 1e-5)

    def test_two_stage_full_matrix_wq_and_zero_image_repair(self) -> None:
        k, d_q, n_inst, s_px = 16, 24, 12, 36
        w_q = torch.randn(k, d_q) * 0.1
        b_q = torch.zeros(k)
        h_f = torch.randn(n_inst, d_q)
        p_c = torch.randn(n_inst, k, s_px)
        y_gt = (torch.rand(n_inst, s_px) > 0.5).float()
        bnd = (torch.rand(n_inst, s_px) > 0.8).float()
        wq_s, bq_s = solve_two_stage_full_matrix_wq(w_q, b_q, h_f, p_c, y_gt, boundary_masks=bnd)
        self.assertEqual(tuple(wq_s.shape), (k, d_q))
        self.assertEqual(tuple(bq_s.shape), (k,))
        self.assertGreater(float((wq_s - w_q).abs().max().item()), 0.0)

        w_proto = torch.randn(32, 16, 3, 3) * 0.1
        w_proto[:4] *= 5.0  # inject channel-norm spike on 4 channels
        w_rep, audit = apply_zero_image_corrupted_head_repair(w_proto, mad_thresh=2.5)
        self.assertEqual(tuple(w_rep.shape), tuple(w_proto.shape))
        self.assertGreaterEqual(audit["channels_norm_winsorized"], 4.0)

    def test_procrustes_bridge_and_strict_weights_only_security(self) -> None:
        m_tgt = torch.randn(16, 32)
        q_orth, _ = torch.linalg.qr(torch.randn(32, 32))
        m_src = m_tgt @ q_orth.t()
        q_rec, _, cos_sim = solve_stiefel_procrustes(m_src, m_tgt)
        self.assertEqual(tuple(q_rec.shape), (32, 32))
        self.assertGreater(cos_sim, 0.999)

        x = torch.randn(128, 24)
        w_true = torch.randn(16, 24)
        y = x @ w_true.t() + 0.02 * torch.randn(128, 16)
        w_br, b_br = solve_mm_fw_ridge(x, y, fg_weights=torch.rand(128) + 0.2)
        self.assertEqual(tuple(w_br.shape), (16, 24))
        self.assertEqual(tuple(b_br.shape), (16,))

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "clean.pt"
            torch.save({"version": "0.1.1", "state_dict": {"w": torch.randn(4, 4)}}, ckpt)
            self.assertEqual(safe_load_checkpoint(ckpt)["version"], "0.1.1")

            # Security check: untrusted custom class pickle MUST be rejected by weights_only=True
            bad = Path(td) / "unsafe.pt"
            torch.save({"payload": _DummyUnsafePayload()}, bad)
            with self.assertRaises(Exception):
                safe_load_checkpoint(bad)

    def test_s2_synergy_and_zero_nms_dual_geom_s3_consensus(self) -> None:
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        scores = torch.tensor([0.85])
        masks = torch.zeros((1, 64, 64), dtype=torch.bool)
        masks[0, 12:48, 12:48] = True
        b_out, s_out, ag = apply_single_pass_box_mask_synergy(boxes, scores, masks)
        self.assertEqual(tuple(b_out.shape), (1, 4))
        self.assertGreater(float(ag[0].item()), 0.75)
        self.assertGreater(float(b_out[0, 0].item()), 10.0)  # tightened inward

        bsz, q_cnt, n_cls, h, w = 1, 20, 80, 16, 16
        prim = {
            "pred_logits": torch.randn(bsz, q_cnt, n_cls) - 2.0,
            "pred_boxes": torch.cat([torch.rand(bsz, q_cnt, 2) * 0.6 + 0.2, torch.rand(bsz, q_cnt, 2) * 0.3 + 0.1], dim=-1),
            "pred_masks": torch.randn(bsz, q_cnt, h, w),
        }
        prim["pred_logits"][0, 0, 5] = 3.0
        aux_hf = {
            "pred_logits": prim["pred_logits"].clone(),
            "pred_boxes": prim["pred_boxes"].clone(),
            "pred_masks": torch.flip(prim["pred_masks"], dims=[-1]),
        }
        aux_hf["pred_boxes"][:, :, 0] = 1.0 - aux_hf["pred_boxes"][:, :, 0]
        fused_s3 = fuse_detr_dual_geom_consensus_zero_nms(prim, [(aux_hf, 0.9, True)])
        self.assertEqual(tuple(fused_s3["pred_masks"].shape), (bsz, q_cnt, h, w))
        self.assertLess(float((fused_s3["pred_masks"][0, 0] - prim["pred_masks"][0, 0]).abs().max().item()), 1e-5)

        fused_box = fuse_detr_soft_mask_consensus_zero_nms(prim, [(aux_hf, 0.9, True)])
        self.assertEqual(tuple(fused_box["pred_masks"].shape), (bsz, q_cnt, h, w))

    def test_iswsp_19_model_continuum_resolution(self) -> None:
        self.assertEqual(len(ISWSP_PAPER_MODELS), 19)
        self.assertEqual(parse_iswsp_budget_spec("0%")[0], "zero")
        self.assertEqual(parse_iswsp_budget_spec("S2")[0], "synergy")
        self.assertEqual(parse_iswsp_budget_spec("-25%")[2], "-20%")
        self.assertEqual(parse_iswsp_budget_spec("full")[0], "unconstrained")

        for mkey in ISWSP_PAPER_MODELS:
            for b in ("-60%", "-30%", "-10%", "0%", "S2", "+10%", "+50%", "+100%", "S3", "full"):
                cfg = resolve_iswsp_budget_config(mkey, budget=b)
                self.assertEqual(cfg["model_key"], mkey)
                self.assertIn("operating_point", cfg)


if __name__ == "__main__":
    unittest.main()
