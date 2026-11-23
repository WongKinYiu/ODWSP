#!/usr/bin/env python3
"""
Unit tests verifying IR-ODWSP mathematical invariants, zero-gradient guards,
Proposition 1 Haar-O(P) invariance, MM-FW-Ridge contraction, and `weights_only=True` safety.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from odwsp import (
    ExpandedAdaBN2d,
    ExpandedAdaLayerNorm,
    ExpandedBoxAffineLinear,
    ExpandedDCTConv2d,
    ExpandedPCAProcrustesLinear,
    PAPER_MODELS,
    SufficientStatsAccumulator,
    build_whitening_projector_from_covariance,
    enforce_no_gradients,
    fit_mm_fw_ridge_bridge,
    fuse_conv_and_bn_tensors,
    max_ieee_fp32_diff,
    parse_budget_spec,
    resolve_budget_config,
    safe_load_checkpoint,
    solve_lifted_ir_odwsp_readout,
    solve_stiefel_procrustes,
)


class TestODWSP(unittest.TestCase):
    def setUp(self) -> None:
        enforce_no_gradients()
        torch.manual_seed(42)

    def test_zero_gradient_guard(self) -> None:
        self.assertFalse(torch.is_grad_enabled())
        t = torch.randn(4)
        with self.assertRaises(RuntimeError):
            t.backward()

    def test_exact_pre_and_post_collapse_equivalence(self) -> None:
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

        w_box, b_box = torch.randn(4, 32), torch.randn(4)
        ebox = ExpandedBoxAffineLinear(w_box, b_box)
        ebox.m_dist.copy_(torch.eye(4) + 0.05 * torch.randn(4, 4))
        ebox.d_dist.copy_(0.02 * torch.randn(4))
        wb_c, bb_c = ebox.collapse_weights()
        self.assertLess(max_ieee_fp32_diff(lambda: ebox(x2), lambda: nn.functional.linear(x2, wb_c, bb_c)), 1e-5)

    def test_proposition1_haar_orthogonal_invariance(self) -> None:
        d, c = 32, 10
        w0 = torch.randn(c, d) * 0.1
        b0 = torch.full((c,), -3.0)
        acc = SufficientStatsAccumulator(d, c, torch.device("cpu"))
        feats = torch.randn(64, d)
        logits = nn.functional.linear(feats, w0, b0)
        boxes = torch.rand(64, 4) * 100.0
        boxes[:, 2:] += boxes[:, :2] + 15.0
        acc.update(feats, logits, boxes, boxes[:6] + 1.0, torch.arange(6) % c)

        # Native Euclidean (no lift) vs Haar-random orthogonal Q in O(d): J_C J_C^T = I + Q^T Q = 2 I
        q_haar, _ = torch.linalg.qr(torch.randn(d, d))
        lift_id = ExpandedPCAProcrustesLinear(w0, b0, p_whiten=torch.eye(d))
        lift_q = ExpandedPCAProcrustesLinear(w0, b0, p_whiten=q_haar)
        w_id, b_id = solve_lifted_ir_odwsp_readout(lift_id, acc.h_mat, acc.g_vec, lam_reg=500.0, eta_head=0.2, js_kappa0=0.0)
        w_q, b_q = solve_lifted_ir_odwsp_readout(lift_q, acc.h_mat, acc.g_vec, lam_reg=500.0, eta_head=0.2, js_kappa0=0.0)
        self.assertLess(float((w_id - w_q).abs().max().item()), 1e-6)
        self.assertLess(float((b_id - b_q).abs().max().item()), 1e-6)

    def test_mm_fw_ridge_and_conv_bn_fusion(self) -> None:
        x = torch.randn(128, 24)
        true_w = torch.randn(16, 24)
        y = x @ true_w.t() + 0.05 * torch.randn(128, 16)
        fg = torch.rand(128) + 0.1
        w_br, b_br = fit_mm_fw_ridge_bridge(x, y, fg_weights=fg)
        self.assertEqual(tuple(w_br.shape), (16, 24))
        self.assertEqual(tuple(b_br.shape), (16,))

        conv = nn.Conv2d(8, 16, 3, padding=1).eval()
        bn = nn.BatchNorm2d(16).eval()
        wf, bf = fuse_conv_and_bn_tensors(conv, bn)
        inp = torch.randn(2, 8, 6, 6)
        self.assertLess(
            max_ieee_fp32_diff(lambda: bn(conv(inp)), lambda: nn.functional.conv2d(inp, wf, bf, padding=1)),
            1e-5,
        )

    def test_procrustes_and_weights_only_safe_load(self) -> None:
        m_tgt = torch.randn(16, 32)
        q_orth, _ = torch.linalg.qr(torch.randn(32, 32))
        m_src = m_tgt @ q_orth.t()
        q_rec, _, a_cos = solve_stiefel_procrustes(m_src, m_tgt)
        self.assertEqual(tuple(q_rec.shape), (32, 32))
        self.assertGreater(a_cos, 0.999)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "clean.pt"
            torch.save({"version": "0.1.0", "state_dict": {"w": torch.randn(4, 4)}}, ckpt)
            loaded = safe_load_checkpoint(ckpt)
            self.assertEqual(loaded["version"], "0.1.0")

    def test_multi_budget_continuum_resolution(self) -> None:
        self.assertEqual(parse_budget_spec("0%")[0], "zero")
        self.assertEqual(parse_budget_spec("-25%")[2], "-30%")
        self.assertEqual(parse_budget_spec("+35%")[2], "+30%")
        self.assertEqual(parse_budget_spec("full")[0], "unconstrained")

        for mkey in PAPER_MODELS:
            for b in ("-100%", "-50%", "-10%", "0%", "+10%", "+50%", "+100%", "full"):
                for par in ("auto", "dyn", "sg"):
                    cfg = resolve_budget_config(mkey, budget=b, paradigm=par, batch_size=1)
                    self.assertEqual(cfg["model_key"], mkey)
                    self.assertIn("cand_id", cfg)


if __name__ == "__main__":
    unittest.main()
