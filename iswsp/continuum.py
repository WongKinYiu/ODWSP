"""
Bidirectional Latency-Budget Continuum Lookup & CLI Budget Parser for IR-ISWSP (19 Segmenters).

All reference latencies and blind MS COCO val2017 Mask AP / Box AP figures are bundled in
`iswsp/continuum_table.json` (zero external path dependencies).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from .models import normalize_iswsp_model_key

_TABLE_CACHE: dict[str, Any] | None = None


def get_iswsp_continuum_table() -> dict[str, Any]:
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        tbl_path = Path(__file__).resolve().parent / "continuum_table.json"
        _TABLE_CACHE = json.loads(tbl_path.read_text(encoding="utf-8"))
    return _TABLE_CACHE


def normalize_iswsp_budget_cli_argv(argv: list[str] | None = None) -> list[str]:
    """Allows negative `--budget -30%` tokens to pass cleanly through `argparse`."""
    raw = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok == "--budget" and i + 1 < len(raw) and raw[i + 1].startswith("-"):
            out.append(f"--budget={raw[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def parse_iswsp_budget_spec(budget: str) -> tuple[str, float | None, str]:
    """Parses a budget string into `(regime, numeric_pct, canonical_tier)`."""
    s = budget.strip().lower()
    if s in ("0", "0%", "+0%", "-0%", "1x", "1.0x", "1.00x"):
        return "zero", 0.0, "0%"
    if s in ("s2", "syn", "synergy"):
        return "synergy", 2.0, "S2"
    if s in ("s3", "dual", "dual_geom"):
        return "dual_geom", 103.4, "S3"
    if s in ("full", "max", "unconstrained", "plus", "iswsp+"):
        return "unconstrained", None, "full"

    val = float(s.rstrip("%"))
    if abs(val) < 1e-9:
        return "zero", 0.0, "0%"
    if val < 0.0:
        snapped = max(-60, min(-10, int(round(val / 10.0)) * 10))
        return "negative", val, f"{snapped}%"
    if val <= 10.0:
        return "positive", val, "+10%"
    if val <= 50.0:
        return "positive", val, "+50%"
    return "positive", val, "+100%"


def resolve_iswsp_budget_config(
    model_key: str,
    budget: str = "0%",
    paradigm: str = "static",
) -> dict[str, Any]:
    """Resolves the Paper-Verified operating point for any of the 19 ISWSP segmenters."""
    mkey = normalize_iswsp_model_key(model_key)
    regime, pct, tier = parse_iswsp_budget_spec(budget)
    tbl = get_iswsp_continuum_table()
    rec = tbl["models"][mkey]

    if regime == "zero":
        entry = rec["zero_1x"]
    elif regime == "synergy":
        entry = rec["s2"]
    elif regime == "dual_geom":
        entry = rec["s3"]
    elif regime == "unconstrained":
        entry = rec["full"]
    elif regime == "negative":
        entry = rec["negative_tiers"][tier]
    else:
        entry = rec["positive_tiers"][tier]

    return {
        "model_key": mkey,
        "display_name": rec["display_name"],
        "family": rec["family"],
        "resolution": rec["resolution"],
        "params_m": rec["params_m"],
        "budget_requested": budget,
        "canonical_tier": tier,
        "regime": regime,
        "paradigm": paradigm,
        "base_pt1_ms": rec["base_pt1_ms"],
        "base_trt1_ms": rec["base_trt1_ms"],
        "official_mask_ap": rec["official"]["mask_ap"],
        "official_box_ap": rec["official"]["box_ap"],
        "operating_point": entry,
    }
