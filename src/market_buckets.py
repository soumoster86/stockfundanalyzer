"""
Market-cap buckets
------------------
Assign Large / Mid / Small (and optional Micro) labels from market_cap.
Thresholds are INR-oriented defaults (crores of rupees → absolute INR).
Also works if market_cap is already in absolute currency units.

Defaults (INR absolute):
  Large  ≥ 20,000 Cr  = 2e11
  Mid    ≥  5,000 Cr  = 5e10
  Small  ≥    500 Cr  = 5e9
  else Micro
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Absolute currency units (e.g. INR). Override via assign_market_cap_bucket thresholds=.
DEFAULT_THRESHOLDS = {
    "large": 200_000_000_000,   # 20,000 Cr
    "mid": 50_000_000_000,      # 5,000 Cr
    "small": 5_000_000_000,     # 500 Cr
}

BUCKET_ORDER = ["Large", "Mid", "Small", "Micro", "Unknown"]


def cap_bucket(value, thresholds: dict | None = None) -> str:
    """Map a single market_cap value to a bucket label."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if np.isnan(v) or v <= 0:
        return "Unknown"
    if v >= t["large"]:
        return "Large"
    if v >= t["mid"]:
        return "Mid"
    if v >= t["small"]:
        return "Small"
    return "Micro"


def assign_market_cap_bucket(
    df: pd.DataFrame,
    col: str = "market_cap",
    out_col: str = "mcap_bucket",
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Add `out_col` with Large/Mid/Small/Micro/Unknown. No-op if col missing."""
    out = df.copy()
    if col not in out.columns:
        out[out_col] = "Unknown"
        return out
    out[out_col] = out[col].map(lambda x: cap_bucket(x, thresholds))
    return out


def filter_by_bucket(
    df: pd.DataFrame,
    bucket: str,
    col: str = "mcap_bucket",
) -> pd.DataFrame:
    """Filter to one bucket; 'All' returns df unchanged."""
    if not bucket or bucket == "All" or col not in df.columns:
        return df
    return df[df[col] == bucket].copy()
