"""
Institutional Scores: Piotroski F-Score & Altman Z-Score
--------------------------------------------------------
Two established, absolute (non-relative) measures that complement the
percentile-based Quality Score.

Piotroski F-Score (0-9): nine binary financial-health tests across
profitability, leverage/liquidity, and operating efficiency. Higher = stronger.
Each test needs the current year and the prior year, so the input must be a
multi-year panel (>=2 rows per ticker).

Altman Z-Score: a weighted formula estimating bankruptcy risk. Traffic light:
  Green  (> 2.99) = safe
  Yellow (1.81-2.99) = grey zone
  Red    (< 1.81) = distress

Both degrade gracefully: if some inputs are missing, the F-Score reports how
many of the nine tests could be evaluated, and the Z-Score returns NaN rather
than a misleading number.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- Piotroski
# Each test maps to a column we try to populate. Inputs needed (current + prior):
#   net_profit, operating_cash_flow, total_assets (for ROA/turnover),
#   total_debt or debt_to_equity (leverage), current_ratio, shares_outstanding,
#   gross_margin, revenue (for asset turnover)
PIOTROSKI_TESTS = [
    "pf_positive_net_income",
    "pf_positive_ocf",
    "pf_roa_improved",
    "pf_ocf_gt_net_income",      # accruals quality
    "pf_lower_leverage",
    "pf_higher_current_ratio",
    "pf_no_dilution",
    "pf_higher_gross_margin",
    "pf_higher_asset_turnover",
]

PIOTROSKI_LABELS = {
    "pf_positive_net_income": "Positive net income",
    "pf_positive_ocf": "Positive operating cash flow",
    "pf_roa_improved": "ROA improved year-over-year",
    "pf_ocf_gt_net_income": "Operating cash flow exceeds net income (clean accruals)",
    "pf_lower_leverage": "Leverage decreased",
    "pf_higher_current_ratio": "Current ratio improved (better liquidity)",
    "pf_no_dilution": "No share dilution",
    "pf_higher_gross_margin": "Gross margin improved",
    "pf_higher_asset_turnover": "Asset turnover improved (efficiency)",
}


def _prev(df, col, by="ticker"):
    return df.groupby(by)[col].shift(1)


def compute_piotroski(df, by="ticker", date_col="date"):
    """
    Compute the Piotroski F-Score per row (using each row vs its prior year).
    Adds:
      pf_<test>      : 1/0/NaN for each of the nine tests
      f_score        : sum of available tests (0-9 if all present)
      f_tests_used   : how many of the nine tests could be evaluated
      f_score_note   : text if partial
    Latest row per ticker carries the most recent comparison.
    """
    d = df.sort_values([by, date_col]).copy() if date_col in df.columns else df.copy()

    # derived inputs
    has_assets = "total_assets" in d.columns
    if has_assets:
        roa = d["net_profit"] / d["total_assets"] if "net_profit" in d.columns else np.nan
        roa_prev = _prev(d.assign(_roa=roa), "_roa", by) if "net_profit" in d.columns else np.nan
        turnover = d["revenue"] / d["total_assets"] if "revenue" in d.columns else np.nan
        turn_prev = _prev(d.assign(_t=turnover), "_t", by) if "revenue" in d.columns else np.nan

    tests = {}

    # 1. positive net income
    if "net_profit" in d.columns:
        tests["pf_positive_net_income"] = (d["net_profit"] > 0).astype(float)
    # 2. positive operating cash flow
    if "operating_cash_flow" in d.columns:
        tests["pf_positive_ocf"] = (d["operating_cash_flow"] > 0).astype(float)
    # 3. ROA improvement
    if has_assets and "net_profit" in d.columns:
        tests["pf_roa_improved"] = (roa > roa_prev).astype(float)
    # 4. OCF > net income (accruals)
    if {"operating_cash_flow", "net_profit"}.issubset(d.columns):
        tests["pf_ocf_gt_net_income"] = (d["operating_cash_flow"] > d["net_profit"]).astype(float)
    # 5. lower leverage (prefer total_debt, else debt_to_equity)
    lev_col = "total_debt" if "total_debt" in d.columns else (
        "debt_to_equity" if "debt_to_equity" in d.columns else None)
    if lev_col:
        tests["pf_lower_leverage"] = (d[lev_col] < _prev(d, lev_col, by)).astype(float)
    # 6. higher current ratio
    if "current_ratio" in d.columns:
        tests["pf_higher_current_ratio"] = (d["current_ratio"] > _prev(d, "current_ratio", by)).astype(float)
    # 7. no dilution (shares not up)
    if "shares_outstanding" in d.columns:
        tests["pf_no_dilution"] = (d["shares_outstanding"] <= _prev(d, "shares_outstanding", by)).astype(float)
    # 8. higher gross margin
    if "gross_margin" in d.columns:
        tests["pf_higher_gross_margin"] = (d["gross_margin"] > _prev(d, "gross_margin", by)).astype(float)
    # 9. higher asset turnover
    if has_assets and "revenue" in d.columns:
        tests["pf_higher_asset_turnover"] = (turnover > turn_prev).astype(float)

    for t in PIOTROSKI_TESTS:
        d[t] = tests.get(t, np.nan)

    test_cols = [t for t in PIOTROSKI_TESTS if t in tests]
    if test_cols:
        # NaN where the prior-year comparison is unavailable (first year): treat as not-evaluable
        present = d[test_cols]
        d["f_score"] = present.sum(axis=1, skipna=True)
        d["f_tests_used"] = present.notna().sum(axis=1)
    else:
        d["f_score"] = np.nan
        d["f_tests_used"] = 0

    d["f_score_note"] = d["f_tests_used"].apply(
        lambda n: "" if n == 9 else (f"partial: {int(n)}/9 tests" if n > 0 else "no data"))
    return d


def f_score_band(score, tests_used=9):
    if pd.isna(score) or tests_used == 0:
        return "N/A"
    if score >= 7:
        return "Strong"
    if score >= 4:
        return "Moderate"
    return "Weak"


# ---------------------------------------------------------------- Altman Z
def compute_altman_z(df):
    """
    Altman Z-Score (manufacturing/classic form):
      Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
      A = working capital / total assets
      B = retained earnings / total assets
      C = EBIT / total assets
      D = market cap / total liabilities
      E = revenue / total assets

    Needs: working_capital (or current_assets - current_liabilities),
           retained_earnings, ebit, market_cap, total_liabilities,
           total_assets, revenue.
    Returns z_score (NaN if inputs missing) and z_band traffic light.
    """
    d = df.copy()

    ta = d["total_assets"] if "total_assets" in d.columns else np.nan

    # working capital
    if "working_capital" in d.columns:
        wc = d["working_capital"]
    elif {"current_assets", "current_liabilities"}.issubset(d.columns):
        wc = d["current_assets"] - d["current_liabilities"]
    else:
        wc = np.nan

    re = d["retained_earnings"] if "retained_earnings" in d.columns else np.nan
    ebit = d["ebit"] if "ebit" in d.columns else (
        d["operating_income"] if "operating_income" in d.columns else np.nan)
    mcap = d["market_cap"] if "market_cap" in d.columns else np.nan
    tl = d["total_liabilities"] if "total_liabilities" in d.columns else np.nan
    rev = d["revenue"] if "revenue" in d.columns else np.nan

    def _safe_div(n, dd):
        out = n / dd
        return out.replace([np.inf, -np.inf], np.nan) if hasattr(out, "replace") else out

    A = _safe_div(wc, ta)
    B = _safe_div(re, ta)
    C = _safe_div(ebit, ta)
    D = _safe_div(mcap, tl)
    E = _safe_div(rev, ta)

    z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    d["z_score"] = z
    d["z_band"] = d["z_score"].apply(z_band)
    return d


def z_band(z):
    if pd.isna(z):
        return "N/A"
    if z > 2.99:
        return "Green"
    if z >= 1.81:
        return "Yellow"
    return "Red"


Z_BAND_TEXT = {
    "Green": "Safe zone — low bankruptcy risk (Z > 3)",
    "Yellow": "Grey zone — some distress risk (Z 1.8–3)",
    "Red": "Distress zone — elevated bankruptcy risk (Z < 1.8)",
    "N/A": "Not computable — missing balance-sheet inputs",
}


# ---------------------------------------------------------------- combine
def blend_with_quality(df, quality_col="quality_score",
                       f_weight=0.15, out_col="quality_plus"):
    """
    Optionally fold the F-Score into the Quality Score as a light tilt.
    F-Score (0-9) is scaled to 0-100 and blended. Rows without an F-Score keep
    their original quality score unchanged.
    """
    d = df.copy()
    if quality_col not in d.columns or "f_score" not in d.columns:
        d[out_col] = d.get(quality_col, np.nan)
        return d
    f_scaled = (d["f_score"] / 9.0) * 100.0
    blended = (1 - f_weight) * d[quality_col] + f_weight * f_scaled
    d[out_col] = np.where(d["f_score"].notna(), blended, d[quality_col])
    return d
