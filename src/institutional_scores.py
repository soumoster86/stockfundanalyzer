"""
Institutional Scores: Piotroski F, Altman Z, Beneish M
-----------------------------------------------------
Absolute (non-relative) measures that complement the percentile Quality Score.

Piotroski F-Score (0-9): nine binary financial-health tests. Higher = stronger.
Needs a multi-year panel (>=2 rows per ticker) for YoY tests.

Altman Z-Score: bankruptcy-risk formula.
  Green  (> 2.99) = safe · Yellow (1.81-2.99) = grey · Red (< 1.81) = distress

Beneish M-Score: earnings-manipulation probability model (8-variable form).
  M > -1.78  →  likely manipulator (red)
  M ≤ -2.22  →  unlikely (green)
  in between →  grey zone
Degrades gracefully when inputs are missing (m_score = NaN).
"""

from __future__ import annotations

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


# ---------------------------------------------------------------- Beneish M
# Classic 8-variable model (Beneish 1999). We compute the indices we can from
# the panel; if fewer than 4 of the 8 indices are available, M is left NaN.
BENEISH_INDICES = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "tata", "lvgi"]

BENEISH_COEF = {
    "intercept": -4.84,
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "tata": 4.679,
    "lvgi": -0.327,
}


def _safe_div(n, d):
    out = n / d
    if hasattr(out, "replace"):
        return out.replace([np.inf, -np.inf], np.nan)
    if np.isscalar(out) and not np.isfinite(out):
        return np.nan
    return out


def compute_beneish(df, by="ticker", date_col="date", min_indices=4):
    """
    Beneish M-Score on a multi-year panel (uses prior year via shift).

    Adds:
      m_dsri, m_gmi, ...  component indices (when computable)
      m_score             composite M (NaN if too few components)
      m_indices_used      count of non-null indices used
      m_band              Green / Yellow / Red / N/A
    """
    d = df.sort_values([by, date_col]).copy() if date_col in df.columns else df.copy()

    def prev(col):
        return _prev(d, col, by) if col in d.columns else np.nan

    # --- indices ---
    idx = {}

    # DSRI: (Receivables/Sales)_t / (Receivables/Sales)_{t-1}
    if {"receivables", "revenue"}.issubset(d.columns):
        r_s = _safe_div(d["receivables"], d["revenue"])
        r_s_prev = _safe_div(prev("receivables"), prev("revenue"))
        idx["dsri"] = _safe_div(r_s, r_s_prev)

    # GMI: GrossMargin_{t-1} / GrossMargin_t  (decline in GM → GMI > 1)
    if "gross_margin" in d.columns:
        # margins may be in %; ratio is unit-free either way
        idx["gmi"] = _safe_div(prev("gross_margin"), d["gross_margin"])

    # AQI: non-current assets quality proxy — needs total_assets + current_assets
    # AQI = [1 - (CA + PPE)/TA]_t / [1 - (CA + PPE)/TA]_{t-1}
    # Without PPE we use CA only as a soft proxy when total_assets present.
    if {"total_assets", "current_assets"}.issubset(d.columns):
        soft = 1.0 - _safe_div(d["current_assets"], d["total_assets"])
        soft_prev = 1.0 - _safe_div(prev("current_assets"), prev("total_assets"))
        idx["aqi"] = _safe_div(soft, soft_prev)

    # SGI: Sales_t / Sales_{t-1}
    if "revenue" in d.columns:
        idx["sgi"] = _safe_div(d["revenue"], prev("revenue"))

    # DEPI: depreciation index — skip unless depreciation columns exist
    if "depreciation" in d.columns and "ppe" in d.columns:
        dep_rate = _safe_div(d["depreciation"], d["depreciation"] + d["ppe"])
        dep_rate_prev = _safe_div(prev("depreciation"), prev("depreciation") + prev("ppe"))
        idx["depi"] = _safe_div(dep_rate_prev, dep_rate)

    # SGAI: SG&A / Sales ratio index
    if "sga" in d.columns and "revenue" in d.columns:
        sga_s = _safe_div(d["sga"], d["revenue"])
        sga_s_prev = _safe_div(prev("sga"), prev("revenue"))
        idx["sgai"] = _safe_div(sga_s, sga_s_prev)

    # TATA: (Net Income - OCF) / Total Assets  (total accruals)
    if {"net_profit", "operating_cash_flow"}.issubset(d.columns):
        accr = d["net_profit"] - d["operating_cash_flow"]
        if "total_assets" in d.columns:
            idx["tata"] = _safe_div(accr, d["total_assets"])
        elif "revenue" in d.columns:
            # weaker scale fallback
            idx["tata"] = _safe_div(accr, d["revenue"])

    # LVGI: leverage_t / leverage_{t-1}
    lev_col = (
        "total_debt" if "total_debt" in d.columns
        else ("debt_to_equity" if "debt_to_equity" in d.columns else None)
    )
    if lev_col:
        idx["lvgi"] = _safe_div(d[lev_col], prev(lev_col))

    for name in BENEISH_INDICES:
        d[f"m_{name}"] = idx.get(name, np.nan)

    # Composite: only where enough indices present
    used = pd.Series(0, index=d.index, dtype=int)
    m = pd.Series(BENEISH_COEF["intercept"], index=d.index, dtype=float)
    for name, coef in BENEISH_COEF.items():
        if name == "intercept":
            continue
        col = f"m_{name}"
        present = d[col].notna()
        used = used + present.astype(int)
        m = m + coef * d[col].fillna(0.0)

    d["m_indices_used"] = used
    d["m_score"] = np.where(used >= min_indices, m, np.nan)
    d["m_band"] = d["m_score"].apply(m_band)
    return d


def m_band(m):
    if pd.isna(m):
        return "N/A"
    if m > -1.78:
        return "Red"       # likely manipulator
    if m > -2.22:
        return "Yellow"
    return "Green"         # unlikely


M_BAND_TEXT = {
    "Green": "Unlikely earnings manipulator (M ≤ −2.22)",
    "Yellow": "Grey zone (−2.22 < M ≤ −1.78) — review accruals",
    "Red": "Likely manipulator (M > −1.78) — forensic review recommended",
    "N/A": "Not computable — need multi-year receivables/revenue/margins (and ideally total assets)",
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
