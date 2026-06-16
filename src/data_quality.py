"""
Data Quality Layer
------------------
Two independent checks that surface where the underlying data can't be trusted,
so a score built on thin or distorted inputs isn't read with false confidence.

1. completeness  -> how many of the core metrics actually populated (0..1 + count)
2. sanity_flags  -> corporate-action distortions and mathematically broken ratios

Neither alters the quality score; they annotate it. The philosophy: never
silently rank a number we have reason to distrust — mark it "verify manually".
"""

import numpy as np
import pandas as pd

# The core metrics we expect a complete fundamental row to carry.
CORE_METRICS = [
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth",
    "ebitda_growth", "roe", "roce", "net_margin", "operating_margin",
    "gross_margin", "debt_to_equity", "interest_coverage", "current_ratio",
    "cash_position", "dividend_yield", "pe", "pb", "ev_ebitda",
    "peg", "price_sales",
]

# Thresholds for sanity checks (tunable).
SANITY_THRESHOLDS = {
    "revenue_jump_pct": 1.5,     # >150% YoY revenue change -> likely corporate action
    "share_jump_pct": 1.5,       # >150% YoY share-count change -> merger/split/issue
    "pe_extreme": 200.0,         # P/E above this is effectively meaningless
    "margin_impossible": 100.0,  # net/operating margin > 100% -> accounting artifact
    "roe_extreme": 100.0,        # ROE above this usually = tiny/negative equity base
}


def data_completeness(df, metrics=CORE_METRICS):
    """
    Add per-row completeness columns:
      data_fields_present (int)  -> how many core metrics are non-null
      data_fields_total   (int)  -> how many core metrics exist as columns
      data_completeness    (0..1) -> ratio
    """
    out = df.copy()
    present_cols = [m for m in metrics if m in out.columns]
    total = len(present_cols)
    if total == 0:
        out["data_fields_present"] = 0
        out["data_fields_total"] = 0
        out["data_completeness"] = 0.0
        return out
    out["data_fields_present"] = out[present_cols].notna().sum(axis=1).astype(int)
    out["data_fields_total"] = total
    out["data_completeness"] = out["data_fields_present"] / total
    return out


def _yoy(df, col, by="ticker"):
    prev = df.groupby(by)[col].shift(1)
    return (df[col] - prev) / prev.abs()


def data_sanity_flags(df, by="ticker", date_col="date", thresholds=None):
    """
    Detect distortions that make a row's fundamentals untrustworthy. Operates on
    the full multi-year panel (corporate-action checks need YoY). Returns the
    frame with added boolean columns and a `data_warnings` list + `data_warning`
    summary flag per row.

    Checks:
      ca_revenue_jump   : |YoY revenue change| > threshold (merger/demerger/spinoff)
      ca_share_jump     : |YoY share-count change| > threshold (split/large issue)
      bad_negative_equity: D/E negative (negative equity base -> ratios meaningless)
      bad_pe_extreme    : P/E absurdly high
      bad_margin        : net or operating margin physically implausible (>100%)
      bad_roe_extreme   : |ROE| implausibly high (usually tiny equity base)
    """
    t = {**SANITY_THRESHOLDS, **(thresholds or {})}
    d = df.sort_values([by, date_col]).copy() if date_col in df.columns else df.copy()
    w = pd.DataFrame(index=d.index)

    # ---- corporate-action distortions (YoY) ----
    if "revenue" in d.columns and date_col in d.columns:
        rev_chg = _yoy(d, "revenue", by).abs()
        w["ca_revenue_jump"] = (rev_chg > t["revenue_jump_pct"]).fillna(False)
    if "shares_outstanding" in d.columns and date_col in d.columns:
        sh_chg = _yoy(d, "shares_outstanding", by).abs()
        w["ca_share_jump"] = (sh_chg > t["share_jump_pct"]).fillna(False)

    # ---- mathematically broken / implausible ratios (point-in-time) ----
    if "debt_to_equity" in d.columns:
        w["bad_negative_equity"] = (d["debt_to_equity"] < 0).fillna(False)
    if "pe" in d.columns:
        w["bad_pe_extreme"] = (d["pe"].abs() > t["pe_extreme"]).fillna(False)
    margin_bad = pd.Series(False, index=d.index)
    for mcol in ("net_margin", "operating_margin", "gross_margin"):
        if mcol in d.columns:
            margin_bad = margin_bad | (d[mcol].abs() > t["margin_impossible"]).fillna(False)
    w["bad_margin"] = margin_bad
    if "roe" in d.columns:
        w["bad_roe_extreme"] = (d["roe"].abs() > t["roe_extreme"]).fillna(False)

    warn_cols = list(w.columns)
    d = pd.concat([d, w], axis=1)
    if warn_cols:
        d["data_warning_count"] = d[warn_cols].sum(axis=1).astype(int)
        d["data_warnings"] = d[warn_cols].apply(
            lambda r: [c for c in warn_cols if r[c]], axis=1
        )
        d["data_warning"] = d["data_warning_count"] > 0
    else:
        d["data_warning_count"] = 0
        d["data_warnings"] = [[] for _ in range(len(d))]
        d["data_warning"] = False
    return d, warn_cols


WARNING_TEXT = {
    "ca_revenue_jump": "Revenue jumped >150% YoY — possible merger/demerger; figures may not be comparable",
    "ca_share_jump": "Share count changed >150% YoY — possible split/large issuance; per-share metrics distorted",
    "bad_negative_equity": "Negative equity — debt-to-equity and ROE are not meaningful",
    "bad_pe_extreme": "Extreme P/E (>200) — earnings near zero; valuation read unreliable",
    "bad_margin": "Margin exceeds 100% — likely a one-off accounting item, not operating reality",
    "bad_roe_extreme": "Extreme ROE (>100%) — usually a tiny or distorted equity base",
}
