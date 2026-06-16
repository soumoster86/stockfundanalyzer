"""
Red Flag Detection
-------------------
Transparent, rule-based forensic checks. Each rule returns a boolean per row
plus a human-readable reason. Works on a time-sorted panel; YoY rules expect
prior-period columns or are computed via groupby(ticker).shift().

Thresholds are configurable; defaults are conventional starting points.
"""

import pandas as pd


DEFAULT_THRESHOLDS = {
    "receivables_vs_revenue_mult": 1.5,   # receivables growth > 1.5x revenue growth
    "share_dilution_pct": 0.05,           # >5% YoY share count increase
    "debt_spike_pct": 0.50,               # >50% YoY debt increase
    "interest_coverage_min": 2.5,         # below this is a flag
    "promoter_pledge_increase_pp": 1.0,   # +1 percentage point pledge
    "insider_net_sell_threshold": 0.0,    # net insider selling < 0
}


def _yoy(df, col, by="ticker"):
    """Year-over-year % change assuming one row per period per ticker, sorted by date."""
    prev = df.groupby(by)[col].shift(1)
    return (df[col] - prev) / prev.abs()


def detect_red_flags(df: pd.DataFrame, by="ticker", date_col="date",
                     thresholds=None) -> pd.DataFrame:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    d = df.sort_values([by, date_col]).copy()
    flags = pd.DataFrame(index=d.index)

    # ---- Earnings Quality ----
    # Rising profits + falling cash flow
    if {"net_profit", "operating_cash_flow"}.issubset(d.columns):
        profit_up = _yoy(d, "net_profit", by) > 0
        ocf_down = _yoy(d, "operating_cash_flow", by) < 0
        flags["eq_profit_up_cashflow_down"] = (profit_up & ocf_down).fillna(False)

    # Excessive receivables growth vs revenue
    if {"receivables", "revenue"}.issubset(d.columns):
        rec_g = _yoy(d, "receivables", by)
        rev_g = _yoy(d, "revenue", by)
        flags["eq_receivables_outpace_revenue"] = (
            (rec_g > t["receivables_vs_revenue_mult"] * rev_g) & (rev_g > 0)
        ).fillna(False)

    # Frequent equity dilution
    if "shares_outstanding" in d.columns:
        sh_g = _yoy(d, "shares_outstanding", by)
        flags["eq_equity_dilution"] = (sh_g > t["share_dilution_pct"]).fillna(False)

    # ---- Financial Risks ----
    if "total_debt" in d.columns:
        debt_g = _yoy(d, "total_debt", by)
        flags["fr_debt_spike"] = (debt_g > t["debt_spike_pct"]).fillna(False)

    if "interest_coverage" in d.columns:
        ic_prev = d.groupby(by)["interest_coverage"].shift(1)
        flags["fr_falling_interest_coverage"] = (
            (d["interest_coverage"] < ic_prev)
            & (d["interest_coverage"] < t["interest_coverage_min"])
        ).fillna(False)

    if "auditor" in d.columns:
        prev_aud = d.groupby(by)["auditor"].shift(1)
        flags["fr_auditor_change"] = (
            (d["auditor"] != prev_aud) & prev_aud.notna()
        ).fillna(False)

    # ---- Governance Risks ----
    if "promoter_pledge_pct" in d.columns:
        pledge_chg = d["promoter_pledge_pct"] - d.groupby(by)["promoter_pledge_pct"].shift(1)
        flags["gov_promoter_pledging"] = (
            pledge_chg > t["promoter_pledge_increase_pp"]
        ).fillna(False)

    if "insider_net_buy" in d.columns:  # negative = net selling
        flags["gov_insider_selling"] = (
            d["insider_net_buy"] < t["insider_net_sell_threshold"]
        ).fillna(False)

    if "related_party_txn_flag" in d.columns:
        flags["gov_related_party_txn"] = (
            d["related_party_txn_flag"].fillna(0).astype(float) > 0
        )

    # Aggregate
    flag_cols = list(flags.columns)
    if flag_cols:
        d = pd.concat([d, flags], axis=1)
        d["red_flag_count"] = d[flag_cols].sum(axis=1)
        d["red_flags"] = d[flag_cols].apply(
            lambda r: [c for c in flag_cols if r[c]], axis=1
        )
    else:
        # No raw columns present to evaluate any rule
        d["red_flag_count"] = 0
        d["red_flags"] = [[] for _ in range(len(d))]
    return d, flag_cols


REASON_TEXT = {
    "eq_profit_up_cashflow_down": "Profits rising while operating cash flow falls",
    "eq_receivables_outpace_revenue": "Receivables growing far faster than revenue",
    "eq_equity_dilution": "Significant share count increase (dilution)",
    "fr_debt_spike": "Sharp YoY increase in total debt",
    "fr_falling_interest_coverage": "Interest coverage falling below safe level",
    "fr_auditor_change": "Auditor changed vs prior period",
    "gov_promoter_pledging": "Increase in promoter share pledging",
    "gov_insider_selling": "Net insider selling",
    "gov_related_party_txn": "Related-party transactions flagged",
}
