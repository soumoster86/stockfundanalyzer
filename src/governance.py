"""
India governance data merge
---------------------------
Yahoo Finance does not provide promoter pledging, insider trades, related-party
flags, or auditor changes. This module merges a separate governance CSV into
the fundamentals panel so red-flag rules can fire.

Expected governance CSV columns (ticker required; date optional):
  ticker, date (optional), auditor, promoter_pledge_pct,
  promoter_holding_change, insider_net_buy, related_party_txn_flag

If `date` is present, merge on (ticker, date), then fall back to same year.
If absent, apply the latest governance row per ticker to all fundamental rows
for that ticker.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

GOVERNANCE_COLS = [
    "auditor",
    "promoter_pledge_pct",
    "promoter_holding_change",
    "insider_net_buy",
    "related_party_txn_flag",
]

GOVERNANCE_TEMPLATE_ROWS = [
    {
        "ticker": "RELIANCE.NS",
        "date": "2024-03-31",
        "auditor": "Auditor A",
        "promoter_pledge_pct": 0.0,
        "promoter_holding_change": 0.0,
        "insider_net_buy": 10.0,
        "related_party_txn_flag": 0,
    },
    {
        "ticker": "EXAMPLE.NS",
        "date": "2024-03-31",
        "auditor": "Auditor B",
        "promoter_pledge_pct": 8.5,
        "promoter_holding_change": -1.2,
        "insider_net_buy": -50.0,
        "related_party_txn_flag": 1,
    },
]


def governance_template_csv() -> bytes:
    return pd.DataFrame(GOVERNANCE_TEMPLATE_ROWS).to_csv(index=False).encode("utf-8")


def load_governance_csv(source) -> pd.DataFrame:
    """Load governance data from a path, file-like, UploadedFile, bytes, or DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(source))
    elif hasattr(source, "read"):
        # Streamlit UploadedFile or file handle
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("Governance CSV must include a `ticker` column.")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
    keep = ["ticker"] + (["date"] if "date" in df.columns else [])
    keep += [c for c in GOVERNANCE_COLS if c in df.columns]
    return df[keep]


def _prefer_overlay(base: pd.Series, overlay) -> pd.Series:
    """Governance values win where present; otherwise keep fundamentals."""
    if isinstance(overlay, pd.Series):
        overlay_s = overlay.reindex(base.index)
    else:
        overlay_s = pd.Series(overlay, index=base.index)
    return overlay_s.combine_first(base)


def merge_governance(
    fundamentals: pd.DataFrame,
    governance,
) -> tuple[pd.DataFrame, dict]:
    """Overlay governance fields onto fundamentals. Returns (merged_df, stats)."""
    fund = fundamentals.copy()
    fund.columns = [str(c).strip().lower() for c in fund.columns]
    gov = load_governance_csv(governance)

    gov_cols = [c for c in GOVERNANCE_COLS if c in gov.columns]
    empty_stats = {
        "n_gov_rows": int(len(gov)),
        "n_tickers_matched": 0,
        "n_rows_with_gov": 0,
        "cols_merged": [],
        "mode": "none",
    }
    if not gov_cols:
        return fund, empty_stats

    for c in gov_cols:
        if c not in fund.columns:
            fund[c] = np.nan

    if (
        "date" in gov.columns
        and gov["date"].notna().any()
        and "date" in fund.columns
    ):
        mode = "by_ticker_date"
        fund = fund.copy()
        fund["_dt"] = pd.to_datetime(fund["date"], errors="coerce")
        g = gov.dropna(subset=["date"]).copy()
        g["_dt"] = pd.to_datetime(g["date"], errors="coerce")

        exact = g[["ticker", "_dt"] + gov_cols].drop_duplicates(
            subset=["ticker", "_dt"], keep="last"
        )
        fund = fund.merge(exact, on=["ticker", "_dt"], how="left", suffixes=("", "_g"))
        for c in gov_cols:
            gc = f"{c}_g"
            if gc in fund.columns:
                fund[c] = _prefer_overlay(fund[c], fund[gc])
                fund.drop(columns=[gc], inplace=True)

        # Year-level fallback only for rows still missing *all* gov fields
        missing_mask = fund[gov_cols].isna().all(axis=1)
        if missing_mask.any():
            g = g.copy()
            g["_year"] = g["_dt"].dt.year
            year_map = (
                g.sort_values("_dt")
                .groupby(["ticker", "_year"], as_index=False)
                .tail(1)[["ticker", "_year"] + gov_cols]
            )
            tmp = fund.loc[missing_mask, ["ticker"]].copy()
            tmp["_year"] = fund.loc[missing_mask, "_dt"].dt.year.to_numpy()
            tmp = tmp.reset_index().merge(year_map, on=["ticker", "_year"], how="left")
            tmp = tmp.set_index("index")
            for c in gov_cols:
                fund.loc[missing_mask, c] = _prefer_overlay(
                    fund.loc[missing_mask, c],
                    tmp[c].reindex(fund.loc[missing_mask].index),
                )

        fund.drop(columns=["_dt"], inplace=True, errors="ignore")
    else:
        mode = "by_ticker_latest"
        g = gov.sort_values("date") if "date" in gov.columns else gov
        latest = g.groupby("ticker", as_index=False).tail(1).set_index("ticker")
        for c in gov_cols:
            mapped = fund["ticker"].map(latest[c])
            fund[c] = _prefer_overlay(fund[c], mapped)

    stats = {
        "n_gov_rows": int(len(gov)),
        "n_tickers_matched": int(len(set(fund["ticker"].astype(str)) & set(gov["ticker"].astype(str)))),
        "n_rows_with_gov": int(fund[gov_cols].notna().any(axis=1).sum()),
        "cols_merged": gov_cols,
        "mode": mode,
    }
    return fund, stats
