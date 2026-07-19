"""
Sector peer panel
-----------------
For a given ticker, return same-sector peers ranked by quality (excluding self).
"""

from __future__ import annotations

import pandas as pd


def sector_peers(
    data: pd.DataFrame,
    ticker: str,
    n: int = 5,
    quality_col: str = "quality_score",
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> tuple[pd.DataFrame, str | None]:
    """
    Return (peer_frame, sector_name).

    peer_frame columns when available: ticker, quality_score, red_flag_count,
    pe, roe, rank_in_sector. Empty frame if no sector or no peers.
    """
    if data is None or data.empty or ticker_col not in data.columns:
        return pd.DataFrame(), None

    row = data[data[ticker_col] == ticker]
    if row.empty:
        # try strip .NS match
        bare = str(ticker).replace(".NS", "")
        row = data[data[ticker_col].astype(str).str.replace(".NS", "", regex=False) == bare]
    if row.empty:
        return pd.DataFrame(), None

    latest = row.sort_values("date").iloc[-1] if "date" in row.columns else row.iloc[-1]
    if sector_col not in data.columns or pd.isna(latest.get(sector_col)):
        return pd.DataFrame(), None

    sector = latest[sector_col]
    peers = data[data[sector_col] == sector].copy()
    if "date" in peers.columns:
        peers = peers.sort_values("date").groupby(ticker_col, as_index=False).tail(1)

    if quality_col in peers.columns:
        peers = peers.sort_values(quality_col, ascending=False, na_position="last")
    peers = peers.reset_index(drop=True)
    peers["rank_in_sector"] = range(1, len(peers) + 1)

    others = peers[peers[ticker_col] != latest[ticker_col]].head(n)
    cols = [
        c
        for c in [
            "rank_in_sector",
            ticker_col,
            quality_col,
            "red_flag_count",
            "roe",
            "pe",
            "f_score",
            "z_band",
            "m_band",
        ]
        if c in others.columns
    ]
    return others[cols].reset_index(drop=True), str(sector)


def peer_context_line(data: pd.DataFrame, ticker: str) -> str:
    peers, sector = sector_peers(data, ticker, n=1)
    if not sector:
        return "No sector peers available."
    # full sector size
    full, _ = sector_peers(data, ticker, n=10_000)
    # rank of this ticker
    row = data[data["ticker"] == ticker]
    if row.empty:
        return f"Sector: {sector}"
    latest = row.iloc[-1]
    q = latest.get("quality_score")
    # recompute rank including self
    sec = data[data["sector"] == sector] if "sector" in data.columns else data.iloc[0:0]
    if "date" in sec.columns:
        sec = sec.sort_values("date").groupby("ticker", as_index=False).tail(1)
    if "quality_score" in sec.columns and len(sec):
        sec = sec.sort_values("quality_score", ascending=False, na_position="last")
        ranks = {t: i + 1 for i, t in enumerate(sec["ticker"])}
        r = ranks.get(ticker)
        n = len(sec)
        if r and pd.notna(q):
            return f"#{r} of {n} in **{sector}** (quality {float(q):.1f})"
    return f"Sector: **{sector}**"
