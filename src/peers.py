"""
Peer panel
----------
For a given ticker, return same-industry (preferred) or same-sector peers
ranked by quality (excluding self).
"""

from __future__ import annotations

import pandas as pd


def _resolve_row(data: pd.DataFrame, ticker: str, ticker_col: str = "ticker") -> pd.Series | None:
    row = data[data[ticker_col] == ticker]
    if row.empty:
        bare = str(ticker).replace(".NS", "")
        row = data[data[ticker_col].astype(str).str.replace(".NS", "", regex=False) == bare]
    if row.empty:
        return None
    if "date" in row.columns:
        return row.sort_values("date").iloc[-1]
    return row.iloc[-1]


def sector_peers(
    data: pd.DataFrame,
    ticker: str,
    n: int = 5,
    quality_col: str = "quality_score",
    ticker_col: str = "ticker",
    sector_col: str = "sector",
    industry_col: str = "industry",
    prefer_industry: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """
    Return (peer_frame, group_label).

    Prefers industry peers when `industry` is populated and has at least 2
    other names; otherwise falls back to sector.

    peer_frame columns when available: ticker, quality_score, red_flag_count,
    pe, roe, rank_in_sector. Empty frame if no group or no peers.
    """
    if data is None or data.empty or ticker_col not in data.columns:
        return pd.DataFrame(), None

    latest = _resolve_row(data, ticker, ticker_col)
    if latest is None:
        return pd.DataFrame(), None

    group_col = None
    group_val = None
    group_label = None

    if (
        prefer_industry
        and industry_col in data.columns
        and pd.notna(latest.get(industry_col))
    ):
        ind = latest[industry_col]
        ind_peers = data[data[industry_col] == ind]
        if "date" in ind_peers.columns:
            ind_peers = ind_peers.sort_values("date").groupby(ticker_col, as_index=False).tail(1)
        n_others = (ind_peers[ticker_col] != latest[ticker_col]).sum()
        if n_others >= 2:
            group_col, group_val = industry_col, ind
            group_label = f"Industry: {ind}"

    if group_col is None:
        if sector_col not in data.columns or pd.isna(latest.get(sector_col)):
            return pd.DataFrame(), None
        group_col = sector_col
        group_val = latest[sector_col]
        group_label = str(group_val)

    peers = data[data[group_col] == group_val].copy()
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
    return others[cols].reset_index(drop=True), group_label


def peer_context_line(data: pd.DataFrame, ticker: str) -> str:
    peers, group_label = sector_peers(data, ticker, n=1)
    if not group_label:
        return "No sector/industry peers available."
    full, _ = sector_peers(data, ticker, n=10_000)
    n = len(full) + 1  # peers exclude self
    latest = _resolve_row(data, ticker)
    if latest is None:
        return f"{group_label}"
    q = latest.get("quality_score")
    # Rank among full group including self
    if "quality_score" in data.columns:
        # Rebuild full ranked group the same way sector_peers does
        full_inc, _ = sector_peers(data, ticker, n=10_000, prefer_industry=True)
        # sector_peers excludes self — estimate rank from quality among peers + self
        better = 0
        if pd.notna(q) and not full_inc.empty and "quality_score" in full_inc.columns:
            better = int((full_inc["quality_score"] > float(q)).sum())
        r = better + 1
        if pd.notna(q):
            return f"#{r} of {n} in **{group_label}** (quality {float(q):.1f})"
    return f"**{group_label}**"
