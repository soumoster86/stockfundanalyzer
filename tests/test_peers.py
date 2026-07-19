"""Sector peer panel."""
import pandas as pd

from src.enrich import enrich
from src.peers import peer_context_line, sector_peers
from src.sample_data import sample_dataframe


def test_sector_peers_excludes_self():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    scored = enrich(df)
    # Both sample tickers have different sectors — peers may be empty
    peers, sector = sector_peers(scored, "TICKER1", n=5)
    assert sector is not None
    if not peers.empty:
        assert "TICKER1" not in peers["ticker"].values


def test_peers_same_sector_multi():
    rows = []
    for i, t in enumerate(["A", "B", "C", "D", "E", "F"]):
        rows.append({
            "ticker": t,
            "date": "2020-03-31",
            "sector": "Technology",
            "roe": 10 + i,
            "pe": 20 - i,
            "revenue_growth": 0.1,
            "net_margin": 10,
            "operating_margin": 12,
            "gross_margin": 30,
            "debt_to_equity": 0.5,
            "interest_coverage": 5,
            "current_ratio": 1.5,
            "cash_position": 100,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    scored = enrich(df)
    peers, sector = sector_peers(scored, "A", n=3)
    assert sector == "Technology"
    assert len(peers) == 3
    assert "A" not in peers["ticker"].values
    assert "rank_in_sector" in peers.columns


def test_peer_context_line():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    scored = enrich(df)
    line = peer_context_line(scored, "TICKER1")
    assert "sector" in line.lower() or "Sector" in line or "#" in line
