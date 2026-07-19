"""Watchlist helpers."""
import pandas as pd

from src.watchlist import (
    add_ticker,
    filter_universe,
    get_watchlist,
    is_watched,
    remove_ticker,
    set_watchlist,
    watchlist_from_csv,
    watchlist_summary,
    watchlist_to_csv,
)


class _SS(dict):
    """Minimal session_state stand-in."""
    pass


def test_add_remove_roundtrip():
    ss = _SS()
    assert add_ticker(ss, "AAA.NS")
    assert not add_ticker(ss, "AAA.NS")
    assert is_watched(ss, "AAA.NS")
    assert get_watchlist(ss) == ["AAA.NS"]
    assert remove_ticker(ss, "AAA.NS")
    assert not is_watched(ss, "AAA.NS")


def test_csv_roundtrip():
    raw = watchlist_to_csv(["A", "B"])
    assert watchlist_from_csv(raw) == ["A", "B"]


def test_filter_and_summary(sample_panel):
    from src.enrich import enrich

    scored = enrich(sample_panel)
    tickers = scored["ticker"].unique().tolist()[:1]
    ss = _SS()
    set_watchlist(ss, tickers)
    sub = filter_universe(scored, get_watchlist(ss))
    assert len(sub) == 1
    s = watchlist_summary(scored, tickers)
    assert s["n"] == 1
    assert pd.notna(s["avg_quality"])
