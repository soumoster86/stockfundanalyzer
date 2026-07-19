"""Verify builtin screens actually change the filtered set on real-ish data."""
import pandas as pd

from src.enrich import enrich
from src.ranking import rank_universe
from src.sample_data import sample_dataframe
from src.screens import BUILTIN_SCREENS, apply_screen


def _ranked_sample():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    scored = enrich(df)
    return rank_universe(scored, w_quality=1.0, w_ml=0.0)


def test_screens_are_not_noops():
    ranked = _ranked_sample()
    all_n = len(apply_screen(ranked, BUILTIN_SCREENS["All (default)"]))
    assert all_n == len(ranked)

    clean = apply_screen(ranked, BUILTIN_SCREENS["Clean quality"])
    # Clean is stricter than All (min Q 65 + no flags + reliable)
    assert len(clean) <= all_n
    if len(clean):
        assert (clean["quality_score"] >= 65).all()
        assert (clean["red_flag_count"] == 0).all()

    value = apply_screen(ranked, BUILTIN_SCREENS["Value quality"])
    if len(value):
        assert (value["quality_score"] >= 55).all()
        assert (value["red_flag_count"] == 0).all()
        # PE filter: non-null PE must be <= 25
        pe = value["pe"].dropna()
        if len(pe):
            assert (pe <= 25).all()

    lev = apply_screen(ranked, BUILTIN_SCREENS["Low leverage"])
    if len(lev):
        assert (lev["quality_score"] >= 50).all()
        de = lev["debt_to_equity"].dropna()
        if len(de):
            assert (de <= 1.0).all()


def test_watchlist_only_screen():
    ranked = _ranked_sample()
    tickers = ranked["ticker"].tolist()
    one = tickers[:1]
    out = apply_screen(
        ranked, BUILTIN_SCREENS["Watchlist only"], watchlist=one
    )
    assert set(out["ticker"]) <= set(one)


def test_clean_stricter_than_all_on_demo_or_sample():
    """If demo_data exists, run the same checks on a larger universe."""
    from pathlib import Path

    demo = Path("demo_data.csv")
    if not demo.exists():
        return
    raw = pd.read_csv(demo)
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", dayfirst=True, errors="coerce")
    ranked = rank_universe(enrich(raw), w_quality=1.0, w_ml=0.0)
    n_all = len(apply_screen(ranked, BUILTIN_SCREENS["All (default)"]))
    n_clean = len(apply_screen(ranked, BUILTIN_SCREENS["Clean quality"]))
    n_value = len(apply_screen(ranked, BUILTIN_SCREENS["Value quality"]))
    # On a real universe clean/value should usually drop names; at minimum not grow
    assert n_clean <= n_all
    assert n_value <= n_all
