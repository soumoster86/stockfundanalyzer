"""Saved ranking screens + funnel."""
import pandas as pd

from src.screens import (
    BUILTIN_SCREENS,
    apply_screen,
    list_screens,
    save_custom_screen,
    screen_funnel,
)


def _frame():
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "quality_score": [80.0, 60.0, 70.0, 40.0, 58.0],
        "red_flag_count": [0, 1, 0, 0, 0],
        "data_warning": [False, False, True, False, False],
        "pe": [15.0, 30.0, 20.0, 12.0, 18.0],
        "debt_to_equity": [0.5, 0.8, 1.5, 0.3, 0.4],
    })


def test_clean_quality_daily_threshold():
    """Clean quality is Q≥55, no flags, reliable — not the elite 65 bar."""
    df = _frame()
    out = apply_screen(df, BUILTIN_SCREENS["Clean quality"])
    # A=80, E=58 pass; D=40 fails Q; B flags; C warning
    assert set(out["ticker"]) == {"A", "E"}


def test_clean_quality_elite_stricter():
    df = _frame()
    daily = apply_screen(df, BUILTIN_SCREENS["Clean quality"])
    elite = apply_screen(df, BUILTIN_SCREENS["Clean quality (elite)"])
    assert set(elite["ticker"]).issubset(set(daily["ticker"]))
    assert set(elite["ticker"]) == {"A"}  # only 80 ≥ 65


def test_value_quality_max_pe():
    df = _frame()
    out = apply_screen(df, BUILTIN_SCREENS["Value quality"])
    assert "A" in set(out["ticker"])
    assert "B" not in set(out["ticker"])  # has flags


def test_top_pct_quality():
    df = _frame()
    # Reliable only first → drop C; remaining A,B,D,E qualities 80,60,40,58
    # Top 25% of 4 ≈ keep highest 1 (80)
    scr = {
        "flag_filter": "All",
        "rel_filter": "Reliable only",
        "min_quality": 0.0,
        "top_pct": 25.0,
        "watchlist_only": False,
    }
    out = apply_screen(df, scr)
    assert "A" in set(out["ticker"])
    assert len(out) <= 2  # ~top quarter


def test_watchlist_only():
    df = _frame()
    scr = dict(BUILTIN_SCREENS["Watchlist only"])
    out = apply_screen(df, scr, watchlist=["C", "D"])
    assert set(out["ticker"]) == {"C", "D"}


def test_screen_funnel_steps():
    df = _frame()
    steps = screen_funnel(df, BUILTIN_SCREENS["Clean quality"])
    assert steps[0]["key"] == "universe"
    assert steps[0]["n"] == 5
    # last step should equal apply_screen count
    final = apply_screen(df, BUILTIN_SCREENS["Clean quality"])
    assert steps[-1]["n"] == len(final)
    labels = [s["label"] for s in steps]
    assert any("No red flags" in L or L == "No red flags" for L in labels)
    assert any("Q≥55" in L for L in labels)


def test_custom_screen_session():
    ss = {}
    save_custom_screen(
        ss,
        "Mine",
        {
            "min_quality": 75,
            "flag_filter": "All",
            "rel_filter": "All",
            "watchlist_only": False,
        },
    )
    names = list_screens(ss)
    assert "Mine" in names
