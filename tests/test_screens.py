"""Saved ranking screens."""
import pandas as pd

from src.screens import BUILTIN_SCREENS, apply_screen, list_screens, save_custom_screen


def _frame():
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "quality_score": [80.0, 60.0, 70.0, 40.0],
        "red_flag_count": [0, 1, 0, 0],
        "data_warning": [False, False, True, False],
        "pe": [15.0, 30.0, 20.0, 12.0],
        "debt_to_equity": [0.5, 0.8, 1.5, 0.3],
    })


def test_clean_quality_screen():
    df = _frame()
    out = apply_screen(df, BUILTIN_SCREENS["Clean quality"])
    assert set(out["ticker"]) == {"A"}  # 80, no flags, reliable


def test_value_quality_max_pe():
    df = _frame()
    out = apply_screen(df, BUILTIN_SCREENS["Value quality"])
    # A: q80 pe15 flags0; C has warning so excluded by reliable only
    assert "A" in set(out["ticker"])
    assert "B" not in set(out["ticker"])  # has flags


def test_watchlist_only():
    df = _frame()
    scr = dict(BUILTIN_SCREENS["Watchlist only"])
    out = apply_screen(df, scr, watchlist=["C", "D"])
    assert set(out["ticker"]) == {"C", "D"}


def test_custom_screen_session():
    ss = {}
    save_custom_screen(ss, "Mine", {"min_quality": 75, "flag_filter": "All",
                                    "rel_filter": "All", "watchlist_only": False})
    names = list_screens(ss)
    assert "Mine" in names
