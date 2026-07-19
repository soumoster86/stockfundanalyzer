"""Scoring pipeline (pure enrich, no Streamlit)."""
import pandas as pd

from src.enrich import FEATURE_COLS, build_history_panel, enrich, is_tickers_only
from src.sample_data import sample_dataframe


def test_is_tickers_only():
    bare = pd.DataFrame({"ticker": ["A"], "date": ["2020-01-01"]})
    full = sample_dataframe()
    assert is_tickers_only(bare)
    assert not is_tickers_only(full)


def test_enrich_produces_quality_and_flags(sample_panel):
    out = enrich(sample_panel, use_sector=False)
    assert "quality_score" in out.columns
    assert "red_flag_count" in out.columns
    assert "f_score" in out.columns or "f_tests_used" in out.columns
    # latest row per ticker
    assert out["ticker"].nunique() == out.shape[0]


def test_build_history_keeps_multi_year(sample_panel):
    hist = build_history_panel(sample_panel, use_sector=False)
    # more rows than unique tickers when multi-year
    assert len(hist) >= sample_panel["ticker"].nunique()
    assert "quality_score" in hist.columns


def test_feature_cols_documented():
    assert "roe" in FEATURE_COLS
    assert "quality_score" in FEATURE_COLS
