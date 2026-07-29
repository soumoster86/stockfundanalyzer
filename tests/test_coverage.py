"""Data coverage summary."""
import pandas as pd

from src.coverage import coverage_summary, coverage_banner_text, coverage_detail_lines
from src.enrich import enrich
from src.sample_data import sample_dataframe


def test_coverage_on_sample():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    scored = enrich(df)
    cov = coverage_summary(scored, raw=df)
    assert cov["n_tickers"] == 2
    assert cov["avg_completeness"] is not None
    assert "pct_f_any" in cov
    text = coverage_banner_text(cov)
    assert "stocks" in text.lower() or "Universe" in text
    lines = coverage_detail_lines(cov)
    assert len(lines) >= 3
