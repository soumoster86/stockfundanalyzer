"""Research note export."""
import pandas as pd

from src.report_export import build_research_note


def test_research_note_contains_ticker_and_scores():
    latest = pd.Series(
        {
            "ticker": "TCS.NS",
            "date": pd.Timestamp("2025-03-31"),
            "sector": "Technology",
            "quality_score": 72.5,
            "red_flag_count": 0,
            "red_flags": [],
            "f_score": 7,
            "f_tests_used": 9,
            "z_score": 3.2,
            "z_band": "Green",
            "m_score": -2.5,
            "m_band": "Green",
            "roe": 25.0,
            "pe": 28.0,
            "debt_to_equity": 0.1,
        }
    )
    note = build_research_note(latest, sector="Technology")
    assert "TCS.NS" in note
    assert "Research note" in note
    assert "72.5" in note
    assert "Technology" in note
