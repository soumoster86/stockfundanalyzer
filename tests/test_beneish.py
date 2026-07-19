"""Beneish M-Score."""
import pandas as pd

from src.enrich import enrich
from src.institutional_scores import M_BAND_TEXT, compute_beneish, m_band
from src.sample_data import sample_dataframe


def test_m_band_thresholds():
    assert m_band(-2.5) == "Green"
    assert m_band(-2.0) == "Yellow"
    assert m_band(-1.5) == "Red"
    assert m_band(float("nan")) == "N/A"
    assert "manipulator" in M_BAND_TEXT["Red"].lower() or "Manipulator" in M_BAND_TEXT["Red"]


def test_beneish_on_sample():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    out = compute_beneish(df)
    assert "m_score" in out.columns
    assert "m_band" in out.columns
    assert "m_indices_used" in out.columns
    # Latest TICKER2 should have several indices (receivables, revenue, margins, debt)
    latest = out[out["ticker"] == "TICKER2"].sort_values("date").iloc[-1]
    assert latest["m_indices_used"] >= 4
    assert pd.notna(latest["m_score"])


def test_enrich_includes_m_score(sample_panel):
    out = enrich(sample_panel)
    assert "m_score" in out.columns or "m_band" in out.columns
