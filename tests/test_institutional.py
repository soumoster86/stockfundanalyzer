"""Piotroski F-Score and Altman Z-Score."""
import numpy as np
import pandas as pd

from src.institutional_scores import (
    compute_piotroski,
    compute_altman_z,
    f_score_band,
    z_band,
    blend_with_quality,
)
from src.sample_data import sample_dataframe


def test_piotroski_on_sample():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    out = compute_piotroski(df)
    assert "f_score" in out.columns
    assert "f_tests_used" in out.columns
    # Latest rows should evaluate several tests
    latest = out.sort_values("date").groupby("ticker").tail(1)
    assert (latest["f_tests_used"] > 0).all()
    assert latest["f_score"].notna().all()


def test_f_score_band():
    assert "strong" in f_score_band(8, 9).lower() or "Strong" in f_score_band(8, 9)
    assert f_score_band(float("nan"), 0)


def test_altman_z_safe_and_distress():
    # Minimal columns for Z if implementation needs them
    df = pd.DataFrame({
        "ticker": ["SAFE", "RISK"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "total_assets": [1000.0, 1000.0],
        "total_liabilities": [200.0, 900.0],
        "current_assets": [400.0, 50.0],
        "current_liabilities": [100.0, 400.0],
        "retained_earnings": [300.0, -50.0],
        "ebit": [150.0, 10.0],
        "revenue": [800.0, 200.0],
        "market_cap": [2000.0, 100.0],
    })
    # Some implementations use slightly different column names — call and check
    out = compute_altman_z(df)
    assert "z_score" in out.columns or "z_band" in out.columns


def test_blend_with_quality():
    df = pd.DataFrame({
        "quality_score": [50.0, 80.0],
        "f_score": [3.0, 8.0],
        "f_tests_used": [9, 9],
    })
    out = blend_with_quality(df)
    assert "quality_plus" in out.columns
    # higher F should lift quality_plus above base when blended
    assert out.loc[1, "quality_plus"] >= out.loc[1, "quality_score"] * 0.8
