"""Quality score engine: percentiles, weights, sector fallback, labels."""
import numpy as np
import pandas as pd

from src.quality_score import (
    compute_quality_score,
    build_config,
    score_label,
    DEFAULT_CATEGORY_WEIGHTS,
    METRIC_CONFIG,
)


def test_score_label_bands():
    assert score_label(85) == "Excellent"
    assert score_label(70) == "Strong"
    assert score_label(55) == "Average"
    assert score_label(40) == "Weak"
    assert score_label(10) == "Poor"
    assert score_label(float("nan")) == "N/A"


def test_build_config_overrides_and_preserves_metrics():
    cfg = build_config({"Valuation": 0.5, "Profitability": 0.1})
    assert cfg["Valuation"]["weight"] == 0.5
    assert cfg["Profitability"]["weight"] == 0.1
    # metric weights unchanged
    assert cfg["Valuation"]["metrics"]["pe"]["weight"] == METRIC_CONFIG["Valuation"]["metrics"]["pe"]["weight"]
    # defaults still present for untouched categories
    assert cfg["Financial Performance"]["weight"] == DEFAULT_CATEGORY_WEIGHTS["Financial Performance"]


def test_compute_quality_score_range_and_columns(sample_panel):
    scored = compute_quality_score(sample_panel, group_cols=("date",))
    assert "quality_score" in scored.columns
    qs = scored["quality_score"].dropna()
    assert len(qs) > 0
    assert qs.min() >= 0 and qs.max() <= 100
    # category scores present
    assert "profitability_score" in scored.columns
    assert "valuation_score" in scored.columns


def test_higher_roe_ranks_higher_within_date():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2020-01-01"] * 5),
        "roe": [5, 10, 15, 20, 25],
        "pe": [30, 25, 20, 15, 10],  # lower better
    })
    scored = compute_quality_score(df, group_cols=("date",))
    order = scored.sort_values("quality_score")["ticker"].tolist()
    # E has best ROE and cheapest PE -> should be at or near top
    assert order[-1] == "E"
    assert order[0] == "A"


def test_valuation_inverted_in_percentiles():
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(6)],
        "date": pd.to_datetime(["2020-01-01"] * 6),
        "pe": [10, 20, 30, 40, 50, 60],
    })
    scored = compute_quality_score(df, group_cols=("date",))
    # cheapest PE should have highest _pct_pe
    cheap = scored.loc[scored["pe"] == 10, "_pct_pe"].iloc[0]
    dear = scored.loc[scored["pe"] == 60, "_pct_pe"].iloc[0]
    assert cheap > dear


def test_small_sector_falls_back(sample_panel):
    # Force tiny sector groups (< min_group_size)
    df = sample_panel.copy()
    scored = compute_quality_score(df, group_cols=("date", "sector"), min_group_size=5)
    assert "quality_score" in scored.columns
    assert scored["quality_score"].notna().any()
