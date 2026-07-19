"""Completeness and sanity guards."""
import numpy as np
import pandas as pd

from src.data_quality import CORE_METRICS, data_completeness, data_sanity_flags


def test_completeness_counts(sample_panel):
    out = data_completeness(sample_panel)
    assert "data_fields_present" in out.columns
    assert "data_completeness" in out.columns
    present_cols = [m for m in CORE_METRICS if m in sample_panel.columns]
    assert out["data_fields_total"].iloc[0] == len(present_cols)
    assert (out["data_completeness"] >= 0).all() and (out["data_completeness"] <= 1).all()
    # sample rows are fully populated for core metrics that exist
    assert (out["data_fields_present"] == len(present_cols)).all()


def test_sparse_row_low_completeness():
    # Columns exist for many core metrics, but only roe is populated
    row = {m: np.nan for m in CORE_METRICS}
    row.update({"ticker": "X", "date": pd.Timestamp("2020-01-01"), "roe": 15.0})
    df = pd.DataFrame([row])
    out = data_completeness(df)
    assert out["data_fields_present"].iloc[0] == 1
    assert out["data_completeness"].iloc[0] < 0.5


def test_sanity_flags_extreme_pe_and_jump():
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2019-01-01", "2020-01-01"]),
        "revenue": [100.0, 400.0],  # +300% -> jump
        "pe": [10.0, 250.0],        # extreme PE
        "shares_outstanding": [100.0, 100.0],
        "roe": [15.0, 15.0],
        "net_margin": [10.0, 10.0],
    })
    out, warn_cols = data_sanity_flags(df)
    assert warn_cols
    latest = out.sort_values("date").iloc[-1]
    assert latest["data_warning"] or latest.get("data_warning_count", 0) > 0
