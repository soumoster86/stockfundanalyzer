"""Schema validation and unit normalization."""
import numpy as np
import pandas as pd

from src.sample_data import sample_dataframe
from src.schema import (
    format_growth_pct,
    normalize_growth_units,
    prepare_panel,
    validate_panel,
)


def test_validate_ok_on_sample():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    res = validate_panel(df)
    assert res.ok
    assert not res.errors


def test_missing_ticker_errors():
    df = pd.DataFrame({"date": ["2020-01-01"], "roe": [10]})
    res = validate_panel(df)
    assert not res.ok
    assert any("ticker" in e for e in res.errors)


def test_empty_errors():
    res = validate_panel(pd.DataFrame())
    assert not res.ok


def test_growth_percent_points_normalized():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "date": ["2020-01-01", "2020-01-01"],
        "revenue_growth": [12.0, 15.0],  # percent points by mistake
        "roe": [18.0, 20.0],
    })
    out, notes = normalize_growth_units(df)
    assert abs(out["revenue_growth"].iloc[0] - 0.12) < 1e-9
    assert notes


def test_prepare_panel_converts_growth():
    df = pd.DataFrame({
        "Ticker": ["A", "B", "C"],
        "Date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "revenue_growth": [10.0, 20.0, 30.0],
        "roe": [15.0, 16.0, 17.0],
    })
    prepared, res, notes = prepare_panel(df)
    assert "ticker" in prepared.columns
    assert prepared["revenue_growth"].max() < 1.0
    assert res.ok


def test_format_growth_pct():
    assert format_growth_pct(0.123) == "12.3%"
    assert format_growth_pct(np.nan) == "—"
