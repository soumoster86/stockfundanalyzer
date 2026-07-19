"""Test setup for the Fundamental Stock Analyzer (src/ + app.py)."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_panel():
    """Two tickers × two years with full metrics (from sample_data)."""
    from src.sample_data import sample_dataframe
    df = sample_dataframe().copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def multi_date_panel():
    """
    Larger synthetic panel for ML time-splits: 10 tickers × 5 year-ends,
    with labels and a few NaNs mixed in.
    """
    rows = []
    years = [2018, 2019, 2020, 2021, 2022]
    for i in range(10):
        ticker = f"T{i:02d}"
        sector = "Tech" if i % 2 == 0 else "Banks"
        for y in years:
            rows.append({
                "ticker": ticker,
                "date": f"{y}-03-31",
                "sector": sector,
                "revenue_growth": 0.05 + 0.01 * i + 0.002 * (y - 2018),
                "eps_growth": 0.04 + 0.01 * i,
                "roe": 10 + i + (y - 2018),
                "roce": 12 + i,
                "net_margin": 8 + 0.5 * i,
                "operating_margin": 12 + 0.3 * i,
                "gross_margin": 30 + i,
                "debt_to_equity": 0.5 + 0.1 * (i % 3),
                "interest_coverage": 5 + i * 0.2,
                "current_ratio": 1.2 + 0.05 * i,
                "cash_position": 100 * (i + 1),
                "pe": 15 + i,
                "pb": 2 + 0.1 * i,
                "ev_ebitda": 10 + 0.2 * i,
                "peg": 1.0 + 0.05 * i,
                "price_sales": 2 + 0.1 * i,
                "quality_score": 40 + i * 2 + (y - 2018),  # optional meta feature
                "fwd_return": 0.1 + 0.02 * i + 0.01 * (y - 2018),
                "bench_fwd_return": 0.12,
                "net_profit": 100 + 10 * i,
                "operating_cash_flow": 90 + 12 * i if i != 3 else 50,  # one weak OCF
                "revenue": 1000 + 50 * i,
                "shares_outstanding": 1000 + 10 * i,
                "total_debt": 200 + 20 * i,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # Sprinkle NaNs so imputation is exercised
    df.loc[df["ticker"] == "T00", "peg"] = float("nan")
    df.loc[df["ticker"] == "T01", "price_sales"] = float("nan")
    return df
