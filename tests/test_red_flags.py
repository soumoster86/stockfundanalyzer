"""Red-flag forensic rules on a multi-year panel."""
import pandas as pd

from src.red_flags import detect_red_flags
from src.sample_data import sample_dataframe


def test_sample_ticker2_trips_flags():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    out, flag_cols = detect_red_flags(df)
    assert flag_cols
    latest = out[out["ticker"] == "TICKER2"].sort_values("date").iloc[-1]
    # TICKER2 latest is crafted to trip several rules
    assert latest["red_flag_count"] >= 3
    assert "eq_profit_up_cashflow_down" in latest["red_flags"] or latest["eq_profit_up_cashflow_down"]
    assert latest.get("fr_debt_spike") or latest.get("eq_equity_dilution") or latest.get("gov_related_party_txn")


def test_clean_ticker_has_few_or_zero_flags():
    df = sample_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    out, _ = detect_red_flags(df)
    latest1 = out[out["ticker"] == "TICKER1"].sort_values("date").iloc[-1]
    latest2 = out[out["ticker"] == "TICKER2"].sort_values("date").iloc[-1]
    assert latest1["red_flag_count"] < latest2["red_flag_count"]


def test_no_raw_columns_yields_zero_flags():
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2019-01-01", "2020-01-01"]),
        "roe": [10, 12],
    })
    out, flag_cols = detect_red_flags(df)
    assert flag_cols == []
    assert (out["red_flag_count"] == 0).all()
    assert all(f == [] for f in out["red_flags"])
