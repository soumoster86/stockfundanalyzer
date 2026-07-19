"""Governance CSV merge."""
import pandas as pd

from src.governance import governance_template_csv, load_governance_csv, merge_governance
from src.sample_data import sample_dataframe


def test_template_loads():
    df = load_governance_csv(governance_template_csv())
    assert "ticker" in df.columns
    assert "promoter_pledge_pct" in df.columns


def test_merge_by_ticker_date():
    fund = sample_dataframe()
    fund["date"] = pd.to_datetime(fund["date"])
    # TICKER2 2020 should pick up high pledge
    gov = pd.DataFrame([
        {
            "ticker": "TICKER2",
            "date": "2020-03-31",
            "auditor": "New Auditor",
            "promoter_pledge_pct": 20.0,
            "insider_net_buy": -100,
            "related_party_txn_flag": 1,
        }
    ])
    merged, stats = merge_governance(fund, gov)
    assert stats["n_tickers_matched"] == 1
    row = merged[(merged["ticker"] == "TICKER2") & (merged["date"] == "2020-03-31")]
    # date may be timestamp
    row = merged[merged["ticker"] == "TICKER2"].sort_values("date").iloc[-1]
    assert row["promoter_pledge_pct"] == 20.0
    assert row["auditor"] == "New Auditor"


def test_merge_by_ticker_only():
    fund = pd.DataFrame({
        "ticker": ["X", "X", "Y"],
        "date": pd.to_datetime(["2019-01-01", "2020-01-01", "2020-01-01"]),
        "roe": [10, 11, 12],
    })
    gov = pd.DataFrame({
        "ticker": ["X"],
        "promoter_pledge_pct": [5.0],
        "related_party_txn_flag": [1],
    })
    merged, stats = merge_governance(fund, gov)
    assert stats["mode"] == "by_ticker_latest"
    assert (merged.loc[merged["ticker"] == "X", "promoter_pledge_pct"] == 5.0).all()
    assert merged.loc[merged["ticker"] == "Y", "promoter_pledge_pct"].isna().all()
