"""Market-cap bucket helpers."""
import pandas as pd

from src.market_buckets import (
    assign_market_cap_bucket,
    cap_bucket,
    filter_by_bucket,
)


def test_cap_bucket_thresholds():
    assert cap_bucket(300_000_000_000) == "Large"
    assert cap_bucket(80_000_000_000) == "Mid"
    assert cap_bucket(10_000_000_000) == "Small"
    assert cap_bucket(1_000_000_000) == "Micro"
    assert cap_bucket(None) == "Unknown"
    assert cap_bucket(float("nan")) == "Unknown"


def test_assign_and_filter():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "market_cap": [300e9, 10e9, None],
        }
    )
    out = assign_market_cap_bucket(df)
    assert list(out["mcap_bucket"]) == ["Large", "Small", "Unknown"]
    large = filter_by_bucket(out, "Large")
    assert list(large["ticker"]) == ["A"]
    all_ = filter_by_bucket(out, "All")
    assert len(all_) == 3


def test_missing_market_cap_column():
    df = pd.DataFrame({"ticker": ["A"]})
    out = assign_market_cap_bucket(df)
    assert out["mcap_bucket"].iloc[0] == "Unknown"
