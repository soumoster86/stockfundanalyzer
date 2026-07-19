"""Label builder pure helpers (no network)."""
import numpy as np
import pandas as pd

from src.build_labels import forward_return, label_coverage, price_on_or_after


def _series():
    idx = pd.date_range("2018-01-01", periods=1200, freq="B")
    prices = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    return prices


def test_price_on_or_after():
    s = _series()
    p = price_on_or_after(s, pd.Timestamp("2019-06-15"))
    assert p is not None and p > 100


def test_forward_return_positive_trend():
    s = _series()
    r = forward_return(s, pd.Timestamp("2018-06-01"), horizon_years=1)
    assert pd.notna(r) and r > 0


def test_forward_return_nan_if_too_recent():
    s = _series()
    r = forward_return(s, s.index.max() - pd.Timedelta(days=5), horizon_years=3)
    assert pd.isna(r)


def test_label_coverage():
    df = pd.DataFrame({
        "fwd_return": [0.1, np.nan, 0.2],
        "bench_fwd_return": [0.05, 0.05, np.nan],
    })
    cov = label_coverage(df)
    assert cov["n_labeled"] == 1
    assert cov["n_rows"] == 3
