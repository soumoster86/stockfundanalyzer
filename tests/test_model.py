"""Outperformance model: labels, time split, NaN-safe train, predict."""
import numpy as np
import pandas as pd
import pytest

from src.model import (
    MIN_TRAIN_ROWS,
    make_label,
    predict_proba,
    time_split,
    train_outperformance_model,
)

FEATURE_COLS = [
    "revenue_growth", "eps_growth", "roe", "roce", "net_margin",
    "operating_margin", "gross_margin", "debt_to_equity", "interest_coverage",
    "current_ratio", "cash_position", "pe", "pb", "ev_ebitda", "peg",
    "price_sales", "quality_score",
]


def test_make_label_binary_and_nan(multi_date_panel):
    df = make_label(multi_date_panel)
    assert "target_outperform" in df.columns
    assert set(df["target_outperform"].dropna().unique()).issubset({0.0, 1.0, 0, 1})
    # missing returns -> nan label
    d2 = multi_date_panel.copy()
    d2.loc[0, "fwd_return"] = np.nan
    lab = make_label(d2)
    assert pd.isna(lab.loc[0, "target_outperform"])


def test_time_split_no_future_in_train(multi_date_panel):
    train, valid, test = time_split(multi_date_panel, date_col="date")
    if len(valid) and len(train):
        assert train["date"].max() <= valid["date"].min()
    if len(test) and len(valid):
        assert valid["date"].max() <= test["date"].min()
    if len(test) and len(train):
        assert train["date"].max() <= test["date"].min()


def test_train_and_predict_with_nans(multi_date_panel):
    df = make_label(multi_date_panel)
    feats = [c for c in FEATURE_COLS if c in df.columns]
    model, report = train_outperformance_model(df, feats, kind="randomforest")
    assert report["n_train"] >= MIN_TRAIN_ROWS
    assert "features" in report
    probs = predict_proba(model, df, feats)
    assert len(probs) == len(df)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()


def test_train_rejects_too_few_rows():
    df = pd.DataFrame({
        "ticker": ["A"] * 5,
        "date": pd.to_datetime([f"{2015 + i}-01-01" for i in range(5)]),
        "roe": [1, 2, 3, 4, 5],
        "fwd_return": [0.1, 0.2, 0.1, 0.3, 0.2],
        "bench_fwd_return": [0.15] * 5,
    })
    df = make_label(df)
    with pytest.raises(ValueError, match="at least"):
        train_outperformance_model(df, ["roe"], kind="randomforest")


def test_train_rejects_missing_features():
    df = pd.DataFrame({
        "ticker": ["A"] * 25,
        "date": pd.to_datetime([f"{2000 + i}-01-01" for i in range(25)]),
        "fwd_return": np.random.default_rng(0).random(25),
        "bench_fwd_return": [0.5] * 25,
    })
    df = make_label(df)
    with pytest.raises(ValueError, match="No feature"):
        train_outperformance_model(df, ["nonexistent"], kind="randomforest")
