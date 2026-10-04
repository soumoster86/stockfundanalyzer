"""Fetcher row construction against a fake yfinance Ticker (no network)."""
import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

YEARS = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31"])


@pytest.fixture
def ff(monkeypatch):
    try:
        import yfinance  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "yfinance", types.ModuleType("yfinance"))
    return importlib.import_module("src.fetch_fundamentals")


def _frame(rows):
    return pd.DataFrame(rows, index=YEARS).T


def _ticker(info=None, cf_overrides=None):
    fin = _frame({
        "Total Revenue": [1000.0, 900.0, 800.0],
        "Operating Income": [200.0, 180.0, 150.0],
        "Net Income": [150.0, 120.0, 100.0],
        "Diluted EPS": [15.0, 12.0, 10.0],
    })
    bs = _frame({
        "Total Assets": [2000.0, 1800.0, 1600.0],
        "Current Liabilities": [300.0, 300.0, 300.0],
        "Stockholders Equity": [1000.0, 900.0, 800.0],
        "Ordinary Shares Number": [11.0, 10.0, 10.0],  # 10% dilution in latest year
    })
    cf_rows = {
        "Free Cash Flow": [120.0, 100.0, 80.0],
        "Operating Cash Flow": [300.0, 250.0, 200.0],
        "Cash Dividends Paid": [-60.0, -50.0, -40.0],
        "Repurchase Of Capital Stock": [-30.0, 0.0, 0.0],
    }
    cf_rows.update(cf_overrides or {})
    return types.SimpleNamespace(
        info=info if info is not None else {
            "marketCap": 3000.0, "sharesOutstanding": 11.5,
            "currency": "INR", "financialCurrency": "INR", "dividendYield": 2.0,
        },
        financials=fin, balance_sheet=bs, cashflow=_frame(cf_rows),
    )


def _rows(ff, monkeypatch, tk):
    monkeypatch.setattr(ff, "yf", types.SimpleNamespace(Ticker=lambda _t: tk), raising=False)
    latest, prior = ff.fetch_one("TEST.NS", "")
    return latest, prior


def test_shares_are_per_year_so_dilution_is_visible(ff, monkeypatch):
    latest, prior = _rows(ff, monkeypatch, _ticker())
    assert latest["shares_outstanding"] == 11.0
    assert prior["shares_outstanding"] == 10.0


def test_fcf_growth_compares_fcf_with_prior_fcf(ff, monkeypatch):
    latest, prior = _rows(ff, monkeypatch, _ticker())
    assert latest["fcf_growth"] == pytest.approx(0.20)   # 120 vs 100, not vs OCF 250
    assert prior["fcf_growth"] == pytest.approx(0.25)    # 100 vs 80


def test_fcf_falls_back_to_ocf_plus_capex(ff, monkeypatch):
    tk = _ticker(cf_overrides={
        "Free Cash Flow": [np.nan, np.nan, np.nan],
        "Capital Expenditure": [-100.0, -100.0, -100.0],
    })
    latest, _ = _rows(ff, monkeypatch, tk)
    assert latest["fcf_growth"] == pytest.approx((200 - 150) / 150)


def test_eps_and_dividend_growth_populated(ff, monkeypatch):
    latest, prior = _rows(ff, monkeypatch, _ticker())
    assert latest["eps_growth"] == pytest.approx(0.25)
    assert latest["dividend_growth"] == pytest.approx(0.20)
    assert prior["dividend_growth"] == pytest.approx(0.25)


def test_buyback_yield_latest_year_only(ff, monkeypatch):
    latest, prior = _rows(ff, monkeypatch, _ticker())
    assert latest["buyback_yield"] == pytest.approx(1.0)  # 30 / 3000 in percent points
    assert np.isnan(prior["buyback_yield"])


def test_buyback_yield_skipped_on_currency_mismatch(ff, monkeypatch):
    info = {"marketCap": 3000.0, "currency": "INR", "financialCurrency": "USD"}
    latest, _ = _rows(ff, monkeypatch, _ticker(info=info))
    assert np.isnan(latest["buyback_yield"])


def test_implausible_dividend_yield_dropped(ff, monkeypatch):
    info = {"marketCap": 3000.0, "dividendYield": 146.48}
    latest, _ = _rows(ff, monkeypatch, _ticker(info=info))
    assert np.isnan(latest["dividend_yield"])
