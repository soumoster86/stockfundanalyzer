"""Watchlist alert rules."""
import pandas as pd

from src.alerts import alert_summary, evaluate_ticker_alerts, evaluate_watchlist_alerts


def test_low_quality_and_flags():
    row = pd.Series(
        {
            "ticker": "AAA.NS",
            "quality_score": 40.0,
            "red_flag_count": 2,
            "z_band": "Green",
            "m_band": "Green",
            "f_score": 7,
            "f_tests_used": 9,
            "data_warning": False,
        }
    )
    alerts = evaluate_ticker_alerts(row)
    codes = {a["code"] for a in alerts}
    assert "low_quality" in codes
    assert "red_flags" in codes


def test_z_m_high_severity():
    row = pd.Series(
        {
            "ticker": "BBB.NS",
            "quality_score": 70.0,
            "red_flag_count": 0,
            "z_band": "Red",
            "m_band": "Red",
            "f_score": 5,
            "f_tests_used": 9,
            "data_warning": False,
        }
    )
    alerts = evaluate_ticker_alerts(row)
    assert any(a["code"] == "z_risk" and a["severity"] == "high" for a in alerts)
    assert any(a["code"] == "m_risk" and a["severity"] == "high" for a in alerts)


def test_watchlist_frame_and_summary():
    df = pd.DataFrame(
        [
            {
                "ticker": "A.NS",
                "quality_score": 30.0,
                "red_flag_count": 1,
                "z_band": "Green",
                "m_band": "Green",
                "f_score": 2,
                "f_tests_used": 8,
                "data_warning": True,
            },
            {
                "ticker": "B.NS",
                "quality_score": 80.0,
                "red_flag_count": 0,
                "z_band": "Green",
                "m_band": "Green",
                "f_score": 8,
                "f_tests_used": 9,
                "data_warning": False,
            },
        ]
    )
    alerts = evaluate_watchlist_alerts(df, tickers=["A.NS", "B.NS"])
    assert not alerts.empty
    assert set(alerts["ticker"]) == {"A.NS"} or "A.NS" in set(alerts["ticker"])
    s = alert_summary(alerts)
    assert s["total"] >= 1
    assert s["names"] >= 1


def test_no_alerts_clean_name():
    df = pd.DataFrame(
        [
            {
                "ticker": "CLEAN.NS",
                "quality_score": 75.0,
                "red_flag_count": 0,
                "z_band": "Green",
                "m_band": "Green",
                "f_score": 8,
                "f_tests_used": 9,
                "data_warning": False,
            }
        ]
    )
    alerts = evaluate_watchlist_alerts(df)
    assert alerts.empty
    assert alert_summary(alerts)["total"] == 0
