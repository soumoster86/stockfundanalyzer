"""
Watchlist alert rules
---------------------
Evaluate a scored universe slice against simple monitoring rules.
Pure functions — no Streamlit dependency.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Default thresholds (tunable by callers)
DEFAULT_RULES: dict[str, Any] = {
    "min_quality": 50.0,       # alert if quality below this
    "max_red_flags": 0,        # alert if red_flag_count > this
    "max_f_score_low": 3,      # alert if F-score <= this (when tests used)
    "min_f_tests": 5,          # only F-alert when enough tests populated
    "z_risk_bands": ("Red",),  # Altman distress
    "m_risk_bands": ("Red",),  # Beneish likely manipulator
    "data_warning": True,      # alert on data_warning True
}


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def evaluate_ticker_alerts(
    row: pd.Series,
    rules: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Return a list of alert dicts for one scored row:
      {ticker, severity, code, message}
    """
    r = {**DEFAULT_RULES, **(rules or {})}
    ticker = str(row.get("ticker", "?"))
    out: list[dict[str, str]] = []

    q = row.get("quality_score")
    if pd.notna(q) and float(q) < float(r["min_quality"]):
        out.append(
            {
                "ticker": ticker,
                "severity": "medium",
                "code": "low_quality",
                "message": f"Quality {float(q):.0f} below {r['min_quality']:.0f}",
            }
        )

    n_flags = row.get("red_flag_count")
    if pd.notna(n_flags) and int(n_flags) > int(r["max_red_flags"]):
        out.append(
            {
                "ticker": ticker,
                "severity": "high",
                "code": "red_flags",
                "message": f"{int(n_flags)} red flag(s)",
            }
        )

    zb = row.get("z_band")
    if pd.notna(zb) and str(zb) in set(r["z_risk_bands"]):
        out.append(
            {
                "ticker": ticker,
                "severity": "high",
                "code": "z_risk",
                "message": f"Altman Z band {zb} (distress risk)",
            }
        )

    mb = row.get("m_band")
    if pd.notna(mb) and str(mb) in set(r["m_risk_bands"]):
        out.append(
            {
                "ticker": ticker,
                "severity": "high",
                "code": "m_risk",
                "message": f"Beneish M band {mb} (manipulation risk)",
            }
        )

    fs = row.get("f_score")
    ftu = row.get("f_tests_used", 0)
    if (
        pd.notna(fs)
        and pd.notna(ftu)
        and int(ftu) >= int(r["min_f_tests"])
        and float(fs) <= float(r["max_f_score_low"])
    ):
        out.append(
            {
                "ticker": ticker,
                "severity": "medium",
                "code": "low_f_score",
                "message": f"Piotroski F-score {int(fs)}/{int(ftu)} weak",
            }
        )

    if r.get("data_warning") and bool(row.get("data_warning", False)):
        out.append(
            {
                "ticker": ticker,
                "severity": "low",
                "code": "data_warning",
                "message": "Data reliability warning — verify filings",
            }
        )

    return out


def evaluate_watchlist_alerts(
    df: pd.DataFrame,
    tickers: list[str] | None = None,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Evaluate alerts for a scored frame (optionally filtered to tickers).

    Returns a DataFrame with columns:
      ticker, severity, code, message
    sorted by severity then ticker.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "severity", "code", "message"])

    work = df
    if tickers is not None:
        wanted = {str(t).strip() for t in tickers if str(t).strip()}
        if "ticker" in work.columns:
            work = work[work["ticker"].astype(str).isin(wanted)]

    if work.empty:
        return pd.DataFrame(columns=["ticker", "severity", "code", "message"])

    # Latest row per ticker if multi-year panel
    if "date" in work.columns and work["ticker"].duplicated().any():
        work = work.sort_values("date").groupby("ticker", as_index=False).tail(1)

    rows: list[dict[str, str]] = []
    for _, row in work.iterrows():
        rows.extend(evaluate_ticker_alerts(row, rules=rules))

    if not rows:
        return pd.DataFrame(columns=["ticker", "severity", "code", "message"])

    out = pd.DataFrame(rows)
    out["_sev"] = out["severity"].map(SEVERITY_ORDER).fillna(9)
    out = out.sort_values(["_sev", "ticker", "code"]).drop(columns=["_sev"])
    return out.reset_index(drop=True)


def alert_summary(alerts: pd.DataFrame) -> dict[str, int]:
    """Counts by severity."""
    if alerts is None or alerts.empty:
        return {"high": 0, "medium": 0, "low": 0, "total": 0, "names": 0}
    return {
        "high": int((alerts["severity"] == "high").sum()),
        "medium": int((alerts["severity"] == "medium").sum()),
        "low": int((alerts["severity"] == "low").sum()),
        "total": len(alerts),
        "names": int(alerts["ticker"].nunique()),
    }
