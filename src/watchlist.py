"""
Watchlist persistence
---------------------
Session-scoped list of tickers the user cares about, with optional CSV
import/export. Cloud deploys are ephemeral; download the CSV to keep a
watchlist across sessions.
"""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd

SESSION_KEY = "watchlist_tickers"


def normalize_ticker(t: str) -> str:
    return str(t).strip()


def get_watchlist(session_state) -> list[str]:
    raw = session_state.get(SESSION_KEY, [])
    out = []
    seen = set()
    for t in raw:
        nt = normalize_ticker(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def set_watchlist(session_state, tickers: Iterable[str]) -> list[str]:
    cleaned = []
    seen = set()
    for t in tickers:
        nt = normalize_ticker(t)
        if nt and nt not in seen:
            seen.add(nt)
            cleaned.append(nt)
    session_state[SESSION_KEY] = cleaned
    return cleaned


def add_ticker(session_state, ticker: str) -> bool:
    """Add one ticker. Returns True if newly added."""
    wl = get_watchlist(session_state)
    nt = normalize_ticker(ticker)
    if not nt or nt in wl:
        return False
    wl.append(nt)
    session_state[SESSION_KEY] = wl
    return True


def remove_ticker(session_state, ticker: str) -> bool:
    wl = get_watchlist(session_state)
    nt = normalize_ticker(ticker)
    if nt not in wl:
        return False
    session_state[SESSION_KEY] = [t for t in wl if t != nt]
    return True


def is_watched(session_state, ticker: str) -> bool:
    return normalize_ticker(ticker) in set(get_watchlist(session_state))


def watchlist_to_csv(tickers: Iterable[str]) -> bytes:
    df = pd.DataFrame({"ticker": list(tickers)})
    return df.to_csv(index=False).encode("utf-8")


def watchlist_from_csv(source) -> list[str]:
    if isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(source))
    elif hasattr(source, "read"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        # allow single-column nameless or first column
        if len(df.columns) == 1:
            col = df.columns[0]
        else:
            raise ValueError("Watchlist CSV needs a `ticker` column.")
    else:
        col = "ticker"
    return [normalize_ticker(t) for t in df[col].dropna().astype(str) if str(t).strip()]


def filter_universe(df: pd.DataFrame, tickers: Iterable[str],
                    ticker_col: str = "ticker") -> pd.DataFrame:
    wanted = {normalize_ticker(t) for t in tickers}
    if not wanted or ticker_col not in df.columns:
        return df.iloc[0:0].copy()
    mask = df[ticker_col].astype(str).map(normalize_ticker).isin(wanted)
    return df.loc[mask].copy()


def watchlist_summary(df: pd.DataFrame, tickers: Iterable[str]) -> dict:
    """Aggregate quality / flags for a watchlist slice of the scored universe."""
    sub = filter_universe(df, tickers)
    if sub.empty:
        return {
            "n": 0,
            "avg_quality": float("nan"),
            "median_quality": float("nan"),
            "avg_flags": float("nan"),
            "n_warnings": 0,
            "sectors": 0,
        }
    return {
        "n": int(sub["ticker"].nunique()) if "ticker" in sub.columns else len(sub),
        "avg_quality": float(sub["quality_score"].mean()) if "quality_score" in sub else float("nan"),
        "median_quality": float(sub["quality_score"].median()) if "quality_score" in sub else float("nan"),
        "avg_flags": float(sub["red_flag_count"].mean()) if "red_flag_count" in sub else float("nan"),
        "n_warnings": int(sub["data_warning"].sum()) if "data_warning" in sub else 0,
        "sectors": int(sub["sector"].nunique()) if "sector" in sub.columns else 0,
    }
