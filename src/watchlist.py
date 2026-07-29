"""
Watchlist persistence
---------------------
Public API used by Report / Ranking / Watchlist UI.

Backends:
  * Supabase (when [supabase] url+key secrets / env are set) — durable per user
  * Streamlit session_state — always used as cache; sole backend if Supabase off

CSV import/export remains for backup and migration.
"""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd

SESSION_KEY = "watchlist_tickers"
HYDRATED_KEY = "_watchlist_hydrated"
BACKEND_KEY = "_watchlist_backend"  # "supabase" | "session"
ERROR_KEY = "_watchlist_backend_error"
USERNAME_KEY = "auth_user"


def normalize_ticker(t: str) -> str:
    return str(t).strip()


def _username(session_state) -> str:
    u = session_state.get(USERNAME_KEY) if session_state is not None else None
    u = (u or "anonymous").strip() or "anonymous"
    return u


def _use_supabase() -> bool:
    try:
        from src.watchlist_supabase import is_configured
        return is_configured()
    except Exception:
        return False


def backend_name(session_state=None) -> str:
    """Active backend label for UI."""
    if session_state is not None and session_state.get(BACKEND_KEY):
        return str(session_state[BACKEND_KEY])
    return "supabase" if _use_supabase() else "session"


def ensure_hydrated(session_state) -> None:
    """
    Load watchlist from Supabase once per session when configured.
    Safe no-op for session-only mode.
    """
    if session_state is None:
        return
    if session_state.get(HYDRATED_KEY):
        return

    session_state[ERROR_KEY] = None
    if not _use_supabase():
        session_state[BACKEND_KEY] = "session"
        session_state[HYDRATED_KEY] = True
        # leave any existing session list as-is
        if SESSION_KEY not in session_state:
            session_state[SESSION_KEY] = []
        return

    try:
        from src import watchlist_supabase as sb

        user = _username(session_state)
        tickers = sb.fetch_tickers(user)
        session_state[SESSION_KEY] = list(tickers)
        session_state[BACKEND_KEY] = "supabase"
    except Exception as e:
        # Fall back to session so the app keeps working
        session_state[BACKEND_KEY] = "session"
        session_state[ERROR_KEY] = str(e)
        if SESSION_KEY not in session_state:
            session_state[SESSION_KEY] = []
    session_state[HYDRATED_KEY] = True


def get_watchlist(session_state) -> list[str]:
    ensure_hydrated(session_state)
    raw = session_state.get(SESSION_KEY, []) if session_state is not None else []
    out = []
    seen = set()
    for t in raw:
        nt = normalize_ticker(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def set_watchlist(session_state, tickers: Iterable[str]) -> list[str]:
    ensure_hydrated(session_state)
    cleaned = []
    seen = set()
    for t in tickers:
        nt = normalize_ticker(t)
        if nt and nt not in seen:
            seen.add(nt)
            cleaned.append(nt)
    session_state[SESSION_KEY] = cleaned

    if _use_supabase():
        try:
            from src import watchlist_supabase as sb

            sb.replace_all(_username(session_state), cleaned)
            session_state[BACKEND_KEY] = "supabase"
            session_state[ERROR_KEY] = None
        except Exception as e:
            session_state[ERROR_KEY] = str(e)
            # keep session list even if sync fails
    return cleaned


def add_ticker(session_state, ticker: str) -> bool:
    """Add one ticker. Returns True if newly added."""
    ensure_hydrated(session_state)
    wl = get_watchlist(session_state)
    nt = normalize_ticker(ticker)
    if not nt or nt in wl:
        return False
    wl.append(nt)
    session_state[SESSION_KEY] = wl

    if _use_supabase():
        try:
            from src import watchlist_supabase as sb

            sb.insert_ticker(_username(session_state), nt)
            session_state[BACKEND_KEY] = "supabase"
            session_state[ERROR_KEY] = None
        except Exception as e:
            session_state[ERROR_KEY] = str(e)
    return True


def remove_ticker(session_state, ticker: str) -> bool:
    ensure_hydrated(session_state)
    wl = get_watchlist(session_state)
    nt = normalize_ticker(ticker)
    if nt not in wl:
        return False
    session_state[SESSION_KEY] = [t for t in wl if t != nt]

    if _use_supabase():
        try:
            from src import watchlist_supabase as sb

            sb.delete_ticker(_username(session_state), nt)
            session_state[BACKEND_KEY] = "supabase"
            session_state[ERROR_KEY] = None
        except Exception as e:
            session_state[ERROR_KEY] = str(e)
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
