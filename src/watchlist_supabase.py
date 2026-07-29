"""
Supabase backend for watchlists
--------------------------------
Reads/writes public.watchlist_items via supabase-py when configured.

Secrets (Streamlit):
    [supabase]
    url = "https://xxxx.supabase.co"
    key = "<service_role or anon key>"

Env fallbacks: SUPABASE_URL, SUPABASE_KEY
"""

from __future__ import annotations

import os
from typing import Any

TABLE = "watchlist_items"


def _secrets_map() -> dict[str, Any]:
    """Best-effort Streamlit secrets + environment."""
    out: dict[str, Any] = {}
    try:
        import streamlit as st

        block = st.secrets.get("supabase", None)
        if block is not None:
            # st.secrets nested objects support dict-like access
            try:
                out["url"] = block["url"]
                out["key"] = block["key"]
            except Exception:
                out.update(dict(block))
    except Exception:
        pass
    out.setdefault("url", os.environ.get("SUPABASE_URL", ""))
    out.setdefault("key", os.environ.get("SUPABASE_KEY", "")
                    or os.environ.get("SUPABASE_SERVICE_KEY", ""))
    return out


def is_configured() -> bool:
    cfg = _secrets_map()
    url = (cfg.get("url") or "").strip()
    key = (cfg.get("key") or "").strip()
    return bool(url and key)


def get_client():
    """Return a Supabase client or raise RuntimeError / ImportError."""
    if not is_configured():
        raise RuntimeError("Supabase is not configured (url/key missing).")
    try:
        from supabase import create_client
    except ImportError as e:
        raise ImportError(
            "supabase package not installed. Run: pip install supabase"
        ) from e
    cfg = _secrets_map()
    return create_client(str(cfg["url"]).strip(), str(cfg["key"]).strip())


def fetch_tickers(username: str, client=None) -> list[str]:
    """Load tickers for username, oldest first."""
    client = client or get_client()
    res = (
        client.table(TABLE)
        .select("ticker,created_at")
        .eq("username", username)
        .order("created_at")
        .execute()
    )
    rows = res.data or []
    out, seen = [], set()
    for row in rows:
        t = str(row.get("ticker") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def insert_ticker(username: str, ticker: str, client=None) -> None:
    client = client or get_client()
    client.table(TABLE).upsert(
        {"username": username, "ticker": ticker},
        on_conflict="username,ticker",
    ).execute()


def delete_ticker(username: str, ticker: str, client=None) -> None:
    client = client or get_client()
    client.table(TABLE).delete().eq("username", username).eq("ticker", ticker).execute()


def replace_all(username: str, tickers: list[str], client=None) -> None:
    """Replace the full list for a user (clear + insert)."""
    client = client or get_client()
    client.table(TABLE).delete().eq("username", username).execute()
    rows = [{"username": username, "ticker": t} for t in tickers if t]
    if rows:
        client.table(TABLE).insert(rows).execute()


def backend_status() -> dict:
    """Diagnostic for UI captions."""
    cfg = _secrets_map()
    has_pkg = True
    try:
        import supabase  # noqa: F401
    except ImportError:
        has_pkg = False
    return {
        "configured": is_configured(),
        "package_installed": has_pkg,
        "url_set": bool((cfg.get("url") or "").strip()),
        "key_set": bool((cfg.get("key") or "").strip()),
    }
