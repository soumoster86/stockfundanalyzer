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
import re
from typing import Any
from urllib.parse import urlparse

TABLE = "watchlist_items"


def normalize_supabase_url(url: str) -> str:
    """
    Project URL only: https://<ref>.supabase.co

    Strips whitespace, quotes, trailing slashes, and accidental /rest/v1
    (create_client already appends the REST path).
    """
    u = (url or "").strip().strip('"').strip("'")
    if not u:
        return ""
    # Common paste mistakes
    u = re.sub(r"/+$", "", u)
    u = re.sub(r"/rest/v1/?$", "", u, flags=re.IGNORECASE)
    u = re.sub(r"/+$", "", u)
    return u


def _pick_key(block: dict[str, Any]) -> str:
    """Accept several secret key names people paste from the dashboard."""
    for k in ("key", "service_role_key", "service_key", "api_key", "secret"):
        v = block.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().strip('"').strip("'")
    return ""


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Normalize Streamlit AttrDict / Mapping into a plain dict."""
    if block is None:
        return {}
    if isinstance(block, dict):
        return dict(block)
    try:
        return dict(block)
    except Exception:
        pass
    out: dict[str, Any] = {}
    for k in ("url", "key", "service_role_key", "service_key", "api_key", "secret"):
        try:
            out[k] = block[k]
        except Exception:
            try:
                out[k] = block.get(k)  # type: ignore[attr-defined]
            except Exception:
                pass
    return out


def _secrets_map() -> dict[str, Any]:
    """Best-effort Streamlit secrets + environment."""
    raw: dict[str, Any] = {}
    try:
        import streamlit as st

        raw = _block_to_dict(st.secrets.get("supabase", None))
    except Exception:
        raw = {}

    url = raw.get("url") or os.environ.get("SUPABASE_URL", "")
    key = _pick_key(raw) or (
        os.environ.get("SUPABASE_KEY", "")
        or os.environ.get("SUPABASE_SERVICE_KEY", "")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    )
    return {
        "url": normalize_supabase_url(str(url or "")),
        "key": str(key or "").strip().strip('"').strip("'"),
    }


def is_configured() -> bool:
    cfg = _secrets_map()
    return bool(cfg.get("url") and cfg.get("key"))


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
    url = cfg["url"]
    # Sanity: must look like a project host, not a DB connection string
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError(
            f"Supabase url must be https://<project-ref>.supabase.co "
            f"(got scheme/host invalid). Check Streamlit secrets [supabase].url"
        )
    if "supabase.co" not in parsed.netloc and "localhost" not in parsed.netloc:
        # still allow custom domains / local, but warn via path of call
        pass
    return create_client(url, cfg["key"])


def _friendly_api_error(exc: Exception) -> Exception:
    """Rewrite common PostgREST errors with setup hints."""
    msg = str(exc)
    code = ""
    if hasattr(exc, "code"):
        code = str(getattr(exc, "code") or "")
    if "PGRST125" in msg or code == "PGRST125":
        cfg = _secrets_map()
        host = urlparse(cfg.get("url") or "").netloc or "(url missing)"
        return RuntimeError(
            f"Supabase path invalid (PGRST125). "
            f"Use Project URL only (Settings → API), e.g. "
            f"https://xxxx.supabase.co — no trailing slash, no /rest/v1. "
            f"Host in secrets: {host}. "
            f"Confirm table public.{TABLE} exists (SQL editor), then run: "
            f"NOTIFY pgrst, 'reload schema';"
        )
    if "PGRST205" in msg or code == "PGRST205" or "schema cache" in msg.lower():
        return RuntimeError(
            f"Supabase cannot find public.{TABLE}. "
            f"Run sql/watchlist_supabase.sql, then: NOTIFY pgrst, 'reload schema';"
        )
    return exc


def fetch_tickers(username: str, client=None) -> list[str]:
    """Load tickers for username, oldest first."""
    client = client or get_client()
    try:
        res = (
            client.table(TABLE)
            .select("ticker,created_at")
            .eq("username", username)
            .order("created_at")
            .execute()
        )
    except Exception as e:
        raise _friendly_api_error(e) from e
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
    try:
        client.table(TABLE).upsert(
            {"username": username, "ticker": ticker},
            on_conflict="username,ticker",
        ).execute()
    except Exception as e:
        raise _friendly_api_error(e) from e


def delete_ticker(username: str, ticker: str, client=None) -> None:
    client = client or get_client()
    try:
        client.table(TABLE).delete().eq("username", username).eq("ticker", ticker).execute()
    except Exception as e:
        raise _friendly_api_error(e) from e


def replace_all(username: str, tickers: list[str], client=None) -> None:
    """Replace the full list for a user (clear + insert)."""
    client = client or get_client()
    try:
        client.table(TABLE).delete().eq("username", username).execute()
        rows = [{"username": username, "ticker": t} for t in tickers if t]
        if rows:
            client.table(TABLE).insert(rows).execute()
    except Exception as e:
        raise _friendly_api_error(e) from e


def backend_status() -> dict:
    """Diagnostic for UI captions."""
    cfg = _secrets_map()
    has_pkg = True
    try:
        import supabase  # noqa: F401
    except ImportError:
        has_pkg = False
    host = urlparse(cfg.get("url") or "").netloc
    return {
        "configured": is_configured(),
        "package_installed": has_pkg,
        "url_set": bool((cfg.get("url") or "").strip()),
        "key_set": bool((cfg.get("key") or "").strip()),
        "url_host": host,
    }
