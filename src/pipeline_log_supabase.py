"""
Log GitHub / local pipeline runs to Supabase
--------------------------------------------
Table: public.pipeline_runs (see sql/pipeline_runs_supabase.sql)

Uses the same [supabase] url/key as watchlists (service_role recommended).
Env: SUPABASE_URL, SUPABASE_KEY
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.watchlist_supabase import get_client, is_configured

TABLE = "pipeline_runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_row(
    *,
    status: str = "success",
    meta: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge fundamentals_meta + score_summary into a pipeline_runs row."""
    meta = meta or {}
    summary = summary or {}
    extra = extra or {}

    max_t = meta.get("max_tickers", extra.get("max_tickers"))
    try:
        max_t_int = int(max_t) if max_t not in (None, "", "0") else None
        if max_t_int == 0:
            max_t_int = None
    except (TypeError, ValueError):
        max_t_int = None

    use_sector = meta.get("use_sector")
    if use_sector is None:
        use_sector = summary.get("use_sector")
    if use_sector is None:
        use_sector = extra.get("use_sector")
    if isinstance(use_sector, str):
        use_sector = use_sector.strip().lower() in ("1", "true", "yes")

    commit_csv = meta.get("commit_csv")
    if commit_csv is None:
        commit_csv = extra.get("commit_csv")
    if isinstance(commit_csv, str):
        commit_csv = commit_csv.strip().lower() in ("1", "true", "yes")

    details = {
        "meta": meta,
        "summary": summary,
        **{k: v for k, v in extra.items() if k not in ("max_tickers", "use_sector", "commit_csv")},
    }

    row: dict[str, Any] = {
        "workflow_run": str(meta.get("workflow_run") or extra.get("workflow_run") or "")
        or None,
        "repository": meta.get("repository") or extra.get("repository"),
        "ref": meta.get("ref") or extra.get("ref"),
        "status": status,
        "source": meta.get("source") or "github-actions daily-fundamentals",
        "finished_at": meta.get("generated_at_utc") or _now_iso(),
        "n_tickers": meta.get("n_tickers")
        if meta.get("n_tickers") is not None
        else summary.get("n_tickers"),
        "n_rows": meta.get("n_rows") if meta.get("n_rows") is not None else summary.get("n_rows"),
        "max_tickers": max_t_int,
        "use_sector": use_sector,
        "avg_quality": summary.get("avg_quality"),
        "median_quality": summary.get("median_quality"),
        "n_data_warnings": summary.get("n_data_warnings"),
        "commit_csv": commit_csv,
        "workflow_url": meta.get("workflow_url") or extra.get("workflow_url"),
        "error_message": error_message,
        "details": details,
    }
    # Drop pure-null optional keys that confuse upsert? Keep them — Postgres accepts nulls.
    return row


def log_pipeline_run(
    row: dict[str, Any],
    *,
    client=None,
) -> dict[str, Any]:
    """
    Insert or upsert a pipeline run. Returns {ok, id?, error?}.
    Upserts on workflow_run when present (unique partial index).
    """
    if not is_configured():
        return {"ok": False, "error": "Supabase not configured"}
    try:
        client = client or get_client()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    payload = dict(row)
    # jsonb: ensure details is a dict
    if not isinstance(payload.get("details"), dict):
        try:
            payload["details"] = json.loads(payload["details"])
        except Exception:
            payload["details"] = {"raw": str(payload.get("details"))}

    try:
        wr = payload.get("workflow_run")
        if wr:
            res = (
                client.table(TABLE)
                .upsert(payload, on_conflict="workflow_run")
                .execute()
            )
        else:
            res = client.table(TABLE).insert(payload).execute()
        data = res.data or []
        rid = data[0].get("id") if data and isinstance(data[0], dict) else None
        return {"ok": True, "id": rid, "data": data}
    except Exception as e:
        # Fallback insert without upsert if unique index missing / conflict name differs
        try:
            res = client.table(TABLE).insert(payload).execute()
            data = res.data or []
            rid = data[0].get("id") if data and isinstance(data[0], dict) else None
            return {"ok": True, "id": rid, "data": data, "note": f"insert fallback: {e}"}
        except Exception as e2:
            return {"ok": False, "error": str(e2)}


def fetch_recent_runs(limit: int = 10, client=None) -> list[dict[str, Any]]:
    """Latest pipeline runs for UI (empty list if unconfigured / error)."""
    if not is_configured():
        return []
    try:
        client = client or get_client()
        res = (
            client.table(TABLE)
            .select(
                "id,workflow_run,status,source,finished_at,n_tickers,n_rows,"
                "avg_quality,n_data_warnings,workflow_url,error_message,ref,repository"
            )
            .order("finished_at", desc=True)
            .limit(int(limit))
            .execute()
        )
        return list(res.data or [])
    except Exception:
        return []


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
