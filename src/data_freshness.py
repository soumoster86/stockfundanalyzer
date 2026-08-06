"""
Data freshness / provenance
---------------------------
Reads `fundamentals_meta.json` written by the GitHub daily-fundamentals workflow
(and optionally local refresh) so the UI can show that rankings use CI data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Committed to the repo root by CI (not under gitignored data/ or artifacts/)
DEFAULT_META_NAME = "fundamentals_meta.json"


def meta_path(project_dir: str | Path | None = None) -> Path:
    root = Path(project_dir) if project_dir else Path(".")
    return root / DEFAULT_META_NAME


def load_fundamentals_meta(
    project_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return meta dict or None if missing/invalid."""
    p = meta_path(project_dir)
    if not p.is_file():
        # Also try next to this package's parent (repo root when deployed)
        alt = Path(__file__).resolve().parents[1] / DEFAULT_META_NAME
        p = alt if alt.is_file() else p
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_github_daily(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    src = str(meta.get("source") or "").lower()
    return "github-actions" in src or "daily-fundamentals" in src


def format_freshness_line(
    meta: dict[str, Any] | None,
    *,
    data_source_label: str | None = None,
    file_mtime: float | None = None,
) -> str:
    """
    One-line status for captions.
    Prefer CI meta; fall back to file mtime / source label.
    """
    if meta and is_github_daily(meta):
        when = meta.get("generated_at_utc") or meta.get("generated_at") or "?"
        n = meta.get("n_tickers")
        run = meta.get("workflow_run") or meta.get("run_id")
        bits = [
            "🔄 **GitHub daily pipeline**",
            f"fetched **{when}** UTC",
        ]
        if n is not None:
            bits.append(f"**{n:,}** tickers" if isinstance(n, int) else f"**{n}** tickers")
        if run:
            bits.append(f"run `{run}`")
        age = _age_hint(str(when))
        if age:
            bits.append(age)
        return " · ".join(bits)

    if meta and meta.get("generated_at_utc"):
        src = meta.get("source") or "local"
        return (
            f"Data panel: **{src}** · generated **{meta.get('generated_at_utc')}** UTC"
        )

    if file_mtime is not None:
        local = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M")
        label = data_source_label or "fundamentals file"
        return f"Data source: **{label}** · file updated **{local}** (local clock)"

    if data_source_label:
        return f"Data source: **{data_source_label}**"

    return "Data source: unknown"


def format_freshness_detail(meta: dict[str, Any] | None) -> list[str]:
    """Bullet lines for an expander."""
    if not meta:
        return [
            "No `fundamentals_meta.json` in the repo yet.",
            "After the first successful **Daily fundamentals + rankings** "
            "GitHub Action (with commit enabled), this file is written next to "
            "`fundamentals.csv` and the banner will show the pipeline run.",
        ]
    lines = []
    for key, label in (
        ("source", "Source"),
        ("generated_at_utc", "Fetched (UTC)"),
        ("n_tickers", "Tickers"),
        ("n_rows", "Rows"),
        ("workflow_run", "Actions run id"),
        ("ref", "Git branch"),
        ("repository", "Repository"),
        ("max_tickers", "Max tickers cap"),
    ):
        if key in meta and meta[key] not in (None, ""):
            lines.append(f"**{label}:** `{meta[key]}`")
    if is_github_daily(meta):
        lines.append(
            "Rankings in the app are scored from this panel when Streamlit "
            "loads the committed `fundamentals.csv`."
        )
    return lines


def _age_hint(iso_utc: str) -> str:
    """Human age like '2h ago' if parseable."""
    try:
        s = iso_utc.strip().replace("Z", "+00:00")
        # allow "YYYY-MM-DDTHH:MM:SS+00:00" or without T space
        if "T" not in s and " " in s:
            s = s.replace(" ", "T", 1)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        secs = max(0, int((now - dt.astimezone(timezone.utc)).total_seconds()))
        if secs < 3600:
            return f"~{secs // 60}m ago"
        if secs < 86400:
            return f"~{secs // 3600}h ago"
        return f"~{secs // 86400}d ago"
    except Exception:
        return ""


def write_fundamentals_meta(
    path: str | Path,
    **fields: Any,
) -> Path:
    """Helper for local scripts / CI to write the meta file."""
    p = Path(path)
    if p.is_dir():
        p = p / DEFAULT_META_NAME
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **fields,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
