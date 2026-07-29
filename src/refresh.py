"""
Local fundamentals refresh
--------------------------
Re-fetch Yahoo fundamentals for tickers in stocks.csv or the current panel.
Intended for local runs (needs yfinance + network). Streamlit Cloud should
not rely on this (rate limits / no long jobs).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def yfinance_available() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_tickers_csv(
    project_dir: str | Path,
    scored_tickers: list[str] | None = None,
) -> tuple[Path | None, list[str]]:
    """
    Prefer stocks.csv; else build a temporary list from scored_tickers.
    Returns (path_or_None, tickers).
    """
    root = Path(project_dir)
    stocks = root / "stocks.csv"
    if stocks.exists():
        df = pd.read_csv(stocks)
        df.columns = [c.strip().lower() for c in df.columns]
        col = next(
            (
                c
                for c in (
                    "ticker",
                    "symbol",
                    "tickers",
                    "symbols",
                    "code",
                    "nse_symbol",
                    "yahoo_symbol",
                )
                if c in df.columns
            ),
            None,
        )
        if col:
            tickers = (
                df[col].dropna().astype(str).str.strip().drop_duplicates().tolist()
            )
            return stocks, tickers
    if scored_tickers:
        return None, list(dict.fromkeys(str(t).strip() for t in scored_tickers if t))
    return None, []


def refresh_fundamentals(
    project_dir: str | Path,
    out_name: str = "fundamentals.csv",
    tickers: list[str] | None = None,
    sleep_s: float = 0.35,
    max_tickers: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Fetch fundamentals for tickers and write project_dir/out_name.

    progress_cb(i, total, ticker) optional.
    Returns {ok, path, n_tickers, n_rows, failures, message}.
    """
    if not yfinance_available():
        return {
            "ok": False,
            "path": None,
            "n_tickers": 0,
            "n_rows": 0,
            "failures": [],
            "message": "yfinance is not installed. Run: pip install yfinance",
        }

    from src.fetch_fundamentals import OUTPUT_COLUMNS, fetch_one, _blank_row

    root = Path(project_dir)
    path, resolved = resolve_tickers_csv(root, tickers)
    if not resolved:
        return {
            "ok": False,
            "path": None,
            "n_tickers": 0,
            "n_rows": 0,
            "failures": [],
            "message": "No tickers found (add stocks.csv or load a panel first).",
        }

    if max_tickers is not None:
        resolved = resolved[: int(max_tickers)]

    all_rows = []
    failures = []
    total = len(resolved)
    for i, tkr in enumerate(resolved, start=1):
        if progress_cb:
            progress_cb(i, total, tkr)
        try:
            all_rows.extend(fetch_one(tkr, ""))
        except Exception:
            failures.append(tkr)
            all_rows.append(_blank_row(tkr, ""))
        if sleep_s:
            time.sleep(sleep_s)

    out = pd.DataFrame(all_rows)
    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUTPUT_COLUMNS]
    out_path = root / out_name
    out.to_csv(out_path, index=False)

    return {
        "ok": True,
        "path": str(out_path),
        "n_tickers": total,
        "n_rows": len(out),
        "failures": failures,
        "message": (
            f"Wrote {len(out)} rows for {total} tickers → {out_path.name}"
            + (f" ({len(failures)} fetch failures)" if failures else "")
        ),
        "source_list": str(path) if path else "current universe tickers",
    }
