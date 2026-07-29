"""
Saved ranking screens (filter presets)
--------------------------------------
Named filter combinations for the Universe Ranking tab. Built-in presets plus
user-defined screens stored in Streamlit session_state.

Supports absolute min_quality and relative top_pct (e.g. top 10% by quality).
``screen_funnel`` returns step-by-step match counts so the UI can show why N remain.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

SESSION_KEY = "custom_screens"

# Built-in screens. Clean quality is "daily usable" (Q≥55); elite is the strict bar.
BUILTIN_SCREENS: dict[str, dict[str, Any]] = {
    "All (default)": {
        "flag_filter": "All",
        "rel_filter": "All",
        "min_quality": 0.0,
        "top_pct": None,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "No extra filters.",
    },
    "Clean quality": {
        "flag_filter": "No red flags",
        "rel_filter": "Reliable only",
        "min_quality": 55.0,
        "top_pct": None,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "Q≥55, no red flags, reliable data (day-to-day shortlist).",
    },
    "Clean quality (elite)": {
        "flag_filter": "No red flags",
        "rel_filter": "Reliable only",
        "min_quality": 65.0,
        "top_pct": None,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "Q≥65, no red flags, reliable — very few names on relative scores.",
    },
    "Top 10% quality": {
        "flag_filter": "All",
        "rel_filter": "Reliable only",
        "min_quality": 0.0,
        "top_pct": 10.0,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "Highest 10% by quality score (relative), reliable data only.",
    },
    "Top 20% quality": {
        "flag_filter": "All",
        "rel_filter": "Reliable only",
        "min_quality": 0.0,
        "top_pct": 20.0,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "Highest 20% by quality score (relative), reliable data only.",
    },
    "Value quality": {
        "flag_filter": "No red flags",
        "rel_filter": "Reliable only",
        "min_quality": 55.0,
        "top_pct": None,
        "max_pe": 25.0,
        "max_de": None,
        "watchlist_only": False,
        "description": "Q≥55, P/E ≤ 25, no red flags, reliable.",
    },
    "Low leverage": {
        "flag_filter": "All",
        "rel_filter": "Reliable only",
        "min_quality": 50.0,
        "top_pct": None,
        "max_pe": None,
        "max_de": 1.0,
        "watchlist_only": False,
        "description": "Q≥50, D/E ≤ 1.0, reliable data.",
    },
    "Watchlist only": {
        "flag_filter": "All",
        "rel_filter": "All",
        "min_quality": 0.0,
        "top_pct": None,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": True,
        "description": "Restrict to your saved watchlist.",
    },
}


def list_screens(session_state=None) -> dict[str, dict[str, Any]]:
    out = deepcopy(BUILTIN_SCREENS)
    if session_state is not None:
        custom = session_state.get(SESSION_KEY, {})
        if isinstance(custom, dict):
            out.update(deepcopy(custom))
    return out


def save_custom_screen(session_state, name: str, screen: dict[str, Any]) -> None:
    custom = dict(session_state.get(SESSION_KEY, {}))
    custom[name] = dict(screen)
    session_state[SESSION_KEY] = custom


def delete_custom_screen(session_state, name: str) -> bool:
    if name in BUILTIN_SCREENS:
        return False
    custom = dict(session_state.get(SESSION_KEY, {}))
    if name not in custom:
        return False
    del custom[name]
    session_state[SESSION_KEY] = custom
    return True


def _apply_watchlist(view: pd.DataFrame, watchlist: list[str] | None) -> pd.DataFrame:
    if watchlist is None:
        return view
    wanted = {str(t).strip() for t in watchlist}
    if "ticker" not in view.columns:
        return view
    return view[view["ticker"].astype(str).str.strip().isin(wanted)]


def _apply_flags(view: pd.DataFrame, flag_filter: str) -> pd.DataFrame:
    if "red_flag_count" not in view.columns or flag_filter == "All":
        return view
    if flag_filter == "No red flags":
        return view[view["red_flag_count"] == 0]
    if flag_filter == "Has red flags":
        return view[view["red_flag_count"] > 0]
    return view


def _apply_reliability(view: pd.DataFrame, rel_filter: str) -> pd.DataFrame:
    if "data_warning" not in view.columns or rel_filter == "All":
        return view
    if rel_filter == "Reliable only":
        return view[~view["data_warning"].astype(bool)]
    if rel_filter == "Warnings only":
        return view[view["data_warning"].astype(bool)]
    return view


def _apply_min_quality(view: pd.DataFrame, min_q: float) -> pd.DataFrame:
    if "quality_score" not in view.columns or min_q <= 0:
        return view
    return view[view["quality_score"].fillna(-1) >= min_q]


def _apply_top_pct(view: pd.DataFrame, top_pct: float | None) -> pd.DataFrame:
    """Keep the highest top_pct% of rows by quality_score (among current view)."""
    if top_pct is None or top_pct <= 0 or "quality_score" not in view.columns:
        return view
    if view.empty:
        return view
    q = view["quality_score"]
    valid = q.dropna()
    if valid.empty:
        return view.iloc[0:0].copy()
    # Threshold = (100 - top_pct)th percentile of *current* view
    thr = float(np.percentile(valid.to_numpy(), 100.0 - float(top_pct)))
    return view[q.fillna(-np.inf) >= thr]


def _apply_pe(view: pd.DataFrame, max_pe) -> pd.DataFrame:
    if max_pe is None or "pe" not in view.columns:
        return view
    return view[view["pe"].isna() | (view["pe"] <= float(max_pe))]


def _apply_de(view: pd.DataFrame, max_de) -> pd.DataFrame:
    if max_de is None or "debt_to_equity" not in view.columns:
        return view
    return view[
        view["debt_to_equity"].isna() | (view["debt_to_equity"] <= float(max_de))
    ]


def apply_screen(
    df: pd.DataFrame,
    screen: dict[str, Any],
    watchlist: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply a screen dict to a ranked (or scored) frame.
    Order: watchlist → flags → reliability → min quality → top % → PE → D/E.
    """
    view = df.copy()
    if bool(screen.get("watchlist_only", False)):
        view = _apply_watchlist(view, watchlist)
    view = _apply_flags(view, screen.get("flag_filter", "All"))
    view = _apply_reliability(view, screen.get("rel_filter", "All"))
    view = _apply_min_quality(view, float(screen.get("min_quality") or 0.0))
    view = _apply_top_pct(view, screen.get("top_pct"))
    view = _apply_pe(view, screen.get("max_pe"))
    view = _apply_de(view, screen.get("max_de"))
    return view


def screen_funnel(
    df: pd.DataFrame,
    screen: dict[str, Any],
    watchlist: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Step-by-step match counts for the active screen (for UI funnel chips).

    Each step: {label, n, delta} where delta is how many were dropped at this step
    (negative = removed).
    """
    steps: list[dict[str, Any]] = []
    view = df.copy()
    n0 = len(view)
    steps.append({"key": "universe", "label": "Universe", "n": n0, "delta": 0})

    def _step(key: str, label: str, new_view: pd.DataFrame):
        nonlocal view
        n_before = len(view)
        view = new_view
        steps.append(
            {
                "key": key,
                "label": label,
                "n": len(view),
                "delta": len(view) - n_before,
            }
        )

    if bool(screen.get("watchlist_only", False)):
        _step("watchlist", "Watchlist", _apply_watchlist(view, watchlist))

    ff = screen.get("flag_filter", "All")
    if ff != "All":
        _step("flags", ff, _apply_flags(view, ff))

    rf = screen.get("rel_filter", "All")
    if rf != "All":
        short = "Reliable" if rf == "Reliable only" else "Warnings"
        _step("data", short, _apply_reliability(view, rf))

    min_q = float(screen.get("min_quality") or 0.0)
    if min_q > 0:
        _step("quality", f"Q≥{min_q:.0f}", _apply_min_quality(view, min_q))

    top_pct = screen.get("top_pct")
    if top_pct is not None and float(top_pct) > 0:
        _step("top_pct", f"Top {float(top_pct):.0f}%", _apply_top_pct(view, float(top_pct)))

    max_pe = screen.get("max_pe")
    if max_pe is not None:
        _step("pe", f"PE≤{float(max_pe):.0f}", _apply_pe(view, max_pe))

    max_de = screen.get("max_de")
    if max_de is not None:
        _step("de", f"D/E≤{float(max_de):.2g}", _apply_de(view, max_de))

    return steps
