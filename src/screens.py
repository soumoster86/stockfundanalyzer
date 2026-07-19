"""
Saved ranking screens (filter presets)
--------------------------------------
Named filter combinations for the Universe Ranking tab. Built-in presets plus
user-defined screens stored in Streamlit session_state (exportable as JSON/CSV
metadata via the UI).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

SESSION_KEY = "custom_screens"

# Built-in screens: values match ranking UI controls + numeric thresholds
BUILTIN_SCREENS: dict[str, dict[str, Any]] = {
    "All (default)": {
        "flag_filter": "All",
        "rel_filter": "All",
        "min_quality": 0.0,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "No extra filters.",
    },
    "Clean quality": {
        "flag_filter": "No red flags",
        "rel_filter": "Reliable only",
        "min_quality": 65.0,
        "max_pe": None,
        "max_de": None,
        "watchlist_only": False,
        "description": "Quality ≥ 65, no red flags, reliable data only.",
    },
    "Value quality": {
        "flag_filter": "No red flags",
        "rel_filter": "Reliable only",
        "min_quality": 55.0,
        "max_pe": 25.0,
        "max_de": None,
        "watchlist_only": False,
        "description": "Quality ≥ 55, P/E ≤ 25, no red flags.",
    },
    "Low leverage": {
        "flag_filter": "All",
        "rel_filter": "Reliable only",
        "min_quality": 50.0,
        "max_pe": None,
        "max_de": 1.0,
        "watchlist_only": False,
        "description": "Quality ≥ 50, D/E ≤ 1.0, reliable data.",
    },
    "Watchlist only": {
        "flag_filter": "All",
        "rel_filter": "All",
        "min_quality": 0.0,
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


def apply_screen(
    df: pd.DataFrame,
    screen: dict[str, Any],
    watchlist: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply a screen dict to a ranked (or scored) frame.
    Expects columns used by ranking filters when present.
    """
    view = df.copy()
    flag_filter = screen.get("flag_filter", "All")
    rel_filter = screen.get("rel_filter", "All")
    min_q = float(screen.get("min_quality") or 0.0)
    max_pe = screen.get("max_pe")
    max_de = screen.get("max_de")
    wl_only = bool(screen.get("watchlist_only", False))

    if wl_only and watchlist is not None:
        wanted = {str(t).strip() for t in watchlist}
        if "ticker" in view.columns:
            view = view[view["ticker"].astype(str).str.strip().isin(wanted)]

    if "red_flag_count" in view.columns:
        if flag_filter == "No red flags":
            view = view[view["red_flag_count"] == 0]
        elif flag_filter == "Has red flags":
            view = view[view["red_flag_count"] > 0]

    if "data_warning" in view.columns:
        if rel_filter == "Reliable only":
            view = view[~view["data_warning"].astype(bool)]
        elif rel_filter == "Warnings only":
            view = view[view["data_warning"].astype(bool)]

    if "quality_score" in view.columns and min_q > 0:
        view = view[view["quality_score"].fillna(-1) >= min_q]

    if max_pe is not None and "pe" in view.columns:
        view = view[view["pe"].isna() | (view["pe"] <= float(max_pe))]

    if max_de is not None and "debt_to_equity" in view.columns:
        view = view[
            view["debt_to_equity"].isna() | (view["debt_to_equity"] <= float(max_de))
        ]

    return view
