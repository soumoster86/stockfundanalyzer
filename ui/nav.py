"""
App navigation: 3 modes × pages
--------------------------------
Research · Context · Tools — fewer top-level choices than a flat 7-page radio.
"""

from __future__ import annotations

from typing import Mapping

# Mode → ordered page labels (stable strings used by _pending_nav jumps)
NAV_BY_MODE: dict[str, list[str]] = {
    "Research": [
        "Single Stock Report",
        "Universe Ranking",
        "Watchlist",
    ],
    "Context": [
        "Compare",
        "Sector Overview",
    ],
    "Tools": [
        "Train Model",
        "Tutorial",
    ],
}

NAV_MODES: list[str] = list(NAV_BY_MODE.keys())

ALL_PAGES: list[str] = [p for pages in NAV_BY_MODE.values() for p in pages]

PAGE_TO_MODE: dict[str, str] = {
    page: mode for mode, pages in NAV_BY_MODE.items() for page in pages
}

# Pages where the universe summary metrics row is useful
SHOW_UNIVERSE_BANNER: frozenset[str] = frozenset(
    {
        "Universe Ranking",
        "Sector Overview",
        "Compare",
        "Watchlist",
    }
)


def resolve_pending_nav(session_state: Mapping) -> None:
    """
    Apply cross-page jumps written as session_state["_pending_nav"].

    Must run *before* mode/page widgets are instantiated.
    """
    pending = session_state.pop("_pending_nav", None)  # type: ignore[attr-defined]
    if not pending or pending not in PAGE_TO_MODE:
        return
    session_state["nav_mode"] = PAGE_TO_MODE[pending]  # type: ignore[index]
    session_state["nav_page"] = pending  # type: ignore[index]


def ensure_page_in_mode(session_state: Mapping, mode: str) -> str:
    """If current page is not in mode, snap to the mode's first page."""
    pages = NAV_BY_MODE.get(mode) or ALL_PAGES
    current = session_state.get("nav_page")  # type: ignore[attr-defined]
    if current not in pages:
        session_state["nav_page"] = pages[0]  # type: ignore[index]
        return pages[0]
    return str(current)
