"""3-mode navigation helpers."""
from ui.nav import (
    ALL_PAGES,
    NAV_BY_MODE,
    NAV_MODES,
    PAGE_TO_MODE,
    SHOW_UNIVERSE_BANNER,
    ensure_page_in_mode,
    resolve_pending_nav,
)


class _SS(dict):
    def pop(self, key, default=None):
        return super().pop(key, default)


def test_all_pages_mapped():
    assert len(ALL_PAGES) == sum(len(v) for v in NAV_BY_MODE.values())
    assert set(PAGE_TO_MODE) == set(ALL_PAGES)
    assert NAV_MODES == ["Research", "Context", "Tools"]


def test_streamlit_has_pills():
    """Nav uses st.pills for capsule UI (Streamlit ≥1.33)."""
    import streamlit as st

    assert hasattr(st, "pills"), "Upgrade streamlit for pill navigation UI"


def test_resolve_pending_nav_sets_mode_and_page():
    ss = _SS()
    ss["_pending_nav"] = "Single Stock Report"
    resolve_pending_nav(ss)
    assert ss["nav_mode"] == "Research"
    assert ss["nav_page"] == "Single Stock Report"
    assert "_pending_nav" not in ss


def test_resolve_pending_nav_tools():
    ss = _SS()
    ss["_pending_nav"] = "Tutorial"
    resolve_pending_nav(ss)
    assert ss["nav_mode"] == "Tools"
    assert ss["nav_page"] == "Tutorial"


def test_ensure_page_in_mode_snaps():
    ss = _SS()
    ss["nav_page"] = "Tutorial"
    page = ensure_page_in_mode(ss, "Research")
    assert page == "Single Stock Report"
    assert ss["nav_page"] == "Single Stock Report"


def test_ensure_page_keeps_valid():
    ss = _SS()
    ss["nav_page"] = "Watchlist"
    page = ensure_page_in_mode(ss, "Research")
    assert page == "Watchlist"


def test_banner_pages():
    assert "Universe Ranking" in SHOW_UNIVERSE_BANNER
    assert "Single Stock Report" not in SHOW_UNIVERSE_BANNER
    assert "Train Model" not in SHOW_UNIVERSE_BANNER
