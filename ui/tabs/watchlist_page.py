"""Watchlist portfolio page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.quality_score import CONCEPT_TOOLTIPS, score_label
from src.watchlist import (
    add_ticker,
    filter_universe,
    get_watchlist,
    remove_ticker,
    set_watchlist,
    watchlist_from_csv,
    watchlist_summary,
    watchlist_to_csv,
)
from ui.theme import badge_row, band_text, section


def render_watchlist(data: pd.DataFrame) -> None:
    section("Watchlist")
    st.caption(
        "Focus on names you care about. Session-scoped — download a CSV to keep it "
        "across restarts. Z/M: 🟢 good · 🟡 watch · 🔴 risk."
    )

    wl = get_watchlist(st.session_state)
    summary = watchlist_summary(data, wl)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Names", summary["n"])
    c2.metric(
        "Avg quality",
        f"{summary['avg_quality']:.1f}" if pd.notna(summary["avg_quality"]) else "—",
        help=CONCEPT_TOOLTIPS.get("quality_score", "Mean quality on the watchlist."),
    )
    c3.metric(
        "Median quality",
        f"{summary['median_quality']:.1f}" if pd.notna(summary["median_quality"]) else "—",
    )
    c4.metric(
        "Avg flags",
        f"{summary['avg_flags']:.1f}" if pd.notna(summary["avg_flags"]) else "—",
    )

    # Manage list
    st.markdown("**Manage**")
    all_tickers = sorted(data["ticker"].unique())
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        pick = st.selectbox(
            "Add ticker",
            ["—"] + all_tickers,
            key="wl_add_pick",
        )
    with m2:
        if st.button("➕ Add", use_container_width=True) and pick != "—":
            if add_ticker(st.session_state, pick):
                st.success(f"Added {pick}")
                st.rerun()
            else:
                st.info("Already on the watchlist.")
    with m3:
        if st.button("🗑️ Clear all", use_container_width=True) and wl:
            set_watchlist(st.session_state, [])
            st.rerun()

    up, down = st.columns(2)
    with up:
        uploaded = st.file_uploader("Import watchlist CSV", type="csv", key="wl_import")
        if uploaded is not None and st.button("Load import"):
            try:
                tickers = watchlist_from_csv(uploaded)
                # keep only those present in universe (plus allow orphans for later)
                set_watchlist(st.session_state, tickers)
                st.success(f"Loaded {len(tickers)} ticker(s).")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with down:
        st.download_button(
            "⬇️ Download watchlist CSV",
            data=watchlist_to_csv(wl),
            file_name="watchlist.csv",
            mime="text/csv",
            disabled=not wl,
        )

    if not wl:
        st.info("Watchlist is empty. Add tickers above or import a CSV with a `ticker` column.")
        return

    # Heat-style table
    sub = filter_universe(data, wl)
    if sub.empty:
        st.warning(
            "Watchlist tickers are not in the current universe. "
            f"List: {', '.join(wl)}"
        )
        st.write(wl)
        return

    # Preserve watchlist order
    order = {t: i for i, t in enumerate(wl)}
    sub = sub.copy()
    sub["_ord"] = sub["ticker"].map(lambda t: order.get(t, 9999))
    sub = sub.sort_values(["_ord", "quality_score"], ascending=[True, False])

    show_cols = [
        c
        for c in [
            "ticker",
            "sector",
            "quality_score",
            "red_flag_count",
            "roe",
            "pe",
            "debt_to_equity",
            "f_score",
            "z_band",
            "m_band",
            "data_warning",
        ]
        if c in sub.columns
    ]
    disp = sub[show_cols].reset_index(drop=True)
    disp["ticker_disp"] = disp["ticker"].str.replace(".NS", "", regex=False)

    st.markdown("**Portfolio snapshot**")
    if "z_band" in disp.columns:
        disp["z_band"] = disp["z_band"].map(lambda x: band_text(x, "z"))
    if "m_band" in disp.columns:
        disp["m_band"] = disp["m_band"].map(lambda x: band_text(x, "m"))
    if "data_warning" in disp.columns:
        disp["data_warning"] = disp["data_warning"].map(
            lambda x: "⚠️ Review" if bool(x) else "✓ OK"
        )
    if "f_score" in disp.columns:
        disp["f_score"] = disp["f_score"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "–"
        )

    colcfg = {
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "quality_score": st.column_config.ProgressColumn(
            "Quality", min_value=0, max_value=100, format="%.1f"
        ),
        "red_flag_count": st.column_config.NumberColumn("Flags", format="%d"),
        "roe": st.column_config.NumberColumn("ROE", format="%.1f"),
        "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        "debt_to_equity": st.column_config.NumberColumn("D/E", format="%.2f"),
        "z_band": st.column_config.TextColumn("Altman Z", width="small"),
        "m_band": st.column_config.TextColumn("Beneish M", width="small"),
        "f_score": st.column_config.TextColumn("F", width="small"),
        "data_warning": st.column_config.TextColumn("Data", width="small"),
    }
    if "sector" in disp.columns:
        colcfg["sector"] = st.column_config.TextColumn("Sector")

    out = disp.drop(columns=["ticker"], errors="ignore").rename(
        columns={"ticker_disp": "ticker"}
    )
    # Summary badges for the portfolio
    n_risk = int((sub["m_band"] == "Red").sum()) if "m_band" in sub.columns else 0
    n_distress = int((sub["z_band"] == "Red").sum()) if "z_band" in sub.columns else 0
    st.markdown(
        badge_row(
            [
                (f"{summary['n']} names", "Moderate"),
                (
                    f"Avg Q {summary['avg_quality']:.0f}"
                    if pd.notna(summary["avg_quality"])
                    else "Avg Q —",
                    "Strong"
                    if pd.notna(summary["avg_quality"]) and summary["avg_quality"] >= 65
                    else "Moderate",
                ),
                (f"{n_distress} Z-risk", "Red" if n_distress else "Green"),
                (f"{n_risk} M-risk", "Red" if n_risk else "Green"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(out, use_container_width=True, hide_index=True, column_config=colcfg)

    # Per-name actions
    st.markdown("**Open or remove**")
    a1, a2 = st.columns(2)
    with a1:
        open_tk = st.selectbox("Open report", wl, key="wl_open")
        if st.button("➡️ Open report"):
            st.session_state["report_ticker"] = open_tk
            st.session_state["_report_jump"] = True
            st.session_state["_pending_nav"] = "Single Stock Report"
            st.rerun()
    with a2:
        rm_tk = st.selectbox("Remove", wl, key="wl_rm")
        if st.button("➖ Remove from watchlist"):
            remove_ticker(st.session_state, rm_tk)
            st.rerun()

    # Sector mix
    if "sector" in sub.columns and sub["sector"].notna().any():
        st.markdown("**Sector mix**")
        mix = (
            sub.groupby("sector", dropna=False)
            .agg(Names=("ticker", "nunique"), Avg_quality=("quality_score", "mean"))
            .reset_index()
            .sort_values("Names", ascending=False)
        )
        st.bar_chart(mix.set_index("sector")["Names"], height=220)
        st.dataframe(mix, use_container_width=True, hide_index=True)

    # Quality labels strip
    st.caption(
        "Labels: "
        + ", ".join(
            f"{r['ticker'].replace('.NS', '')}={score_label(r['quality_score'])}"
            for _, r in sub.iterrows()
            if pd.notna(r.get("quality_score"))
        )
    )
