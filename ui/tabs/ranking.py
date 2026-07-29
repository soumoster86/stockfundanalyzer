"""Universe Ranking tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.quality_score import CONCEPT_TOOLTIPS, METRIC_TOOLTIPS
from src.ranking import rank_universe
from src.screens import (
    BUILTIN_SCREENS,
    apply_screen,
    delete_custom_screen,
    list_screens,
    save_custom_screen,
    screen_funnel,
)
from src.watchlist import add_ticker, get_watchlist
from ui.theme import band_text, section


def render_ranking(data: pd.DataFrame) -> None:
    section("Multi-factor ranking")
    st.caption(
        "Latest fiscal-year figures, ranked together. "
        "Use **screens** for repeatable filters · "
        "Z: bankruptcy risk · M: earnings-manipulation risk · "
        "🟢 ok · 🟡 caution · 🔴 elevated risk"
    )
    has_ml_scores = (
        "outperform_proba" in data.columns and data["outperform_proba"].notna().any()
    )
    if has_ml_scores:
        wq = st.slider(
            "Weight: Quality Score",
            0.0,
            1.0,
            0.5,
            0.05,
            help="How much the composite ranking leans on quality vs ML probability.",
        )
        w_ml = 1.0 - wq
    else:
        wq, w_ml = 1.0, 0.0
        st.caption(
            "Ranking is pure quality (no ML scores yet). Train and score in **Train Model**."
        )
    ranked = rank_universe(data, w_quality=wq, w_ml=w_ml, as_of_date=None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks ranked", len(ranked))
    m2.metric(
        "Top score",
        f"{ranked['composite_score'].max():.1f}",
        help=CONCEPT_TOOLTIPS["composite_score"],
    )
    m3.metric("Median", f"{ranked['composite_score'].median():.1f}")
    if "data_warning" in ranked.columns:
        m4.metric("Data warnings", int(ranked["data_warning"].sum()))
    else:
        m4.metric("Clean (0 flags)", int((ranked["red_flag_count"] == 0).sum()))

    with st.expander("📊 Score distribution", expanded=False):
        vals = ranked["composite_score"].dropna()
        if len(vals):
            bins = np.arange(0, 105, 5)
            counts, edges = np.histogram(vals, bins=bins)
            hist_df = pd.DataFrame(
                {"count": counts},
                index=[
                    f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(counts))
                ],
            )
            st.bar_chart(hist_df, height=200)

    # ---- Compact screener toolbar (dense rows, no tall empty columns) ----
    screens = list_screens(st.session_state)
    screen_names = list(screens.keys())
    has_sector_col = "sector" in ranked.columns
    flag_opts = ["All", "No red flags", "Has red flags"]
    rel_opts = ["All", "Reliable only", "Warnings only"]
    sort_opts = {
        "Rank": ("rank", True),
        "Quality (raw)": ("quality_score", False),
        "ROE": ("roe", False),
        "P/E (low→high)": ("pe", True),
        "Debt/Equity (low→high)": ("debt_to_equity", True),
        "Revenue growth": ("revenue_growth", False),
    }

    # Row 1: preset + search + sector
    r1a, r1b, r1c = st.columns([1.4, 1.1, 1.5])
    with r1a:
        screen_name = st.selectbox(
            "Screen",
            screen_names,
            help="Filter preset. Choosing a screen resets thresholds & flag/data filters.",
            key="rank_screen",
        )
    screen = screens[screen_name]

    # When the Screen preset changes, push its defaults into widget session keys
    # BEFORE those widgets are created. Otherwise Streamlit keeps old values and
    # "Clean quality" / "Value quality" appear broken (description changes, filters don't).
    if st.session_state.get("_last_rank_screen") != screen_name:
        st.session_state["rank_min_q"] = float(screen.get("min_quality") or 0.0)
        st.session_state["rank_max_pe"] = float(screen.get("max_pe") or 0.0)
        st.session_state["rank_max_de"] = float(screen.get("max_de") or 0.0)
        st.session_state["rank_flags"] = screen.get("flag_filter", "All")
        st.session_state["rank_data"] = screen.get("rel_filter", "All")
        st.session_state["rank_top_pct"] = float(screen.get("top_pct") or 0.0)
        st.session_state["_last_rank_screen"] = screen_name
    # Ensure new widget keys exist even if screen name was already synced
    if "rank_top_pct" not in st.session_state:
        st.session_state["rank_top_pct"] = float(screen.get("top_pct") or 0.0)

    with r1b:
        search = st.text_input(
            "Search", "", placeholder="e.g. TCS", key="rank_search"
        ).strip().upper()
    with r1c:
        if has_sector_col:
            sectors = ["All sectors"] + sorted(
                ranked["sector"].dropna().unique().tolist()
            )
            sector_pick = st.selectbox("Sector", sectors, key="rank_sector")
        else:
            sector_pick = "All sectors"
            st.selectbox("Sector", ["All sectors"], disabled=True, key="rank_sector_na")

    # Row 2: thresholds (values come from session after screen sync above)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        min_quality = st.number_input(
            "Min quality",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key="rank_min_q",
            help="Absolute quality floor. Use 0 with Top % for relative screens.",
        )
    with t2:
        max_pe = st.number_input(
            "Max P/E (0=off)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="rank_max_pe",
        )
    with t3:
        max_de = st.number_input(
            "Max D/E (0=off)",
            min_value=0.0,
            max_value=20.0,
            step=0.1,
            key="rank_max_de",
        )
    with t4:
        top_pct = st.number_input(
            "Top % quality (0=off)",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key="rank_top_pct",
            help="Keep highest N% by quality among names that passed prior filters.",
        )
    if screen.get("description"):
        st.caption(screen["description"])

    # Row 3: flags / data / sort / show
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        flag_filter = st.selectbox("Flags", flag_opts, key="rank_flags")
    with f2:
        rel_filter = st.selectbox("Data", rel_opts, key="rank_data")
    with f3:
        sort_by = st.selectbox("Sort by", list(sort_opts.keys()), key="rank_sort")
    with f4:
        top_n = st.selectbox(
            "Show", ["Top 25", "Top 50", "Top 100", "All"], index=1, key="rank_topn"
        )

    active_screen = {
        "flag_filter": flag_filter,
        "rel_filter": rel_filter,
        "min_quality": float(min_quality),
        "top_pct": float(top_pct) if float(top_pct) > 0 else None,
        "max_pe": float(max_pe) if float(max_pe) > 0 else None,
        "max_de": float(max_de) if float(max_de) > 0 else None,
        "watchlist_only": bool(screen.get("watchlist_only", False)),
        "description": screen.get("description", ""),
    }

    with st.expander("Save / delete custom screen", expanded=False):
        s1, s2 = st.columns([2, 1])
        with s1:
            new_name = st.text_input(
                "Name", value="", key="rank_screen_name", placeholder="My screen"
            )
        with s2:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Save screen", use_container_width=True) and new_name.strip():
                save_custom_screen(
                    st.session_state,
                    new_name.strip(),
                    {**active_screen, "description": f"Custom: {new_name.strip()}"},
                )
                st.success(f"Saved “{new_name.strip()}”.")
                st.rerun()
        custom_names = [n for n in screens if n not in BUILTIN_SCREENS]
        if custom_names:
            d1, d2 = st.columns([2, 1])
            with d1:
                del_name = st.selectbox("Delete", custom_names, key="rank_del_screen")
            with d2:
                st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                if st.button("Delete", use_container_width=True):
                    delete_custom_screen(st.session_state, del_name)
                    st.rerun()

    wl = get_watchlist(st.session_state)

    # Funnel: how many survive each filter step (proves the screener is alive)
    funnel = screen_funnel(ranked, active_screen, watchlist=wl)
    view = apply_screen(ranked, active_screen, watchlist=wl)
    n_after_screen = len(view)

    # Visual funnel chips
    if len(funnel) > 1:
        st.markdown("**Screen funnel**")
        cols = st.columns(min(len(funnel), 6))
        for i, step in enumerate(funnel[:6]):
            with cols[i]:
                delta = step["delta"]
                delta_txt = (
                    ""
                    if i == 0
                    else (f"{delta:+,}" if delta != 0 else "—")
                )
                st.metric(step["label"], f"{step['n']:,}", delta_txt if i else None)
        if len(funnel) > 6:
            extra = " → ".join(f"{s['label']} {s['n']:,}" for s in funnel[6:])
            st.caption(f"… {extra}")
    else:
        st.caption(f"**Screen `{screen_name}`:** {n_after_screen:,} / {len(ranked):,} stocks.")

    if search:
        view = view[view["ticker"].str.upper().str.contains(search)]
    if sector_pick != "All sectors" and has_sector_col:
        view = view[view["sector"] == sector_pick]
    sort_col, asc = sort_opts[sort_by]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=asc, na_position="last")

    total_matched = len(view)
    limit_map = {"Top 25": 25, "Top 50": 50, "Top 100": 100, "All": len(view)}
    view = view.head(limit_map[top_n])

    if total_matched != n_after_screen:
        st.caption(
            f"After search/sector: **{total_matched:,}** names "
            f"(showing {min(total_matched, limit_map[top_n]):,})."
        )

    if screen_name != "All (default)" and n_after_screen == 0:
        st.warning(
            f"No stocks match **{screen_name}**. Loosen Min quality / Flags / Data, "
            "or try **Top 20% quality** (relative) instead of a high absolute Q cut."
        )
    elif "elite" in screen_name.lower() and 0 < n_after_screen <= 15:
        st.info(
            f"**{screen_name}** is intentionally strict. Only **{n_after_screen}** "
            "name(s) pass — the funnel above shows where names drop off. "
            "For a larger shortlist use **Clean quality** (Q≥55) or **Top 20% quality**."
        )

    candidate_cols = ["rank", "ticker"]
    if has_sector_col:
        candidate_cols.append("sector")
    candidate_cols += ["composite_score", "quality_score"]
    if "outperform_proba" in view.columns:
        candidate_cols.append("outperform_proba")
    candidate_cols += [
        "roe",
        "pe",
        "debt_to_equity",
        "net_margin",
        "revenue_growth",
        "f_score",
        "z_band",
        "m_band",
        "red_flag_count",
        "data_fields_present",
        "data_warning_count",
    ]
    show = [c for c in candidate_cols if c in view.columns]
    disp = view[show].reset_index(drop=True).copy()
    # Keep full ticker for navigation; display strip .NS
    disp["_full_ticker"] = disp["ticker"]
    disp["ticker"] = disp["ticker"].str.replace(".NS", "", regex=False)
    # Growth stored as decimals → show as percent points (0.12 → 12.0)
    if "revenue_growth" in disp.columns:
        disp["revenue_growth"] = pd.to_numeric(disp["revenue_growth"], errors="coerce") * 100.0
    if "f_score" in disp.columns:
        disp["f_score"] = disp["f_score"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "–"
        )
    if "z_band" in disp.columns:
        disp["z_band"] = disp["z_band"].map(lambda x: band_text(x, "z"))
    if "m_band" in disp.columns:
        disp["m_band"] = disp["m_band"].map(lambda x: band_text(x, "m"))
    if "data_fields_present" in disp.columns and "data_fields_total" in view.columns:
        tot = int(view["data_fields_total"].iloc[0]) if len(view) else 0
        disp["data_fields_present"] = disp["data_fields_present"].apply(
            lambda x: f"{int(x)}/{tot}" if pd.notna(x) else "–"
        )
    if "red_flag_count" in disp.columns:
        disp["red_flag_count"] = disp["red_flag_count"].apply(
            lambda n: "—" if (pd.isna(n) or n == 0) else "🚩" * int(min(n, 3))
        )
    if "data_warning_count" in disp.columns:
        disp["data_warning_count"] = disp["data_warning_count"].apply(
            lambda n: "" if (pd.isna(n) or n == 0) else "⚠️"
        )

    colcfg = {
        "rank": st.column_config.NumberColumn("#", width="small", help="Position in the ranking."),
        "ticker": st.column_config.TextColumn("Ticker", width="small", help="Stock symbol (NSE)."),
        "composite_score": st.column_config.ProgressColumn(
            "Score",
            min_value=0,
            max_value=100,
            format="%.1f",
            help=CONCEPT_TOOLTIPS["composite_score"],
        ),
        "quality_score": st.column_config.ProgressColumn(
            "Quality",
            min_value=0,
            max_value=100,
            format="%.1f",
            help=CONCEPT_TOOLTIPS["quality_score"],
        ),
        "roe": st.column_config.NumberColumn("ROE", format="%.1f", help=METRIC_TOOLTIPS["roe"]),
        "pe": st.column_config.NumberColumn("P/E", format="%.1f", help=METRIC_TOOLTIPS["pe"]),
        "debt_to_equity": st.column_config.NumberColumn(
            "D/E", format="%.2f", help=METRIC_TOOLTIPS["debt_to_equity"]
        ),
        "net_margin": st.column_config.NumberColumn(
            "Net%", format="%.1f", help=METRIC_TOOLTIPS["net_margin"]
        ),
        "revenue_growth": st.column_config.NumberColumn(
            "Rev gr %",
            format="%.1f",
            help="YoY revenue growth in percent points (12 = 12%). Stored as decimals in CSV.",
        ),
        "f_score": st.column_config.TextColumn(
            "F",
            width="small",
            help="Piotroski F-Score (0–9).",
        ),
        "z_band": st.column_config.TextColumn(
            "Altman Z",
            width="medium",
            help="Bankruptcy risk. 🟢 Safe (Z>3) · 🟡 Caution (1.8–3) · 🔴 Distress (Z<1.8).",
        ),
        "m_band": st.column_config.TextColumn(
            "Beneish M",
            width="medium",
            help="Earnings-manipulation model (Beneish). "
                 "🟢 Unlikely (M≤−2.22) · 🟡 Caution (−2.22<M≤−1.78) · "
                 "🔴 Likely manip. (M>−1.78). Not a proven fraud flag — review filings.",
        ),
        "red_flag_count": st.column_config.TextColumn(
            "Flags", width="small", help=CONCEPT_TOOLTIPS["red_flags"]
        ),
        "data_fields_present": st.column_config.TextColumn(
            "Data", width="small", help=CONCEPT_TOOLTIPS["data_completeness"]
        ),
        "data_warning_count": st.column_config.TextColumn(
            "⚠", width="small", help=CONCEPT_TOOLTIPS["data_warning"]
        ),
    }
    if has_sector_col:
        colcfg["sector"] = st.column_config.TextColumn(
            "Sector", width="medium", help=CONCEPT_TOOLTIPS["scored_vs"]
        )
    if "outperform_proba" in disp.columns:
        colcfg["outperform_proba"] = st.column_config.NumberColumn(
            "Outperf.", format="%.0f%%", help=CONCEPT_TOOLTIPS["outperform_proba"]
        )

    showing_txt = f"Showing {len(disp)} of {total_matched} matched" + (
        f" ({len(ranked)} total)" if total_matched != len(ranked) else ""
    )
    st.caption(
        showing_txt
        + " · Score = quality − red-flag penalty · Rev gr % is percent points."
    )
    display_cols = [c for c in disp.columns if c != "_full_ticker"]
    st.dataframe(
        disp[display_cols],
        use_container_width=True,
        hide_index=True,
        height=min(620, 48 + 35 * max(len(disp), 1)),
        column_config=colcfg,
    )

    # Click-through: open report for a ticker in the current view
    if len(disp):
        options = list(disp["ticker"])
        full_map = dict(zip(disp["ticker"], disp["_full_ticker"]))
        pick = st.selectbox(
            "Open in Single Stock Report",
            ["—"] + options,
            help="Jump to the full report for a name in the table above.",
            key="rank_open_ticker",
        )
        b1, b2 = st.columns(2)
        with b1:
            if pick != "—" and st.button("➡️ Open report", key="rank_open_btn"):
                st.session_state["report_ticker"] = full_map.get(pick, pick)
                st.session_state["_report_jump"] = True
                # Do not set nav_page here — radio widget already owns that key.
                # Apply on next run before the radio is created.
                st.session_state["_pending_nav"] = "Single Stock Report"
                st.rerun()
        with b2:
            if pick != "—" and st.button("⭐ Add to watchlist", key="rank_wl_btn"):
                full = full_map.get(pick, pick)
                if add_ticker(st.session_state, full):
                    st.success(f"Added {pick} to watchlist.")
                else:
                    st.info("Already on watchlist.")

    st.download_button(
        "⬇️ Download this ranking (CSV)",
        data=disp[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="stock_rankings.csv",
        mime="text/csv",
    )
