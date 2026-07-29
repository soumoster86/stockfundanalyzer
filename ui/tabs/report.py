"""Single Stock Report tab — progressive sub-tabs to reduce scroll overload."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_quality import WARNING_TEXT
from src.institutional_scores import (
    M_BAND_TEXT,
    PIOTROSKI_LABELS,
    PIOTROSKI_TESTS,
    Z_BAND_TEXT,
    f_score_band,
    m_band,
    z_band,
)
from src.peers import peer_context_line, sector_peers
from src.quality_score import (
    CATEGORY_TOOLTIPS,
    CONCEPT_TOOLTIPS,
    METRIC_CONFIG,
    METRIC_LABELS,
    METRIC_TOOLTIPS,
    explain_score,
    explanation_sentence,
    score_label,
    ticker_trend,
)
from src.red_flags import REASON_TEXT
from src.report_export import build_research_note
from src.watchlist import add_ticker, is_watched, remove_ticker
from ui.gauges import CATEGORY_SHORT, category_radar_svg, quality_gauge_svg
from ui.theme import (
    badge,
    badge_row,
    ok_banner,
    peers_html_table,
    quality_color,
    red_flag_block,
    render_f_score_breakdown,
    score_card,
    section,
)


def render_report(data: pd.DataFrame, history_panel: pd.DataFrame, custom_config) -> None:
    section("Single stock report")
    tickers = sorted(data["ticker"].unique())
    # Only force the selectbox when another page requests a jump (ranking /
    # watchlist / peers). Overwriting report_ticker_select every run from
    # report_ticker was locking the dropdown on the previous value.
    if st.session_state.pop("_report_jump", None):
        pref = st.session_state.get("report_ticker")
        if pref in tickers:
            st.session_state["report_ticker_select"] = pref
    if (
        "report_ticker_select" not in st.session_state
        or st.session_state["report_ticker_select"] not in tickers
    ):
        st.session_state["report_ticker_select"] = tickers[0]

    tk = st.selectbox(
        "Select stock",
        tickers,
        key="report_ticker_select",
        help="Choose a stock. Tip: open a name from Ranking or Watchlist.",
    )
    st.session_state["report_ticker"] = tk
    latest = data[data["ticker"] == tk].sort_values("date").iloc[-1]

    qscore = float(latest["quality_score"]) if pd.notna(latest["quality_score"]) else 0.0
    q_label = score_label(qscore)
    n_flags = int(latest["red_flag_count"]) if pd.notna(latest.get("red_flag_count")) else 0
    fs = latest.get("f_score")
    ftu = (
        int(latest.get("f_tests_used", 0))
        if pd.notna(latest.get("f_tests_used", 0))
        else 0
    )
    f_band_lbl = f_score_band(fs, ftu) if pd.notna(fs) and ftu > 0 else "N/A"
    zs = latest.get("z_score")
    zb = z_band(zs)
    ms = latest.get("m_score")
    mb = latest.get("m_band") if "m_band" in latest.index else m_band(ms)
    miu = (
        int(latest.get("m_indices_used", 0))
        if pd.notna(latest.get("m_indices_used", 0))
        else 0
    )

    # Always-visible chrome: actions + status
    wl_col, dl_col, _ = st.columns([1.2, 1.2, 1.6])
    with wl_col:
        if is_watched(st.session_state, tk):
            if st.button("⭐ On watchlist — remove", key="rpt_wl_rm"):
                remove_ticker(st.session_state, tk)
                st.rerun()
        else:
            if st.button("☆ Add to watchlist", key="rpt_wl_add"):
                add_ticker(st.session_state, tk)
                st.rerun()
    with dl_col:
        peers_for_note, peer_group = sector_peers(data, tk, n=5)
        trend_for_note = ticker_trend(history_panel, tk)
        note_md = build_research_note(
            latest,
            history_points=trend_for_note.get("points") or None,
            peers=peers_for_note,
            sector=peer_group
            or (latest.get("sector") if "sector" in latest.index else None),
            config=custom_config,
        )
        safe_name = str(tk).replace("/", "_").replace("\\", "_")
        st.download_button(
            "⬇️ Research note",
            data=note_md.encode("utf-8"),
            file_name=f"{safe_name}_research_note.md",
            mime="text/markdown",
            key="rpt_note_dl",
            help="Markdown one-pager you can share or convert to PDF.",
        )

    badge_items = [
        (f"Quality {q_label}", q_label if q_label != "N/A" else "N/A"),
        (
            f"{n_flags} red flag{'s' if n_flags != 1 else ''}",
            "Red" if n_flags else "Green",
        ),
        (f"F: {f_band_lbl}", f_band_lbl if f_band_lbl != "N/A" else "N/A"),
        (f"Z: {zb}", zb if zb else "N/A"),
        (f"M: {mb if mb else 'N/A'}", mb if mb else "N/A"),
    ]
    st.markdown(badge_row(badge_items), unsafe_allow_html=True)

    tab_overview, tab_forensics, tab_drivers, tab_context = st.tabs(
        ["Overview", "Forensics", "Drivers", "Context"]
    )

    # ---------- Overview ----------
    with tab_overview:
        gcol, mcol = st.columns([1, 2])
        with gcol:
            st.markdown(quality_gauge_svg(qscore), unsafe_allow_html=True)
        with mcol:
            c2, c3 = st.columns(2)
            c2.metric(
                "Red Flags",
                n_flags,
                help=CONCEPT_TOOLTIPS["red_flags"],
            )
            if "outperform_proba" in data.columns and pd.notna(
                latest.get("outperform_proba")
            ):
                c3.metric(
                    "Outperform Prob.",
                    f"{latest['outperform_proba'] * 100:.0f}%",
                    help=CONCEPT_TOOLTIPS["outperform_proba"],
                )
            else:
                c3.metric(
                    "Quality Score",
                    f"{qscore:.1f}",
                    q_label,
                    help=CONCEPT_TOOLTIPS["quality_score"],
                )
            sector_txt = (
                latest.get("sector", "Unknown") if "sector" in latest.index else "Unknown"
            )
            scored_vs = latest.get("scored_vs", "all stocks (by date)")
            st.caption(
                f"Sector: **{sector_txt}** · Scored vs **{scored_vs}** · "
                f"{peer_context_line(data, tk)}"
            )

        # Compact key metrics
        key_metrics = [
            ("ROE", "roe", "%.1f"),
            ("P/E", "pe", "%.1f"),
            ("D/E", "debt_to_equity", "%.2f"),
            ("Net %", "net_margin", "%.1f"),
        ]
        present_keys = [(lab, col, fmt) for lab, col, fmt in key_metrics if col in latest.index]
        if present_keys:
            cols = st.columns(len(present_keys))
            for col_ui, (lab, col, fmt) in zip(cols, present_keys):
                val = latest.get(col)
                if pd.notna(val):
                    col_ui.metric(lab, fmt % float(val))
                else:
                    col_ui.metric(lab, "—")

        present = int(latest.get("data_fields_present", 0) or 0)
        total = int(latest.get("data_fields_total", 0) or 0)
        completeness = float(latest.get("data_completeness", 0.0) or 0.0)
        if total:
            if completeness < 0.5:
                st.warning(
                    f"Sparse data: only {present}/{total} core metrics — "
                    "open **Forensics** for details."
                )
            else:
                st.caption(
                    f"Data completeness: {present}/{total} ({completeness * 100:.0f}%). "
                    "Deep dive in **Forensics** · drivers in **Drivers** · peers in **Context**."
                )
        else:
            st.caption("Use the tabs above for forensics, score drivers, trend, and peers.")

    # ---------- Forensics ----------
    with tab_forensics:
        st.caption(
            "Institutional scores, data reliability, and rule-based red flags. "
            "Z = bankruptcy risk · M = earnings-manipulation risk."
        )
        fcol, zcol, mcol_inst = st.columns(3)
        with fcol:
            if pd.notna(fs) and ftu > 0:
                accent = {
                    "Strong": "#1D9E75",
                    "Moderate": "#378ADD",
                    "Weak": "#E24B4A",
                }.get(f_band_lbl, "#30363D")
                st.markdown(
                    score_card(
                        "Piotroski F-Score",
                        f"{int(fs)} / 9",
                        f"{badge(f_band_lbl, f_band_lbl)}"
                        + (f" · {ftu}/9 tests" if ftu < 9 else ""),
                        accent=accent,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    score_card(
                        "Piotroski F-Score", "N/A", "Need multi-year fundamentals"
                    ),
                    unsafe_allow_html=True,
                )
        with zcol:
            if pd.notna(zs):
                accent = {"Green": "#1D9E75", "Yellow": "#E0A82E", "Red": "#E24B4A"}.get(
                    zb, "#30363D"
                )
                st.markdown(
                    score_card(
                        "Altman Z-Score",
                        f"{zs:.2f}",
                        f"{badge(zb, zb)} · {Z_BAND_TEXT.get(zb, '')}",
                        accent=accent,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    score_card(
                        "Altman Z-Score",
                        "N/A",
                        "Need balance-sheet fields (assets, EBIT, mkt cap)",
                    ),
                    unsafe_allow_html=True,
                )
        with mcol_inst:
            if pd.notna(ms):
                accent = {"Green": "#1D9E75", "Yellow": "#E0A82E", "Red": "#E24B4A"}.get(
                    mb, "#30363D"
                )
                sub = f"{badge(str(mb), mb)} · {M_BAND_TEXT.get(mb, '')}"
                if miu < 8:
                    sub += f" · {miu}/8 indices"
                st.markdown(
                    score_card("Beneish M-Score", f"{ms:.2f}", sub, accent=accent),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    score_card(
                        "Beneish M-Score",
                        "N/A",
                        M_BAND_TEXT.get("N/A", ""),
                    ),
                    unsafe_allow_html=True,
                )

        if pd.notna(fs) and ftu > 0:
            with st.expander("F-Score test breakdown", expanded=False):
                st.caption(
                    "Nine binary health checks. Green = pass · Red = fail · Grey = n/a."
                )
                render_f_score_breakdown(
                    latest,
                    PIOTROSKI_TESTS,
                    PIOTROSKI_LABELS,
                    score=float(fs),
                    tests_used=ftu,
                )

        present = int(latest.get("data_fields_present", 0) or 0)
        total = int(latest.get("data_fields_total", 0) or 0)
        completeness = float(latest.get("data_completeness", 0.0) or 0.0)
        warnings = (
            latest.get("data_warnings", []) if "data_warnings" in latest.index else []
        )
        st.markdown("**Data quality**")
        if total:
            if completeness < 0.5:
                st.warning(
                    f"Sparse data: only {present}/{total} core metrics available — "
                    "treat this score with low confidence."
                )
            else:
                st.markdown(
                    f'<p class="sfa-muted">Completeness: {present}/{total} core metrics '
                    f"({completeness * 100:.0f}%). "
                    f'{badge(f"{completeness * 100:.0f}% complete", "Green" if completeness >= 0.7 else "Yellow")}'
                    f"</p>",
                    unsafe_allow_html=True,
                )
        if isinstance(warnings, list) and warnings:
            st.warning(
                "Data reliability warnings — verify against filings before trusting this score:"
            )
            for wkey in warnings:
                st.write("• " + WARNING_TEXT.get(wkey, wkey))

        st.markdown("**Red flags**")
        flags = latest.get("red_flags", [])
        if isinstance(flags, list) and flags:
            for f in flags:
                st.markdown(
                    red_flag_block(REASON_TEXT.get(f, f)), unsafe_allow_html=True
                )
        else:
            st.markdown(ok_banner("No red flags detected."), unsafe_allow_html=True)

    # ---------- Drivers ----------
    with tab_drivers:
        st.markdown("**Why this score?**")
        ex = explain_score(latest, config=custom_config)
        st.write(explanation_sentence(ex, ticker=tk))

        if ex["drivers"]:
            col_up, col_down = st.columns(2)
            expl_cfg = {
                "Metric": st.column_config.TextColumn(
                    "Metric",
                    help="Fundamental metric (see glossary expander).",
                ),
                "Percentile": st.column_config.NumberColumn(
                    "Percentile",
                    format="%.0f",
                    help="Rank vs comparison group, 0–100.",
                ),
                "Value": st.column_config.NumberColumn(
                    "Value", help="Raw metric value."
                ),
            }
            with col_up:
                st.markdown("**Top strengths** (vs peers)")
                up_df = pd.DataFrame(ex["drivers"])[
                    ["metric", "percentile", "raw_value"]
                ]
                up_df.columns = ["Metric", "Percentile", "Value"]
                st.dataframe(
                    up_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config=expl_cfg,
                )
            with col_down:
                st.markdown("**Weakest areas** (vs peers)")
                dn_df = pd.DataFrame(ex["drags"])[["metric", "percentile", "raw_value"]]
                dn_df.columns = ["Metric", "Percentile", "Value"]
                st.dataframe(
                    dn_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config=expl_cfg,
                )
            st.caption(
                "Percentile = rank vs comparison group (100 = best). "
                "Valuation metrics are inverted so cheaper = higher percentile."
            )
            with st.expander("Metric glossary"):
                gloss = pd.DataFrame(
                    [
                        {"Metric": METRIC_LABELS.get(k, k), "Meaning": v}
                        for k, v in METRIC_TOOLTIPS.items()
                    ]
                )
                st.dataframe(gloss, hide_index=True, use_container_width=True)

        st.markdown("**Category scores**")
        cat_score_map = {}
        for c in METRIC_CONFIG:
            col = f"{c.replace(' ', '_').lower()}_score"
            if col in latest.index:
                cat_score_map[c] = latest[col]
        if cat_score_map:
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.markdown(category_radar_svg(cat_score_map), unsafe_allow_html=True)
            with rc2:
                for c, v in cat_score_map.items():
                    vv = 0 if pd.isna(v) else float(v)
                    st.metric(
                        CATEGORY_SHORT[c], f"{vv:.0f}", help=CATEGORY_TOOLTIPS.get(c)
                    )
                    st.markdown(
                        f'<div style="height:8px;background:#30363D;border-radius:4px;'
                        f'overflow:hidden;margin:-0.4rem 0 0.6rem 0;">'
                        f'<div style="width:{min(100, vv):.0f}%;height:100%;'
                        f'background:{quality_color(vv)};"></div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No category scores available.")

    # ---------- Context ----------
    with tab_context:
        st.markdown("**Quality trend**")
        trend = ticker_trend(history_panel, tk)
        if trend["direction"] == "insufficient data":
            npts = len(trend["points"])
            if npts == 0:
                st.caption(
                    "Not enough complete fiscal years to chart a trend "
                    "(common for banks/financials with sparse Yahoo data)."
                )
            else:
                st.caption(
                    "Only one scored fiscal year available — need at least two "
                    "to show a trend."
                )
        else:
            arrow = {
                "improving": "📈",
                "declining": "📉",
                "stable": "➡️",
            }[trend["direction"]]
            st.markdown(
                f"{arrow} Quality is **{trend['direction']}** — "
                f"{trend['delta']:+.1f} points from "
                f"{trend['points'][0]['date'].year} to "
                f"{trend['points'][-1]['date'].year}."
            )
            tdf = pd.DataFrame(
                [
                    {
                        "Year": str(p["date"].year),
                        "Quality": round(p["quality_score"], 1),
                    }
                    for p in trend["points"]
                ]
            ).set_index("Year")
            st.line_chart(tdf, height=220)
            if trend["category_deltas"]:
                moves = sorted(
                    trend["category_deltas"].items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
                bits = ", ".join(
                    f"{cat.split()[0]} {d:+.0f}" for cat, d in moves if abs(d) >= 1
                )
                if bits:
                    st.caption(
                        "Biggest category moves: "
                        + bits
                        + " (points, latest vs earliest)."
                    )
            st.caption(
                "Each year is scored against that year's peers — changing "
                "fundamentals, not market timing."
            )
            if trend.get("caveat"):
                st.caption("⚠️ " + trend["caveat"])

        st.markdown("**Peers**")
        peers, sector = sector_peers(data, tk, n=5)
        if peers is None or peers.empty or not sector:
            st.caption("No sector/industry peer group available for this stock.")
        else:
            st.caption(
                f"Top quality names in **{sector}** (excluding this stock). "
                "Industry used when available; otherwise sector. "
                "Z = bankruptcy risk · M = earnings-manipulation risk · "
                "🟢 ok · 🟡 caution · 🔴 elevated"
            )
            st.markdown(peers_html_table(peers), unsafe_allow_html=True)
            peer_pick = st.selectbox(
                "Open a peer report",
                ["—"] + peers["ticker"].tolist()
                if "ticker" in peers.columns
                else ["—"],
                key="peer_open",
            )
            if peer_pick != "—" and st.button("➡️ Open peer", key="peer_open_btn"):
                st.session_state["report_ticker"] = peer_pick
                st.session_state["_report_jump"] = True
                st.rerun()
