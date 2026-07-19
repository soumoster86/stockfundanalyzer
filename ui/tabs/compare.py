"""Compare Stocks tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.quality_score import CONCEPT_TOOLTIPS, METRIC_CONFIG, score_label
from src.red_flags import REASON_TEXT
from src.institutional_scores import z_band, m_band
from ui.gauges import category_radar_svg
from ui.theme import badge_row, section, quality_color


def render_compare(data: pd.DataFrame) -> None:
    section("Compare stocks")
    st.caption(
        "Pick 2–5 stocks for side-by-side radars, quality badges, and key metrics."
    )
    all_tickers = sorted(data["ticker"].str.replace(".NS", "", regex=False).unique())
    picks = st.multiselect(
        "Stocks to compare",
        all_tickers,
        max_selections=5,
        help="Pick up to 5 stocks to see their category radars and key metrics side by side.",
        default=all_tickers[:3] if len(all_tickers) >= 3 else all_tickers,
    )
    if len(picks) < 2:
        st.info("Select at least two stocks to compare.")
        return

    sel = data[data["ticker"].str.replace(".NS", "", regex=False).isin(picks)].copy()
    sel = sel.sort_values("date").groupby("ticker", as_index=False).tail(1)
    sel["disp_ticker"] = sel["ticker"].str.replace(".NS", "", regex=False)

    cols = st.columns(len(sel))
    palette = ["#1D9E75", "#378ADD", "#E0A82E", "#C175E0", "#E24B4A"]
    for i, (_, row) in enumerate(sel.iterrows()):
        with cols[i]:
            st.markdown(f"**{row['disp_ticker']}**")
            q = row.get("quality_score")
            q_lab = score_label(q) if pd.notna(q) else "N/A"
            zb = row.get("z_band") if "z_band" in row.index else z_band(row.get("z_score"))
            mb = row.get("m_band") if "m_band" in row.index else m_band(row.get("m_score"))
            n_flags = int(row["red_flag_count"]) if pd.notna(row.get("red_flag_count")) else 0
            st.markdown(
                badge_row(
                    [
                        (q_lab, q_lab if q_lab != "N/A" else "N/A"),
                        (f"Z {zb}", zb or "N/A"),
                        (f"M {mb}", mb or "N/A"),
                        (f"{n_flags}🚩", "Red" if n_flags else "Green"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            cmap = {}
            for c in METRIC_CONFIG:
                col = f"{c.replace(' ', '_').lower()}_score"
                if col in row.index:
                    cmap[c] = row[col]
            st.markdown(
                category_radar_svg(cmap, color=palette[i % len(palette)], size=210),
                unsafe_allow_html=True,
            )
            st.metric(
                "Quality",
                f"{q:.1f}" if pd.notna(q) else "—",
                q_lab if pd.notna(q) else "",
                help=CONCEPT_TOOLTIPS["quality_score"],
            )
            if pd.notna(q):
                st.markdown(
                    f'<div style="height:8px;background:#30363D;border-radius:4px;'
                    f'overflow:hidden;margin-top:-0.5rem;">'
                    f'<div style="width:{min(100, float(q)):.0f}%;height:100%;'
                    f'background:{quality_color(float(q))};"></div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("**Metrics side by side**")
    metric_rows = {
        "Quality score": "quality_score",
        "Red flags": "red_flag_count",
        "ROE %": "roe",
        "ROCE %": "roce",
        "Net margin %": "net_margin",
        "Oper. margin %": "operating_margin",
        "D/E": "debt_to_equity",
        "Interest cover": "interest_coverage",
        "Current ratio": "current_ratio",
        "P/E": "pe",
        "P/B": "pb",
        "EV/EBITDA": "ev_ebitda",
        "Rev growth %": "revenue_growth",
        "Dividend yield %": "dividend_yield",
        "Sector": "sector",
    }
    table = {}
    for _, row in sel.iterrows():
        colvals = {}
        for label, key in metric_rows.items():
            if key not in row.index:
                colvals[label] = "—"
                continue
            v = row[key]
            if pd.isna(v):
                colvals[label] = "—"
            elif key == "sector":
                colvals[label] = str(v)
            elif key == "revenue_growth":
                colvals[label] = f"{float(v) * 100:.1f}"
            elif key == "red_flag_count":
                colvals[label] = str(int(v))
            else:
                colvals[label] = f"{float(v):.2f}"
        table[row["disp_ticker"]] = colvals
    st.dataframe(pd.DataFrame(table), use_container_width=True)

    st.markdown("**Red flags**")
    for _, row in sel.iterrows():
        rf = row.get("red_flags", [])
        if isinstance(rf, list) and rf:
            st.write(
                f"**{row['disp_ticker']}**: "
                + "; ".join(REASON_TEXT.get(f, f) for f in rf)
            )
        else:
            st.write(f"**{row['disp_ticker']}**: none")
