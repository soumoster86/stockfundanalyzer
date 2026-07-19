"""Compare Stocks tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.quality_score import METRIC_CONFIG, score_label, CONCEPT_TOOLTIPS
from src.red_flags import REASON_TEXT
from ui.gauges import category_radar_svg


def render_compare(data: pd.DataFrame) -> None:
    st.subheader("Compare Stocks")
    st.caption(
        "Pick 2–5 stocks to see their scores, categories, and key metrics side by side."
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
            cmap = {}
            for c in METRIC_CONFIG:
                col = f"{c.replace(' ', '_').lower()}_score"
                if col in row.index:
                    cmap[c] = row[col]
            st.markdown(
                category_radar_svg(cmap, color=palette[i % len(palette)], size=210),
                unsafe_allow_html=True,
            )
            q = row.get("quality_score")
            st.metric(
                "Quality",
                f"{q:.1f}" if pd.notna(q) else "—",
                score_label(q) if pd.notna(q) else "",
                help=CONCEPT_TOOLTIPS["quality_score"],
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
