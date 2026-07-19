"""Single Stock Report tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.quality_score import (
    METRIC_CONFIG,
    explain_score,
    explanation_sentence,
    score_label,
    ticker_trend,
    METRIC_TOOLTIPS,
    CATEGORY_TOOLTIPS,
    CONCEPT_TOOLTIPS,
    METRIC_LABELS,
)
from src.red_flags import REASON_TEXT
from src.data_quality import WARNING_TEXT
from src.institutional_scores import (
    f_score_band,
    z_band,
    Z_BAND_TEXT,
    PIOTROSKI_LABELS,
    PIOTROSKI_TESTS,
)
from ui.gauges import category_radar_svg, quality_gauge_svg, CATEGORY_SHORT


def render_report(data: pd.DataFrame, history_panel: pd.DataFrame, custom_config) -> None:
    tickers = sorted(data["ticker"].unique())
    tk = st.selectbox(
        "Select stock",
        tickers,
        help="Choose a stock to see its quality score, category breakdown, "
             "trend, and red flags.",
    )
    latest = data[data["ticker"] == tk].sort_values("date").iloc[-1]

    qscore = float(latest["quality_score"]) if pd.notna(latest["quality_score"]) else 0.0

    gcol, mcol = st.columns([1, 2])
    with gcol:
        st.markdown(quality_gauge_svg(qscore), unsafe_allow_html=True)
    with mcol:
        c2, c3 = st.columns(2)
        c2.metric(
            "Red Flags",
            int(latest["red_flag_count"]),
            help=CONCEPT_TOOLTIPS["red_flags"],
        )
        if "outperform_proba" in data.columns and pd.notna(latest.get("outperform_proba")):
            c3.metric(
                "Outperform Prob.",
                f"{latest['outperform_proba'] * 100:.0f}%",
                help=CONCEPT_TOOLTIPS["outperform_proba"],
            )
        else:
            c3.metric(
                "Quality Score",
                f"{qscore:.1f}",
                score_label(qscore),
                help=CONCEPT_TOOLTIPS["quality_score"],
            )
        sector_txt = latest.get("sector", "Unknown") if "sector" in latest.index else "Unknown"
        scored_vs = latest.get("scored_vs", "all stocks (by date)")
        st.caption(f"Sector: **{sector_txt}** · Scored vs **{scored_vs}**")

    # ---- Institutional scores ----
    fcol, zcol = st.columns(2)
    with fcol:
        fs = latest.get("f_score")
        ftu = (
            int(latest.get("f_tests_used", 0))
            if pd.notna(latest.get("f_tests_used", 0))
            else 0
        )
        if pd.notna(fs) and ftu > 0:
            st.metric(
                "Piotroski F-Score",
                f"{int(fs)} / 9",
                f_score_band(fs, ftu),
                help="Nine binary financial-health tests (profitability, leverage/"
                     "liquidity, efficiency). 7–9 strong, 4–6 moderate, 0–3 weak.",
            )
            if ftu < 9:
                st.caption(
                    f"⚠️ Computed from {ftu}/9 tests "
                    "(missing inputs — re-fetch for total-assets-based tests)."
                )
            with st.expander("F-Score test breakdown"):
                for t in PIOTROSKI_TESTS:
                    v = latest.get(t)
                    if pd.isna(v):
                        st.write(f"➖ {PIOTROSKI_LABELS[t]} — *not evaluable*")
                    elif v >= 1:
                        st.write(f"✅ {PIOTROSKI_LABELS[t]}")
                    else:
                        st.write(f"❌ {PIOTROSKI_LABELS[t]}")
        else:
            st.metric(
                "Piotroski F-Score",
                "N/A",
                help="Needs at least two years of data with net income, cash flow, "
                     "margins, etc.",
            )
    with zcol:
        zs = latest.get("z_score")
        zb = z_band(zs)
        if pd.notna(zs):
            emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(zb, "")
            st.metric(
                "Altman Z-Score",
                f"{zs:.2f}",
                f"{emoji} {zb}",
                help="Bankruptcy-risk score. 🟢 > 3 safe · 🟡 1.8–3 grey zone · "
                     "🔴 < 1.8 distress.",
            )
            st.caption(Z_BAND_TEXT.get(zb, ""))
        else:
            st.metric(
                "Altman Z-Score",
                "N/A",
                help="Needs balance-sheet inputs (total assets/liabilities, retained "
                     "earnings, EBIT, market cap).",
            )
            st.caption("Re-run the fetcher to capture the balance-sheet fields this needs.")

    # ---- Data quality ----
    present = int(latest.get("data_fields_present", 0) or 0)
    total = int(latest.get("data_fields_total", 0) or 0)
    completeness = float(latest.get("data_completeness", 0.0) or 0.0)
    warnings = latest.get("data_warnings", []) if "data_warnings" in latest.index else []
    if total:
        if completeness < 0.5:
            st.warning(
                f"⚠️ Sparse data: only {present}/{total} core metrics available — "
                "treat this score with low confidence."
            )
        else:
            st.caption(
                f"Data completeness: {present}/{total} core metrics populated "
                f"({completeness * 100:.0f}%)."
            )
    if isinstance(warnings, list) and warnings:
        st.warning(
            "⚠️ **Data reliability warnings** — verify against filings before trusting this score:"
        )
        for wkey in warnings:
            st.write("• " + WARNING_TEXT.get(wkey, wkey))

    # ---- Explainability ----
    st.subheader("Why this score?")
    ex = explain_score(latest, config=custom_config)
    st.write(explanation_sentence(ex, ticker=tk))

    if ex["drivers"]:
        col_up, col_down = st.columns(2)
        expl_cfg = {
            "Metric": st.column_config.TextColumn(
                "Metric",
                help="The fundamental metric (hover the glossary below for definitions).",
            ),
            "Percentile": st.column_config.NumberColumn(
                "Percentile",
                format="%.0f",
                help="How this stock ranks vs its comparison group, 0–100.",
            ),
            "Value": st.column_config.NumberColumn(
                "Value", help="The raw underlying value of the metric."
            ),
        }
        with col_up:
            st.markdown("**🟢 Top strengths** (vs peers)")
            up_df = pd.DataFrame(ex["drivers"])[["metric", "percentile", "raw_value"]]
            up_df.columns = ["Metric", "Percentile", "Value"]
            st.dataframe(
                up_df, hide_index=True, use_container_width=True, column_config=expl_cfg
            )
        with col_down:
            st.markdown("**🔴 Weakest areas** (vs peers)")
            dn_df = pd.DataFrame(ex["drags"])[["metric", "percentile", "raw_value"]]
            dn_df.columns = ["Metric", "Percentile", "Value"]
            st.dataframe(
                dn_df, hide_index=True, use_container_width=True, column_config=expl_cfg
            )
        st.caption(
            "Percentile = how this stock ranks against its comparison group "
            "(100 = best). Valuation metrics are inverted so cheaper = higher percentile."
        )
        with st.expander("📖 Metric glossary — what each metric means"):
            gloss = pd.DataFrame(
                [
                    {"Metric": METRIC_LABELS.get(k, k), "Meaning": v}
                    for k, v in METRIC_TOOLTIPS.items()
                ]
            )
            st.dataframe(gloss, hide_index=True, use_container_width=True)

    st.subheader("Category Scores")
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
                st.metric(CATEGORY_SHORT[c], f"{vv:.0f}", help=CATEGORY_TOOLTIPS.get(c))
                st.progress(min(1.0, vv / 100.0))
    else:
        st.caption("No category scores available.")

    # ---- Quality trend ----
    st.subheader("Quality Trend")
    trend = ticker_trend(history_panel, tk)
    if trend["direction"] == "insufficient data":
        npts = len(trend["points"])
        if npts == 0:
            st.caption(
                "Not enough complete fiscal years to chart a trend "
                "(this stock's available years fall below the data-completeness "
                "bar — common for banks/financials with sparse Yahoo data)."
            )
        else:
            st.caption(
                "Only one scored fiscal year available — need at least two "
                "to show a trend."
            )
    else:
        arrow = {"improving": "📈", "declining": "📉", "stable": "➡️"}[trend["direction"]]
        st.markdown(
            f"{arrow} Quality is **{trend['direction']}** — "
            f"{trend['delta']:+.1f} points from "
            f"{trend['points'][0]['date'].year} to {trend['points'][-1]['date'].year}."
        )
        tdf = pd.DataFrame(
            [
                {"Year": str(p["date"].year), "Quality": round(p["quality_score"], 1)}
                for p in trend["points"]
            ]
        ).set_index("Year")
        st.line_chart(tdf, height=220)
        if trend["category_deltas"]:
            moves = sorted(
                trend["category_deltas"].items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            bits = ", ".join(
                f"{cat.split()[0]} {d:+.0f}" for cat, d in moves if abs(d) >= 1
            )
            if bits:
                st.caption(
                    "Biggest category moves: " + bits + " (points, latest vs earliest)."
                )
        st.caption(
            "Each year is scored against that year's peers, so the trend "
            "reflects changing fundamentals, not market timing."
        )
        if trend.get("caveat"):
            st.caption("⚠️ " + trend["caveat"])

    st.subheader("Red Flags")
    flags = latest.get("red_flags", [])
    if isinstance(flags, list) and flags:
        for f in flags:
            st.error(REASON_TEXT.get(f, f))
    else:
        st.success("No red flags detected.")
