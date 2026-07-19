"""Universe Ranking tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ranking import rank_universe
from src.quality_score import METRIC_TOOLTIPS, CONCEPT_TOOLTIPS


def render_ranking(data: pd.DataFrame) -> None:
    st.subheader("Multi-Factor Ranking")
    st.caption("Each stock's most recent fiscal-year figures, ranked together.")
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
            help="How much the composite ranking leans on the fundamental quality "
                 "score vs the ML outperformance probability. At 1.0 the ranking is "
                 "pure quality; lower values blend in the model.",
        )
        w_ml = 1.0 - wq
    else:
        wq, w_ml = 1.0, 0.0
        st.caption(
            "Ranking is pure quality (no ML scores yet). Train and score the universe "
            "in the **Train Model** tab to enable the quality/ML blend slider."
        )
    ranked = rank_universe(data, w_quality=wq, w_ml=w_ml, as_of_date=None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks ranked", len(ranked), help="Total stocks in the current ranking.")
    m2.metric(
        "Top score",
        f"{ranked['composite_score'].max():.1f}",
        help=CONCEPT_TOOLTIPS["composite_score"],
    )
    m3.metric(
        "Median",
        f"{ranked['composite_score'].median():.1f}",
        help="Median composite score — the middle of the pack.",
    )
    if "data_warning" in ranked.columns:
        m4.metric(
            "Data warnings",
            int(ranked["data_warning"].sum()),
            help="Stocks with distorted/implausible figures — verify manually.",
        )
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
            st.caption(
                f"Distribution of composite scores across {len(vals)} stocks "
                f"(median {vals.median():.1f}). Most stocks cluster mid-range "
                "because scores are relative percentiles."
            )

    has_sector_col = "sector" in ranked.columns
    r1c1, r1c2 = st.columns([2, 2])
    search = r1c1.text_input("Search ticker", "", placeholder="e.g. TCS").strip().upper()
    if has_sector_col:
        sectors = ["All sectors"] + sorted(ranked["sector"].dropna().unique().tolist())
        sector_pick = r1c2.selectbox(
            "Sector",
            sectors,
            help="Filter the ranking to one sector.",
        )
    else:
        sector_pick = "All sectors"

    r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1.3, 1])
    flag_filter = r2c1.selectbox(
        "Flags",
        ["All", "No red flags", "Has red flags"],
        help="Filter by forensic red flags.",
    )
    rel_filter = r2c2.selectbox(
        "Data",
        ["All", "Reliable only", "Warnings only"],
        help="'Reliable only' hides stocks with data-quality warnings.",
    )
    sort_opts = {
        "Rank": ("rank", True),
        "Quality (raw)": ("quality_score", False),
        "ROE": ("roe", False),
        "P/E (low→high)": ("pe", True),
        "Debt/Equity (low→high)": ("debt_to_equity", True),
        "Revenue growth": ("revenue_growth", False),
    }
    sort_by = r2c3.selectbox(
        "Sort by", list(sort_opts.keys()), help="Reorder the table by any metric."
    )
    top_n = r2c4.selectbox(
        "Show",
        ["Top 25", "Top 50", "Top 100", "All"],
        index=1,
        help="Limit how many rows render.",
    )

    view = ranked.copy()
    if search:
        view = view[view["ticker"].str.upper().str.contains(search)]
    if sector_pick != "All sectors" and has_sector_col:
        view = view[view["sector"] == sector_pick]
    if flag_filter == "No red flags":
        view = view[view["red_flag_count"] == 0]
    elif flag_filter == "Has red flags":
        view = view[view["red_flag_count"] > 0]
    if "data_warning" in view.columns:
        if rel_filter == "Reliable only":
            view = view[~view["data_warning"]]
        elif rel_filter == "Warnings only":
            view = view[view["data_warning"]]
    sort_col, asc = sort_opts[sort_by]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=asc, na_position="last")

    total_matched = len(view)
    limit_map = {"Top 25": 25, "Top 50": 50, "Top 100": 100, "All": len(view)}
    view = view.head(limit_map[top_n])

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
        "red_flag_count",
        "data_fields_present",
        "data_warning_count",
    ]
    show = [c for c in candidate_cols if c in view.columns]
    disp = view[show].reset_index(drop=True).copy()
    disp["ticker"] = disp["ticker"].str.replace(".NS", "", regex=False)
    if "revenue_growth" in disp.columns:
        disp["revenue_growth"] = disp["revenue_growth"] * 100.0
    if "f_score" in disp.columns:
        disp["f_score"] = disp["f_score"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "–"
        )
    if "z_band" in disp.columns:
        disp["z_band"] = (
            disp["z_band"]
            .map({"Green": "🟢", "Yellow": "🟡", "Red": "🔴", "N/A": "–"})
            .fillna("–")
        )
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
            "Rev gr%", format="%.1f", help=METRIC_TOOLTIPS["revenue_growth"]
        ),
        "f_score": st.column_config.TextColumn(
            "F",
            width="small",
            help="Piotroski F-Score (0–9).",
        ),
        "z_band": st.column_config.TextColumn(
            "Z",
            width="small",
            help="Altman Z-Score bankruptcy risk: 🟢 safe · 🟡 grey zone · 🔴 distress.",
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
    st.caption(showing_txt + " · Score = quality minus red-flag penalty.")
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=min(620, 48 + 35 * max(len(disp), 1)),
        column_config=colcfg,
    )

    st.download_button(
        "⬇️ Download this ranking (CSV)",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name="stock_rankings.csv",
        mime="text/csv",
    )
