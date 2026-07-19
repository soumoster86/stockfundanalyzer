"""Sector Overview tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ranking import rank_universe


def render_sector(data: pd.DataFrame) -> None:
    st.subheader("Sector Overview")
    if "sector" not in data.columns or data["sector"].dropna().nunique() < 2:
        st.info(
            "No sector data available. Re-run the fetcher to populate the "
            "`sector` column, then sector analytics will appear here."
        )
        return

    sec_ranked = rank_universe(data, w_quality=1.0, w_ml=0.0, as_of_date=None)
    grp = sec_ranked.groupby("sector")
    summary = pd.DataFrame(
        {
            "Stocks": grp.size(),
            "Avg quality": grp["quality_score"].mean(),
            "Median quality": grp["quality_score"].median(),
            "Best score": grp["quality_score"].max(),
            "Avg red flags": grp["red_flag_count"].mean(),
        }
    ).reset_index()
    idx = grp["quality_score"].idxmax()
    best = sec_ranked.loc[idx, ["sector", "ticker"]].copy()
    best["ticker"] = best["ticker"].str.replace(".NS", "", regex=False)
    summary = summary.merge(best, on="sector", how="left").rename(
        columns={"sector": "Sector", "ticker": "Top stock"}
    )
    summary = summary.sort_values("Avg quality", ascending=False).reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sectors", len(summary))
    c1_best = summary.iloc[0]
    c2.metric(
        "Strongest sector",
        c1_best["Sector"],
        f"avg {c1_best['Avg quality']:.1f}",
    )
    c3.metric(
        "Largest sector",
        summary.loc[summary["Stocks"].idxmax(), "Sector"],
        f"{int(summary['Stocks'].max())} stocks",
    )

    st.markdown("**Average quality score by sector**")
    st.bar_chart(summary.set_index("Sector")["Avg quality"], height=300)

    st.markdown("**Sector detail**")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Stocks": st.column_config.NumberColumn("Stocks", width="small"),
            "Avg quality": st.column_config.ProgressColumn(
                "Avg quality", min_value=0, max_value=100, format="%.1f"
            ),
            "Median quality": st.column_config.NumberColumn("Median", format="%.1f"),
            "Best score": st.column_config.NumberColumn("Best", format="%.1f"),
            "Avg red flags": st.column_config.NumberColumn("Avg flags", format="%.2f"),
            "Top stock": st.column_config.TextColumn("Top stock", width="small"),
        },
    )
    st.caption(
        "Tip: use the Sector filter in the Universe Ranking tab to drill "
        "into any sector's individual stocks."
    )
