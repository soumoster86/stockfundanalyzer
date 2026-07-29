"""Watchlist portfolio page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.alerts import alert_summary, evaluate_watchlist_alerts
from src.quality_score import CONCEPT_TOOLTIPS, score_label
from src.watchlist import (
    ERROR_KEY,
    add_ticker,
    backend_name,
    filter_universe,
    get_watchlist,
    remove_ticker,
    set_watchlist,
    watchlist_from_csv,
    watchlist_summary,
    watchlist_to_csv,
)
from src.watchlist_supabase import backend_status
from ui.theme import badge_row, band_text, section


def render_watchlist(data: pd.DataFrame) -> None:
    section("Watchlist")
    backend = backend_name(st.session_state)
    if backend == "supabase":
        st.caption(
            "Storage: **Supabase** (per user, survives restarts). "
            "Z/M: 🟢 good · 🟡 watch · 🔴 risk."
        )
    else:
        st.caption(
            "Storage: **this browser session** (ephemeral). "
            "Configure Supabase secrets for a durable per-user list, "
            "or download CSV as backup. Z/M: 🟢 good · 🟡 watch · 🔴 risk."
        )
        status = backend_status()
        if not status["configured"]:
            with st.expander("How to enable Supabase watchlists", expanded=False):
                st.markdown(
                    """
1. Create a free project at [supabase.com](https://supabase.com).  
2. Run the SQL in `sql/watchlist_supabase.sql` in the SQL editor.  
3. Add to `.streamlit/secrets.toml` (or Streamlit Cloud secrets):

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
key = "YOUR_SERVICE_ROLE_OR_ANON_KEY"
```

4. Install: `pip install supabase`  
5. Restart the app and sign in — your list loads from Supabase.
                    """
                )
        elif not status["package_installed"]:
            st.warning("Supabase secrets found but package missing: `pip install supabase`")

    err = st.session_state.get(ERROR_KEY)
    if err:
        st.warning(f"Watchlist cloud sync issue (using session cache): {err}")

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

    # Alerts (monitoring rules)
    st.markdown("**Alerts**")
    with st.expander("Alert thresholds", expanded=False):
        a1, a2, a3 = st.columns(3)
        min_q = a1.number_input(
            "Min quality",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            key="wl_alert_min_q",
            help="Alert when quality falls below this.",
        )
        max_flags = a2.number_input(
            "Max red flags",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="wl_alert_max_flags",
            help="Alert when flag count exceeds this.",
        )
        max_f = a3.number_input(
            "Weak F-score ≤",
            min_value=0,
            max_value=9,
            value=3,
            step=1,
            key="wl_alert_max_f",
            help="Alert when Piotroski F is at or below this (needs enough tests).",
        )
    alerts = evaluate_watchlist_alerts(
        sub,
        tickers=wl,
        rules={
            "min_quality": float(min_q),
            "max_red_flags": int(max_flags),
            "max_f_score_low": int(max_f),
        },
    )
    asum = alert_summary(alerts)
    st.markdown(
        badge_row(
            [
                (f"{asum['high']} high", "Red" if asum["high"] else "Green"),
                (f"{asum['medium']} medium", "Yellow" if asum["medium"] else "Green"),
                (f"{asum['low']} low", "Moderate" if asum["low"] else "Green"),
                (f"{asum['names']} names", "Moderate"),
            ]
        ),
        unsafe_allow_html=True,
    )
    if alerts.empty:
        st.success("No alerts on the current watchlist with these thresholds.")
    else:
        show_a = alerts.copy()
        show_a["ticker"] = show_a["ticker"].astype(str).str.replace(".NS", "", regex=False)
        st.dataframe(
            show_a.rename(
                columns={
                    "ticker": "Ticker",
                    "severity": "Severity",
                    "code": "Rule",
                    "message": "Message",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Sector mix / concentration
    if "sector" in sub.columns and sub["sector"].notna().any():
        st.markdown("**Sector mix & concentration**")
        mix = (
            sub.groupby("sector", dropna=False)
            .agg(Names=("ticker", "nunique"), Avg_quality=("quality_score", "mean"))
            .reset_index()
            .sort_values("Names", ascending=False)
        )
        mix["Weight_%"] = (mix["Names"] / mix["Names"].sum() * 100).round(1)
        top_w = float(mix["Weight_%"].iloc[0]) if len(mix) else 0.0
        if top_w >= 50 and len(mix) > 1:
            st.warning(
                f"Concentration: **{mix.iloc[0]['sector']}** is {top_w:.0f}% of the list."
            )
        st.bar_chart(mix.set_index("sector")["Names"], height=220)
        st.dataframe(mix, use_container_width=True, hide_index=True)

    # Market-cap mix when available
    if "mcap_bucket" in sub.columns and (sub["mcap_bucket"] != "Unknown").any():
        st.markdown("**Market-cap mix**")
        cap_mix = (
            sub.groupby("mcap_bucket", dropna=False)
            .agg(Names=("ticker", "nunique"), Avg_quality=("quality_score", "mean"))
            .reset_index()
        )
        st.dataframe(cap_mix, use_container_width=True, hide_index=True)

    # Quality labels strip
    st.caption(
        "Labels: "
        + ", ".join(
            f"{r['ticker'].replace('.NS', '')}={score_label(r['quality_score'])}"
            for _, r in sub.iterrows()
            if pd.notna(r.get("quality_score"))
        )
    )
