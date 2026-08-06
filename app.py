"""
Fundamental Stock Analyzer - Streamlit App
==========================================
Run:  streamlit run app.py

UI lives in ui/; scoring pipeline in src/enrich.py.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.auth import login_gate, logout_button
from src.enrich import build_history_panel, config_from_weights, enrich, is_tickers_only
from src.governance import governance_template_csv, merge_governance
from src.quality_score import (
    CATEGORY_TOOLTIPS,
    DEFAULT_CATEGORY_WEIGHTS,
    build_config,
)
from src.sample_data import COLUMN_DOCS, sample_csv_bytes, sample_dataframe
from src.schema import prepare_panel
from ui.nav import (
    ALL_PAGES,
    NAV_BY_MODE,
    NAV_MODES,
    SHOW_UNIVERSE_BANNER,
    ensure_page_in_mode,
    resolve_pending_nav,
)
from ui.tabs import (
    render_compare,
    render_ranking,
    render_report,
    render_sector,
    render_train,
    render_tutorial,
    render_watchlist,
)
from ui.theme import inject_global_css

# ---------------------------------------------------------------------------
# Page config + auth
# ---------------------------------------------------------------------------
_PAGE_ICON = "📊"
_logo_png = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(_logo_png):
    try:
        from PIL import Image as _PILImage

        _PAGE_ICON = _PILImage.open(_logo_png)
    except Exception:
        _PAGE_ICON = "📊"

st.set_page_config(
    page_title="Fundamental Stock Analyzer",
    page_icon=_PAGE_ICON,
    layout="wide",
)
inject_global_css()

login_gate()
logout_button()

# ---------------------------------------------------------------------------
# Cached pipeline wrappers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    return df


@st.cache_data(show_spinner="Scoring universe…")
def cached_enrich(raw_df: pd.DataFrame, use_sector: bool, weights_tuple: tuple):
    cfg = config_from_weights(weights_tuple)
    return enrich(raw_df, use_sector=use_sector, config=cfg)


@st.cache_data(show_spinner=False)
def cached_history(raw_df: pd.DataFrame, use_sector: bool, weights_tuple: tuple):
    cfg = config_from_weights(weights_tuple)
    return build_history_panel(raw_df, use_sector=use_sector, config=cfg)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
from ui.landing import render_hero

render_hero(
    st,
    subtitle=(
        "Quality · red flags · Piotroski / Altman / Beneish · screens · "
        "watchlist · compare · sectors · optional ML"
    ),
)
# Full how-to lives under nav **Tutorial** / sidebar — not on the main chrome

# ---------------------------------------------------------------------------
# Sidebar: upload + governance + scoring controls
# ---------------------------------------------------------------------------
# Default data file shipped / generated next to app.py
DEFAULT_FUNDAMENTALS = "fundamentals.csv"
LABELED_DEFAULT = "labeled.csv"  # optional pre-labeled panel
DEMO_PATH = "demo_data.csv"

default_fund_path = None
for candidate in (DEFAULT_FUNDAMENTALS, LABELED_DEFAULT):
    p = os.path.join(os.path.dirname(__file__) or ".", candidate)
    if os.path.exists(p):
        default_fund_path = p
        break
    if os.path.exists(candidate):
        default_fund_path = candidate
        break

demo_available = os.path.exists(DEMO_PATH) or os.path.exists(
    os.path.join(os.path.dirname(__file__) or ".", DEMO_PATH)
)

# Collapse data chrome after a successful prior load so research pages stay clean.
_data_ready = bool(st.session_state.get("_data_loaded"))
with st.sidebar.expander("📂 Data source", expanded=not _data_ready):
    if _data_ready and default_fund_path is not None:
        st.caption(
            f"Loaded: **`{os.path.basename(default_fund_path)}`** "
            "(upload below to override)."
        )
    uploaded = st.file_uploader(
        "Upload fundamentals CSV (optional)",
        type="csv",
        help="Leave empty to use the project’s fundamentals.csv automatically.",
        key="sidebar_fundamentals_upload",
    )

    use_demo = False  # set by demo buttons below when shown
    if default_fund_path is None and demo_available and uploaded is None:
        use_demo = st.button(
            "▶️ Load demo data",
            help="Explore the app with a bundled sample universe.",
            key="sidebar_load_demo",
        )
    elif default_fund_path is not None and uploaded is None:
        if not _data_ready:
            st.caption(
                f"Using project file **`{os.path.basename(default_fund_path)}`** "
                "by default."
            )
        if demo_available:
            use_demo = st.button(
                "▶️ Load demo data instead",
                help="Ignore fundamentals.csv and load the small demo sample.",
                key="sidebar_load_demo_alt",
            )

    st.download_button(
        "⬇️ Sample fundamentals template",
        data=sample_csv_bytes(),
        file_name="stock_analyzer_template.csv",
        mime="text/csv",
        help="Pre-filled example with 2 tickers × 2 years.",
    )
    st.caption(
        "CSV needs `ticker`, `date`, metric columns. Growth = decimals (0.12); "
        "ROE/margins = percent points."
    )

    st.markdown("**India governance (optional)**")
    st.caption(
        "Yahoo lacks pledge / insider / auditor / related-party. Upload an overlay "
        "so those red flags can fire."
    )
    st.download_button(
        "⬇️ Governance template",
        data=governance_template_csv(),
        file_name="governance_template.csv",
        mime="text/csv",
        key="gov_template_dl",
    )
    gov_file = st.file_uploader(
        "Upload governance CSV",
        type="csv",
        key="gov_upload",
    )

    # Local Yahoo refresh (optional — needs yfinance + network)
    from src.refresh import refresh_fundamentals, yfinance_available

    st.markdown("**Refresh from Yahoo**")
    if yfinance_available():
        _max_refresh = st.number_input(
            "Max tickers to refresh",
            min_value=1,
            max_value=5000,
            value=100,
            step=50,
            help="Prefer offline `python -m src.fetch_fundamentals` for large lists.",
            key="refresh_max_tickers",
        )
        if st.button(
            "Refresh fundamentals (Yahoo)",
            use_container_width=True,
            help="Re-fetch from Yahoo into fundamentals.csv (local only).",
            key="refresh_yahoo_btn",
        ):
            root = os.path.dirname(os.path.abspath(__file__))
            prog = st.progress(0.0, text="Starting…")
            status = st.empty()

            def _cb(i, total, tkr):
                prog.progress(min(1.0, i / max(total, 1)), text=f"{i}/{total} {tkr}")
                status.caption(f"Fetching {tkr}…")

            with st.spinner("Refreshing fundamentals from Yahoo…"):
                result = refresh_fundamentals(
                    root,
                    out_name="fundamentals.csv",
                    tickers=None,
                    sleep_s=0.35,
                    max_tickers=int(_max_refresh),
                    progress_cb=_cb,
                )
            prog.empty()
            status.empty()
            if result["ok"]:
                st.success(result["message"])
                st.cache_data.clear()
                st.session_state["_data_loaded"] = True
                st.rerun()
            else:
                st.error(result["message"])
    else:
        st.caption(
            "Install `yfinance` for one-click refresh, or run "
            "`python -m src.fetch_fundamentals`."
        )

if st.sidebar.button("📖 Open Tutorial", use_container_width=True):
    st.session_state["_pending_nav"] = "Tutorial"
    st.rerun()
# Resolve which source to load (upload > demo button > project fundamentals > empty)
data_source_label = None
data_source_path = None
file_mtime = None

if uploaded is not None:
    raw_in = load_data(uploaded)
    data_source_label = f"Upload: {getattr(uploaded, 'name', 'uploaded.csv')}"
elif use_demo:
    demo_p = DEMO_PATH if os.path.exists(DEMO_PATH) else os.path.join(
        os.path.dirname(__file__) or ".", DEMO_PATH
    )
    raw_in = load_data(demo_p)
    data_source_label = f"Demo: {os.path.basename(demo_p)}"
    data_source_path = demo_p
    file_mtime = os.path.getmtime(demo_p)
elif default_fund_path is not None:
    raw_in = load_data(default_fund_path)
    data_source_label = f"Project: {os.path.basename(default_fund_path)}"
    data_source_path = default_fund_path
    file_mtime = os.path.getmtime(default_fund_path)
else:
    from ui.landing import render_feature_sections, render_hero

    render_hero(
        st,
        subtitle="Load a fundamentals panel to start scoring, screening, and researching.",
    )
    st.info(
        "No `fundamentals.csv` found in the project folder. "
        "Upload a CSV in the sidebar"
        + (", click **Load demo data**," if demo_available else "")
        + ", or generate one:\n\n"
        "`python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv`"
    )

    st.markdown("##### ✨ What's new")
    st.caption(
        "Expand a section for capabilities. "
        "After you load data, use the **Tutorial** page for step-by-step usage."
    )
    render_feature_sections(st, compact=False, expand_first=False)

    with st.expander("📋 Preview the sample template & column guide", expanded=False):
        st.dataframe(sample_dataframe(), use_container_width=True)
        st.markdown("**Column reference**")
        st.dataframe(
            pd.DataFrame(
                [{"column": k, "description": v} for k, v in COLUMN_DOCS.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            "**Units:** growth / fwd returns = decimals · "
            "ROE, margins, dividend yield = percent points · "
            "ratios (P/E, D/E) = multiples."
        )
    st.stop()

raw, validation, unit_notes = prepare_panel(raw_in)

if validation.errors:
    st.error("**Upload failed validation**")
    for e in validation.errors:
        st.markdown(f"- {e}")
    st.stop()

for w in validation.warnings:
    st.warning(w)
for note in validation.info:
    st.caption(f"ℹ️ {note}")

# Data freshness / provenance (GitHub daily pipeline vs upload/demo)
from src.data_freshness import (
    format_freshness_detail,
    format_freshness_line,
    is_github_daily,
    load_fundamentals_meta,
)

_date_min = _date_max = None
if "date" in raw.columns and raw["date"].notna().any():
    _date_min = pd.to_datetime(raw["date"], errors="coerce").min()
    _date_max = pd.to_datetime(raw["date"], errors="coerce").max()

_mtime_txt = ""
if file_mtime is not None:
    _mtime_txt = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M")
elif uploaded is not None:
    _mtime_txt = "this session (upload)"

_fiscal_txt = "—"
if _date_min is not None and pd.notna(_date_min):
    if _date_max is not None and pd.notna(_date_max) and _date_min != _date_max:
        _fiscal_txt = (
            f"{_date_min.strftime('%Y-%m-%d')} → {_date_max.strftime('%Y-%m-%d')}"
        )
    else:
        _fiscal_txt = _date_min.strftime("%Y-%m-%d")

_fund_meta = load_fundamentals_meta(os.path.dirname(os.path.abspath(__file__)) or ".")
# Only claim GitHub daily provenance when loading the project fundamentals file
_using_project_fundamentals = (
    uploaded is None
    and not use_demo
    and data_source_path is not None
    and os.path.basename(str(data_source_path)).lower() in (
        "fundamentals.csv",
        "labeled.csv",
    )
)
_ci_active = _using_project_fundamentals and is_github_daily(_fund_meta)

if _ci_active:
    st.success(format_freshness_line(_fund_meta))
    _run_url = (_fund_meta or {}).get("workflow_url")
    if _run_url:
        st.caption(f"[Open this GitHub Actions run]({_run_url})")
elif uploaded is not None:
    st.info(
        f"**Data source:** {data_source_label} · **session upload** "
        "(not the GitHub daily pipeline for this session)"
        + f" · **Fiscal dates:** {_fiscal_txt}"
    )
elif use_demo:
    st.info(
        f"**Data source:** {data_source_label} · demo sample · "
        f"**Fiscal dates:** {_fiscal_txt}"
    )
else:
    # Project file present but no CI meta yet (or meta from non-CI)
    st.info(
        f"**Data source:** {data_source_label}"
        + (f" · **File updated:** {_mtime_txt}" if _mtime_txt else "")
        + f" · **Fiscal dates:** {_fiscal_txt}"
        + (
            " · Upload a CSV in the sidebar to override for this session."
            if uploaded is None and default_fund_path is not None
            else ""
        )
    )
    if _using_project_fundamentals and not is_github_daily(_fund_meta):
        st.caption(
            "No GitHub daily-pipeline stamp yet. After **Actions → Daily fundamentals "
            "+ rankings** commits `fundamentals_meta.json`, this banner will show the "
            "pipeline run time and id."
        )

with st.expander("Data provenance details", expanded=False):
    if _ci_active:
        for line in format_freshness_detail(_fund_meta):
            st.markdown(f"- {line}")
    else:
        for line in format_freshness_detail(
            _fund_meta if _using_project_fundamentals else None
        ):
            st.markdown(f"- {line}")
    st.caption(
        f"Session source label: `{data_source_label}` · fiscal range: {_fiscal_txt}"
    )
    # Recent GitHub pipeline runs stored in Supabase (if configured)
    try:
        from src.pipeline_log_supabase import fetch_recent_runs
        from src.watchlist_supabase import is_configured as _sb_ok

        if _sb_ok():
            st.markdown("**Recent pipeline runs (Supabase)**")
            _runs = fetch_recent_runs(limit=8)
            if not _runs:
                st.caption(
                    "No rows in `pipeline_runs` yet — run the daily Action after "
                    "applying `sql/pipeline_runs_supabase.sql` and setting "
                    "GitHub secrets `SUPABASE_URL` / `SUPABASE_KEY`."
                )
            else:
                import pandas as _pd

                _df_runs = _pd.DataFrame(_runs)
                _show = [
                    c
                    for c in (
                        "finished_at",
                        "status",
                        "n_tickers",
                        "avg_quality",
                        "n_data_warnings",
                        "workflow_run",
                        "ref",
                    )
                    if c in _df_runs.columns
                ]
                st.dataframe(
                    _df_runs[_show] if _show else _df_runs,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.caption(
                "Configure Supabase secrets to load pipeline history from "
                "`pipeline_runs`."
            )
    except Exception as _e:
        st.caption(f"Pipeline history unavailable: {_e}")

# Optional governance overlay (before scoring so red flags see the fields)
if gov_file is not None:
    try:
        raw, gov_stats = merge_governance(raw, gov_file)
        st.sidebar.success(
            f"Governance merged ({gov_stats['mode']}): "
            f"{gov_stats['n_tickers_matched']} tickers, "
            f"{gov_stats['n_rows_with_gov']} rows, "
            f"cols={', '.join(gov_stats['cols_merged']) or '—'}."
        )
        # Bust enrich cache key by tagging session
        st.session_state["gov_merged"] = True
    except Exception as e:
        st.sidebar.error(f"Governance merge failed: {e}")

# Keep multi-year panel for training labels
st.session_state["raw_panel"] = raw

if is_tickers_only(raw):
    st.warning("This looks like a **tickers-only** file (just ticker/date, no metrics).")
    st.markdown(
        "Fetch fundamentals locally, then upload the result:\n\n"
        "1. `pip install yfinance`\n"
        "2. Run:"
    )
    st.code(
        "python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv",
        language="bash",
    )
    st.markdown(
        "3. Optional labels:\n"
        "```bash\n"
        "python -m src.build_labels --in fundamentals.csv --out labeled.csv "
        "--horizon-years 3 --benchmark ^NSEI\n"
        "```\n"
        "4. Optional governance: upload a governance CSV in the sidebar.\n"
        "5. Upload `fundamentals.csv` / `labeled.csv` here."
    )
    st.info(f"Detected {raw['ticker'].nunique()} unique tickers in your upload.")
    st.stop()

# Mark data loaded so next run collapses the data expander
st.session_state["_data_loaded"] = True

has_sector = "sector" in raw.columns and raw["sector"].notna().any()
use_sector = False
if has_sector:
    use_sector = st.sidebar.checkbox(
        "Rank within sector (peer-relative)",
        value=True,
        help="Compares each stock against sector peers. Small sectors fall back "
             "to overall ranking.",
    )
else:
    st.sidebar.caption(
        "No `sector` column — ranking vs full universe."
    )

# ---- Configurable category weights (sliders collapsed unless Custom) ----
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring weights")
preset = st.sidebar.selectbox(
    "Preset",
    [
        "Balanced (default)",
        "Value tilt",
        "Quality tilt",
        "Growth tilt",
        "Safety tilt",
        "Custom",
    ],
    help="Quickly bias the score toward a style.",
)
PRESETS = {
    "Value tilt": {
        "Valuation": 0.40,
        "Profitability": 0.20,
        "Financial Performance": 0.15,
        "Financial Strength": 0.15,
        "Shareholder Metrics": 0.10,
    },
    "Quality tilt": {
        "Profitability": 0.40,
        "Financial Strength": 0.25,
        "Valuation": 0.15,
        "Financial Performance": 0.15,
        "Shareholder Metrics": 0.05,
    },
    "Growth tilt": {
        "Financial Performance": 0.45,
        "Profitability": 0.25,
        "Valuation": 0.15,
        "Financial Strength": 0.10,
        "Shareholder Metrics": 0.05,
    },
    "Safety tilt": {
        "Financial Strength": 0.40,
        "Profitability": 0.25,
        "Valuation": 0.15,
        "Shareholder Metrics": 0.10,
        "Financial Performance": 0.10,
    },
}
if preset == "Balanced (default)":
    base_weights = dict(DEFAULT_CATEGORY_WEIGHTS)
elif preset in PRESETS:
    base_weights = PRESETS[preset]
else:
    base_weights = dict(DEFAULT_CATEGORY_WEIGHTS)

weights = {}
with st.sidebar.expander(
    "Category sliders",
    expanded=(preset == "Custom"),
):
    if preset != "Custom":
        st.caption("Switch preset to **Custom** to edit sliders.")
    for cat in DEFAULT_CATEGORY_WEIGHTS:
        weights[cat] = st.slider(
            cat,
            0.0,
            1.0,
            float(base_weights.get(cat, DEFAULT_CATEGORY_WEIGHTS[cat])),
            0.05,
            disabled=(preset != "Custom"),
            key=f"w_{cat}",
            help=CATEGORY_TOOLTIPS.get(cat),
        )
total_w = sum(weights.values())
if total_w > 0:
    st.sidebar.caption(
        "Mix: "
        + ", ".join(
            f"{cat.split()[0]} {weights[cat] / total_w * 100:.0f}%" for cat in weights
        )
    )
else:
    st.sidebar.warning("All weights are zero — set at least one above zero.")
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)

weights_tuple = tuple(sorted(weights.items()))
custom_config = build_config(weights)

# Include gov merge flag in cache key via a hash of gov-related columns presence
# Streamlit cache keys on raw_df content, so a governance-updated frame busts cache.
data = cached_enrich(raw, use_sector, weights_tuple)

if "outperform_by_ticker" in st.session_state:
    data = data.copy()
    data["outperform_proba"] = data["ticker"].map(st.session_state["outperform_by_ticker"])

history_panel = cached_history(raw, use_sector, weights_tuple)

# ---------------------------------------------------------------------------
# Summary banner + data coverage + navigation
# ---------------------------------------------------------------------------
from src.coverage import coverage_banner_text, coverage_detail_lines, coverage_summary

_cov = coverage_summary(data, raw=raw)
st.caption(coverage_banner_text(_cov))
with st.expander("📋 Data coverage details", expanded=False):
    for line in coverage_detail_lines(_cov):
        st.markdown(f"- {line}")
    st.caption(
        "Sparse = <50% of core metrics filled. F/Z/M need multi-year and balance-sheet "
        "fields — re-fetch or refresh if coverage is low."
    )

# Navigation: apply cross-page jumps *before* mode/page widgets are created
# (Streamlit forbids mutating a widget key after the widget is instantiated).
resolve_pending_nav(st.session_state)
if "nav_mode" not in st.session_state:
    st.session_state["nav_mode"] = NAV_MODES[0]
if "nav_page" not in st.session_state or st.session_state["nav_page"] not in ALL_PAGES:
    st.session_state["nav_page"] = NAV_BY_MODE[NAV_MODES[0]][0]

mode = st.radio(
    "Mode",
    NAV_MODES,
    horizontal=True,
    key="nav_mode",
    help="Research = daily work · Context = compare & sectors · Tools = train & tutorial",
)
page = ensure_page_in_mode(st.session_state, mode)
page_options = NAV_BY_MODE[mode]
page = st.radio(
    "Page",
    page_options,
    horizontal=True,
    key="nav_page",
    label_visibility="collapsed",
)

# Universe metrics only where they help (not on deep single-stock report / tools)
if page in SHOW_UNIVERSE_BANNER:
    _n_stocks = data["ticker"].nunique()
    _n_sectors = data["sector"].nunique() if "sector" in data.columns else 0
    _avg_q = (
        data["quality_score"].mean() if "quality_score" in data.columns else float("nan")
    )
    _warned = int(data["data_warning"].sum()) if "data_warning" in data.columns else 0
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Universe", f"{_n_stocks:,} stocks")
    b2.metric("Sectors", _n_sectors if _n_sectors else "—")
    b3.metric("Avg quality", f"{_avg_q:.1f}" if pd.notna(_avg_q) else "—")
    b4.metric("Data warnings", _warned)
    st.divider()

if page == "Single Stock Report":
    render_report(data, history_panel, custom_config)
elif page == "Universe Ranking":
    render_ranking(data)
elif page == "Watchlist":
    render_watchlist(data)
elif page == "Compare":
    render_compare(data)
elif page == "Sector Overview":
    render_sector(data)
elif page == "Train Model":
    render_train(data)
elif page == "Tutorial":
    render_tutorial()
