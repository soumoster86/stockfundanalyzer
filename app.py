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

NAV_PAGES = [
    "Single Stock Report",
    "Universe Ranking",
    "Watchlist",
    "Compare",
    "Sector Overview",
    "Train Model",
    "Tutorial",
]

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

st.sidebar.markdown("### 📂 Data source")
uploaded = st.sidebar.file_uploader(
    "Upload a different fundamentals CSV (optional)",
    type="csv",
    help="Leave empty to use the project’s fundamentals.csv automatically. "
         "Upload here to override with a new file (e.g. labeled.csv).",
)

use_demo = False
if default_fund_path is None and demo_available and uploaded is None:
    use_demo = st.sidebar.button(
        "▶️ Load demo data",
        help="Explore the app with a bundled sample universe.",
    )
elif default_fund_path is not None and uploaded is None:
    st.sidebar.caption(
        f"Using project file **`{os.path.basename(default_fund_path)}`** by default. "
        "Upload above only if you want to replace it for this session."
    )
    if demo_available:
        use_demo = st.sidebar.button(
            "▶️ Load demo data instead",
            help="Ignore fundamentals.csv and load the small demo sample.",
        )

st.sidebar.download_button(
    "⬇️ Sample fundamentals template",
    data=sample_csv_bytes(),
    file_name="stock_analyzer_template.csv",
    mime="text/csv",
    help="Pre-filled example with 2 tickers × 2 years.",
)

with st.sidebar.expander("🇮🇳 India governance CSV (optional)"):
    st.caption(
        "Yahoo does not provide promoter pledge, insider trades, auditor, or "
        "related-party flags. Upload a governance overlay so those red flags can fire."
    )
    st.download_button(
        "⬇️ Governance template",
        data=governance_template_csv(),
        file_name="governance_template.csv",
        mime="text/csv",
    )
    gov_file = st.file_uploader(
        "Upload governance CSV",
        type="csv",
        key="gov_upload",
    )

st.sidebar.markdown(
    "CSV needs `ticker`, `date`, metric columns. Growth fields are **decimals** "
    "(0.12 = 12%); ROE/margins are **percent points** (12 = 12%). "
    "For ML labels run `python -m src.build_labels`."
)

# Local Yahoo refresh (optional — needs yfinance + network)
from src.refresh import refresh_fundamentals, yfinance_available

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Refresh data")
if yfinance_available():
    _max_refresh = st.sidebar.number_input(
        "Max tickers to refresh",
        min_value=1,
        max_value=5000,
        value=100,
        step=50,
        help="Full universe refreshes can take a long time. Prefer stocks.csv offline "
             "via `python -m src.fetch_fundamentals` for thousands of names.",
        key="refresh_max_tickers",
    )
    if st.sidebar.button(
        "Refresh fundamentals (Yahoo)",
        use_container_width=True,
        help="Re-fetch from Yahoo into fundamentals.csv (local only).",
    ):
        root = os.path.dirname(os.path.abspath(__file__))
        prog = st.sidebar.progress(0.0, text="Starting…")
        status = st.sidebar.empty()

        def _cb(i, total, tkr):
            prog.progress(min(1.0, i / max(total, 1)), text=f"{i}/{total} {tkr}")
            status.caption(f"Fetching {tkr}…")

        with st.spinner("Refreshing fundamentals from Yahoo…"):
            # Uses stocks.csv when present; otherwise pass None (refresh will error
            # clearly if neither stocks.csv nor tickers exist).
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
            st.sidebar.success(result["message"])
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(result["message"])
else:
    st.sidebar.caption(
        "Install `yfinance` locally to enable one-click refresh, or run:\n"
        "`python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv`"
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

# Data freshness banner (file mtime + fiscal date range in the panel)
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
        "ℹ️ No `sector` column — ranking vs full universe. Re-run the fetcher to add sectors."
    )

# ---- Configurable category weights ----
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
for cat in DEFAULT_CATEGORY_WEIGHTS:
    weights[cat] = st.sidebar.slider(
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
        "Effective mix: "
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

_n_stocks = data["ticker"].nunique()
_n_sectors = data["sector"].nunique() if "sector" in data.columns else 0
_avg_q = data["quality_score"].mean() if "quality_score" in data.columns else float("nan")
_warned = int(data["data_warning"].sum()) if "data_warning" in data.columns else 0
b1, b2, b3, b4 = st.columns(4)
b1.metric("Universe", f"{_n_stocks:,} stocks")
b2.metric("Sectors", _n_sectors if _n_sectors else "—")
b3.metric("Avg quality", f"{_avg_q:.1f}" if pd.notna(_avg_q) else "—")
b4.metric("Data warnings", _warned)
st.divider()

# Navigation radio uses key "nav_page". Streamlit forbids changing that key
# after the widget is created in the same run, so page jumps (Ranking → Report)
# write to "_pending_nav" and we apply it *here* before instantiating the radio.
if "_pending_nav" in st.session_state:
    pending = st.session_state.pop("_pending_nav")
    if pending in NAV_PAGES:
        st.session_state["nav_page"] = pending
elif "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_PAGES[0]

page = st.radio(
    "Navigate",
    NAV_PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)

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
