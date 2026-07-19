"""
Fundamental Stock Analyzer - Streamlit App
==========================================
Run:  streamlit run app.py

Tabs:
  1. Single Stock Report  - quality score breakdown + red flags
  2. Universe Ranking      - multi-factor leaderboard
  3. Compare               - side-by-side radars
  4. Sector Overview       - sector aggregates
  5. Train Model           - (re)train the global outperformance model

UI lives in ui/; scoring pipeline in src/enrich.py.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.auth import login_gate, logout_button
from src.enrich import enrich, build_history_panel, config_from_weights, is_tickers_only
from src.quality_score import (
    DEFAULT_CATEGORY_WEIGHTS,
    CATEGORY_TOOLTIPS,
    build_config,
)
from src.sample_data import sample_csv_bytes, sample_dataframe, COLUMN_DOCS
from ui.tabs import (
    render_report,
    render_ranking,
    render_compare,
    render_sector,
    render_train,
)

# ---------------------------------------------------------------------------
# Page config + auth (must run before any other Streamlit body)
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

login_gate()
logout_button()

# ---------------------------------------------------------------------------
# Cached pipeline wrappers (hashable weights_tuple for Streamlit cache keys)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
    return df


@st.cache_data(show_spinner="Scoring universe…")
def cached_enrich(raw_df: pd.DataFrame, use_sector: bool, weights_tuple: tuple):
    """Cache enrich across reruns when data + weights + sector toggle are unchanged."""
    cfg = config_from_weights(weights_tuple)
    return enrich(raw_df, use_sector=use_sector, config=cfg)


@st.cache_data(show_spinner=False)
def cached_history(raw_df: pd.DataFrame, use_sector: bool, weights_tuple: tuple):
    cfg = config_from_weights(weights_tuple)
    return build_history_panel(raw_df, use_sector=use_sector, config=cfg)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_HEADER_LOGO = (
    "<svg width='44' height='44' viewBox='0 0 120 120' style='vertical-align:middle;margin-right:12px;'>"
    "<rect x='0' y='0' width='120' height='120' rx='26' fill='#0E1A14'/>"
    "<rect x='1' y='1' width='118' height='118' rx='25' fill='none' stroke='#1D9E75' stroke-width='2'/>"
    "<rect x='30' y='74' width='14' height='22' rx='3' fill='#2E6E55'/>"
    "<rect x='53' y='58' width='14' height='38' rx='3' fill='#26B583'/>"
    "<rect x='76' y='40' width='14' height='56' rx='3' fill='#1D9E75'/>"
    "<path d='M30 58 L52 42 L70 30 L94 26' fill='none' stroke='#7CF0C0' stroke-width='4' "
    "stroke-linecap='round' stroke-linejoin='round'/>"
    "<circle cx='94' cy='26' r='5.5' fill='#7CF0C0'/></svg>"
)
st.markdown(
    f"<h1 style='display:inline-flex;align-items:center;margin-bottom:0;'>"
    f"{_HEADER_LOGO}<span>Fundamental Stock Analyzer</span></h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Quality Score Engine · Self-learning Outperformance Model · "
    "Multi-Factor Ranking · Red-Flag Detection"
)

# ---------------------------------------------------------------------------
# Sidebar: upload + scoring controls
# ---------------------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload fundamentals panel (CSV)", type="csv")

DEMO_PATH = "demo_data.csv"
demo_available = os.path.exists(DEMO_PATH)
use_demo = False
if demo_available and uploaded is None:
    use_demo = st.sidebar.button(
        "▶️ Load demo data",
        help="Explore the app with a bundled sample universe.",
    )

st.sidebar.download_button(
    "⬇️ Download sample CSV template",
    data=sample_csv_bytes(),
    file_name="stock_analyzer_template.csv",
    mime="text/csv",
    help="Pre-filled example with 2 tickers x 2 years. Replace with your own data.",
)
st.sidebar.markdown(
    "CSV must include: `ticker`, `date`, the metric columns, and for training "
    "`fwd_return` + `bench_fwd_return`. Use at least 2 years per ticker so the "
    "year-over-year red-flag rules can compute."
)

if uploaded is None and not use_demo:
    st.info(
        "Upload a CSV to begin"
        + (", click **Load demo data** in the sidebar," if demo_available else "")
        + " or download the sample template, fill it with your data, and upload it back."
    )
    with st.expander("📋 Preview the sample template & column guide", expanded=True):
        st.dataframe(sample_dataframe(), use_container_width=True)
        st.markdown("**Column reference**")
        st.dataframe(
            pd.DataFrame(
                [{"column": k, "description": v} for k, v in COLUMN_DOCS.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.stop()

raw = load_data(DEMO_PATH if use_demo else uploaded)

if is_tickers_only(raw):
    st.warning("This looks like a **tickers-only** file (just ticker/date, no metrics).")
    st.markdown(
        "To analyze, you first need to fetch the fundamentals. This runs **locally** "
        "(it needs internet access to Yahoo Finance), then you upload the result here.\n\n"
        "**Steps:**\n"
        "1. Install: `pip install yfinance`\n"
        "2. Run the fetcher on your tickers file:\n"
    )
    st.code(
        "python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv",
        language="bash",
    )
    st.markdown(
        "3. Upload the generated `fundamentals.csv` here instead.\n\n"
        "**Note:** Yahoo provides the financial metrics, profitability, valuation, and the "
        "earnings/financial red-flag inputs. India-specific governance fields are **not** "
        "available from free sources and will stay blank unless you add that data yourself."
    )
    st.info(f"Detected {raw['ticker'].nunique()} unique tickers in your upload.")
    st.stop()

has_sector = "sector" in raw.columns and raw["sector"].notna().any()
use_sector = False
if has_sector:
    use_sector = st.sidebar.checkbox(
        "Rank within sector (peer-relative)",
        value=True,
        help="Compares each stock against others in its own sector rather than the "
             "whole universe. Small sectors (<5 stocks) fall back to overall ranking.",
    )
else:
    st.sidebar.caption(
        "ℹ️ No `sector` column found — ranking against the full universe. "
        "Re-run the fetcher to populate sectors automatically."
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

data = cached_enrich(raw, use_sector, weights_tuple)

# Re-apply ML scores from session (training is ephemeral; enrich rebuilds every run)
if "outperform_by_ticker" in st.session_state:
    data = data.copy()
    data["outperform_proba"] = data["ticker"].map(st.session_state["outperform_by_ticker"])

history_panel = cached_history(raw, use_sector, weights_tuple)

# ---------------------------------------------------------------------------
# Summary banner + tabs
# ---------------------------------------------------------------------------
_n_stocks = data["ticker"].nunique()
_n_sectors = data["sector"].nunique() if "sector" in data.columns else 0
_avg_q = data["quality_score"].mean() if "quality_score" in data.columns else float("nan")
_warned = int(data["data_warning"].sum()) if "data_warning" in data.columns else 0
b1, b2, b3, b4 = st.columns(4)
b1.metric("Universe", f"{_n_stocks:,} stocks", help="Number of unique stocks loaded and scored.")
b2.metric("Sectors", _n_sectors if _n_sectors else "—", help="Distinct sectors represented.")
b3.metric(
    "Avg quality",
    f"{_avg_q:.1f}" if pd.notna(_avg_q) else "—",
    help="Mean quality score across the universe (0–100).",
)
b4.metric(
    "Data warnings",
    _warned,
    help="Stocks with distorted/implausible figures flagged for manual review.",
)
st.divider()

tab1, tab2, tab_compare, tab_sector, tab3 = st.tabs(
    [
        "Single Stock Report",
        "Universe Ranking",
        "Compare",
        "Sector Overview",
        "Train Model",
    ]
)

with tab1:
    render_report(data, history_panel, custom_config)
with tab2:
    render_ranking(data)
with tab_compare:
    render_compare(data)
with tab_sector:
    render_sector(data)
with tab3:
    render_train(data)
