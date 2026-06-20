"""
Fundamental Stock Analyzer - Streamlit App
==========================================
Run:  streamlit run app.py

Tabs:
  1. Single Stock Report  - quality score breakdown + red flags
  2. Universe Ranking      - multi-factor leaderboard
  3. Train Model           - (re)train the global outperformance model
"""

import os
import joblib
import pandas as pd
import streamlit as st

from src.quality_score import (compute_quality_score, score_label, METRIC_CONFIG,
                               explain_score, explanation_sentence,
                               build_config, DEFAULT_CATEGORY_WEIGHTS,
                               quality_history, ticker_trend,
                               METRIC_TOOLTIPS, CATEGORY_TOOLTIPS, CONCEPT_TOOLTIPS)
from src.red_flags import detect_red_flags, REASON_TEXT
from src.model import (make_label, train_outperformance_model, predict_proba)
from src.ranking import rank_universe
from src.data_quality import (data_completeness, data_sanity_flags, WARNING_TEXT)
from src.institutional_scores import (compute_piotroski, compute_altman_z,
                                      f_score_band, z_band, Z_BAND_TEXT,
                                      PIOTROSKI_LABELS, PIOTROSKI_TESTS,
                                      blend_with_quality)
from src.sample_data import sample_csv_bytes, sample_dataframe, COLUMN_DOCS
from src.auth import login_gate, logout_button

_PAGE_ICON = "📊"
_logo_png = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(_logo_png):
    try:
        from PIL import Image as _PILImage
        _PAGE_ICON = _PILImage.open(_logo_png)
    except Exception:
        _PAGE_ICON = "📊"

st.set_page_config(page_title="Fundamental Stock Analyzer",
                   page_icon=_PAGE_ICON, layout="wide")

# ---- Access gate: nothing below renders until authenticated ----
login_gate()
logout_button()

import tempfile
MODEL_PATH = os.path.join(tempfile.gettempdir(), "outperformance_model.joblib")

FEATURE_COLS = [
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth",
    "ebitda_growth", "roe", "roce", "net_margin", "operating_margin",
    "gross_margin", "debt_to_equity", "interest_coverage", "current_ratio",
    "cash_position", "dividend_yield", "dividend_growth", "buyback_yield",
    "pe", "pb", "ev_ebitda", "peg", "price_sales", "quality_score",
]


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        # handle both DD-MM-YYYY and YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], format="mixed",
                                    dayfirst=True, errors="coerce")
    return df


CATEGORY_SHORT = {
    "Financial Performance": "Growth",
    "Profitability": "Profit",
    "Financial Strength": "Strength",
    "Shareholder Metrics": "Shareholder",
    "Valuation": "Value",
}


def category_radar_svg(scores: dict, color="#1D9E75", size=320):
    """
    Render a 5-axis radar/spider chart for category scores (0-100) as inline SVG.
    `scores` maps category name -> score. Missing/None categories plot at 0.
    """
    import math
    cats = list(CATEGORY_SHORT.keys())
    n = len(cats)
    cx = cy = size / 2
    r_max = size * 0.34
    # grid rings
    rings = ""
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            x = cx + r_max * frac * math.cos(ang)
            y = cy + r_max * frac * math.sin(ang)
            pts.append(f"{x:.1f},{y:.1f}")
        rings += f'<polygon points="{" ".join(pts)}" fill="none" stroke="#3a3a3a" stroke-width="1"/>'
    # spokes + labels
    spokes, labels = "", ""
    for i, cat in enumerate(cats):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x = cx + r_max * math.cos(ang)
        y = cy + r_max * math.sin(ang)
        spokes += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#3a3a3a" stroke-width="1"/>'
        lx = cx + (r_max + 22) * math.cos(ang)
        ly = cy + (r_max + 22) * math.sin(ang)
        anchor = "middle"
        if math.cos(ang) > 0.3:
            anchor = "start"
        elif math.cos(ang) < -0.3:
            anchor = "end"
        labels += (f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
                   f'font-size="12" fill="#aaa" dominant-baseline="middle">'
                   f'{CATEGORY_SHORT[cat]}</text>')
    # data polygon
    dpts = []
    for i, cat in enumerate(cats):
        v = scores.get(cat)
        v = 0.0 if (v is None or pd.isna(v)) else max(0.0, min(100.0, float(v)))
        ang = -math.pi / 2 + 2 * math.pi * i / n
        rr = r_max * v / 100.0
        dpts.append(f"{cx + rr*math.cos(ang):.1f},{cy + rr*math.sin(ang):.1f}")
    data_poly = (f'<polygon points="{" ".join(dpts)}" fill="{color}" '
                 f'fill-opacity="0.28" stroke="{color}" stroke-width="2"/>')
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
            f'{rings}{spokes}{data_poly}{labels}</svg>')


def is_tickers_only(df):
    """True if the upload has just ticker/date and no metric columns."""
    metric_cols = {"revenue_growth", "roe", "pe", "net_margin", "revenue"}
    return not (metric_cols & set(df.columns))


def enrich(df, use_sector=False, config=METRIC_CONFIG):
    # Data-quality layer needs the full multi-year panel (corporate-action checks
    # compare year-over-year), so run sanity + red flags on all rows first.
    df, _flag_cols = detect_red_flags(df)
    df, _warn_cols = data_sanity_flags(df)
    df = data_completeness(df)
    # Piotroski F-Score also needs the multi-year panel (year-over-year tests),
    # so compute it before collapsing to the latest row.
    df = compute_piotroski(df)
    # Keep the latest row per ticker that clears a minimum completeness bar, so a
    # stock whose newest fiscal row is mostly empty (e.g. not-yet-fully-reported)
    # falls back to its last substantially-populated year rather than ranking on a
    # thin row. If no row clears the bar, keep the most complete available.
    if "date" in df.columns:
        df = df.sort_values("date")
        MIN_FIELDS = 12  # of ~20 core metrics
        good = df[df["data_fields_present"] >= MIN_FIELDS]
        latest_good = good.groupby("ticker", as_index=False).tail(1)
        missing = set(df["ticker"]) - set(latest_good["ticker"])
        if missing:
            # for tickers with no row clearing the bar, take their most complete row
            rest = df[df["ticker"].isin(missing)].sort_values(
                ["ticker", "data_fields_present", "date"])
            rest = rest.groupby("ticker", as_index=False).tail(1)
            df = pd.concat([latest_good, rest], ignore_index=True)
        else:
            df = latest_good
    # Altman Z-Score is point-in-time -> compute on the latest cross-section.
    df = compute_altman_z(df)
    # Stocks may have different fiscal-end dates; for a fair cross-sectional
    # comparison they must all be ranked together, not split into tiny per-date
    # groups. Use a single snapshot key (optionally sector) instead of raw date.
    df = df.copy()
    df["_snapshot"] = "latest"
    group = ("_snapshot", "sector") if (use_sector and "sector" in df.columns) else ("_snapshot",)
    df = compute_quality_score(df, group_cols=group, config=config)
    df = df.drop(columns=["_snapshot"], errors="ignore")
    # Light blend of F-Score into a "quality_plus" column (does not overwrite quality_score)
    df = blend_with_quality(df)
    return df


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
st.caption("Quality Score Engine · Self-learning Outperformance Model · Multi-Factor Ranking · Red-Flag Detection")

uploaded = st.sidebar.file_uploader("Upload fundamentals panel (CSV)", type="csv")

# Optional bundled demo dataset for a public deployment
DEMO_PATH = "demo_data.csv"
demo_available = os.path.exists(DEMO_PATH)
use_demo = False
if demo_available and uploaded is None:
    use_demo = st.sidebar.button("▶️ Load demo data",
                                 help="Explore the app with a bundled sample universe.")

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
    st.info("Upload a CSV to begin"
            + (", click **Load demo data** in the sidebar," if demo_available else "")
            + " or download the sample template, fill it with your data, and upload it back.")
    with st.expander("📋 Preview the sample template & column guide", expanded=True):
        st.dataframe(sample_dataframe(), use_container_width=True)
        st.markdown("**Column reference**")
        st.dataframe(
            pd.DataFrame(
                [{"column": k, "description": v} for k, v in COLUMN_DOCS.items()]
            ),
            use_container_width=True, hide_index=True,
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
    st.code("python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv", language="bash")
    st.markdown(
        "3. Upload the generated `fundamentals.csv` here instead.\n\n"
        "**Note:** Yahoo provides the financial metrics, profitability, valuation, and the "
        "earnings/financial red-flag inputs. India-specific governance fields (promoter "
        "pledging/holding, insider trades, related-party, auditor) are **not** available from "
        "free sources and will stay blank — those red-flag rules simply won't fire unless you "
        "add that data yourself."
    )
    st.info(f"Detected {raw['ticker'].nunique()} unique tickers in your upload.")
    st.stop()

has_sector = "sector" in raw.columns and raw["sector"].notna().any()
use_sector = False
if has_sector:
    use_sector = st.sidebar.checkbox(
        "Rank within sector (peer-relative)", value=True,
        help="Compares each stock against others in its own sector rather than the "
             "whole universe. Small sectors (<5 stocks) fall back to overall ranking.",
    )
else:
    st.sidebar.caption("ℹ️ No `sector` column found — ranking against the full universe. "
                       "Re-run the fetcher to populate sectors automatically.")

# ---- Configurable category weights ----
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring weights")
preset = st.sidebar.selectbox(
    "Preset", ["Balanced (default)", "Value tilt", "Quality tilt",
               "Growth tilt", "Safety tilt", "Custom"],
    help="Quickly bias the score toward a style: Value favours cheap stocks, "
         "Quality favours profitability/strength, Growth favours expansion, "
         "Safety favours balance-sheet resilience. Pick Custom to set sliders yourself.")
PRESETS = {
    "Value tilt":   {"Valuation": 0.40, "Profitability": 0.20, "Financial Performance": 0.15,
                     "Financial Strength": 0.15, "Shareholder Metrics": 0.10},
    "Quality tilt": {"Profitability": 0.40, "Financial Strength": 0.25, "Valuation": 0.15,
                     "Financial Performance": 0.15, "Shareholder Metrics": 0.05},
    "Growth tilt":  {"Financial Performance": 0.45, "Profitability": 0.25, "Valuation": 0.15,
                     "Financial Strength": 0.10, "Shareholder Metrics": 0.05},
    "Safety tilt":  {"Financial Strength": 0.40, "Profitability": 0.25, "Valuation": 0.15,
                     "Shareholder Metrics": 0.10, "Financial Performance": 0.10},
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
        cat, 0.0, 1.0, float(base_weights.get(cat, DEFAULT_CATEGORY_WEIGHTS[cat])), 0.05,
        disabled=(preset != "Custom"),
        key=f"w_{cat}",
        help=CATEGORY_TOOLTIPS.get(cat),
    )
total_w = sum(weights.values())
if total_w > 0:
    st.sidebar.caption(
        "Effective mix: " + ", ".join(
            f"{cat.split()[0]} {weights[cat]/total_w*100:.0f}%" for cat in weights))
else:
    st.sidebar.warning("All weights are zero — set at least one above zero.")
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)

custom_config = build_config(weights)

data = enrich(raw, use_sector=use_sector, config=custom_config)

# Per-year scored panel for quality trends (each fiscal year scored vs its peers)
@st.cache_data(show_spinner=False)
def build_history(raw_df, use_sector, weights_tuple):
    cfg = build_config(dict(weights_tuple))
    d, _ = detect_red_flags(raw_df)
    d, _ = data_sanity_flags(d)
    d = data_completeness(d)
    gc = ("date", "sector") if (use_sector and "sector" in d.columns) else ("date",)
    return quality_history(d, group_cols=gc, config=cfg)

history_panel = build_history(raw, use_sector, tuple(sorted(weights.items())))

# ---- summary banner (oriented overview the moment data loads) ----
_n_stocks = data["ticker"].nunique()
_n_sectors = data["sector"].nunique() if "sector" in data.columns else 0
_avg_q = data["quality_score"].mean() if "quality_score" in data.columns else float("nan")
_warned = int(data["data_warning"].sum()) if "data_warning" in data.columns else 0
b1, b2, b3, b4 = st.columns(4)
b1.metric("Universe", f"{_n_stocks:,} stocks", help="Number of unique stocks loaded and scored.")
b2.metric("Sectors", _n_sectors if _n_sectors else "—", help="Distinct sectors represented in the data.")
b3.metric("Avg quality", f"{_avg_q:.1f}" if pd.notna(_avg_q) else "—",
          help="Mean quality score across the universe (0–100).")
b4.metric("Data warnings", _warned,
          help="Stocks with distorted/implausible figures flagged for manual review.")
st.divider()

tab1, tab2, tab_compare, tab_sector, tab3 = st.tabs(
    ["Single Stock Report", "Universe Ranking", "Compare",
     "Sector Overview", "Train Model"])

# ----------------------------------------------------------------- Tab 1
with tab1:
    tickers = sorted(data["ticker"].unique())
    tk = st.selectbox("Select stock", tickers,
                      help="Choose a stock to see its quality score, category breakdown, "
                           "trend, and red flags.")
    latest = data[data["ticker"] == tk].sort_values("date").iloc[-1]

    qscore = float(latest["quality_score"]) if pd.notna(latest["quality_score"]) else 0.0

    def quality_gauge_svg(score):
        """Semicircular gauge SVG for a 0-100 score."""
        score = max(0.0, min(100.0, score))
        if score >= 65:
            color = "#1D9E75"
        elif score >= 50:
            color = "#378ADD"
        elif score >= 35:
            color = "#E0A82E"
        else:
            color = "#E24B4A"
        import math
        # semicircle from 180° (left) to 0° (right)
        ang = math.pi * (1 - score / 100.0)
        cx, cy, r = 100, 100, 80
        x = cx + r * math.cos(ang)
        y = cy - r * math.sin(ang)
        large = 0  # always minor arc for a semicircle segment
        bg = ("M 20 100 A 80 80 0 0 1 180 100")
        fg = f"M 20 100 A 80 80 0 {large} 1 {x:.1f} {y:.1f}"
        return f"""
        <svg viewBox="0 0 200 130" width="220" height="143">
          <path d="{bg}" fill="none" stroke="#3a3a3a" stroke-width="14" stroke-linecap="round"/>
          <path d="{fg}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"/>
          <text x="100" y="92" text-anchor="middle" font-size="36" font-weight="600" fill="{color}">{score:.0f}</text>
          <text x="100" y="115" text-anchor="middle" font-size="14" fill="#888">{score_label(score)}</text>
        </svg>"""

    gcol, mcol = st.columns([1, 2])
    with gcol:
        st.markdown(quality_gauge_svg(qscore), unsafe_allow_html=True)
    with mcol:
        c2, c3 = st.columns(2)
        c2.metric("Red Flags", int(latest["red_flag_count"]),
                  help=CONCEPT_TOOLTIPS["red_flags"])
        if "outperform_proba" in data.columns:
            c3.metric("Outperform Prob.", f"{latest['outperform_proba']*100:.0f}%",
                      help=CONCEPT_TOOLTIPS["outperform_proba"])
        else:
            c3.metric("Quality Score", f"{qscore:.1f}", score_label(qscore),
                      help=CONCEPT_TOOLTIPS["quality_score"])
        sector_txt = latest.get("sector", "Unknown") if "sector" in latest else "Unknown"
        scored_vs = latest.get("scored_vs", "all stocks (by date)")
        st.caption(f"Sector: **{sector_txt}** · Scored vs **{scored_vs}**")

    # ---- Institutional scores: Piotroski F & Altman Z ----
    fcol, zcol = st.columns(2)
    with fcol:
        fs = latest.get("f_score")
        ftu = int(latest.get("f_tests_used", 0)) if pd.notna(latest.get("f_tests_used", 0)) else 0
        if pd.notna(fs) and ftu > 0:
            st.metric(f"Piotroski F-Score", f"{int(fs)} / 9",
                      f_score_band(fs, ftu),
                      help="Nine binary financial-health tests (profitability, leverage/"
                           "liquidity, efficiency). 7–9 strong, 4–6 moderate, 0–3 weak. "
                           "Institutional investors use it to spot improving fundamentals.")
            if ftu < 9:
                st.caption(f"⚠️ Computed from {ftu}/9 tests "
                           "(missing inputs — re-fetch for total-assets-based tests).")
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
            st.metric("Piotroski F-Score", "N/A",
                      help="Needs at least two years of data with net income, cash flow, "
                           "margins, etc.")
    with zcol:
        zs = latest.get("z_score")
        zb = z_band(zs)
        if pd.notna(zs):
            emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(zb, "")
            st.metric("Altman Z-Score", f"{zs:.2f}", f"{emoji} {zb}",
                      help="Bankruptcy-risk score. 🟢 > 3 safe · 🟡 1.8–3 grey zone · "
                           "🔴 < 1.8 distress. Especially useful for small caps and cyclicals.")
            st.caption(Z_BAND_TEXT.get(zb, ""))
        else:
            st.metric("Altman Z-Score", "N/A",
                      help="Needs balance-sheet inputs (total assets/liabilities, retained "
                           "earnings, EBIT, market cap). Re-fetch with the updated fetcher to enable.")
            st.caption("Re-run the fetcher to capture the balance-sheet fields this needs.")

    # ---- Data quality ----
    present = int(latest.get("data_fields_present", 0))
    total = int(latest.get("data_fields_total", 0))
    completeness = float(latest.get("data_completeness", 0.0))
    warnings = latest.get("data_warnings", []) if "data_warnings" in latest else []
    if total:
        if completeness < 0.5:
            st.warning(f"⚠️ Sparse data: only {present}/{total} core metrics available — "
                       "treat this score with low confidence.")
        else:
            st.caption(f"Data completeness: {present}/{total} core metrics populated "
                       f"({completeness*100:.0f}%).")
    if warnings:
        st.warning("⚠️ **Data reliability warnings** — verify against filings before trusting this score:")
        for wkey in warnings:
            st.write("• " + WARNING_TEXT.get(wkey, wkey))


    # ---- Why this score? (explainability) ----
    st.subheader("Why this score?")
    ex = explain_score(latest, config=custom_config)
    st.write(explanation_sentence(ex, ticker=tk))

    if ex["drivers"]:
        col_up, col_down = st.columns(2)
        expl_cfg = {
            "Metric": st.column_config.TextColumn("Metric", help="The fundamental metric (hover the glossary below for definitions)."),
            "Percentile": st.column_config.NumberColumn("Percentile", format="%.0f",
                help="How this stock ranks vs its comparison group, 0–100. Higher = better; valuation metrics are inverted so cheaper ranks higher."),
            "Value": st.column_config.NumberColumn("Value", help="The raw underlying value of the metric."),
        }
        with col_up:
            st.markdown("**🟢 Top strengths** (vs peers)")
            up_df = pd.DataFrame(ex["drivers"])[["metric", "percentile", "raw_value"]]
            up_df.columns = ["Metric", "Percentile", "Value"]
            st.dataframe(up_df, hide_index=True, use_container_width=True, column_config=expl_cfg)
        with col_down:
            st.markdown("**🔴 Weakest areas** (vs peers)")
            dn_df = pd.DataFrame(ex["drags"])[["metric", "percentile", "raw_value"]]
            dn_df.columns = ["Metric", "Percentile", "Value"]
            st.dataframe(dn_df, hide_index=True, use_container_width=True, column_config=expl_cfg)
        st.caption("Percentile = how this stock ranks against its comparison group "
                   "(100 = best). Valuation metrics are inverted so cheaper = higher percentile.")
        with st.expander("📖 Metric glossary — what each metric means"):
            from src.quality_score import METRIC_LABELS
            gloss = pd.DataFrame(
                [{"Metric": METRIC_LABELS.get(k, k), "Meaning": v}
                 for k, v in METRIC_TOOLTIPS.items()])
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
                st.metric(CATEGORY_SHORT[c], f"{vv:.0f}",
                          help=CATEGORY_TOOLTIPS.get(c))
                st.progress(min(1.0, vv / 100.0))
    else:
        st.caption("No category scores available.")

    # ---- Quality trend over time ----
    st.subheader("Quality Trend")
    trend = ticker_trend(history_panel, tk)
    if trend["direction"] == "insufficient data":
        npts = len(trend["points"])
        if npts == 0:
            st.caption("Not enough complete fiscal years to chart a trend "
                       "(this stock's available years fall below the data-completeness "
                       "bar — common for banks/financials with sparse Yahoo data).")
        else:
            st.caption("Only one scored fiscal year available — need at least two "
                       "to show a trend.")
    else:
        arrow = {"improving": "📈", "declining": "📉", "stable": "➡️"}[trend["direction"]]
        st.markdown(f"{arrow} Quality is **{trend['direction']}** — "
                    f"{trend['delta']:+.1f} points from "
                    f"{trend['points'][0]['date'].year} to {trend['points'][-1]['date'].year}.")
        tdf = pd.DataFrame([
            {"Year": str(p["date"].year), "Quality": round(p["quality_score"], 1)}
            for p in trend["points"]
        ]).set_index("Year")
        st.line_chart(tdf, height=220)
        if trend["category_deltas"]:
            moves = sorted(trend["category_deltas"].items(),
                           key=lambda kv: abs(kv[1]), reverse=True)
            bits = ", ".join(f"{cat.split()[0]} {d:+.0f}" for cat, d in moves if abs(d) >= 1)
            if bits:
                st.caption("Biggest category moves: " + bits + " (points, latest vs earliest).")
        st.caption("Each year is scored against that year's peers, so the trend "
                   "reflects changing fundamentals, not market timing.")
        if trend.get("caveat"):
            st.caption("⚠️ " + trend["caveat"])

    st.subheader("Red Flags")
    if latest["red_flags"]:
        for f in latest["red_flags"]:
            st.error(REASON_TEXT.get(f, f))
    else:
        st.success("No red flags detected.")

# ----------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Multi-Factor Ranking")
    st.caption("Each stock's most recent fiscal-year figures, ranked together.")
    wq = st.slider("Weight: Quality Score", 0.0, 1.0, 0.5, 0.05,
                   help="How much the composite ranking leans on the fundamental quality "
                        "score vs the ML outperformance probability. At 1.0 the ranking is "
                        "pure quality; lower values blend in the model (only meaningful once trained).")
    ranked = rank_universe(data, w_quality=wq, w_ml=1 - wq, as_of_date=None)

    # ---- summary cards ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks ranked", len(ranked), help="Total stocks in the current ranking.")
    m2.metric("Top score", f"{ranked['composite_score'].max():.1f}",
              help=CONCEPT_TOOLTIPS["composite_score"])
    m3.metric("Median", f"{ranked['composite_score'].median():.1f}",
              help="Median composite score — the middle of the pack.")
    if "data_warning" in ranked.columns:
        m4.metric("Data warnings", int(ranked["data_warning"].sum()),
                  help="Stocks with distorted/implausible figures — verify manually.")
    else:
        m4.metric("Clean (0 flags)", int((ranked['red_flag_count'] == 0).sum()))

    # ---- score distribution histogram ----
    with st.expander("📊 Score distribution", expanded=False):
        import numpy as _np
        vals = ranked["composite_score"].dropna()
        if len(vals):
            bins = _np.arange(0, 105, 5)
            counts, edges = _np.histogram(vals, bins=bins)
            hist_df = pd.DataFrame({
                "count": counts,
            }, index=[f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))])
            st.bar_chart(hist_df, height=200)
            st.caption(f"Distribution of composite scores across {len(vals)} stocks "
                       f"(median {vals.median():.1f}). Most stocks cluster mid-range "
                       "because scores are relative percentiles.")

    # ---- controls (row 1: search + sector; row 2: filters + sort + limit) ----
    has_sector_col = "sector" in ranked.columns
    r1c1, r1c2 = st.columns([2, 2])
    search = r1c1.text_input("Search ticker", "", placeholder="e.g. TCS").strip().upper()
    if has_sector_col:
        sectors = ["All sectors"] + sorted(ranked["sector"].dropna().unique().tolist())
        sector_pick = r1c2.selectbox("Sector", sectors,
            help="Filter the ranking to one sector. With sector scoring on, stocks are ranked against sector peers.")
    else:
        sector_pick = "All sectors"

    r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1.3, 1])
    flag_filter = r2c1.selectbox("Flags", ["All", "No red flags", "Has red flags"],
        help="Filter by forensic red flags. 'No red flags' shows only clean names.")
    rel_filter = r2c2.selectbox("Data", ["All", "Reliable only", "Warnings only"],
                                help="'Reliable only' hides stocks with data-quality warnings.")
    sort_opts = {
        "Rank": ("rank", True),
        "Quality (raw)": ("quality_score", False),
        "ROE": ("roe", False),
        "P/E (low→high)": ("pe", True),
        "Debt/Equity (low→high)": ("debt_to_equity", True),
        "Revenue growth": ("revenue_growth", False),
    }
    sort_by = r2c3.selectbox("Sort by", list(sort_opts.keys()),
        help="Reorder the table by any metric. Rank uses the composite score.")
    top_n = r2c4.selectbox("Show", ["Top 25", "Top 50", "Top 100", "All"], index=1,
        help="Limit how many rows render. Fewer rows = faster on a 2,000+ stock universe.")

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

    # ---- build display frame ----
    candidate_cols = ["rank", "ticker"]
    if has_sector_col:
        candidate_cols.append("sector")
    candidate_cols += ["composite_score", "quality_score"]
    if "outperform_proba" in view.columns:
        candidate_cols.append("outperform_proba")
    candidate_cols += ["roe", "pe", "debt_to_equity", "net_margin",
                       "revenue_growth", "f_score", "z_band", "red_flag_count",
                       "data_fields_present", "data_warning_count"]
    show = [c for c in candidate_cols if c in view.columns]
    disp = view[show].reset_index(drop=True).copy()
    disp["ticker"] = disp["ticker"].str.replace(".NS", "", regex=False)
    if "revenue_growth" in disp.columns:
        disp["revenue_growth"] = disp["revenue_growth"] * 100.0
    if "f_score" in disp.columns:
        disp["f_score"] = disp["f_score"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "–")
    if "z_band" in disp.columns:
        disp["z_band"] = disp["z_band"].map(
            {"Green": "🟢", "Yellow": "🟡", "Red": "🔴", "N/A": "–"}).fillna("–")
    if "data_fields_present" in disp.columns and "data_fields_total" in view.columns:
        tot = int(view["data_fields_total"].iloc[0]) if len(view) else 0
        disp["data_fields_present"] = disp["data_fields_present"].apply(
            lambda x: f"{int(x)}/{tot}" if pd.notna(x) else "–")
    # flag/warning badges as small text indicators
    if "red_flag_count" in disp.columns:
        disp["red_flag_count"] = disp["red_flag_count"].apply(
            lambda n: "—" if (pd.isna(n) or n == 0) else "🚩" * int(min(n, 3)))
    if "data_warning_count" in disp.columns:
        disp["data_warning_count"] = disp["data_warning_count"].apply(
            lambda n: "" if (pd.isna(n) or n == 0) else "⚠️")

    # ---- column config: progress bars for scores ----
    colcfg = {
        "rank": st.column_config.NumberColumn("#", width="small", help="Position in the ranking."),
        "ticker": st.column_config.TextColumn("Ticker", width="small", help="Stock symbol (NSE)."),
        "composite_score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.1f",
            help=CONCEPT_TOOLTIPS["composite_score"]),
        "quality_score": st.column_config.ProgressColumn(
            "Quality", min_value=0, max_value=100, format="%.1f",
            help=CONCEPT_TOOLTIPS["quality_score"]),
        "roe": st.column_config.NumberColumn("ROE", format="%.1f", help=METRIC_TOOLTIPS["roe"]),
        "pe": st.column_config.NumberColumn("P/E", format="%.1f", help=METRIC_TOOLTIPS["pe"]),
        "debt_to_equity": st.column_config.NumberColumn("D/E", format="%.2f", help=METRIC_TOOLTIPS["debt_to_equity"]),
        "net_margin": st.column_config.NumberColumn("Net%", format="%.1f", help=METRIC_TOOLTIPS["net_margin"]),
        "revenue_growth": st.column_config.NumberColumn("Rev gr%", format="%.1f", help=METRIC_TOOLTIPS["revenue_growth"]),
        "f_score": st.column_config.TextColumn("F", width="small",
            help="Piotroski F-Score (0–9). Financial-health checklist; higher = stronger fundamentals."),
        "z_band": st.column_config.TextColumn("Z", width="small",
            help="Altman Z-Score bankruptcy risk: 🟢 safe · 🟡 grey zone · 🔴 distress."),
        "red_flag_count": st.column_config.TextColumn("Flags", width="small", help=CONCEPT_TOOLTIPS["red_flags"]),
        "data_fields_present": st.column_config.TextColumn("Data", width="small", help=CONCEPT_TOOLTIPS["data_completeness"]),
        "data_warning_count": st.column_config.TextColumn("⚠", width="small", help=CONCEPT_TOOLTIPS["data_warning"]),
    }
    if has_sector_col:
        colcfg["sector"] = st.column_config.TextColumn("Sector", width="medium",
                                                       help=CONCEPT_TOOLTIPS["scored_vs"])
    if "outperform_proba" in disp.columns:
        colcfg["outperform_proba"] = st.column_config.NumberColumn(
            "Outperf.", format="%.0f%%", help=CONCEPT_TOOLTIPS["outperform_proba"])

    showing_txt = (f"Showing {len(disp)} of {total_matched} matched"
                   + (f" ({len(ranked)} total)" if total_matched != len(ranked) else ""))
    st.caption(showing_txt + " · Score = quality minus red-flag penalty.")
    st.dataframe(disp, use_container_width=True, hide_index=True,
                 height=min(620, 48 + 35 * len(disp)),
                 column_config=colcfg)

    st.download_button(
        "⬇️ Download this ranking (CSV)",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name="stock_rankings.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------- Compare tab
with tab_compare:
    st.subheader("Compare Stocks")
    st.caption("Pick 2–5 stocks to see their scores, categories, and key metrics side by side.")
    all_tickers = sorted(data["ticker"].str.replace(".NS", "", regex=False).unique())
    picks = st.multiselect("Stocks to compare", all_tickers, max_selections=5,
                           help="Pick up to 5 stocks to see their category radars and "
                                "key metrics side by side.",
                           default=all_tickers[:3] if len(all_tickers) >= 3 else all_tickers)
    if len(picks) < 2:
        st.info("Select at least two stocks to compare.")
    else:
        # map back to full tickers
        sel = data[data["ticker"].str.replace(".NS", "", regex=False).isin(picks)].copy()
        sel = sel.sort_values("date").groupby("ticker", as_index=False).tail(1)
        sel["disp_ticker"] = sel["ticker"].str.replace(".NS", "", regex=False)

        # radar overlay-style: one radar per stock in columns
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
                st.markdown(category_radar_svg(cmap, color=palette[i % len(palette)], size=210),
                            unsafe_allow_html=True)
                q = row.get("quality_score")
                st.metric("Quality", f"{q:.1f}" if pd.notna(q) else "—",
                          score_label(q) if pd.notna(q) else "",
                          help=CONCEPT_TOOLTIPS["quality_score"])

        # side-by-side metric table
        st.markdown("**Metrics side by side**")
        metric_rows = {
            "Quality score": "quality_score", "Red flags": "red_flag_count",
            "ROE %": "roe", "ROCE %": "roce", "Net margin %": "net_margin",
            "Oper. margin %": "operating_margin", "D/E": "debt_to_equity",
            "Interest cover": "interest_coverage", "Current ratio": "current_ratio",
            "P/E": "pe", "P/B": "pb", "EV/EBITDA": "ev_ebitda",
            "Rev growth %": "revenue_growth", "Dividend yield %": "dividend_yield",
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
                    colvals[label] = f"{float(v)*100:.1f}"
                elif key == "red_flag_count":
                    colvals[label] = str(int(v))
                else:
                    colvals[label] = f"{float(v):.2f}"
            table[row["disp_ticker"]] = colvals
        cmp_df = pd.DataFrame(table)
        st.dataframe(cmp_df, use_container_width=True)

        # red flags per stock
        st.markdown("**Red flags**")
        for _, row in sel.iterrows():
            rf = row.get("red_flags", [])
            if isinstance(rf, list) and rf:
                st.write(f"**{row['disp_ticker']}**: "
                         + "; ".join(REASON_TEXT.get(f, f) for f in rf))
            else:
                st.write(f"**{row['disp_ticker']}**: none")

# ----------------------------------------------------------------- Sector tab
with tab_sector:
    st.subheader("Sector Overview")
    if "sector" not in data.columns or data["sector"].dropna().nunique() < 2:
        st.info("No sector data available. Re-run the fetcher to populate the "
                "`sector` column, then sector analytics will appear here.")
    else:
        sec_ranked = rank_universe(data, w_quality=0.5, w_ml=0.5, as_of_date=None)
        grp = sec_ranked.groupby("sector")
        summary = pd.DataFrame({
            "Stocks": grp.size(),
            "Avg quality": grp["quality_score"].mean(),
            "Median quality": grp["quality_score"].median(),
            "Best score": grp["quality_score"].max(),
            "Avg red flags": grp["red_flag_count"].mean(),
        }).reset_index()
        # best stock name per sector
        idx = grp["quality_score"].idxmax()
        best = sec_ranked.loc[idx, ["sector", "ticker"]].copy()
        best["ticker"] = best["ticker"].str.replace(".NS", "", regex=False)
        summary = summary.merge(best, on="sector", how="left").rename(
            columns={"sector": "Sector", "ticker": "Top stock"})
        summary = summary.sort_values("Avg quality", ascending=False).reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sectors", len(summary))
        c1_best = summary.iloc[0]
        c2.metric("Strongest sector", c1_best["Sector"], f"avg {c1_best['Avg quality']:.1f}")
        c3.metric("Largest sector",
                  summary.loc[summary["Stocks"].idxmax(), "Sector"],
                  f"{int(summary['Stocks'].max())} stocks")

        st.markdown("**Average quality score by sector**")
        chart_df = summary.set_index("Sector")["Avg quality"]
        st.bar_chart(chart_df, height=300)

        st.markdown("**Sector detail**")
        st.dataframe(
            summary, use_container_width=True, hide_index=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Stocks": st.column_config.NumberColumn("Stocks", width="small"),
                "Avg quality": st.column_config.ProgressColumn(
                    "Avg quality", min_value=0, max_value=100, format="%.1f"),
                "Median quality": st.column_config.NumberColumn("Median", format="%.1f"),
                "Best score": st.column_config.NumberColumn("Best", format="%.1f"),
                "Avg red flags": st.column_config.NumberColumn("Avg flags", format="%.2f"),
                "Top stock": st.column_config.TextColumn("Top stock", width="small"),
            })
        st.caption("Tip: use the Sector filter in the Universe Ranking tab to drill "
                   "into any sector's individual stocks.")


# ----------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Train Global Outperformance Model")
    st.info("⚠️ This needs `fwd_return` + `bench_fwd_return` columns (the realized "
            "forward returns). With current-dated fundamentals these labels don't yet "
            "exist, so the model can't be meaningfully trained until enough time has "
            "passed for forward windows to elapse. On the cloud, a trained model lives "
            "only for the current session (it isn't persisted across redeploys).")
    from src.model import HAS_LGBM, HAS_XGB
    algo_opts = ["randomforest"]
    if HAS_LGBM:
        algo_opts.insert(0, "lightgbm")
    if HAS_XGB:
        algo_opts.append("xgboost")
    kind = st.selectbox("Algorithm", algo_opts,
                        help="Which model to train. RandomForest is always available; "
                             "LightGBM/XGBoost appear only if installed. All predict the "
                             "probability of beating the benchmark.")
    if not (HAS_LGBM or HAS_XGB):
        st.caption("ℹ️ LightGBM/XGBoost aren't installed in this deployment — using "
                   "scikit-learn RandomForest. (They're excluded from requirements.txt "
                   "to keep the cloud build fast.)")
    if st.button("Train"):
        if not {"fwd_return", "bench_fwd_return"}.issubset(data.columns):
            st.error("Training needs `fwd_return` and `bench_fwd_return` columns.")
        else:
            train_df = make_label(data)
            feats = [c for c in FEATURE_COLS if c in train_df.columns]
            model, report = train_outperformance_model(train_df, feats, kind=kind)
            joblib.dump({"model": model, "features": feats}, MODEL_PATH)
            st.success("Model trained and saved.")
            for split in ("valid", "test"):
                if split in report:
                    st.write(f"**{split.upper()}** — AUC: {report[split]['auc']:.3f} "
                             f"(n={report[split]['n']}, base={report[split]['base_rate']:.2f})")
            if "feature_importance" in report:
                st.bar_chart(report["feature_importance"].head(15))

    if os.path.exists(MODEL_PATH) and st.button("Score universe with saved model"):
        bundle = joblib.load(MODEL_PATH)
        data["outperform_proba"] = predict_proba(bundle["model"], data, bundle["features"])
        st.session_state["scored"] = True
        st.success("Universe scored. Switch to Ranking tab.")
