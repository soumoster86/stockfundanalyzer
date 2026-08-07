"""
Shared visual language for the Streamlit UI
-------------------------------------------
Colored risk/quality badges, score chips, section chrome, and global CSS.
Keep colors aligned with brand primary #1D9E75 and traffic-light bands.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

# Brand + traffic light palette
GREEN = "#1D9E75"
GREEN_SOFT = "rgba(29, 158, 117, 0.18)"
GREEN_BORDER = "#26B583"
BLUE = "#378ADD"
BLUE_SOFT = "rgba(55, 138, 221, 0.18)"
YELLOW = "#E0A82E"
YELLOW_SOFT = "rgba(224, 168, 46, 0.20)"
RED = "#E24B4A"
RED_SOFT = "rgba(226, 75, 74, 0.18)"
MUTED = "#8B949E"
MUTED_SOFT = "rgba(139, 148, 158, 0.15)"
SURFACE = "#161B22"
SURFACE_2 = "#0D1117"
BORDER = "#30363D"
TEXT = "#E6EDF3"

BAND_STYLE = {
    "Green": (GREEN, GREEN_SOFT, "Safe"),
    "Yellow": (YELLOW, YELLOW_SOFT, "Watch"),
    "Red": (RED, RED_SOFT, "Risk"),
    "N/A": (MUTED, MUTED_SOFT, "n/a"),
    "Strong": (GREEN, GREEN_SOFT, "Strong"),
    "Moderate": (BLUE, BLUE_SOFT, "Moderate"),
    "Weak": (RED, RED_SOFT, "Weak"),
    "Excellent": (GREEN, GREEN_SOFT, "Excellent"),
    "Average": (BLUE, BLUE_SOFT, "Average"),
    "Poor": (RED, RED_SOFT, "Poor"),
}

BAND_EMOJI = {
    "Green": "🟢",
    "Yellow": "🟡",
    "Red": "🔴",
    "N/A": "⚪",
    "Strong": "🟢",
    "Moderate": "🔵",
    "Weak": "🔴",
}


def inject_global_css() -> None:
    """Once-per-session visual polish for Streamlit chrome."""
    # Bump version when CSS rules change so open sessions pick up new styles
    _CSS_VER = 4
    if st.session_state.get("_css_ver") == _CSS_VER:
        return
    st.session_state["_css_ver"] = _CSS_VER
    st.markdown(
        f"""
<style>
  /* ---- Base shell ---- */
  .stApp {{
    background: radial-gradient(1200px 600px at 10% -10%, rgba(29,158,117,0.07), transparent 55%),
                radial-gradient(900px 500px at 100% 0%, rgba(55,138,221,0.05), transparent 50%),
                {SURFACE_2} !important;
  }}
  .block-container {{
    padding-top: 1.1rem;
    padding-bottom: 3.5rem;
    max-width: 1380px;
  }}
  h1, h2, h3 {{
    letter-spacing: -0.025em;
    font-weight: 700 !important;
  }}
  .stMarkdown, .stCaption, p {{
    line-height: 1.45;
  }}

  /* ---- Metrics ---- */
  div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, #1a212b 0%, {SURFACE} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.18);
  }}
  div[data-testid="stMetric"] label {{ color: {MUTED} !important; font-size: 0.8rem !important; }}
  div[data-testid="stMetricValue"] {{ font-weight: 700; letter-spacing: -0.02em; }}

  /* =========================================================
     Native st.pills — dark capsules (Mode / Page nav)
     Streamlit 1.33+; radios cannot be restyled reliably in 1.56
     ========================================================= */
  div[data-testid="stPills"] {{
    margin-bottom: 0.35rem !important;
  }}
  div[data-testid="stPills"] > label {{
    font-size: 0.8rem !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
  }}
  /* Pill button row */
  div[data-testid="stPills"] [role="group"],
  div[data-testid="stPills"] [data-baseweb="button-group"],
  div[data-testid="stPills"] > div > div {{
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.45rem !important;
    row-gap: 0.45rem !important;
  }}
  /* Individual pill (button) */
  div[data-testid="stPills"] button,
  div[data-testid="stPills"] [data-baseweb="button"],
  div[data-testid="stPills"] [role="button"] {{
    background: #12171f !important;
    border: 1px solid #2a3340 !important;
    border-radius: 999px !important;
    color: {TEXT} !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    min-height: 2.4rem !important;
    padding: 0.4rem 1.15rem !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset,
                0 4px 14px rgba(0,0,0,0.22) !important;
    transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
  }}
  div[data-testid="stPills"] button:hover,
  div[data-testid="stPills"] [data-baseweb="button"]:hover {{
    border-color: #3d4a5c !important;
    background: #161c26 !important;
  }}
  /* Selected pill */
  div[data-testid="stPills"] button[aria-pressed="true"],
  div[data-testid="stPills"] button[kind="primary"],
  div[data-testid="stPills"] [aria-checked="true"],
  div[data-testid="stPills"] button[data-selected="true"] {{
    background: linear-gradient(180deg, rgba(29,158,117,0.22), rgba(29,158,117,0.10)) !important;
    border-color: {GREEN} !important;
    color: {TEXT} !important;
    box-shadow: 0 0 0 1px rgba(29,158,117,0.28),
                0 4px 16px rgba(29,158,117,0.14) !important;
  }}

  /* Fallback: if any st.radio remains, try capsule labels */
  div[data-testid="stRadio"] div[role="radiogroup"] {{
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
  }}
  div[data-testid="stRadio"] label[data-baseweb="radio"],
  div[data-testid="stRadio"] > div > label {{
    background: #12171f !important;
    border: 1px solid #2a3340 !important;
    border-radius: 999px !important;
    padding: 0.4rem 1rem !important;
  }}

  /* ---- Tabs (Report sub-tabs etc.) ---- */
  button[data-baseweb="tab"] {{
    border-radius: 999px !important;
    padding: 0.4rem 0.95rem !important;
    font-weight: 600 !important;
    color: {MUTED} !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    background: {GREEN_SOFT} !important;
    color: {GREEN_BORDER} !important;
  }}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.35rem !important;
    background: transparent !important;
    border-bottom: 1px solid {BORDER} !important;
    padding-bottom: 0.35rem !important;
  }}
  div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    display: none !important;
  }}
  div[data-testid="stTabs"] [data-baseweb="tab-border"] {{
    display: none !important;
  }}

  /* ---- Primary / secondary buttons ---- */
  div[data-testid="stButton"] > button {{
    border-radius: 999px !important;
    border: 1px solid {BORDER} !important;
    font-weight: 600 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
  }}
  div[data-testid="stButton"] > button:hover {{
    border-color: {GREEN} !important;
    transform: translateY(-1px);
  }}
  div[data-testid="stButton"] > button[kind="primary"],
  div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
    background: linear-gradient(180deg, #22b885 0%, {GREEN} 100%) !important;
    border-color: {GREEN_BORDER} !important;
    color: #04120c !important;
  }}
  div[data-testid="stDownloadButton"] > button {{
    border-radius: 999px !important;
    border: 1px solid {BORDER} !important;
  }}

  /* ---- Expanders ---- */
  div[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    overflow: hidden;
    margin-bottom: 0.55rem !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
  }}
  div[data-testid="stExpander"] details summary {{
    font-weight: 600 !important;
  }}

  /* ---- Inputs / select ---- */
  div[data-testid="stTextInput"] input,
  div[data-testid="stNumberInput"] input,
  div[data-baseweb="select"] > div,
  div[data-testid="stSelectbox"] > div > div {{
    border-radius: 12px !important;
    border-color: {BORDER} !important;
    background-color: #12171f !important;
  }}
  div[data-testid="stTextInput"] input:focus {{
    border-color: {GREEN} !important;
    box-shadow: 0 0 0 1px {GREEN}55 !important;
  }}

  /* ---- Sidebar ---- */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #11161e 0%, #0c1016 100%) !important;
    border-right: 1px solid {BORDER} !important;
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.25rem;
  }}

  /* ---- Alerts / status ---- */
  div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
  }}
  div[data-testid="stSuccess"] {{
    background: rgba(29,158,117,0.12) !important;
    border-color: rgba(29,158,117,0.4) !important;
  }}
  div[data-testid="stInfo"] {{
    background: rgba(55,138,221,0.10) !important;
    border-color: rgba(55,138,221,0.35) !important;
  }}

  /* ---- Dataframes ---- */
  div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
  }}

  /* ---- Dividers ---- */
  hr {{
    border-color: {BORDER} !important;
    opacity: 0.85;
  }}

  /* Section card */
  .sfa-card {{
    background: linear-gradient(180deg, #1a212b 0%, {SURFACE} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
  }}
  .sfa-card-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED};
    margin-bottom: 0.35rem;
  }}
  .sfa-card-value {{
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.15;
    color: {TEXT};
  }}
  .sfa-card-sub {{
    font-size: 0.85rem;
    color: {MUTED};
    margin-top: 0.25rem;
  }}

  /* Badges / pills */
  .sfa-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
    line-height: 1.3;
    white-space: nowrap;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
  }}
  .sfa-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.35rem 0 0.6rem 0;
  }}
  .sfa-chip {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 8px;
    font-size: 0.8rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    border: 1px solid {BORDER};
    background: {SURFACE_2};
  }}
  .sfa-chip-ok {{ border-color: {GREEN}; background: {GREEN_SOFT}; color: {GREEN_BORDER}; }}
  .sfa-chip-bad {{ border-color: {RED}; background: {RED_SOFT}; color: #ff8a88; }}
  .sfa-chip-na {{ color: {MUTED}; }}

  /* ---- Piotroski F-Score breakdown ---- */
  .sfa-fbd {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 0.85rem 1rem 1rem 1rem;
    margin-top: 0.35rem;
  }}
  .sfa-fbd-head {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
  }}
  .sfa-fbd-score {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {TEXT};
  }}
  .sfa-fbd-score span {{ color: {MUTED}; font-weight: 500; font-size: 0.95rem; }}
  .sfa-fbd-bar {{
    flex: 1;
    min-width: 120px;
    height: 10px;
    background: {BORDER};
    border-radius: 6px;
    overflow: hidden;
  }}
  .sfa-fbd-bar > i {{
    display: block;
    height: 100%;
    border-radius: 6px;
  }}
  .sfa-fbd-dots {{
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
    margin-bottom: 0.85rem;
  }}
  .sfa-fbd-dot {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    border: 2px solid transparent;
  }}
  .sfa-fbd-dot.ok {{
    background: {GREEN_SOFT};
    border-color: {GREEN};
    color: {GREEN_BORDER};
  }}
  .sfa-fbd-dot.bad {{
    background: {RED_SOFT};
    border-color: {RED};
    color: #ff8a88;
  }}
  .sfa-fbd-dot.na {{
    background: {MUTED_SOFT};
    border-color: {BORDER};
    color: {MUTED};
  }}
  .sfa-fbd-cat {{
    margin-top: 0.65rem;
  }}
  .sfa-fbd-cat-title {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED};
    margin-bottom: 0.35rem;
    font-weight: 600;
  }}
  .sfa-fbd-row {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0.55rem;
    border-radius: 10px;
    border: 1px solid {BORDER};
    margin-bottom: 0.35rem;
    background: {SURFACE_2};
  }}
  .sfa-fbd-row.ok {{
    border-color: rgba(29, 158, 117, 0.45);
    background: {GREEN_SOFT};
  }}
  .sfa-fbd-row.bad {{
    border-color: rgba(226, 75, 74, 0.4);
    background: {RED_SOFT};
  }}
  .sfa-fbd-row.na {{
    opacity: 0.75;
  }}
  .sfa-fbd-icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    font-weight: 700;
  }}
  .sfa-fbd-row.ok .sfa-fbd-icon {{ background: {GREEN}; color: #0D1117; }}
  .sfa-fbd-row.bad .sfa-fbd-icon {{ background: {RED}; color: #fff; }}
  .sfa-fbd-row.na .sfa-fbd-icon {{ background: {BORDER}; color: {MUTED}; }}
  .sfa-fbd-label {{
    flex: 1;
    font-size: 0.88rem;
    color: {TEXT};
    line-height: 1.3;
  }}
  .sfa-fbd-status {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    flex-shrink: 0;
  }}
  .sfa-fbd-row.ok .sfa-fbd-status {{ color: {GREEN_BORDER}; }}
  .sfa-fbd-row.bad .sfa-fbd-status {{ color: #ff8a88; }}
  .sfa-fbd-row.na .sfa-fbd-status {{ color: {MUTED}; }}
  .sfa-fbd-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 0.55rem;
    font-size: 0.75rem;
    color: {MUTED};
  }}

  .sfa-section {{
    margin-top: 0.55rem;
    margin-bottom: 0.45rem;
    padding: 0.15rem 0 0.4rem 0;
    border-bottom: 1px solid {BORDER};
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {TEXT};
  }}
  .sfa-muted {{ color: {MUTED}; font-size: 0.88rem; }}
  .sfa-flag {{
    border-left: 3px solid {RED};
    background: {RED_SOFT};
    padding: 0.55rem 0.75rem;
    border-radius: 0 12px 12px 0;
    margin: 0.35rem 0;
    font-size: 0.9rem;
  }}
  .sfa-ok-banner {{
    border-left: 3px solid {GREEN};
    background: {GREEN_SOFT};
    padding: 0.55rem 0.75rem;
    border-radius: 0 12px 12px 0;
    margin: 0.35rem 0;
  }}
  .sfa-peer-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    background: {SURFACE};
    border-radius: 12px;
    overflow: hidden;
  }}
  .sfa-peer-table th {{
    text-align: left;
    color: {MUTED};
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.55rem 0.55rem;
    border-bottom: 1px solid {BORDER};
    background: #12171f;
  }}
  .sfa-peer-table td {{
    padding: 0.55rem 0.55rem;
    border-bottom: 1px solid {BORDER};
    vertical-align: middle;
  }}
  .sfa-peer-table tr:hover td {{ background: rgba(255,255,255,0.025); }}
  .sfa-qbar {{
    height: 6px;
    border-radius: 4px;
    background: {BORDER};
    overflow: hidden;
    min-width: 64px;
  }}
  .sfa-qbar > span {{
    display: block;
    height: 100%;
    border-radius: 4px;
  }}
  .sfa-ticker {{
    font-weight: 650;
    color: {TEXT};
  }}
</style>
        """,
        unsafe_allow_html=True,
    )


def _band_colors(band: str | None) -> tuple[str, str, str]:
    if band is None or (isinstance(band, float) and pd.isna(band)):
        return BAND_STYLE["N/A"]
    key = str(band).strip()
    return BAND_STYLE.get(key, (MUTED, MUTED_SOFT, key or "n/a"))


def badge(label: str, band: str | None = None, emoji: bool = True) -> str:
    """HTML pill. `band` picks traffic-light colors (Green/Yellow/Red/…)."""
    color, bg, default_label = _band_colors(band if band is not None else label)
    text = label if label else default_label
    em = ""
    if emoji:
        em_key = band if band in BAND_EMOJI else label
        em = BAND_EMOJI.get(str(em_key), "") + (" " if BAND_EMOJI.get(str(em_key)) else "")
    return (
        f'<span class="sfa-badge" style="color:{color};background:{bg};'
        f'border-color:{color}33;">{em}{text}</span>'
    )


def badge_row(items: list[tuple[str, str | None]]) -> str:
    """items: list of (label, band)."""
    inner = "".join(badge(lab, band) for lab, band in items)
    return f'<div class="sfa-badge-row">{inner}</div>'


def band_text(band: str | None, kind: str = "z") -> str:
    """Plain-text badge for st.dataframe columns (emoji + short label).

    Beneish M (kind='m'): earnings-manipulation model
      Green  → Unlikely (M ≤ −2.22)
      Yellow → Caution  (−2.22 < M ≤ −1.78)  [amber/yellow traffic light]
      Red    → Likely manip. (M > −1.78)

    Altman Z (kind='z'): bankruptcy risk
      Green / Yellow / Red → Safe / Caution / Distress
    """
    if band is None or (isinstance(band, float) and pd.isna(band)):
        return "⚪ n/a"
    b = str(band)
    em = BAND_EMOJI.get(b, "⚪")
    if kind == "m":
        # Avoid cryptic "Manip?" / "Grey" — spell out meaning
        labels = {
            "Green": "Unlikely",
            "Yellow": "Caution",
            "Red": "Likely manip.",
            "N/A": "n/a",
        }
    elif kind == "f":
        labels = {"Strong": "Strong", "Moderate": "Mod", "Weak": "Weak", "N/A": "n/a"}
    else:
        labels = {
            "Green": "Safe",
            "Yellow": "Caution",
            "Red": "Distress",
            "N/A": "n/a",
        }
    return f"{em} {labels.get(b, b)}"


def format_band_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy frame with z_band / m_band as readable emoji labels for tables."""
    out = df.copy()
    if "z_band" in out.columns:
        out["z_band"] = out["z_band"].map(lambda x: band_text(x, "z"))
    if "m_band" in out.columns:
        out["m_band"] = out["m_band"].map(lambda x: band_text(x, "m"))
    return out


def quality_color(score: float | None) -> str:
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return MUTED
    s = float(score)
    if s >= 65:
        return GREEN
    if s >= 50:
        return BLUE
    if s >= 35:
        return YELLOW
    return RED


def score_card(title: str, value: str, subtitle: str = "", accent: str | None = None) -> str:
    border = accent or BORDER
    return (
        f'<div class="sfa-card" style="border-color:{border};'
        f'border-top: 3px solid {border};">'
        f'<div class="sfa-card-title">{title}</div>'
        f'<div class="sfa-card-value">{value}</div>'
        f'<div class="sfa-card-sub">{subtitle}</div></div>'
    )


def section(title: str) -> None:
    st.markdown(f'<div class="sfa-section">{title}</div>', unsafe_allow_html=True)


def quality_bar_html(score: float | None, width_pct: float | None = None) -> str:
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return '<div class="sfa-qbar"><span style="width:0"></span></div>'
    s = max(0.0, min(100.0, float(score)))
    col = quality_color(s)
    w = width_pct if width_pct is not None else s
    return (
        f'<div class="sfa-qbar" title="{s:.0f}">'
        f'<span style="width:{w:.0f}%;background:{col};"></span></div>'
    )


def peers_html_table(peers: pd.DataFrame) -> str:
    """Colored HTML table for sector peers (Z/M badges + quality bar)."""
    if peers is None or peers.empty:
        return '<p class="sfa-muted">No peers to show.</p>'

    rows = []
    for _, r in peers.iterrows():
        ticker = str(r.get("ticker", "")).replace(".NS", "")
        q = r.get("quality_score")
        q_txt = f"{float(q):.0f}" if pd.notna(q) else "—"
        z = band_text(r.get("z_band"), "z") if "z_band" in peers.columns else "—"
        m = band_text(r.get("m_band"), "m") if "m_band" in peers.columns else "—"
        # colored badge HTML for Z/M
        zb = r.get("z_band") if "z_band" in peers.columns else None
        mb = r.get("m_band") if "m_band" in peers.columns else None
        z_html = badge(band_text(zb, "z").split(" ", 1)[-1] if zb else "n/a", zb or "N/A")
        m_html = badge(band_text(mb, "m").split(" ", 1)[-1] if mb else "n/a", mb or "N/A")
        flags = r.get("red_flag_count", 0)
        try:
            flags_i = int(flags) if pd.notna(flags) else 0
        except (TypeError, ValueError):
            flags_i = 0
        flag_html = (
            badge(f"{flags_i} flag{'s' if flags_i != 1 else ''}", "Red" if flags_i else "Green")
        )
        rank = r.get("rank_in_sector", "—")
        roe = r.get("roe")
        pe = r.get("pe")
        roe_s = f"{float(roe):.1f}" if pd.notna(roe) else "—"
        pe_s = f"{float(pe):.1f}" if pd.notna(pe) else "—"
        rows.append(
            "<tr>"
            f"<td>#{rank}</td>"
            f'<td><span class="sfa-ticker">{ticker}</span></td>'
            f"<td>{q_txt} {quality_bar_html(q if pd.notna(q) else None)}</td>"
            f"<td>{roe_s}</td>"
            f"<td>{pe_s}</td>"
            f"<td>{z_html}</td>"
            f"<td>{m_html}</td>"
            f"<td>{flag_html}</td>"
            "</tr>"
        )

    return (
        '<table class="sfa-peer-table">'
        "<thead><tr>"
        "<th>#</th><th>Ticker</th><th>Quality</th><th>ROE</th><th>P/E</th>"
        "<th>Altman Z</th><th>Beneish M</th><th>Flags</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# Classic Piotroski grouping for the visual breakdown
PIOTROSKI_GROUPS = [
    (
        "Profitability",
        [
            "pf_positive_net_income",
            "pf_positive_ocf",
            "pf_roa_improved",
            "pf_ocf_gt_net_income",
        ],
    ),
    (
        "Leverage & liquidity",
        [
            "pf_lower_leverage",
            "pf_higher_current_ratio",
            "pf_no_dilution",
        ],
    ),
    (
        "Operating efficiency",
        [
            "pf_higher_gross_margin",
            "pf_higher_asset_turnover",
        ],
    ),
]


def _f_test_state(latest: Any, key: str) -> str:
    """Return 'ok' | 'bad' | 'na' for one Piotroski test."""
    if hasattr(latest, "get"):
        v = latest.get(key)
    elif hasattr(latest, "index") and key in latest.index:
        v = latest[key]
    else:
        v = None
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "na"
    try:
        return "ok" if float(v) >= 1 else "bad"
    except (TypeError, ValueError):
        return "na"


def f_score_chips(latest: Any, tests: list[str], labels: dict) -> str:
    """Compact chips with inline styles (Streamlit-safe)."""
    parts = []
    for t in tests:
        state = _f_test_state(latest, t)
        lab = labels.get(t, t)
        if state == "ok":
            bg, fg, border, mark = GREEN_SOFT, GREEN_BORDER, GREEN, "✓"
        elif state == "bad":
            bg, fg, border, mark = RED_SOFT, "#ff8a88", RED, "✗"
        else:
            bg, fg, border, mark = MUTED_SOFT, MUTED, BORDER, "–"
        parts.append(
            f'<span style="display:inline-block;margin:3px 4px 3px 0;padding:4px 10px;'
            f"border-radius:999px;font-size:0.8rem;font-weight:600;"
            f'background:{bg};color:{fg};border:1px solid {border};">'
            f"{mark} {lab}</span>"
        )
    return "<div>" + "".join(parts) + "</div>"


def f_score_breakdown_html(
    latest: Any,
    tests: list[str],
    labels: dict,
    score: float | None = None,
    tests_used: int | None = None,
) -> str:
    """
    Streamlit-safe HTML: all styling is inline (class-based CSS is often stripped).
    """
    states = [(t, _f_test_state(latest, t), labels.get(t, t)) for t in tests]
    n_ok = sum(1 for _, s, _ in states if s == "ok")
    n_bad = sum(1 for _, s, _ in states if s == "bad")
    n_na = sum(1 for _, s, _ in states if s == "na")
    if score is not None and pd.notna(score):
        passed = int(score)
    else:
        passed = n_ok
    denom = 9
    pct = max(0.0, min(100.0, 100.0 * passed / denom))
    bar_color = quality_color(100.0 * passed / denom)

    used_note = ""
    if tests_used is not None and int(tests_used) < 9:
        used_note = f" · {int(tests_used)}/9 evaluable"
    elif n_na:
        used_note = f" · {n_na} not evaluable"

    # Numbered status cells
    dots = []
    for i, (t, state, lab) in enumerate(states, start=1):
        if state == "ok":
            bg, fg, bd = GREEN_SOFT, GREEN_BORDER, GREEN
        elif state == "bad":
            bg, fg, bd = RED_SOFT, "#ff8a88", RED
        else:
            bg, fg, bd = MUTED_SOFT, MUTED, BORDER
        title = lab.replace('"', "&quot;")
        dots.append(
            f'<td title="{title}" style="text-align:center;padding:6px 4px;">'
            f'<div style="width:30px;height:30px;line-height:30px;margin:0 auto;'
            f"border-radius:50%;font-size:0.75rem;font-weight:700;"
            f'background:{bg};color:{fg};border:2px solid {bd};">{i}</div></td>'
        )

    # Grouped rows as table (tables survive Streamlit sanitizer better than flex divs)
    body_parts = []
    for cat_name, keys in PIOTROSKI_GROUPS:
        body_parts.append(
            f'<tr><td colspan="3" style="padding:14px 8px 6px 8px;font-size:0.72rem;'
            f"text-transform:uppercase;letter-spacing:0.06em;color:{MUTED};"
            f'font-weight:700;border-bottom:1px solid {BORDER};">{cat_name}</td></tr>'
        )
        for t in keys:
            state = _f_test_state(latest, t)
            lab = labels.get(t, t)
            if state == "ok":
                bg, bd, icon, status, sc = (
                    "rgba(29,158,117,0.14)",
                    GREEN,
                    "✓",
                    "PASS",
                    GREEN_BORDER,
                )
                ic_bg, ic_fg = GREEN, "#0D1117"
            elif state == "bad":
                bg, bd, icon, status, sc = (
                    "rgba(226,75,74,0.14)",
                    RED,
                    "✗",
                    "FAIL",
                    "#ff8a88",
                )
                ic_bg, ic_fg = RED, "#ffffff"
            else:
                bg, bd, icon, status, sc = (
                    "rgba(139,148,158,0.08)",
                    BORDER,
                    "–",
                    "N/A",
                    MUTED,
                )
                ic_bg, ic_fg = BORDER, MUTED
            body_parts.append(
                f'<tr style="background:{bg};">'
                f'<td style="width:48px;padding:8px;border-bottom:1px solid {BORDER};'
                f'border-left:3px solid {bd};">'
                f'<div style="width:28px;height:28px;line-height:28px;text-align:center;'
                f"border-radius:8px;background:{ic_bg};color:{ic_fg};"
                f'font-weight:700;font-size:0.95rem;">{icon}</div></td>'
                f'<td style="padding:10px 8px;color:{TEXT};font-size:0.9rem;'
                f'border-bottom:1px solid {BORDER};">{lab}</td>'
                f'<td style="width:64px;padding:10px 8px;text-align:right;'
                f"font-size:0.72rem;font-weight:700;letter-spacing:0.04em;"
                f'color:{sc};border-bottom:1px solid {BORDER};">{status}</td>'
                f"</tr>"
            )

    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:14px;padding:14px 16px;margin-top:4px;">'
        # header + bar
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:1.35rem;font-weight:700;color:{TEXT};margin-bottom:8px;">'
        f'{passed} <span style="color:{MUTED};font-weight:500;font-size:0.95rem;">'
        f"/ {denom}{used_note}</span></div>"
        f'<div style="height:10px;background:{BORDER};border-radius:6px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{bar_color};'
        f'border-radius:6px;"></div></div></div>'
        # dots row
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:10px;">'
        f"<tr>{''.join(dots)}</tr></table>"
        # detail table
        f'<table style="width:100%;border-collapse:collapse;">'
        f"{''.join(body_parts)}</table>"
        # legend
        f'<div style="margin-top:12px;font-size:0.78rem;color:{MUTED};">'
        f'<span style="margin-right:14px;">✓ Pass ({n_ok})</span>'
        f'<span style="margin-right:14px;">✗ Fail ({n_bad})</span>'
        f"<span>– Not evaluable ({n_na})</span>"
        f"</div></div>"
    )


def render_f_score_breakdown(
    latest: Any,
    tests: list[str],
    labels: dict,
    score: float | None = None,
    tests_used: int | None = None,
) -> None:
    """
    Render F-Score breakdown with Streamlit-native widgets + inline-styled HTML.
    Reliable in Streamlit (class-based CSS is often stripped from markdown HTML).
    """
    if score is not None and pd.notna(score):
        passed = int(score)
    else:
        passed = sum(1 for t in tests if _f_test_state(latest, t) == "ok")

    n_ok = sum(1 for t in tests if _f_test_state(latest, t) == "ok")
    n_bad = sum(1 for t in tests if _f_test_state(latest, t) == "bad")
    n_na = sum(1 for t in tests if _f_test_state(latest, t) == "na")

    head_l, head_r = st.columns([1, 3])
    with head_l:
        st.metric("Tests passed", f"{passed} / 9")
    with head_r:
        st.progress(min(1.0, passed / 9.0))
        note = ""
        if tests_used is not None and int(tests_used) < 9:
            note = f"{int(tests_used)}/9 evaluable · "
        st.caption(f"{note}✓ {n_ok} pass · ✗ {n_bad} fail · – {n_na} n/a")

    # 9 status tiles
    cols = st.columns(9)
    for i, t in enumerate(tests):
        state = _f_test_state(latest, t)
        lab = labels.get(t, t)
        if state == "ok":
            bg, fg, bd = "rgba(29,158,117,0.22)", "#26B583", "#1D9E75"
        elif state == "bad":
            bg, fg, bd = "rgba(226,75,74,0.22)", "#ff8a88", "#E24B4A"
        else:
            bg, fg, bd = "rgba(139,148,158,0.15)", "#8B949E", "#30363D"
        with cols[i]:
            st.markdown(
                f'<div title="{lab}" style="text-align:center;padding:8px 2px;'
                f"border-radius:10px;background:{bg};border:2px solid {bd};"
                f'color:{fg};font-weight:700;font-size:0.95rem;">{i + 1}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    # Grouped detail — Streamlit bordered containers when available
    for cat_name, keys in PIOTROSKI_GROUPS:
        st.markdown(
            f'<p style="margin:12px 0 6px 0;font-size:0.75rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:#8B949E;">{cat_name}</p>',
            unsafe_allow_html=True,
        )
        for t in keys:
            state = _f_test_state(latest, t)
            lab = labels.get(t, t)
            if state == "ok":
                bg, bd, icon, status, sc = (
                    "rgba(29,158,117,0.14)",
                    "#1D9E75",
                    "✓",
                    "PASS",
                    "#26B583",
                )
                ic_bg, ic_fg = "#1D9E75", "#0D1117"
            elif state == "bad":
                bg, bd, icon, status, sc = (
                    "rgba(226,75,74,0.14)",
                    "#E24B4A",
                    "✗",
                    "FAIL",
                    "#ff8a88",
                )
                ic_bg, ic_fg = "#E24B4A", "#ffffff"
            else:
                bg, bd, icon, status, sc = (
                    "rgba(139,148,158,0.08)",
                    "#30363D",
                    "–",
                    "N/A",
                    "#8B949E",
                )
                ic_bg, ic_fg = "#30363D", "#8B949E"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f"padding:10px 12px;margin:0 0 6px 0;border-radius:10px;"
                f'background:{bg};border:1px solid {bd};border-left:4px solid {bd};">'
                f'<div style="width:30px;height:30px;min-width:30px;border-radius:8px;'
                f"background:{ic_bg};color:{ic_fg};font-weight:700;font-size:1rem;"
                f'text-align:center;line-height:30px;">{icon}</div>'
                f'<div style="flex:1;color:#E6EDF3;font-size:0.92rem;">{lab}</div>'
                f'<div style="color:{sc};font-size:0.75rem;font-weight:700;'
                f'letter-spacing:0.04em;">{status}</div></div>',
                unsafe_allow_html=True,
            )


def red_flag_block(text: str) -> str:
    return f'<div class="sfa-flag">🚩 {text}</div>'


def ok_banner(text: str) -> str:
    return f'<div class="sfa-ok-banner">✅ {text}</div>'
