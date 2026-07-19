"""
Landing / marketing UI for login and empty-data states.
Feature sections use Streamlit expanders (reliable collapse).
Hero uses responsive title so it is not clipped by the top chrome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import streamlit as st

# Full feature catalog shown on login + empty landing
FEATURE_GROUPS = [
    {
        "title": "Score & explain",
        "icon": "📊",
        "items": [
            ("📊", "Quality Score Engine",
             "22 metrics blended into a 0–100 score, percentile-ranked vs the universe or sector peers."),
            ("⚖️", "Configurable weights",
             "Presets (value, quality, growth, safety) or custom category sliders — renormalized live."),
            ("💬", "Explainability",
             "Top strengths and weaknesses vs peers, plain-English summary, category radar."),
            ("📈", "Quality trends",
             "Score trajectory across fiscal years with peer-group caveats."),
        ],
    },
    {
        "title": "Risk & forensics",
        "icon": "🛡️",
        "items": [
            ("🚩", "Red-flag detection",
             "Rule-based earnings quality, leverage, dilution, auditor, and governance checks."),
            ("🛡️", "Data-quality guards",
             "Completeness % and sanity flags (corporate actions, extreme PE/margins)."),
            ("📗", "Piotroski F-Score",
             "Nine financial-health tests with visual pass/fail breakdown."),
            ("📙", "Altman Z-Score",
             "Bankruptcy-risk traffic light for small caps and cyclicals."),
            ("📕", "Beneish M-Score",
             "Earnings-manipulation risk: unlikely / caution / likely manip."),
        ],
    },
    {
        "title": "Research workflow",
        "icon": "🔍",
        "items": [
            ("🏆", "Universe ranking",
             "Multi-factor leaderboard with progress bars, Z/M bands, export CSV."),
            ("🎯", "Screen presets",
             "Clean quality, value quality, low leverage, watchlist — plus save your own."),
            ("⭐", "Watchlist",
             "Session watchlist with portfolio snapshot, sector mix, CSV import/export."),
            ("🔀", "Compare",
             "2–5 stocks side by side with radars, badges, and metrics table."),
            ("🏭", "Sector overview",
             "Average/median quality by sector, strongest names, drill-down path."),
        ],
    },
    {
        "title": "Data & ML",
        "icon": "🤖",
        "items": [
            ("📂", "Auto-load fundamentals",
             "Uses project fundamentals.csv by default; upload to override anytime."),
            ("🇮🇳", "Governance overlay",
             "Merge India-specific pledge / insider / auditor fields Yahoo cannot provide."),
            ("🤖", "Outperformance model",
             "Train a global classifier (with optional labels from build_labels), download/upload joblib."),
            ("🏷️", "Label builder",
             "Offline script attaches forward returns vs Nifty for honest train/valid splits."),
        ],
    },
]


def _feature_card_html(icon: str, title: str, desc: str, compact: bool = False) -> str:
    pad = "8px 10px" if compact else "12px 14px"
    title_sz = "0.9rem" if compact else "0.95rem"
    desc_sz = "0.76rem" if compact else "0.82rem"
    return (
        f'<div style="background:#161B22;border:1px solid #30363D;border-radius:10px;'
        f'padding:{pad};border-left:3px solid #1D9E75;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:flex-start;gap:8px;">'
        f'<div style="font-size:1.15rem;line-height:1.2;flex-shrink:0;">{icon}</div>'
        f'<div style="min-width:0;">'
        f'<div style="font-weight:650;color:#E6EDF3;font-size:{title_sz};'
        f'margin-bottom:3px;word-wrap:break-word;">{title}</div>'
        f'<div style="color:#8B949E;font-size:{desc_sz};line-height:1.4;'
        f'word-wrap:break-word;">{desc}</div>'
        f"</div></div></div>"
    )


def render_feature_sections(
    st_module,
    *,
    compact: bool = True,
    expand_first: bool = False,
) -> None:
    """
    Collapsible feature groups via st.expander (avoids a long scrolling wall).
    Only the first group is open when expand_first=True.
    """
    for i, group in enumerate(FEATURE_GROUPS):
        label = f"{group.get('icon', '•')}  {group['title']}"
        with st_module.expander(label, expanded=(expand_first and i == 0)):
            for icon, title, desc in group["items"]:
                st_module.markdown(
                    _feature_card_html(icon, title, desc, compact=compact),
                    unsafe_allow_html=True,
                )


def features_grid_html(compact: bool = False) -> str:
    """Non-collapsible HTML fallback (prefer render_feature_sections)."""
    blocks = []
    for group in FEATURE_GROUPS:
        cards = [
            _feature_card_html(icon, title, desc, compact=compact)
            for icon, title, desc in group["items"]
        ]
        blocks.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.06em;color:#8B949E;margin:8px 0 6px 0;">{group["title"]}</div>'
            f"{''.join(cards)}</div>"
        )
    return f'<div style="max-width:100%;">{"".join(blocks)}</div>'


def features_grid_2col_html() -> str:
    """Wide mosaic of feature cards (used inside expanders / empty landing)."""
    cards = []
    for group in FEATURE_GROUPS:
        for icon, title, desc in group["items"]:
            cards.append(
                f'<div style="background:#161B22;border:1px solid #30363D;border-radius:12px;'
                f'padding:12px 14px;border-top:3px solid #1D9E75;">'
                f'<div style="font-size:1.25rem;margin-bottom:6px;">{icon}</div>'
                f'<div style="font-weight:650;color:#E6EDF3;font-size:0.92rem;margin-bottom:6px;'
                f'word-wrap:break-word;">{title}</div>'
                f'<div style="color:#8B949E;font-size:0.78rem;line-height:1.45;'
                f'word-wrap:break-word;">{desc}</div>'
                f'<div style="margin-top:8px;font-size:0.65rem;color:#6e7681;'
                f'text-transform:uppercase;letter-spacing:0.05em;">{group["title"]}</div>'
                f"</div>"
            )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));'
        f'gap:10px;margin:8px 0 12px 0;">{"".join(cards)}</div>'
    )


def hero_html(subtitle: str | None = None) -> str:
    """Lightweight HTML hero (prefer render_hero for login — uses st.title)."""
    sub = subtitle or (
        "Quality scoring · forensic red flags · institutional scores · "
        "sector ranking · watchlists · optional ML outperformance"
    )
    return (
        f'<div style="padding:0.25rem 0 0.4rem 0;max-width:100%;">'
        f'<div style="font-size:clamp(1.2rem, 2.8vw, 1.65rem);font-weight:800;'
        f"color:#E6EDF3;letter-spacing:-0.02em;line-height:1.3;"
        f'word-wrap:break-word;overflow-wrap:break-word;white-space:normal;">'
        f"Fundamental Stock Analyzer</div>"
        f'<div style="color:#8B949E;font-size:0.9rem;margin-top:6px;line-height:1.4;'
        f'word-wrap:break-word;white-space:normal;">{sub}</div></div>'
    )


def render_hero(st_module, subtitle: str | None = None) -> None:
    """
    Streamlit-native title so the name is never clipped by custom HTML / Deploy bar.
    """
    sub = subtitle or (
        "Quality scoring · forensic red flags · institutional scores · "
        "sector ranking · watchlists · optional ML outperformance"
    )
    # Emoji prefix keeps brand without a large SVG that can push text off-screen
    st_module.title("📊  Fundamental Stock Analyzer")
    st_module.caption(sub)


def how_it_works_html() -> str:
    steps = [
        ("1", "Load data", "Auto-loads fundamentals.csv or upload your panel / demo sample."),
        ("2", "Score & filter", "Quality + red flags + F/Z/M. Screen with Clean / Value / Low leverage."),
        ("3", "Research", "Single-stock report, peers, watchlist, compare, sector overview."),
        ("4", "Optional ML", "Add labels via build_labels, train in-app, rank with blend weights."),
    ]
    cells = []
    for n, title, body in steps:
        cells.append(
            f'<div style="flex:1;min-width:140px;background:#161B22;border:1px solid #30363D;'
            f'border-radius:12px;padding:12px;">'
            f'<div style="width:26px;height:26px;border-radius:50%;background:rgba(29,158,117,0.2);'
            f'border:1px solid #1D9E75;color:#26B583;font-weight:700;text-align:center;'
            f'line-height:26px;margin-bottom:8px;font-size:0.85rem;">{n}</div>'
            f'<div style="font-weight:650;color:#E6EDF3;margin-bottom:4px;font-size:0.9rem;">{title}</div>'
            f'<div style="color:#8B949E;font-size:0.78rem;line-height:1.4;">{body}</div>'
            f"</div>"
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;">'
        f"{''.join(cells)}</div>"
    )
