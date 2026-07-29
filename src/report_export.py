"""
Single-stock research note export
---------------------------------
Build a plain-text / Markdown one-pager for download (no PDF dependency).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.institutional_scores import M_BAND_TEXT, Z_BAND_TEXT, f_score_band, m_band, z_band
from src.quality_score import METRIC_CONFIG, explain_score, explanation_sentence, score_label
from src.red_flags import REASON_TEXT


def build_research_note(
    latest: pd.Series,
    *,
    history_points: list | None = None,
    peers: pd.DataFrame | None = None,
    sector: str | None = None,
    config=None,
) -> str:
    """Return a Markdown research note for one scored row."""
    tk = str(latest.get("ticker", "?"))
    date = latest.get("date")
    date_s = ""
    if pd.notna(date):
        try:
            date_s = pd.Timestamp(date).strftime("%Y-%m-%d")
        except Exception:
            date_s = str(date)

    q = latest.get("quality_score")
    q_f = float(q) if pd.notna(q) else float("nan")
    q_lbl = score_label(q_f) if pd.notna(q) else "N/A"
    n_flags = int(latest["red_flag_count"]) if pd.notna(latest.get("red_flag_count")) else 0

    fs = latest.get("f_score")
    ftu = int(latest.get("f_tests_used", 0) or 0) if pd.notna(latest.get("f_tests_used", 0)) else 0
    f_lbl = f_score_band(fs, ftu) if pd.notna(fs) and ftu > 0 else "N/A"
    zs = latest.get("z_score")
    zb = latest.get("z_band") if "z_band" in latest.index else z_band(zs)
    ms = latest.get("m_score")
    mb = latest.get("m_band") if "m_band" in latest.index else m_band(ms)

    lines = [
        f"# Research note — {tk}",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Snapshot",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Ticker | {tk} |",
        f"| As-of | {date_s or '—'} |",
        f"| Sector | {sector or latest.get('sector', '—')} |",
    ]
    if "industry" in latest.index and pd.notna(latest.get("industry")):
        lines.append(f"| Industry | {latest.get('industry')} |")
    if "mcap_bucket" in latest.index:
        lines.append(f"| Market-cap bucket | {latest.get('mcap_bucket')} |")
    lines += [
        f"| Quality score | {q_f:.1f} ({q_lbl}) |" if pd.notna(q) else "| Quality score | — |",
        f"| Red flags | {n_flags} |",
        f"| Piotroski F | {int(fs) if pd.notna(fs) else '—'} ({f_lbl}) |",
        f"| Altman Z | {float(zs):.2f} ({zb}) |" if pd.notna(zs) else f"| Altman Z | — ({zb or 'N/A'}) |",
        f"| Beneish M | {float(ms):.2f} ({mb}) |" if pd.notna(ms) else f"| Beneish M | — ({mb or 'N/A'}) |",
        "",
    ]

    # Key metrics
    metric_rows = []
    for key, label in (
        ("roe", "ROE"),
        ("roce", "ROCE"),
        ("net_margin", "Net margin"),
        ("debt_to_equity", "Debt/Equity"),
        ("interest_coverage", "Interest coverage"),
        ("pe", "P/E"),
        ("pb", "P/B"),
        ("revenue_growth", "Revenue growth"),
        ("eps_growth", "EPS growth"),
    ):
        if key in latest.index and pd.notna(latest.get(key)):
            metric_rows.append((label, latest.get(key)))
    if metric_rows:
        lines += ["## Key metrics", ""]
        lines += ["| Metric | Value |", "|--------|-------|"]
        for lab, val in metric_rows:
            try:
                lines.append(f"| {lab} | {float(val):.2f} |")
            except (TypeError, ValueError):
                lines.append(f"| {lab} | {val} |")
        lines.append("")

    # Explainability
    try:
        ex = explain_score(latest, config=config or METRIC_CONFIG)
        lines += ["## Why this score?", "", explanation_sentence(ex, ticker=tk), ""]
        if ex.get("drivers"):
            lines.append("**Strengths**")
            for d in ex["drivers"][:5]:
                lines.append(
                    f"- {d.get('metric')}: percentile {d.get('percentile', 0):.0f}"
                )
            lines.append("")
        if ex.get("drags"):
            lines.append("**Weak areas**")
            for d in ex["drags"][:5]:
                lines.append(
                    f"- {d.get('metric')}: percentile {d.get('percentile', 0):.0f}"
                )
            lines.append("")
    except Exception:
        pass

    # Red flags
    flags = latest.get("red_flags", [])
    if isinstance(flags, list) and flags:
        lines += ["## Red flags", ""]
        for f in flags:
            lines.append(f"- {REASON_TEXT.get(f, f)}")
        lines.append("")
    elif n_flags == 0:
        lines += ["## Red flags", "", "None triggered.", ""]

    # Institutional notes
    lines += ["## Institutional scores", ""]
    if zb and zb in Z_BAND_TEXT:
        lines.append(f"- **Altman Z:** {Z_BAND_TEXT[zb]}")
    if mb and mb in M_BAND_TEXT:
        lines.append(f"- **Beneish M:** {M_BAND_TEXT[mb]}")
    lines.append("")

    # Trend
    if history_points and len(history_points) >= 2:
        lines += ["## Quality trend", ""]
        for p in history_points:
            d = p.get("date")
            ds = pd.Timestamp(d).strftime("%Y") if d is not None else "?"
            qs = p.get("quality_score")
            lines.append(f"- {ds}: {float(qs):.1f}" if pd.notna(qs) else f"- {ds}: —")
        lines.append("")

    # Peers
    if peers is not None and not peers.empty:
        lines += ["## Sector / industry peers", ""]
        cols = [c for c in ("ticker", "quality_score", "roe", "pe", "red_flag_count") if c in peers.columns]
        if cols:
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join("---" for _ in cols) + " |")
            for _, pr in peers.head(8).iterrows():
                cells = []
                for c in cols:
                    v = pr.get(c)
                    if c == "ticker":
                        cells.append(str(v).replace(".NS", ""))
                    elif pd.isna(v):
                        cells.append("—")
                    elif isinstance(v, float):
                        cells.append(f"{v:.1f}")
                    else:
                        cells.append(str(v))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines += [
        "---",
        "",
        "_This note is generated by Fundamental Stock Analyzer. "
        "It is not investment advice. Verify against filings._",
        "",
    ]
    return "\n".join(lines)
