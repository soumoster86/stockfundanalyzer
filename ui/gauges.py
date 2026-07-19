"""Inline SVG gauges used by report / compare tabs."""

from __future__ import annotations

import math

import pandas as pd

from src.quality_score import score_label

CATEGORY_SHORT = {
    "Financial Performance": "Growth",
    "Profitability": "Profit",
    "Financial Strength": "Strength",
    "Shareholder Metrics": "Shareholder",
    "Valuation": "Value",
}


def category_radar_svg(scores: dict, color: str = "#1D9E75", size: int = 320) -> str:
    """
    Render a 5-axis radar/spider chart for category scores (0-100) as inline SVG.
    `scores` maps category name -> score. Missing/None categories plot at 0.
    """
    cats = list(CATEGORY_SHORT.keys())
    n = len(cats)
    cx = cy = size / 2
    r_max = size * 0.34
    rings = ""
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            x = cx + r_max * frac * math.cos(ang)
            y = cy + r_max * frac * math.sin(ang)
            pts.append(f"{x:.1f},{y:.1f}")
        rings += (
            f'<polygon points="{" ".join(pts)}" fill="none" '
            f'stroke="#3a3a3a" stroke-width="1"/>'
        )
    spokes, labels = "", ""
    for i, cat in enumerate(cats):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x = cx + r_max * math.cos(ang)
        y = cy + r_max * math.sin(ang)
        spokes += (
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="#3a3a3a" stroke-width="1"/>'
        )
        lx = cx + (r_max + 22) * math.cos(ang)
        ly = cy + (r_max + 22) * math.sin(ang)
        anchor = "middle"
        if math.cos(ang) > 0.3:
            anchor = "start"
        elif math.cos(ang) < -0.3:
            anchor = "end"
        labels += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="12" fill="#aaa" dominant-baseline="middle">'
            f"{CATEGORY_SHORT[cat]}</text>"
        )
    dpts = []
    for i, cat in enumerate(cats):
        v = scores.get(cat)
        v = 0.0 if (v is None or pd.isna(v)) else max(0.0, min(100.0, float(v)))
        ang = -math.pi / 2 + 2 * math.pi * i / n
        rr = r_max * v / 100.0
        dpts.append(f"{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}")
    data_poly = (
        f'<polygon points="{" ".join(dpts)}" fill="{color}" '
        f'fill-opacity="0.28" stroke="{color}" stroke-width="2"/>'
    )
    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f"{rings}{spokes}{data_poly}{labels}</svg>"
    )


def quality_gauge_svg(score: float) -> str:
    """Semicircular gauge SVG for a 0-100 score."""
    score = max(0.0, min(100.0, float(score) if score is not None and not pd.isna(score) else 0.0))
    if score >= 65:
        color = "#1D9E75"
    elif score >= 50:
        color = "#378ADD"
    elif score >= 35:
        color = "#E0A82E"
    else:
        color = "#E24B4A"
    ang = math.pi * (1 - score / 100.0)
    cx, cy, r = 100, 100, 80
    x = cx + r * math.cos(ang)
    y = cy - r * math.sin(ang)
    large = 0
    bg = "M 20 100 A 80 80 0 0 1 180 100"
    fg = f"M 20 100 A 80 80 0 {large} 1 {x:.1f} {y:.1f}"
    return f"""
    <svg viewBox="0 0 200 130" width="220" height="143">
      <path d="{bg}" fill="none" stroke="#3a3a3a" stroke-width="14" stroke-linecap="round"/>
      <path d="{fg}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"/>
      <text x="100" y="92" text-anchor="middle" font-size="36" font-weight="600" fill="{color}">{score:.0f}</text>
      <text x="100" y="115" text-anchor="middle" font-size="14" fill="#888">{score_label(score)}</text>
    </svg>"""
