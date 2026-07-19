"""
Scoring pipeline
----------------
Turns a multi-year fundamentals panel into a latest-row-per-ticker frame with
red flags, data-quality annotations, institutional scores, and quality scores.

Pure functions — no Streamlit dependency. Caching lives in the app shell.
"""

from __future__ import annotations

import pandas as pd

from src.quality_score import compute_quality_score, METRIC_CONFIG, build_config, quality_history
from src.red_flags import detect_red_flags
from src.data_quality import data_completeness, data_sanity_flags
from src.institutional_scores import (
    compute_piotroski,
    compute_altman_z,
    blend_with_quality,
)

# Features offered to the outperformance model (subset of available columns).
FEATURE_COLS = [
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth",
    "ebitda_growth", "roe", "roce", "net_margin", "operating_margin",
    "gross_margin", "debt_to_equity", "interest_coverage", "current_ratio",
    "cash_position", "dividend_yield", "dividend_growth", "buyback_yield",
    "pe", "pb", "ev_ebitda", "peg", "price_sales", "quality_score",
]

MIN_FIELDS = 12  # of ~20 core metrics for "substantially populated" latest row


def is_tickers_only(df: pd.DataFrame) -> bool:
    """True if the upload has just ticker/date and no metric columns."""
    metric_cols = {"revenue_growth", "roe", "pe", "net_margin", "revenue"}
    return not (metric_cols & set(df.columns))


def _latest_populated(df: pd.DataFrame, min_fields: int = MIN_FIELDS) -> pd.DataFrame:
    """
    Keep the latest row per ticker that clears a minimum completeness bar.
    Stocks whose newest fiscal row is mostly empty fall back to their last
    substantially-populated year.
    """
    if "date" not in df.columns:
        return df
    df = df.sort_values("date")
    good = df[df["data_fields_present"] >= min_fields]
    latest_good = good.groupby("ticker", as_index=False).tail(1)
    missing = set(df["ticker"]) - set(latest_good["ticker"])
    if missing:
        rest = df[df["ticker"].isin(missing)].sort_values(
            ["ticker", "data_fields_present", "date"]
        )
        rest = rest.groupby("ticker", as_index=False).tail(1)
        return pd.concat([latest_good, rest], ignore_index=True)
    return latest_good


def enrich(df: pd.DataFrame, use_sector: bool = False, config=None) -> pd.DataFrame:
    """
    Full enrichment pipeline on a multi-year panel → latest cross-section.

    Steps: red flags → sanity → completeness → Piotroski → latest row pick →
    Altman Z → quality score (optionally sector-relative) → F-score blend.
    """
    if config is None:
        config = METRIC_CONFIG

    # Data-quality layer needs the full multi-year panel (YoY checks).
    df, _flag_cols = detect_red_flags(df)
    df, _warn_cols = data_sanity_flags(df)
    df = data_completeness(df)
    # Piotroski needs prior-year comparisons before collapsing.
    df = compute_piotroski(df)
    df = _latest_populated(df)
    # Altman Z is point-in-time on the latest cross-section.
    df = compute_altman_z(df)
    # Single snapshot key so mixed fiscal year-ends rank together.
    df = df.copy()
    df["_snapshot"] = "latest"
    group = (
        ("_snapshot", "sector")
        if (use_sector and "sector" in df.columns)
        else ("_snapshot",)
    )
    df = compute_quality_score(df, group_cols=group, config=config)
    df = df.drop(columns=["_snapshot"], errors="ignore")
    df = blend_with_quality(df)
    return df


def build_history_panel(
    raw_df: pd.DataFrame,
    use_sector: bool = False,
    config=None,
) -> pd.DataFrame:
    """
    Per-year quality scores for trend charts (each fiscal year vs its peers).
    Does not collapse to latest row.
    """
    if config is None:
        config = METRIC_CONFIG
    d, _ = detect_red_flags(raw_df)
    d, _ = data_sanity_flags(d)
    d = data_completeness(d)
    gc = (
        ("date", "sector")
        if (use_sector and "sector" in d.columns)
        else ("date",)
    )
    return quality_history(d, group_cols=gc, config=config)


def config_from_weights(weights_tuple) -> dict:
    """Rebuild metric config from a hashable (category, weight) tuple."""
    return build_config(dict(weights_tuple))
