"""
Quality Score Engine
---------------------
Normalizes each fundamental metric cross-sectionally (within a date/sector group),
applies weights, and aggregates into category sub-scores and a composite 0-100 score.

Expected input: a pandas DataFrame where each row is (ticker, date) and columns are
the raw metrics listed in METRIC_CONFIG.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric configuration
#   direction = +1  -> higher is better
#   direction = -1  -> lower is better (valuation multiples)
#   weight    = within-category weight (each category sums to 1.0)
# ---------------------------------------------------------------------------
METRIC_CONFIG = {
    "Financial Performance": {
        "weight": 0.25,
        "metrics": {
            "revenue_growth":       {"direction": +1, "weight": 0.25},
            "eps_growth":           {"direction": +1, "weight": 0.25},
            "operating_profit_growth": {"direction": +1, "weight": 0.20},
            "fcf_growth":           {"direction": +1, "weight": 0.15},
            "ebitda_growth":        {"direction": +1, "weight": 0.15},
        },
    },
    "Profitability": {
        "weight": 0.25,
        "metrics": {
            "roe":              {"direction": +1, "weight": 0.25},
            "roce":             {"direction": +1, "weight": 0.25},
            "net_margin":       {"direction": +1, "weight": 0.20},
            "operating_margin": {"direction": +1, "weight": 0.15},
            "gross_margin":     {"direction": +1, "weight": 0.15},
        },
    },
    "Financial Strength": {
        "weight": 0.20,
        "metrics": {
            "debt_to_equity":          {"direction": -1, "weight": 0.30},
            "interest_coverage":       {"direction": +1, "weight": 0.30},
            "current_ratio":           {"direction": +1, "weight": 0.20},
            "cash_position":           {"direction": +1, "weight": 0.20},
        },
    },
    "Shareholder Metrics": {
        "weight": 0.10,
        "metrics": {
            "dividend_yield":          {"direction": +1, "weight": 0.30},
            "dividend_growth":         {"direction": +1, "weight": 0.30},
            "buyback_yield":           {"direction": +1, "weight": 0.20},
            "promoter_holding_change": {"direction": +1, "weight": 0.20},  # India-specific
        },
    },
    "Valuation": {
        "weight": 0.20,
        "metrics": {
            "pe":          {"direction": -1, "weight": 0.25},
            "pb":          {"direction": -1, "weight": 0.20},
            "ev_ebitda":   {"direction": -1, "weight": 0.25},
            "peg":         {"direction": -1, "weight": 0.15},
            "price_sales": {"direction": -1, "weight": 0.15},
        },
    },
}


def _percentile_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0,1]; NaNs stay NaN."""
    return s.rank(pct=True)


# Default category weights, exposed for UI defaults
DEFAULT_CATEGORY_WEIGHTS = {cat: cfg["weight"] for cat, cfg in METRIC_CONFIG.items()}


def build_config(category_weights=None, base_config=METRIC_CONFIG):
    """
    Return a deep copy of the metric config with category weights overridden.
    `category_weights` is a dict like {"Profitability": 0.30, "Valuation": 0.10, ...}.
    Per-metric weights within each category are preserved. Missing categories
    keep their default weight. Weights need not sum to 1 — the scorer
    renormalizes by total category weight.
    """
    import copy
    cfg = copy.deepcopy(base_config)
    if category_weights:
        for cat, w in category_weights.items():
            if cat in cfg:
                cfg[cat]["weight"] = float(w)
    return cfg



def compute_quality_score(
    df: pd.DataFrame,
    group_cols=("date",),          # add 'sector' to rank within sector+date
    config=METRIC_CONFIG,
    min_group_size=5,              # below this, fall back to date-only ranking
) -> pd.DataFrame:
    """
    Returns df with added columns:
      _pct_<metric>     (0-1)    direction-adjusted percentile for each metric
      <category>_score  (0-100)  for each category
      quality_score     (0-100)  composite

    Normalization is done per group (e.g. per date, or per date+sector) so
    scores are relative to peers at the same point in time. If a (date, sector)
    group has fewer than `min_group_size` members, ranking within it is
    statistically meaningless, so those rows fall back to date-only ranking.
    """
    out = df.copy()
    group_cols = list(group_cols)

    # Decide effective grouping: if sector grouping requested but groups are
    # too small, fall back to ranking by date only for the whole frame.
    effective_groups = group_cols
    used_sector = "sector" in group_cols and "sector" in out.columns
    if "sector" in group_cols and "sector" not in out.columns:
        # requested but unavailable -> drop it
        effective_groups = [c for c in group_cols if c != "sector"]
        used_sector = False
    elif used_sector:
        sizes = out.groupby(group_cols)["sector"].transform("size")
        if (sizes < min_group_size).any():
            # mixed case: rank small-sector rows by date-only, others by sector
            def _mkkey(row):
                return "|".join(str(v) for v in row)
            out["_grp_key"] = out[group_cols].apply(_mkkey, axis=1)
            date_only = [c for c in group_cols if c != "sector"]
            small_mask = sizes < min_group_size
            out.loc[small_mask, "_grp_key"] = (
                out.loc[small_mask, date_only].apply(_mkkey, axis=1)
                + "|__ALL_SECTORS__"
            )
            effective_groups = ["_grp_key"]

    category_scores = {}
    for cat, cat_cfg in config.items():
        metric_norm_cols = []
        for metric, m_cfg in cat_cfg["metrics"].items():
            if metric not in out.columns:
                continue
            direction = m_cfg["direction"]
            # percentile rank within the group; invert if lower-is-better
            ranked = out.groupby(effective_groups)[metric].transform(_percentile_rank)
            if direction < 0:
                ranked = 1.0 - ranked
            # retain the raw percentile for explainability
            out[f"_pct_{metric}"] = ranked
            col = f"_norm_{metric}"
            out[col] = ranked * m_cfg["weight"]
            metric_norm_cols.append(col)

        if metric_norm_cols:
            # renormalize weights for available metrics, then 0-100
            total_w = sum(
                config[cat]["metrics"][c.replace("_norm_", "")]["weight"]
                for c in metric_norm_cols
            )
            cat_score = out[metric_norm_cols].sum(axis=1) / total_w * 100.0
            out[f"{cat.replace(' ', '_').lower()}_score"] = cat_score
            category_scores[cat] = cat_score

    # Composite using category weights
    comp = pd.Series(0.0, index=out.index)
    total_cat_w = 0.0
    for cat, score in category_scores.items():
        w = config[cat]["weight"]
        comp = comp.add(score * w, fill_value=0.0)
        total_cat_w += w
    out["quality_score"] = comp / total_cat_w if total_cat_w else np.nan

    # record how each row was ranked (for transparency)
    out["scored_vs"] = "sector peers" if used_sector else "all stocks (by date)"

    # cleanup temp cols (keep _pct_ for explainability)
    drop = [c for c in out.columns if c.startswith("_norm_")] + (["_grp_key"] if "_grp_key" in out.columns else [])
    out = out.drop(columns=drop)
    return out


def score_label(score: float) -> str:
    if pd.isna(score):
        return "N/A"
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Strong"
    if score >= 50:
        return "Average"
    if score >= 35:
        return "Weak"
    return "Poor"


# Human-readable metric names for explanations
METRIC_LABELS = {
    "revenue_growth": "Revenue Growth", "eps_growth": "EPS Growth",
    "operating_profit_growth": "Operating Profit Growth", "fcf_growth": "Free Cash Flow Growth",
    "ebitda_growth": "EBITDA Growth", "roe": "ROE", "roce": "ROCE",
    "net_margin": "Net Margin", "operating_margin": "Operating Margin",
    "gross_margin": "Gross Margin", "debt_to_equity": "Debt-to-Equity",
    "interest_coverage": "Interest Coverage", "current_ratio": "Current Ratio",
    "cash_position": "Cash Position", "dividend_yield": "Dividend Yield",
    "dividend_growth": "Dividend Growth", "buyback_yield": "Buyback Yield",
    "promoter_holding_change": "Promoter Holding Change", "pe": "P/E",
    "pb": "P/B", "ev_ebitda": "EV/EBITDA", "peg": "PEG", "price_sales": "Price/Sales",
}


def explain_score(row, config=METRIC_CONFIG, top_n=5):
    """
    Explain why a scored row got its quality_score.

    Returns a dict with:
      drivers   : list of (metric_label, percentile_0_100, weighted_contribution, raw_value)
                  sorted best-first  -> what pushed the score UP
      drags     : same, sorted worst-first -> what pulled it DOWN
      categories: {category_name: category_score}

    `row` must come from a DataFrame processed by compute_quality_score
    (it needs the _pct_<metric> columns).
    """
    contributions = []
    for cat, cat_cfg in config.items():
        cat_w = cat_cfg["weight"]
        for metric, m_cfg in cat_cfg["metrics"].items():
            pct_col = f"_pct_{metric}"
            if pct_col not in row or pd.isna(row[pct_col]):
                continue
            pct = float(row[pct_col])                 # 0-1, direction-adjusted
            # contribution to composite = metric weight within cat * cat weight * pct
            contrib = m_cfg["weight"] * cat_w * pct
            raw = row[metric] if metric in row and pd.notna(row.get(metric)) else None
            contributions.append({
                "metric": METRIC_LABELS.get(metric, metric),
                "category": cat,
                "percentile": round(pct * 100, 1),
                "contribution": round(contrib * 100, 2),
                "raw_value": raw,
            })

    if not contributions:
        return {"drivers": [], "drags": [], "categories": {}}

    # Drivers = highest percentile (best vs peers); drags = lowest percentile
    by_pct = sorted(contributions, key=lambda d: d["percentile"], reverse=True)
    drivers = by_pct[:top_n]
    drags = [d for d in reversed(by_pct)][:top_n]

    categories = {}
    for cat in config:
        col = f"{cat.replace(' ', '_').lower()}_score"
        if col in row and pd.notna(row[col]):
            categories[cat] = round(float(row[col]), 1)

    return {"drivers": drivers, "drags": drags, "categories": categories}


def explanation_sentence(explain_result, ticker="This stock"):
    """Turn an explain_score result into a one-paragraph plain-English summary."""
    if not explain_result["drivers"]:
        return f"{ticker} has insufficient data to explain its score."
    top = explain_result["drivers"][:3]
    bot = explain_result["drags"][:3]
    strong = ", ".join(f"{d['metric']} ({d['percentile']:.0f}th pct)" for d in top)
    weak = ", ".join(f"{d['metric']} ({d['percentile']:.0f}th pct)" for d in bot)
    return (f"{ticker} scores strongest on {strong}. "
            f"Its weakest areas versus peers are {weak}.")


def config_default():
    return METRIC_CONFIG


def config_categories(config):
    return list(config.keys())


def quality_history(df, group_cols=("date",), config=METRIC_CONFIG,
                    min_group_size=5, min_fields=12):
    """
    Score EACH fiscal-year cross-section separately, so every (ticker, date) row
    gets a quality_score computed against its same-year peers. This is the basis
    for per-stock quality trends over time.

    Returns the full panel (all years) with quality_score + category scores per
    row. Each distinct date is scored as its own cross-section; rows below
    `min_fields` completeness are excluded from scoring so a sparse stub year
    doesn't produce a misleading point.

    Note: requires a `data_fields_present` column (from data_completeness). If
    absent, all rows are scored.
    """
    panel = df.copy()
    if "data_fields_present" in panel.columns:
        scorable = panel[panel["data_fields_present"] >= min_fields].copy()
    else:
        scorable = panel.copy()

    if scorable.empty or "date" not in scorable.columns:
        panel["quality_score"] = np.nan
        return panel

    scored_parts = []
    for dt, grp in scorable.groupby("date"):
        g = grp.copy()
        g["_snap"] = "y"
        gc = ["_snap", "sector"] if ("sector" in group_cols and "sector" in g.columns) else ["_snap"]
        g = compute_quality_score(g, group_cols=tuple(gc), config=config,
                                  min_group_size=min_group_size)
        g = g.drop(columns=["_snap"], errors="ignore")
        scored_parts.append(g)
    scored = pd.concat(scored_parts, ignore_index=True)
    return scored


def ticker_trend(scored_panel, ticker, ticker_col="ticker", date_col="date",
                 min_peer_ratio=0.5):
    """
    Given a panel scored by quality_history, describe one ticker's quality
    trajectory:
      points    : list of {date, quality_score} sorted by date
      delta     : latest minus earliest quality_score (None if <2 pts)
      direction : 'improving' | 'declining' | 'stable' | 'insufficient data'
      category_deltas : per-category score change latest vs earliest
      caveat    : note when peer-group sizes differ a lot between the two years

    `min_peer_ratio`: if the smaller year's peer count is below this fraction of
    the larger year's, the two scores aren't cleanly comparable, so flag it.
    """
    rows = scored_panel[scored_panel[ticker_col] == ticker].sort_values(date_col)
    rows = rows[rows["quality_score"].notna()]
    pts = [{"date": pd.to_datetime(r[date_col]), "quality_score": float(r["quality_score"])}
           for _, r in rows.iterrows()]
    if len(pts) < 2:
        return {"points": pts, "delta": None, "direction": "insufficient data",
                "category_deltas": {}, "caveat": None}

    year_sizes = scored_panel[scored_panel["quality_score"].notna()].groupby(date_col).size()
    caveat = None
    try:
        s_first = year_sizes.get(rows.iloc[0][date_col])
        s_last = year_sizes.get(rows.iloc[-1][date_col])
        if s_first and s_last:
            lo, hi = min(s_first, s_last), max(s_first, s_last)
            if hi > 0 and (lo / hi) < min_peer_ratio:
                caveat = (f"Peer group differs in size between years "
                          f"({int(s_first)} vs {int(s_last)} stocks scored) — "
                          f"the trend is less reliable.")
    except Exception:
        pass

    delta = pts[-1]["quality_score"] - pts[0]["quality_score"]
    if delta >= 3:
        direction = "improving"
    elif delta <= -3:
        direction = "declining"
    else:
        direction = "stable"
    cat_deltas = {}
    first, last = rows.iloc[0], rows.iloc[-1]
    for cat in config_categories(config_default()):
        col = f"{cat.replace(' ', '_').lower()}_score"
        if col in rows.columns and pd.notna(first.get(col)) and pd.notna(last.get(col)):
            cat_deltas[cat] = round(float(last[col]) - float(first[col]), 1)
    return {"points": pts, "delta": round(delta, 1), "direction": direction,
            "category_deltas": cat_deltas, "caveat": caveat}
