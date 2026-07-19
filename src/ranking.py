"""
Multi-Factor Ranking
--------------------
Combines the Quality Score and the ML outperformance probability into a single
ranked leaderboard. Red flags apply a penalty / demotion.
"""

from __future__ import annotations

import pandas as pd


def rank_universe(df, quality_col="quality_score", proba_col="outperform_proba",
                  red_flag_col="red_flag_count",
                  w_quality=0.5, w_ml=0.5, flag_penalty=5.0,
                  as_of_date=None, date_col="date"):
    """
    Returns a ranked DataFrame (best first) with a composite_score in 0-100.
    flag_penalty is subtracted per red flag.

    When `proba_col` is missing or entirely NaN, ML weight is forced to 0 and
    quality weight to 1 so the quality/ML slider cannot scale scores without
    changing order (which would be a no-op that confuses users).
    """
    d = df.copy()
    if as_of_date is not None:
        d = d[d[date_col] == as_of_date]

    has_ml = (
        proba_col in d.columns
        and d[proba_col].notna().any()
    )
    if not has_ml:
        w_quality, w_ml = 1.0, 0.0

    # Renormalize if both weights zero
    total_w = w_quality + w_ml
    if total_w <= 0:
        w_quality, w_ml = 1.0, 0.0
        total_w = 1.0
    w_quality, w_ml = w_quality / total_w, w_ml / total_w

    ml_component = (d[proba_col] * 100.0) if has_ml else 0.0
    q_component = d[quality_col] if quality_col in d else 0.0

    d["composite_score"] = w_quality * q_component + w_ml * ml_component
    if red_flag_col in d:
        d["composite_score"] = d["composite_score"] - flag_penalty * d[red_flag_col].fillna(0)

    d = d.sort_values("composite_score", ascending=False, na_position="last")
    d["rank"] = range(1, len(d) + 1)
    d.attrs["used_ml"] = bool(has_ml)
    return d
