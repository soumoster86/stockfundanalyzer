"""
Multi-Factor Ranking
--------------------
Combines the Quality Score and the ML outperformance probability into a single
ranked leaderboard. Red flags apply a penalty / demotion.
"""

import pandas as pd


def rank_universe(df, quality_col="quality_score", proba_col="outperform_proba",
                  red_flag_col="red_flag_count",
                  w_quality=0.5, w_ml=0.5, flag_penalty=5.0,
                  as_of_date=None, date_col="date"):
    """
    Returns a ranked DataFrame (best first) with a composite_score in 0-100.
    flag_penalty is subtracted per red flag.
    """
    d = df.copy()
    if as_of_date is not None:
        d = d[d[date_col] == as_of_date]

    # ML proba is 0-1 -> scale to 0-100
    ml_component = d[proba_col] * 100.0 if proba_col in d else 0.0
    q_component = d[quality_col] if quality_col in d else 0.0

    d["composite_score"] = w_quality * q_component + w_ml * ml_component
    if red_flag_col in d:
        d["composite_score"] = d["composite_score"] - flag_penalty * d[red_flag_col]

    d = d.sort_values("composite_score", ascending=False)
    d["rank"] = range(1, len(d) + 1)
    return d
