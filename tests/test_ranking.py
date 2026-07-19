"""Multi-factor ranking: quality, ML blend, flag penalty, no-ML fallback."""
import pandas as pd

from src.ranking import rank_universe


def _frame(with_ml=True):
    d = {
        "ticker": ["A", "B", "C"],
        "quality_score": [80.0, 60.0, 70.0],
        "red_flag_count": [0, 0, 2],
    }
    if with_ml:
        d["outperform_proba"] = [0.9, 0.5, 0.8]
    return pd.DataFrame(d)


def test_ranks_by_quality_when_no_ml():
    ranked = rank_universe(_frame(with_ml=False), w_quality=0.5, w_ml=0.5)
    # Without ML, order should follow quality minus flag penalty
    # C has quality 70 but 2 flags * 5 = 10 penalty -> 60; B is 60; A is 80
    assert ranked.iloc[0]["ticker"] == "A"
    assert list(ranked["rank"]) == [1, 2, 3]
    assert ranked.attrs.get("used_ml") is False


def test_flag_penalty_demotes():
    ranked = rank_universe(_frame(with_ml=False), flag_penalty=5.0)
    scores = ranked.set_index("ticker")["composite_score"]
    assert scores["C"] == 70.0 - 10.0  # 2 flags * 5


def test_ml_blend_changes_order():
    ranked = rank_universe(_frame(with_ml=True), w_quality=0.0, w_ml=1.0, flag_penalty=0.0)
    # Pure ML: A 90, C 80, B 50
    assert list(ranked["ticker"]) == ["A", "C", "B"]
    assert ranked.attrs.get("used_ml") is True


def test_all_nan_ml_treated_as_missing():
    df = _frame(with_ml=True)
    df["outperform_proba"] = float("nan")
    ranked = rank_universe(df, w_quality=0.3, w_ml=0.7)
    assert ranked.attrs.get("used_ml") is False
    assert ranked.iloc[0]["ticker"] == "A"
