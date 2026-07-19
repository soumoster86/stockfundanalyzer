"""Vectorized flag-list aggregation."""
import numpy as np
import pandas as pd

from src.flag_lists import active_names_per_row, attach_active_lists


def test_active_names_basic():
    df = pd.DataFrame(
        {
            "a": [True, False, True],
            "b": [False, True, True],
            "c": [False, False, False],
        }
    )
    lists = active_names_per_row(df, ["a", "b", "c"])
    assert lists == [["a"], ["b"], ["a", "b"]]


def test_active_names_nan_as_false():
    df = pd.DataFrame({"x": [True, np.nan, False], "y": [np.nan, True, True]})
    lists = active_names_per_row(df, ["x", "y"])
    assert lists[0] == ["x"]
    assert lists[1] == ["y"]
    assert lists[2] == ["y"]


def test_empty_cols():
    df = pd.DataFrame({"z": [1, 2]})
    assert active_names_per_row(df, []) == [[], []]


def test_attach_active_lists():
    df = pd.DataFrame({"f1": [True, False], "f2": [True, True]})
    out = attach_active_lists(df.copy(), ["f1", "f2"], "flags", "flag_count")
    assert out["flags"].tolist() == [["f1", "f2"], ["f2"]]
    assert out["flag_count"].tolist() == [2, 1]


def test_red_flags_uses_vectorized_lists(sample_panel):
    from src.red_flags import detect_red_flags

    out, cols = detect_red_flags(sample_panel)
    assert "red_flags" in out.columns
    for flags, count in zip(out["red_flags"], out["red_flag_count"]):
        assert isinstance(flags, list)
        assert len(flags) == int(count)
