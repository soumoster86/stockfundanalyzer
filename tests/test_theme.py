"""UI theme helpers."""
import pandas as pd

from ui.theme import (
    band_text,
    badge,
    format_band_columns,
    peers_html_table,
    quality_color,
    f_score_breakdown_html,
    _f_test_state,
)


def test_band_text_z_m():
    assert "Safe" in band_text("Green", "z")
    assert "Distress" in band_text("Red", "z")
    assert "Caution" in band_text("Yellow", "z")
    assert "Unlikely" in band_text("Green", "m")
    assert "Likely manip" in band_text("Red", "m")
    assert "Caution" in band_text("Yellow", "m")
    assert "n/a" in band_text(None, "z").lower() or "n/a" in band_text(float("nan"), "z")


def test_badge_html():
    html = badge("Safe", "Green")
    assert "sfa-badge" in html
    assert "Safe" in html


def test_quality_color_bands():
    assert quality_color(80).startswith("#")
    assert quality_color(20) != quality_color(80)


def test_format_band_columns():
    df = pd.DataFrame({"z_band": ["Green", "Red"], "m_band": ["Yellow", "N/A"]})
    out = format_band_columns(df)
    assert "Safe" in out["z_band"].iloc[0]
    assert "Caution" in out["m_band"].iloc[0]


def test_peers_html_has_badges():
    peers = pd.DataFrame(
        {
            "rank_in_sector": [1, 2],
            "ticker": ["AAA.NS", "BBB.NS"],
            "quality_score": [72.0, 55.0],
            "roe": [18.0, 12.0],
            "pe": [20.0, 25.0],
            "z_band": ["Green", "Yellow"],
            "m_band": ["Green", "Red"],
            "red_flag_count": [0, 2],
        }
    )
    html = peers_html_table(peers)
    assert "sfa-peer-table" in html
    assert "sfa-badge" in html
    assert "AAA" in html


def test_f_score_breakdown_visual():
    from src.institutional_scores import PIOTROSKI_TESTS, PIOTROSKI_LABELS

    row = {t: 1.0 for t in PIOTROSKI_TESTS}
    row["pf_no_dilution"] = 0.0
    row["pf_higher_asset_turnover"] = float("nan")
    assert _f_test_state(row, "pf_positive_net_income") == "ok"
    assert _f_test_state(row, "pf_no_dilution") == "bad"
    assert _f_test_state(row, "pf_higher_asset_turnover") == "na"
    html = f_score_breakdown_html(
        row, PIOTROSKI_TESTS, PIOTROSKI_LABELS, score=7, tests_used=8
    )
    # Inline-styled HTML (class-based CSS is stripped by Streamlit)
    assert "PASS" in html
    assert "FAIL" in html
    assert "Profitability" in html
    assert "background:" in html
    assert "border-radius" in html
