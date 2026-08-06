"""Fundamentals meta / provenance helpers."""
import json
from pathlib import Path

from src.data_freshness import (
    format_freshness_line,
    is_github_daily,
    load_fundamentals_meta,
    write_fundamentals_meta,
)


def test_is_github_daily():
    assert is_github_daily(
        {"source": "github-actions daily-fundamentals", "generated_at_utc": "2026-01-01T00:00:00Z"}
    )
    assert not is_github_daily({"source": "local refresh"})
    assert not is_github_daily(None)


def test_write_and_load_meta(tmp_path: Path):
    write_fundamentals_meta(
        tmp_path,
        source="github-actions daily-fundamentals",
        n_tickers=100,
        workflow_run="12345",
    )
    meta = load_fundamentals_meta(tmp_path)
    assert meta is not None
    assert meta["n_tickers"] == 100
    assert is_github_daily(meta)
    line = format_freshness_line(meta)
    assert "GitHub daily pipeline" in line
    assert "12345" in line


def test_format_without_meta():
    line = format_freshness_line(None, data_source_label="Project: fundamentals.csv")
    assert "fundamentals" in line.lower() or "Project" in line


def test_meta_from_pipeline_row():
    from src.data_freshness import meta_from_pipeline_row

    m = meta_from_pipeline_row(
        {
            "source": "github-actions daily-fundamentals",
            "finished_at": "2026-08-06T16:05:16+00:00",
            "n_tickers": 2367,
            "workflow_run": "31115210975",
            "status": "success",
        }
    )
    assert m["from_supabase"] is True
    assert is_github_daily(m)
    line = format_freshness_line(m)
    assert "2,367" in line or "2367" in line
    assert "31115210975" in line