"""Pipeline run logging helpers (mocked Supabase)."""
from src.pipeline_log_supabase import build_run_row, log_pipeline_run


def test_build_run_row_merges_meta_and_summary():
    row = build_run_row(
        status="success",
        meta={
            "source": "github-actions daily-fundamentals",
            "workflow_run": "99",
            "n_tickers": 100,
            "n_rows": 200,
            "use_sector": "true",
            "commit_csv": "true",
            "workflow_url": "https://example.com/run/99",
            "generated_at_utc": "2026-07-01T02:00:00Z",
        },
        summary={
            "avg_quality": 55.5,
            "median_quality": 54.0,
            "n_data_warnings": 3,
            "n_tickers": 100,
        },
    )
    assert row["workflow_run"] == "99"
    assert row["status"] == "success"
    assert row["n_tickers"] == 100
    assert row["avg_quality"] == 55.5
    assert row["use_sector"] is True
    assert row["commit_csv"] is True
    assert row["details"]["meta"]["workflow_run"] == "99"


def test_log_pipeline_run_not_configured(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline_log_supabase.is_configured", lambda: False
    )
    out = log_pipeline_run({"status": "success", "source": "test"})
    assert out["ok"] is False
    assert "not configured" in out["error"].lower()


def test_log_pipeline_run_upsert(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline_log_supabase.is_configured", lambda: True
    )

    class _Q:
        def __init__(self):
            self._payload = None

        def upsert(self, payload, on_conflict=None):
            self._payload = payload
            self._conflict = on_conflict
            return self

        def insert(self, payload):
            self._payload = payload
            return self

        def execute(self):
            class R:
                data = [{"id": 7}]

            return R()

    class _Client:
        def table(self, name):
            assert name == "pipeline_runs"
            return _Q()

    monkeypatch.setattr(
        "src.pipeline_log_supabase.get_client", lambda: _Client()
    )
    row = build_run_row(
        status="success",
        meta={"workflow_run": "42", "n_tickers": 10},
        summary={"avg_quality": 60.0},
    )
    out = log_pipeline_run(row)
    assert out["ok"] is True
    assert out["id"] == 7
