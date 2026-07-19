"""Offline rescore script smoke test."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rescore_demo(tmp_path):
    out = tmp_path / "art"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rescore.py"),
            "--in",
            str(ROOT / "demo_data.csv"),
            "--out-dir",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out / "rankings_latest.csv").exists()
    summary = json.loads((out / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["n_tickers"] >= 1
