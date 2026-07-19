#!/usr/bin/env python3
"""
Offline / CI re-score job
------------------------
Scores a fundamentals panel and writes ranked CSV + summary JSON.

Used by GitHub Actions monthly schedule and for local batch runs:

    python scripts/rescore.py --in demo_data.csv --out-dir artifacts
    python scripts/rescore.py --in fundamentals.csv --out-dir artifacts --use-sector

Does not fetch from Yahoo (keep CI free of secrets/network flakiness).
Point --in at a refreshed fundamentals CSV produced locally or by another job.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enrich import enrich  # noqa: E402
from src.ranking import rank_universe  # noqa: E402
from src.schema import prepare_panel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a fundamentals panel offline")
    ap.add_argument("--in", dest="inp", required=True, help="Input fundamentals CSV")
    ap.add_argument("--out-dir", default="artifacts", help="Output directory")
    ap.add_argument("--use-sector", action="store_true", help="Sector-relative quality")
    args = ap.parse_args(argv)

    inp = Path(args.inp)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(inp)
    prepared, validation, _notes = prepare_panel(raw)
    if not validation.ok:
        print("Validation errors:", file=sys.stderr)
        for e in validation.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    scored = enrich(prepared, use_sector=args.use_sector)
    ranked = rank_universe(scored, w_quality=1.0, w_ml=0.0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ranked_path = out_dir / f"rankings_{stamp}.csv"
    latest_path = out_dir / "rankings_latest.csv"
    summary_path = out_dir / "score_summary.json"

    ranked.to_csv(ranked_path, index=False)
    ranked.to_csv(latest_path, index=False)

    summary = {
        "generated_at_utc": stamp,
        "source": str(inp),
        "n_rows": int(len(ranked)),
        "n_tickers": int(ranked["ticker"].nunique()) if "ticker" in ranked else 0,
        "avg_quality": float(ranked["quality_score"].mean())
        if "quality_score" in ranked
        else None,
        "median_quality": float(ranked["quality_score"].median())
        if "quality_score" in ranked
        else None,
        "n_data_warnings": int(ranked["data_warning"].sum())
        if "data_warning" in ranked
        else 0,
        "top10": ranked.head(10)[["ticker", "quality_score", "red_flag_count"]].to_dict(
            orient="records"
        )
        if {"ticker", "quality_score"}.issubset(ranked.columns)
        else [],
        "validation_warnings": validation.warnings,
        "use_sector": bool(args.use_sector),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {ranked_path}")
    print(f"Wrote {latest_path}")
    print(f"Wrote {summary_path}")
    print(
        f"Universe: {summary['n_tickers']} tickers, "
        f"avg quality={summary['avg_quality']:.1f}"
        if summary["avg_quality"] is not None
        else f"Universe: {summary['n_tickers']} tickers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
