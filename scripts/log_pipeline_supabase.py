#!/usr/bin/env python3
"""
Log a pipeline run to Supabase (CI or local).

  python scripts/log_pipeline_supabase.py \\
    --meta fundamentals_meta.json \\
    --summary artifacts/score_summary.json \\
    --status success

Requires SUPABASE_URL + SUPABASE_KEY (or Streamlit [supabase] secrets when run in app).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_log_supabase import (  # noqa: E402
    build_run_row,
    load_json,
    log_pipeline_run,
)
from src.watchlist_supabase import is_configured  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Log pipeline run to Supabase")
    ap.add_argument("--meta", default="fundamentals_meta.json")
    ap.add_argument("--summary", default="artifacts/score_summary.json")
    ap.add_argument(
        "--status",
        default="success",
        choices=["success", "failed", "partial"],
    )
    ap.add_argument("--error", default="", help="Error message when status=failed")
    ap.add_argument(
        "--require-config",
        action="store_true",
        help="Exit non-zero if Supabase is not configured",
    )
    args = ap.parse_args(argv)

    if not is_configured():
        msg = "Supabase not configured (set SUPABASE_URL and SUPABASE_KEY)"
        print(msg, file=sys.stderr)
        return 2 if args.require_config else 0

    meta = load_json(args.meta)
    summary = load_json(args.summary)
    row = build_run_row(
        status=args.status,
        meta=meta,
        summary=summary,
        error_message=args.error or None,
    )
    result = log_pipeline_run(row)
    if result.get("ok"):
        print(f"Logged pipeline run to Supabase id={result.get('id')} status={args.status}")
        return 0
    print(f"Failed to log pipeline run: {result.get('error')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
