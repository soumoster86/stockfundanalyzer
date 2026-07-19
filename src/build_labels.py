"""
Forward-return label builder
----------------------------
Attaches ML training labels to a fundamentals panel using historical prices
from Yahoo Finance:

  fwd_return       = price(date + horizon) / price(date) - 1
  bench_fwd_return = same for the benchmark index

USAGE (local, needs network + yfinance):

    pip install yfinance
    python -m src.build_labels --in fundamentals.csv --out labeled.csv
    python -m src.build_labels --in fundamentals.csv --out labeled.csv \\
        --horizon-years 3 --benchmark ^NSEI

Only rows whose forward window has fully elapsed receive labels; recent rows
stay blank (inference-only). Re-upload the labeled CSV into the app to train.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

DEFAULT_BENCHMARK = "^NSEI"  # Nifty 50
DEFAULT_HORIZON_YEARS = 3
TRADING_DAYS_PER_YEAR = 252


def _require_yf():
    if yf is None:
        raise SystemExit("yfinance not installed. Run:  pip install yfinance")


def _download_close(symbol: str, start: str, end: str) -> pd.Series:
    """Adjusted close series indexed by date (tz-naive)."""
    _require_yf()
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        # single ticker: ('Close', symbol) or level0 Close
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = raw.iloc[:, 0]
    else:
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    s = close.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index()
    s.name = symbol
    return s


def price_on_or_after(series: pd.Series, when: pd.Timestamp) -> Optional[float]:
    """First available close on or after `when`."""
    if series is None or series.empty or pd.isna(when):
        return None
    when = pd.Timestamp(when).tz_localize(None) if getattr(when, "tzinfo", None) else pd.Timestamp(when)
    fut = series.loc[series.index >= when]
    if fut.empty:
        return None
    return float(fut.iloc[0])


def forward_return(
    series: pd.Series,
    as_of: pd.Timestamp,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
) -> float:
    """Total return over horizon_years from as_of. NaN if either price missing."""
    p0 = price_on_or_after(series, as_of)
    p1 = price_on_or_after(series, as_of + pd.DateOffset(years=horizon_years))
    if p0 is None or p1 is None or p0 == 0:
        return np.nan
    return p1 / p0 - 1.0


def attach_labels(
    df: pd.DataFrame,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
    benchmark: str = DEFAULT_BENCHMARK,
    sleep_s: float = 0.15,
    price_cache: dict | None = None,
) -> pd.DataFrame:
    """
    Add/overwrite fwd_return and bench_fwd_return columns.
    `price_cache` maps symbol -> close Series (mutated for reuse).
    """
    _require_yf()
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "ticker" not in out.columns or "date" not in out.columns:
        raise ValueError("Panel needs `ticker` and `date` columns.")
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True, errors="coerce")

    cache = price_cache if price_cache is not None else {}
    min_d = out["date"].min() - pd.DateOffset(days=30)
    max_d = out["date"].max() + pd.DateOffset(years=horizon_years + 1)
    start, end = str(min_d.date()), str(max_d.date())

    if benchmark not in cache:
        cache[benchmark] = _download_close(benchmark, start, end)

    bench = cache[benchmark]
    fwd = []
    bench_fwd = []

    for i, row in out.iterrows():
        sym = row["ticker"]
        dt = row["date"]
        if pd.isna(dt) or not sym:
            fwd.append(np.nan)
            bench_fwd.append(np.nan)
            continue
        if sym not in cache:
            try:
                cache[sym] = _download_close(str(sym), start, end)
                if sleep_s:
                    time.sleep(sleep_s)
            except Exception:
                cache[sym] = pd.Series(dtype=float)
        fwd.append(forward_return(cache[sym], dt, horizon_years))
        bench_fwd.append(forward_return(bench, dt, horizon_years))

    out["fwd_return"] = fwd
    out["bench_fwd_return"] = bench_fwd
    return out


def label_coverage(df: pd.DataFrame) -> dict:
    both = df["fwd_return"].notna() & df["bench_fwd_return"].notna() if (
        "fwd_return" in df.columns and "bench_fwd_return" in df.columns
    ) else pd.Series(False, index=df.index)
    return {
        "n_rows": len(df),
        "n_labeled": int(both.sum()),
        "pct_labeled": float(both.mean() * 100) if len(df) else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Attach forward-return ML labels via yfinance")
    ap.add_argument("--in", dest="inp", required=True, help="Input fundamentals CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV with labels")
    ap.add_argument("--horizon-years", type=float, default=DEFAULT_HORIZON_YEARS)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                    help="Yahoo symbol for benchmark (default ^NSEI)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Pause between ticker downloads")
    args = ap.parse_args(argv)

    _require_yf()
    df = pd.read_csv(args.inp)
    print(f"Loaded {len(df)} rows from {args.inp}")
    print(f"Fetching prices (horizon={args.horizon_years}y, bench={args.benchmark})…")
    labeled = attach_labels(
        df,
        horizon_years=args.horizon_years,
        benchmark=args.benchmark,
        sleep_s=args.sleep,
    )
    cov = label_coverage(labeled)
    labeled.to_csv(args.out, index=False)
    print(
        f"Wrote {args.out}: {cov['n_labeled']}/{cov['n_rows']} rows labeled "
        f"({cov['pct_labeled']:.1f}%)."
    )
    if cov["n_labeled"] == 0:
        print(
            "WARNING: no labels attached. Rows may be too recent for the horizon, "
            "or Yahoo had no price history for these tickers.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
