"""
Panel schema validation and unit normalization
----------------------------------------------
Validates uploaded fundamentals CSVs before scoring, and normalizes common
unit mistakes (e.g. growth stored as 12 instead of 0.12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLS = ("ticker", "date")

# Core scoring metrics (at least some should be present for a useful run)
SCORE_METRICS = (
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth",
    "ebitda_growth", "roe", "roce", "net_margin", "operating_margin",
    "gross_margin", "debt_to_equity", "interest_coverage", "current_ratio",
    "cash_position", "dividend_yield", "pe", "pb", "ev_ebitda", "peg",
    "price_sales",
)

# Growth-like columns expected as *decimals* (0.12 = 12%)
GROWTH_DECIMAL_COLS = (
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth",
    "ebitda_growth", "dividend_growth", "fwd_return", "bench_fwd_return",
)

# Margin / return columns expected as *percent points* (12 = 12%)
PERCENT_POINT_COLS = (
    "roe", "roce", "net_margin", "operating_margin", "gross_margin",
    "dividend_yield", "buyback_yield", "promoter_pledge_pct",
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def raise_if_errors(self):
        if self.errors:
            raise ValueError("; ".join(self.errors))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/strip column names (idempotent)."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(
            out[date_col], format="mixed", dayfirst=True, errors="coerce"
        )
    return out


def normalize_growth_units(df: pd.DataFrame, threshold: float = 2.0) -> tuple[pd.DataFrame, list[str]]:
    """
    If a growth column's median absolute value is > threshold, treat values as
    percent points (12 = 12%) and convert to decimals. Returns (df, notes).
    """
    out = df.copy()
    notes = []
    for col in GROWTH_DECIMAL_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        med = s.abs().median(skipna=True)
        if pd.notna(med) and med > threshold:
            out[col] = s / 100.0
            notes.append(
                f"`{col}` looked like percent points (median |x|={med:.1f}); "
                f"divided by 100 → decimals."
            )
    return out, notes


def validate_panel(df: pd.DataFrame) -> ValidationResult:
    """
    Check a fundamentals panel for hard errors and soft warnings.
    Does not mutate the frame.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    if df is None or df.empty:
        return ValidationResult(False, errors=["Uploaded file is empty."])

    cols = set(df.columns)
    missing_req = [c for c in REQUIRED_COLS if c not in cols]
    if missing_req:
        errors.append(
            "Missing required column(s): "
            + ", ".join(f"`{c}`" for c in missing_req)
            + ". Expected at least `ticker` and `date`."
        )

    if "ticker" in cols:
        n_blank = df["ticker"].isna().sum() + (df["ticker"].astype(str).str.strip() == "").sum()
        if n_blank:
            errors.append(f"{int(n_blank)} row(s) have a blank `ticker`.")
        n_tickers = df["ticker"].nunique(dropna=True)
        info.append(f"{n_tickers} unique ticker(s), {len(df)} row(s).")

    if "date" in cols:
        dates = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
        n_bad = int(dates.isna().sum())
        if n_bad == len(df):
            errors.append(
                "No parseable dates in `date` column. Use YYYY-MM-DD or DD-MM-YYYY."
            )
        elif n_bad:
            warnings.append(f"{n_bad} row(s) have unparseable `date` values (will be dropped or ignored).")

    present_metrics = [c for c in SCORE_METRICS if c in cols]
    if not present_metrics:
        # tickers-only is handled separately; still a warning if they have extra junk
        warnings.append(
            "No scoring metric columns found (e.g. `roe`, `pe`, `revenue_growth`). "
            "If this is tickers-only, run the fetcher first."
        )
    else:
        info.append(f"{len(present_metrics)} scoring metric column(s) present.")

    # Unit sanity: growth as percent points
    for col in GROWTH_DECIMAL_COLS:
        if col not in cols:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        med = s.abs().median(skipna=True)
        if pd.notna(med) and med > 2.0:
            warnings.append(
                f"`{col}` median |value| is {med:.1f} — likely percent points "
                f"(12 for 12%). Will auto-convert to decimals (0.12)."
            )

    # Margins stored as decimals by mistake (0.12 instead of 12)
    for col in ("roe", "net_margin", "operating_margin", "gross_margin"):
        if col not in cols:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) and s.abs().median() < 1.0 and s.abs().max() <= 1.5:
            warnings.append(
                f"`{col}` looks like a decimal fraction (median "
                f"{s.median():.3f}). Expected percent points (e.g. 12 for 12%). "
                "Scoring still works relatively, but tooltips/labels assume %."
            )

    if "fwd_return" in cols and "bench_fwd_return" in cols:
        n_lab = int(
            (
                pd.to_numeric(df["fwd_return"], errors="coerce").notna()
                & pd.to_numeric(df["bench_fwd_return"], errors="coerce").notna()
            ).sum()
        )
        if n_lab:
            info.append(f"{n_lab} row(s) have ML labels (`fwd_return` + `bench_fwd_return`).")
        else:
            info.append("ML labels missing — Train tab needs labels (see build_labels).")

    # Multi-year coverage for YoY rules
    if "ticker" in cols and "date" in cols:
        vc = df.groupby("ticker").size()
        single = int((vc < 2).sum())
        if single:
            warnings.append(
                f"{single} ticker(s) have only one fiscal row — YoY red flags / "
                f"Piotroski trends will be limited for those names."
            )

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, info=info)


def prepare_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult, list[str]]:
    """
    Normalize columns/dates/growth units, then validate.
    Returns (prepared_df, validation, unit_notes).
    """
    out = normalize_columns(df)
    out = parse_dates(out)
    out, unit_notes = normalize_growth_units(out)
    result = validate_panel(out)
    result.info.extend(unit_notes)
    return out, result, unit_notes


def format_growth_pct(value) -> str:
    """Display helper: growth decimal → '12.3%'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def format_pp(value, digits: int = 1) -> str:
    """Display helper: percent-point metric → '12.3'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"
