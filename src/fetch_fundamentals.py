"""
Fundamentals Fetcher (yfinance)
-------------------------------
Input : a CSV with columns `ticker`, `date` (the date is informational; yfinance
        returns the latest available statements, not point-in-time history).
Output: a fundamentals panel matching the analyzer's expected schema, written to
        a CSV you then upload into the Streamlit app.

USAGE
    python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv

NOTES / LIMITATIONS
  * Covers the ~18 financial metrics + red-flag raw inputs that yfinance exposes.
  * India-specific governance fields (promoter pledging/holding, insider trades,
    related-party, auditor) are NOT available from yfinance and are left blank.
  * yfinance gives ~4 yrs of annual statements; for YoY red-flag rules we emit
    the two most recent fiscal years per ticker when available.
  * Yahoo data can be patchy for smaller caps / ETFs (e.g. *BEES, *ETF, *CASE
    tickers in your list are funds and will mostly return NaN — that's expected).
"""

import argparse
import time
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not installed. Run:  pip install yfinance")


OUTPUT_COLUMNS = [
    "ticker", "date", "sector",
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth", "ebitda_growth",
    "roe", "roce", "net_margin", "operating_margin", "gross_margin",
    "debt_to_equity", "interest_coverage", "current_ratio", "cash_position",
    "dividend_yield", "dividend_growth", "buyback_yield", "promoter_holding_change",
    "pe", "pb", "ev_ebitda", "peg", "price_sales",
    "net_profit", "operating_cash_flow", "receivables", "revenue", "shares_outstanding",
    "total_debt", "auditor", "promoter_pledge_pct", "insider_net_buy", "related_party_txn_flag",
    "fwd_return", "bench_fwd_return",
]

# Fields yfinance cannot provide -> left as NaN (governance/India-specific + labels)
UNAVAILABLE = [
    "promoter_holding_change", "auditor", "promoter_pledge_pct",
    "insider_net_buy", "related_party_txn_flag", "fwd_return", "bench_fwd_return",
]


def _safe(df, row_names, col):
    """Fetch a value from a yfinance statement frame by trying several row labels."""
    if df is None or df.empty or col not in df.columns:
        return np.nan
    for name in row_names:
        if name in df.index:
            v = df.loc[name, col]
            if pd.notna(v):
                return float(v)
    return np.nan


def _pct_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev) or prev == 0:
        return np.nan
    return (curr - prev) / abs(prev)


def fetch_one(ticker: str, date_str: str) -> list:
    """Return one or two rows (latest two fiscal years) for a ticker."""
    rows = []
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fin = tk.financials                 # income statement (cols = years)
        bs = tk.balance_sheet
        cf = tk.cashflow
    except Exception as e:
        print(f"  ! {ticker}: fetch failed ({e})")
        return [_blank_row(ticker, date_str)]

    if fin is None or fin.empty:
        return [_blank_row(ticker, date_str)]

    all_cols = list(fin.columns)
    years = all_cols[:2]            # emit up to 2 most recent fiscal years

    for i, ycol in enumerate(years):
        # previous year column for growth calc: the next column in the full list
        pcol = all_cols[i + 1] if (i + 1) < len(all_cols) else None

        revenue = _safe(fin, ["Total Revenue", "Operating Revenue"], ycol)
        gross_profit = _safe(fin, ["Gross Profit"], ycol)
        op_income = _safe(fin, ["Operating Income", "Total Operating Income As Reported"], ycol)
        ebitda = _safe(fin, ["EBITDA", "Normalized EBITDA"], ycol)
        net_income = _safe(fin, ["Net Income", "Net Income Common Stockholders"], ycol)
        interest_exp = _safe(fin, ["Interest Expense", "Interest Expense Non Operating"], ycol)
        eps = info.get("trailingEps", np.nan) if i == 0 else np.nan

        total_debt = _safe(bs, ["Total Debt"], ycol)
        equity = _safe(bs, ["Stockholders Equity", "Total Stockholder Equity",
                            "Common Stock Equity"], ycol)
        cash = _safe(bs, ["Cash And Cash Equivalents",
                          "Cash Cash Equivalents And Short Term Investments"], ycol)
        cur_assets = _safe(bs, ["Current Assets", "Total Current Assets"], ycol)
        cur_liab = _safe(bs, ["Current Liabilities", "Total Current Liabilities"], ycol)
        receivables = _safe(bs, ["Receivables", "Accounts Receivable",
                                 "Gross Accounts Receivable"], ycol)
        capital_employed = np.nan
        total_assets = _safe(bs, ["Total Assets"], ycol)
        if pd.notna(total_assets) and pd.notna(cur_liab):
            capital_employed = total_assets - cur_liab

        op_cf = _safe(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], ycol)
        capex = _safe(cf, ["Capital Expenditure"], ycol)
        fcf = _safe(cf, ["Free Cash Flow"], ycol)
        if pd.isna(fcf) and pd.notna(op_cf) and pd.notna(capex):
            fcf = op_cf + capex  # capex is negative

        # growth vs previous column
        rev_p = _safe(fin, ["Total Revenue", "Operating Revenue"], pcol) if pcol else np.nan
        op_p = _safe(fin, ["Operating Income"], pcol) if pcol else np.nan
        ebitda_p = _safe(fin, ["EBITDA", "Normalized EBITDA"], pcol) if pcol else np.nan
        ni_p = _safe(fin, ["Net Income"], pcol) if pcol else np.nan
        opcf_p = _safe(cf, ["Operating Cash Flow"], pcol) if pcol else np.nan

        shares = info.get("sharesOutstanding", np.nan)

        row = {c: np.nan for c in OUTPUT_COLUMNS}
        row.update(dict(
            ticker=ticker,
            date=pd.to_datetime(ycol).strftime("%Y-%m-%d"),
            sector=info.get("sector") or "Unknown",
            revenue_growth=_pct_change(revenue, rev_p),
            eps_growth=np.nan,  # needs per-year EPS history; trailing only
            operating_profit_growth=_pct_change(op_income, op_p),
            fcf_growth=_pct_change(fcf, opcf_p) if pd.notna(fcf) else np.nan,
            ebitda_growth=_pct_change(ebitda, ebitda_p),
            roe=(net_income / equity * 100) if pd.notna(net_income) and pd.notna(equity) and equity else np.nan,
            roce=(op_income / capital_employed * 100) if pd.notna(op_income) and pd.notna(capital_employed) and capital_employed else np.nan,
            net_margin=(net_income / revenue * 100) if pd.notna(net_income) and pd.notna(revenue) and revenue else np.nan,
            operating_margin=(op_income / revenue * 100) if pd.notna(op_income) and pd.notna(revenue) and revenue else np.nan,
            gross_margin=(gross_profit / revenue * 100) if pd.notna(gross_profit) and pd.notna(revenue) and revenue else np.nan,
            debt_to_equity=(total_debt / equity) if pd.notna(total_debt) and pd.notna(equity) and equity else np.nan,
            interest_coverage=(op_income / abs(interest_exp)) if pd.notna(op_income) and pd.notna(interest_exp) and interest_exp else np.nan,
            current_ratio=(cur_assets / cur_liab) if pd.notna(cur_assets) and pd.notna(cur_liab) and cur_liab else np.nan,
            cash_position=cash,
            dividend_yield=(info.get("dividendYield", np.nan) or np.nan) if i == 0 else np.nan,
            buyback_yield=np.nan,
            pe=info.get("trailingPE", np.nan) if i == 0 else np.nan,
            pb=info.get("priceToBook", np.nan) if i == 0 else np.nan,
            ev_ebitda=info.get("enterpriseToEbitda", np.nan) if i == 0 else np.nan,
            peg=info.get("pegRatio", np.nan) if i == 0 else np.nan,
            price_sales=info.get("priceToSalesTrailing12Months", np.nan) if i == 0 else np.nan,
            net_profit=net_income,
            operating_cash_flow=op_cf,
            receivables=receivables,
            revenue=revenue,
            shares_outstanding=shares,
            total_debt=total_debt,
        ))
        rows.append(row)

    return rows if rows else [_blank_row(ticker, date_str)]


def _blank_row(ticker, date_str):
    row = {c: np.nan for c in OUTPUT_COLUMNS}
    row["ticker"] = ticker
    row["date"] = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
    row["date"] = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else date_str
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="CSV with ticker,date")
    ap.add_argument("--out", dest="outfile", default="fundamentals.csv")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    args = ap.parse_args()

    src = pd.read_csv(args.infile)
    src.columns = [c.strip().lower() for c in src.columns]
    src = src.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    print(f"Fetching {len(src)} unique tickers...")

    all_rows = []
    failures = []
    for n, r in src.iterrows():
        tkr = str(r["ticker"]).strip()
        print(f"[{n+1}/{len(src)}] {tkr}")
        try:
            all_rows.extend(fetch_one(tkr, str(r.get("date", ""))))
        except Exception as e:
            print(f"  ! {tkr}: skipped ({e})")
            failures.append(tkr)
            all_rows.append(_blank_row(tkr, str(r.get("date", ""))))
        time.sleep(args.sleep)

    out = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
    out.to_csv(args.outfile, index=False)

    filled = out.drop(columns=["ticker", "date"]).notna().mean().mean()
    print(f"\nSaved {len(out)} rows -> {args.outfile}")
    print(f"Average field fill rate: {filled*100:.0f}% "
          f"(governance/label fields intentionally blank)")
    if failures:
        print(f"{len(failures)} tickers had no data (likely ETFs/funds or delisted): "
              f"{', '.join(failures)}")


if __name__ == "__main__":
    main()
