"""
Sample template generator
--------------------------
Builds a downloadable CSV with the full expected schema, two tickers and two
years each (so YoY red-flag rules have data). TICKER2's latest row is crafted
to trip several red flags as a demonstration.
"""

import pandas as pd

COLUMNS = [
    "ticker", "date", "sector",
    # Financial Performance (growth, decimals: 0.12 = 12%)
    "revenue_growth", "eps_growth", "operating_profit_growth", "fcf_growth", "ebitda_growth",
    # Profitability (%)
    "roe", "roce", "net_margin", "operating_margin", "gross_margin",
    # Financial Strength
    "debt_to_equity", "interest_coverage", "current_ratio", "cash_position",
    # Shareholder
    "dividend_yield", "dividend_growth", "buyback_yield", "promoter_holding_change",
    # Valuation
    "pe", "pb", "ev_ebitda", "peg", "price_sales",
    # Raw columns used by red-flag detection
    "net_profit", "operating_cash_flow", "receivables", "revenue", "shares_outstanding",
    "total_debt", "auditor", "promoter_pledge_pct", "insider_net_buy", "related_party_txn_flag",
    # ML training labels (forward 3-5yr returns, decimals)
    "fwd_return", "bench_fwd_return",
]

_ROWS = [
    dict(ticker="TICKER1", date="2019-03-31", sector="Technology", revenue_growth=0.12, eps_growth=0.15,
         operating_profit_growth=0.13, fcf_growth=0.10, ebitda_growth=0.14,
         roe=18.5, roce=20.1, net_margin=12.0, operating_margin=18.0, gross_margin=42.0,
         debt_to_equity=0.45, interest_coverage=8.2, current_ratio=1.8, cash_position=1200,
         dividend_yield=1.5, dividend_growth=0.08, buyback_yield=0.5, promoter_holding_change=0.0,
         pe=22.0, pb=3.1, ev_ebitda=12.5, peg=1.3, price_sales=2.8,
         net_profit=240, operating_cash_flow=260, receivables=180, revenue=2000,
         shares_outstanding=1000, total_debt=900, auditor="Auditor A",
         promoter_pledge_pct=0.0, insider_net_buy=50, related_party_txn_flag=0,
         fwd_return=0.65, bench_fwd_return=0.40),
    dict(ticker="TICKER1", date="2020-03-31", sector="Technology", revenue_growth=0.10, eps_growth=0.11,
         operating_profit_growth=0.09, fcf_growth=0.07, ebitda_growth=0.10,
         roe=17.0, roce=19.0, net_margin=11.5, operating_margin=17.0, gross_margin=41.0,
         debt_to_equity=0.50, interest_coverage=7.5, current_ratio=1.7, cash_position=1300,
         dividend_yield=1.6, dividend_growth=0.06, buyback_yield=0.0, promoter_holding_change=-0.5,
         pe=20.0, pb=2.9, ev_ebitda=11.8, peg=1.4, price_sales=2.6,
         net_profit=265, operating_cash_flow=255, receivables=210, revenue=2200,
         shares_outstanding=1010, total_debt=980, auditor="Auditor A",
         promoter_pledge_pct=0.0, insider_net_buy=20, related_party_txn_flag=0,
         fwd_return=0.55, bench_fwd_return=0.38),
    dict(ticker="TICKER2", date="2019-03-31", sector="Financial Services", revenue_growth=0.05, eps_growth=0.03,
         operating_profit_growth=0.02, fcf_growth=-0.05, ebitda_growth=0.04,
         roe=9.0, roce=10.0, net_margin=5.0, operating_margin=8.0, gross_margin=28.0,
         debt_to_equity=1.6, interest_coverage=3.0, current_ratio=1.1, cash_position=300,
         dividend_yield=0.8, dividend_growth=0.0, buyback_yield=0.0, promoter_holding_change=-1.0,
         pe=35.0, pb=4.5, ev_ebitda=18.0, peg=2.5, price_sales=4.0,
         net_profit=80, operating_cash_flow=70, receivables=120, revenue=1500,
         shares_outstanding=2000, total_debt=1400, auditor="Auditor B",
         promoter_pledge_pct=5.0, insider_net_buy=-30, related_party_txn_flag=0,
         fwd_return=0.20, bench_fwd_return=0.40),
    dict(ticker="TICKER2", date="2020-03-31", sector="Financial Services", revenue_growth=0.04, eps_growth=0.05,
         operating_profit_growth=0.03, fcf_growth=-0.02, ebitda_growth=0.03,
         roe=8.0, roce=9.0, net_margin=6.0, operating_margin=7.5, gross_margin=27.0,
         debt_to_equity=2.1, interest_coverage=2.2, current_ratio=0.9, cash_position=250,
         dividend_yield=0.7, dividend_growth=0.0, buyback_yield=0.0, promoter_holding_change=-2.0,
         pe=38.0, pb=4.8, ev_ebitda=19.5, peg=2.8, price_sales=4.3,
         net_profit=95, operating_cash_flow=60, receivables=200, revenue=1560,
         shares_outstanding=2200, total_debt=2300, auditor="Auditor C",
         promoter_pledge_pct=12.0, insider_net_buy=-80, related_party_txn_flag=1,
         fwd_return=0.15, bench_fwd_return=0.38),
]


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(_ROWS)[COLUMNS]


def sample_csv_bytes() -> bytes:
    return sample_dataframe().to_csv(index=False).encode("utf-8")


# Column documentation shown in the UI
COLUMN_DOCS = {
    "ticker": "Stock symbol (text)",
    "date": "Reporting/period date, YYYY-MM-DD",
    "sector": "GICS sector (auto-filled by fetcher; enables peer-relative scoring)",
    "revenue_growth": "YoY revenue growth, decimal (0.12 = 12%)",
    "eps_growth": "YoY EPS growth, decimal",
    "operating_profit_growth": "YoY operating profit growth, decimal",
    "fcf_growth": "YoY free cash flow growth, decimal",
    "ebitda_growth": "YoY EBITDA growth, decimal",
    "roe": "Return on equity, %",
    "roce": "Return on capital employed, %",
    "net_margin": "Net profit margin, %",
    "operating_margin": "Operating margin, %",
    "gross_margin": "Gross margin, %",
    "debt_to_equity": "Debt-to-equity ratio (x)",
    "interest_coverage": "Interest coverage ratio (x)",
    "current_ratio": "Current ratio (x)",
    "cash_position": "Cash & equivalents (currency units)",
    "dividend_yield": "Dividend yield, %",
    "dividend_growth": "YoY dividend growth, decimal",
    "buyback_yield": "Buyback yield, %",
    "promoter_holding_change": "Change in promoter holding, percentage points (India)",
    "pe": "Price-to-earnings (x)",
    "pb": "Price-to-book (x)",
    "ev_ebitda": "EV/EBITDA (x)",
    "peg": "PEG ratio (x)",
    "price_sales": "Price-to-sales (x)",
    "net_profit": "Net profit (currency) — red-flag input",
    "operating_cash_flow": "Operating cash flow (currency) — red-flag input",
    "receivables": "Trade receivables (currency) — red-flag input",
    "revenue": "Total revenue (currency) — red-flag input",
    "shares_outstanding": "Shares outstanding — red-flag input (dilution)",
    "total_debt": "Total debt (currency) — red-flag input",
    "auditor": "Auditor name (text) — red-flag input (auditor change)",
    "promoter_pledge_pct": "Promoter shares pledged, % — red-flag input",
    "insider_net_buy": "Net insider buying (negative = selling) — red-flag input",
    "related_party_txn_flag": "1 if material related-party txn flagged, else 0",
    "fwd_return": "Forward 3-5yr stock return, decimal — ML label",
    "bench_fwd_return": "Forward 3-5yr benchmark return, decimal — ML label",
}
