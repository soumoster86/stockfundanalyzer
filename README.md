# Fundamental Stock Analyzer (ML)

A self-learning fundamental analysis tool: Quality Score Engine + global ML
outperformance model + multi-factor ranking + forensic red-flag detection.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected CSV schema (one row per ticker per period)

Required keys: `ticker`, `date`

Metric columns (all optional but more = better scoring):

| Group | Columns |
|---|---|
| Financial Performance | revenue_growth, eps_growth, operating_profit_growth, fcf_growth, ebitda_growth |
| Profitability | roe, roce, net_margin, operating_margin, gross_margin |
| Financial Strength | debt_to_equity, interest_coverage, current_ratio, cash_position |
| Shareholder | dividend_yield, dividend_growth, buyback_yield, promoter_holding_change |
| Valuation | pe, pb, ev_ebitda, peg, price_sales |

For red flags (raw levels, so YoY can be computed): net_profit,
operating_cash_flow, receivables, revenue, shares_outstanding, total_debt,
auditor, promoter_pledge_pct, insider_net_buy, related_party_txn_flag.

For training the ML model: `fwd_return` (the stock's forward 3-5yr return)
and `bench_fwd_return` (benchmark forward return over the same window),
both measured as-of `date`.

## Sector-relative scoring & explainability

**Sector grouping:** if a `sector` column is present (the fetcher fills it
automatically from Yahoo), the app can rank each stock against its *sector
peers* rather than the whole universe — so a bank's ROE and debt levels are
judged against other banks, not against software firms. Toggle it in the
sidebar. Sectors with fewer than 5 stocks fall back to whole-universe ranking
automatically (ranking 2 stocks against each other is meaningless).

**Explainability:** the Single Stock Report shows *why* a stock got its score —
its top strengths and weakest areas versus peers (as percentiles), a plain-
English summary, and the category breakdown. This makes the score auditable
instead of a black box.

## Data-quality layer (completeness & sanity guards)

Two checks surface where the underlying data can't be trusted, so a score built
on thin or distorted inputs isn't read with false confidence (`src/data_quality.py`):

**Completeness** — each stock shows how many of the ~20 core metrics actually
populated (e.g. "13/20"). Stocks under 50% are flagged low-confidence. The app
also picks each stock's latest *substantially-populated* fiscal row, so a stock
whose newest year hasn't fully reported falls back to its last complete year
rather than ranking on a stub.

**Sanity / corporate-action guards** — flags rows whose figures are distorted:
revenue or share count jumping >150% year-over-year (likely merger/demerger/
split), negative equity (D/E and ROE meaningless), P/E above 200, margins above
100%, or ROE above 100% (tiny equity base). These get a "verify manually"
warning and can be filtered out in the ranking via the **Data** filter.

*Honest limitation:* a clean demerger where the data provider restates the prior
year (so no year-over-year jump appears) is invisible to these checks — only the
resulting weak fundamentals show. Always sanity-check restructured companies
against their actual filings.

## Configurable scoring weights

The Quality Score is a weighted blend of five categories (Financial Performance,
Profitability, Financial Strength, Shareholder Metrics, Valuation). The sidebar
lets you change that blend live and re-score the whole universe:

- **Presets** — Balanced (default), Value tilt, Quality tilt, Growth tilt,
  Safety tilt — express common emphases in one click.
- **Custom** — set each category weight by slider. Weights need not sum to 1;
  the engine renormalizes. The sidebar shows the effective percentage mix.

Changing weights instantly reshapes the ranking and the per-stock
strengths/weaknesses, so the tool reflects *your* view of what matters rather
than one fixed opinion. (Per-metric weights within a category stay at their
defaults; `build_config()` in `src/quality_score.py` is the seam if you want to
expose those too.)

## Interface

- **Single Stock Report** — a semicircular **quality gauge** (color-coded by
  band), red-flag and data-quality panels, the "Why this score?" explainability
  breakdown, category scores, and a **quality trend** showing whether the
  stock's score is improving or declining across its available fiscal years
  (each year scored against that year's peers; flags when peer-group sizes
  differ enough to make the comparison less reliable).
- **Universe Ranking** — searchable, **sector-filterable** leaderboard with
  visual score/quality bars, flag (🚩) and data-warning (⚠️) badges, a Top
  25/50/100/All limiter (so a 2,000+ stock universe stays responsive), and CSV
  export of the current view.
- **Compare** — put 2–5 stocks side by side: a category **radar chart** each,
  quality scores, a metrics comparison table, and per-stock red flags.
- **Sector Overview** — average/median/best quality per sector, stock counts,
  the top stock in each sector, and a bar chart of average quality by sector, so
  you can see where quality clusters before drilling in.

A summary banner (universe size, sectors, average quality, data warnings) shows
the moment data loads. The Single Stock Report uses a **category radar chart**;
the Universe Ranking includes an expandable **score-distribution histogram**.

## Access control (login gate)

The app is protected by a lightweight login (`src/auth.py`). It keeps casual
visitors out — a *soft* gate, not strong security (fine for public-fundamentals
data, not for anything sensitive).

**Set it up:**
1. Generate a password hash locally:
   ```bash
   python -c "import hashlib; print(hashlib.sha256('YOURPASSWORD'.encode()).hexdigest())"
   ```
2. Add credentials to **Streamlit Cloud → app → Settings → Secrets** (or a local
   `.streamlit/secrets.toml`, which is gitignored):
   ```toml
   [auth.users]
   soumo = "your_sha256_hash_here"
   ```
   Add as many `username = "hash"` lines as you want.
3. If no secrets are set, the app runs in **demo mode** (login `demo` / `demo`)
   with a visible warning — so set real credentials before sharing the link.

Passwords are never stored or committed in plaintext (only SHA-256 hashes, only
in secrets). A "Log out" control appears in the sidebar once signed in.

## Deploying to Streamlit Community Cloud

The app is deploy-ready (fetch fundamentals locally, then upload/analyze on the cloud).

1. **Push to GitHub** — this folder, including `src/`, `requirements.txt`,
   `.streamlit/config.toml`, and (optionally) `demo_data.csv`. `.gitignore`
   excludes caches.
2. **Sign in** at https://share.streamlit.io with GitHub.
3. **New app** → pick the repo, branch `main`, main file `app.py` → Deploy.
4. First build takes a few minutes; every later `git push` redeploys.

Notes:
- `requirements.txt` is intentionally slim (streamlit, pandas, numpy,
  scikit-learn). LightGBM/XGBoost and yfinance are **excluded** — they're heavy
  on the free tier and not needed by the hosted app (the model falls back to
  RandomForest; the fetcher runs locally).
- **Workflow:** run `fetch_fundamentals.py` on your machine to produce a CSV,
  then upload that CSV to the live app. The fetcher is not part of the deployed
  app.
- Bundling `demo_data.csv` gives visitors a **Load demo data** button so they
  can explore without uploading.
- Model training on the cloud is **ephemeral** — a trained model lives only for
  the session and isn't persisted across redeploys (and isn't meaningful until
  forward-return labels exist anyway).
- Public Community Cloud apps are visible to anyone with the link; don't commit
  anything sensitive.

## How the pieces fit

1. **Quality Score Engine** (`src/quality_score.py`) — cross-sectional
   percentile-normalizes each metric within a date (optionally sector),
   inverts valuation multiples, weights into category scores, then a 0-100
   composite.
2. **Global ML model** (`src/model.py`) — one model over the whole universe
   predicting P(outperform). Uses **time-based splits** to avoid look-ahead
   leakage. LightGBM / XGBoost / RandomForest selectable.
3. **Multi-Factor Ranking** (`src/ranking.py`) — blends Quality Score and ML
   probability, penalizes red flags, ranks the universe.
4. **Red-Flag Detection** (`src/red_flags.py`) — transparent rule layer for
   earnings-quality, financial, and governance risks.

## Auto-fetch fundamentals (yfinance)

Instead of filling the full template by hand, you can upload a CSV with just
`ticker,date` and fetch the financials automatically:

```bash
pip install yfinance
python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv
```

Then upload `fundamentals.csv` into the app. The fetcher must run **locally**
(it needs internet to reach Yahoo Finance). For Indian stocks use the `.NS`
suffix (e.g. `RELIANCE.NS`).

**What yfinance provides:** revenue/operating-profit/EBITDA/FCF growth, margins,
ROE, ROCE, D/E, current ratio, interest coverage, cash, PE/PB/EV-EBITDA/PEG/PS,
dividend yield, and the earnings/financial red-flag raw inputs.

**What it does NOT provide (left blank):** promoter holding/pledging, insider
trades, related-party flags, auditor changes (India governance data — needs a
paid provider like Tijori/Trendlyne or manual entry), and `fwd_return` /
`bench_fwd_return` (the ML labels — these require historical data where the
3-5yr forward window has already elapsed; you cannot label current-dated rows).

## Other data sources
- India governance/forensic: screener.in exports, NSE/BSE filings, Tijori.
- Global/Denmark: Financial Modeling Prep, EOD Historical Data, SimFin, Refinitiv.

## Important modelling caveats
- **No look-ahead bias**: features must reflect only data publicly available
  at `date` (account for reporting lags).
- **Survivorship bias**: include delisted/bankrupt names in training history.
- **3-5yr labels** mean recent rows can't be labelled yet — they're for
  inference only.
- Retrain periodically as new history accrues — that's the "learns on its own"
  loop (schedule the Train step).
