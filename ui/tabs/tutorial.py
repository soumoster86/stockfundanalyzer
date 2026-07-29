"""
In-app tutorial — how to use Fundamental Stock Analyzer.
"""

from __future__ import annotations

import streamlit as st

from ui.theme import section


def _step(n: int, title: str, body: str) -> None:
    st.markdown(
        f'<div style="display:flex;gap:12px;align-items:flex-start;margin:0 0 12px 0;">'
        f'<div style="width:28px;height:28px;min-width:28px;border-radius:50%;'
        f"background:rgba(29,158,117,0.2);border:1px solid #1D9E75;color:#26B583;"
        f'font-weight:700;text-align:center;line-height:28px;font-size:0.85rem;">{n}</div>'
        f'<div style="min-width:0;">'
        f'<div style="font-weight:650;color:#E6EDF3;margin-bottom:4px;">{title}</div>'
        f'<div style="color:#8B949E;font-size:0.9rem;line-height:1.45;">{body}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_tutorial(*, show_header: bool = True, show_jump_buttons: bool = True) -> None:
    """
    Full how-to-use guide. Used as the Tutorial nav page, login left column,
    and other surfaces that previously showed “What this app offers”.
    """
    if show_header:
        section("Tutorial — how to use this app")
    st.caption(
        "A short guided tour. Expand any section. "
        "This app scores **fundamentals** (financial statements), not day-trading charts."
    )

    # ---- Quick start ----
    with st.expander("⚡ Quick start (5 minutes)", expanded=True):
        _step(
            1,
            "Get data into the app",
            "By default the app loads <b>fundamentals.csv</b> from the project folder. "
            "Or upload a CSV in the sidebar. No file? Run the fetcher locally (see “Load data” below) "
            "or click <b>Load demo data</b>.",
        )
        _step(
            2,
            "Open Universe Ranking",
            "See the whole universe sorted by composite score. Use a <b>Screen</b> preset "
            "(e.g. Clean quality) to filter. Click a ticker → <b>Open report</b>.",
        )
        _step(
            3,
            "Read Single Stock Report",
            "Quality gauge, red flags, Piotroski F / Altman Z / Beneish M, category radar, "
            "trend chart, and sector peers. Add names to your <b>Watchlist</b> with ☆.",
        )
        _step(
            4,
            "Optional: Train Model",
            "Only useful if your CSV has <code>fwd_return</code> and <code>bench_fwd_return</code> "
            "(build with <code>python -m src.build_labels</code>). Otherwise ignore this tab.",
        )
        st.info(
            "**Tip:** Scores are **relative** (percentiles vs peers). A “65” means stronger "
            "than most of the comparison group — not an absolute grade like a school exam."
        )

    # ---- Data ----
    with st.expander("📂 1. Load data", expanded=False):
        st.markdown(
            """
**Where data comes from (priority order)**

1. CSV you **upload** in the sidebar (overrides everything for this session)  
2. Project file **`fundamentals.csv`** (or `labeled.csv` if fundamentals is missing)  
3. **Demo data** button  

**Generate fundamentals from a ticker list (local machine)**

```bash
pip install yfinance
python -m src.fetch_fundamentals --in stocks.csv --out fundamentals.csv
```

- Input: CSV with a `ticker` column (e.g. `RELIANCE.NS`, `TCS.NS`)  
- Output: multi-year panel ready to upload / leave in the project folder  

**Units in the CSV**

| Kind | Examples | Storage |
|------|----------|---------|
| Growth / returns | `revenue_growth`, `fwd_return` | Decimal (`0.12` = 12%) |
| Profitability | `roe`, margins | Percent points (`12` = 12%) |
| Multiples | `pe`, `debt_to_equity` | Ratio |

**India governance** (optional): sidebar → governance template → upload. Yahoo does not provide
promoter pledge, insider, auditor, related-party.

**Banner after load:** shows data source, file updated time, and fiscal date range in the panel.
            """
        )

    # ---- Navigation ----
    with st.expander("🧭 2. Navigate the app", expanded=False):
        st.markdown(
            """
Use the **pill bar** under the header:

| Page | Use it for |
|------|------------|
| **Single Stock Report** | Deep dive on one name |
| **Universe Ranking** | Leaderboard + screens + export |
| **Watchlist** | Your short list & portfolio snapshot |
| **Compare** | 2–5 stocks side by side |
| **Sector Overview** | Where quality clusters by sector |
| **Train Model** | ML outperformance (needs labels) |
| **Tutorial** | This guide |

**Sidebar:** data source, scoring weight presets, sector peer ranking toggle, logout.
            """
        )

    # ---- Report ----
    with st.expander("📄 3. Single Stock Report", expanded=False):
        st.markdown(
            """
1. Pick a **ticker** (or open one from Ranking / Watchlist / peers).  
2. Read the **status badges** (quality band, flags, F / Z / M).  
3. **Quality gauge** — 0–100 composite; label Strong / Average / Weak etc.  
4. **Institutional scores**
   - **Piotroski F (0–9):** financial health checklist — expand *F-Score test breakdown* for pass/fail.  
   - **Altman Z:** bankruptcy risk — 🟢 Safe · 🟡 Caution · 🔴 Distress.  
   - **Beneish M:** earnings-manipulation model — 🟢 Unlikely · 🟡 Caution · 🔴 Likely manip.  
5. **Why this score?** — top strengths / weakest metrics vs peers.  
6. **Category radar** — Growth, Profit, Strength, Shareholder, Value.  
7. **Quality trend** — multi-year score path (needs ≥2 fiscal years).  
8. **Red flags** — rule-based issues (not the same as Beneish M).  
9. **Sector peers** — top quality names in the same sector; open a peer report.  
10. **☆ Watchlist** — save the name for later.
            """
        )

    # ---- Ranking ----
    with st.expander("🏆 4. Universe Ranking & screens", expanded=False):
        st.markdown(
            """
**Leaderboard:** one row per stock (latest solid fiscal year), sorted by composite score  
(quality minus red-flag penalty; optional ML blend if you trained a model).

**Screen presets** (dropdown) — choosing a screen resets thresholds & flag filters:

| Screen | Rules |
|--------|--------|
| **All (default)** | No extra filters |
| **Clean quality** | Q≥55, no red flags, reliable (day-to-day shortlist) |
| **Clean quality (elite)** | Q≥65, no red flags, reliable (very few names) |
| **Top 10% / 20% quality** | Relative cut by quality percentile + reliable |
| **Value quality** | Q≥55, P/E ≤ 25, no red flags, reliable |
| **Low leverage** | Q≥50, D/E ≤ 1, reliable |
| **Watchlist only** | Only names on your watchlist |

After you pick a screen, the **Screen funnel** metrics show how many names
survive each filter step (universe → flags → reliable → Q cut → …).

You can still tweak **Min quality / Max P/E / Max D/E / Flags / Data** after selecting a screen.  
Save custom combos under **Save / delete custom screen**.

**Open report:** select a ticker under the table → **Open report** (jumps to Single Stock Report).
            """
        )

    # ---- Watchlist / compare / sector ----
    with st.expander("⭐ 5. Watchlist, Compare, Sector Overview", expanded=False):
        st.markdown(
            """
**Watchlist**
- Add from Report (☆) or Ranking (**Add to watchlist**).  
- See avg quality, flags, sector mix.  
- Default storage is this browser session; enable Supabase secrets for a durable
  per-user list (see Watchlist page expander / README).  
- Import/export CSV (`ticker` column) as backup or migration.  
- Open or remove names from this page.

**Compare**
- Choose 2–5 stocks.  
- Radars + quality/Z/M badges + side-by-side metrics + red flags.

**Sector Overview**
- Average/median quality by sector, strongest sector, top stock per sector.  
- Use Ranking’s **Sector** filter to drill into one industry.
            """
        )

    # ---- Weights ----
    with st.expander("⚖️ 6. Scoring weights (sidebar)", expanded=False):
        st.markdown(
            """
Quality score blends five categories:

- Financial Performance (growth)  
- Profitability  
- Financial Strength  
- Shareholder Metrics  
- Valuation  

**Presets:** Balanced · Value · Quality · Growth · Safety · Custom sliders.  
Weights are **renormalized** (they need not sum to 1).  
**Rank within sector** — judge each name vs its sector peers (small sectors fall back to full universe).

Changing weights re-scores the universe (cached when inputs are unchanged).
            """
        )

    # ---- ML ----
    with st.expander("🤖 7. Train Model (optional)", expanded=False):
        st.markdown(
            """
Only useful if rows have **realized** forward returns:

```bash
python -m src.build_labels --in fundamentals.csv --out labeled.csv \\
  --horizon-years 3 --benchmark ^NSEI
```

Then upload `labeled.csv` (or replace `fundamentals.csv`).

**Train Model tab**
1. Choose algorithm (RandomForest always; LightGBM/XGBoost if installed via `requirements-ml.txt`).  
2. Prefer **not** including `quality_score` as a feature (default).  
3. **Train** → see valid/test AUC and edge vs random.  
4. **Download** `.joblib` to reuse later; **Score universe** to fill Outperf. column in Ranking.  
5. On Ranking, use the quality vs ML weight slider when scores exist.

Models are **session-scoped** on Streamlit Cloud unless you download the joblib.
            """
        )

    # ---- FAQ ----
    with st.expander("❓ FAQ & tips", expanded=False):
        st.markdown(
            """
**Why is Clean quality almost empty?**  
Quality is peer-relative and often clustered around 30–50. Q≥65 + no flags + reliable data
can leave only a handful of names. The match-count caption shows the filter is working.

**Quality vs red flags vs F / Z / M**  
- **Quality** — relative multi-factor score.  
- **Red flags** — explicit forensic rules.  
- **F / Z / M** — absolute institutional models (health / bankruptcy / manipulation risk).

**What do Z and M colors mean?**  
- 🟢 ok · 🟡 caution · 🔴 elevated risk  
- M “Likely manip.” is a **model signal**, not proof of fraud — review filings.

**Can I use this without login secrets?**  
Local only: `STOCKFUN_DEMO=1 streamlit run app.py` (demo / demo).  
For any shared deploy, set PBKDF2 hashes via `python -m src.auth yourpassword`.

**Educational only**  
Not investment advice. Always check filings and your own research.
            """
        )

    if show_jump_buttons:
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("→ Open Universe Ranking", use_container_width=True, key="tut_goto_rank"):
                st.session_state["_pending_nav"] = "Universe Ranking"
                st.rerun()
        with c2:
            if st.button("→ Open Single Stock Report", use_container_width=True, key="tut_goto_report"):
                st.session_state["_pending_nav"] = "Single Stock Report"
                st.rerun()
        with c3:
            if st.button("→ Open Watchlist", use_container_width=True, key="tut_goto_wl"):
                st.session_state["_pending_nav"] = "Watchlist"
                st.rerun()
