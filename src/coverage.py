"""
Universe data-coverage summary
------------------------------
How complete the loaded panel is for scoring, institutional metrics, labels,
and governance — used in the main data banner and optional sidebar.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

GOV_COLS = (
    "auditor",
    "promoter_pledge_pct",
    "promoter_holding_change",
    "insider_net_buy",
    "related_party_txn_flag",
)


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def coverage_summary(
    scored: pd.DataFrame,
    raw: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Summarize coverage on the *scored* latest-row universe (one row per ticker).

    Also peeks at raw multi-year panel for label / row counts when provided.
    """
    n = int(scored["ticker"].nunique()) if "ticker" in scored.columns else len(scored)
    out: dict[str, Any] = {
        "n_tickers": n,
        "n_rows_scored": len(scored),
    }

    # Completeness
    if "data_completeness" in scored.columns:
        c = scored["data_completeness"].fillna(0)
        out["avg_completeness"] = float(c.mean())
        out["pct_sparse"] = _pct(int((c < 0.5).sum()), len(scored))
        out["pct_completeish"] = _pct(int((c >= 0.7).sum()), len(scored))
    else:
        out["avg_completeness"] = None
        out["pct_sparse"] = None
        out["pct_completeish"] = None

    if "data_warning" in scored.columns:
        out["n_data_warnings"] = int(scored["data_warning"].astype(bool).sum())
        out["pct_data_warnings"] = _pct(out["n_data_warnings"], len(scored))
    else:
        out["n_data_warnings"] = 0
        out["pct_data_warnings"] = 0.0

    # Piotroski
    if "f_tests_used" in scored.columns:
        ftu = scored["f_tests_used"].fillna(0)
        out["pct_f_full"] = _pct(int((ftu >= 9).sum()), len(scored))
        out["pct_f_any"] = _pct(int((ftu > 0).sum()), len(scored))
        out["avg_f_tests"] = float(ftu.mean())
    else:
        out["pct_f_full"] = out["pct_f_any"] = out["avg_f_tests"] = None

    # Altman Z
    if "z_score" in scored.columns:
        out["pct_z"] = _pct(int(scored["z_score"].notna().sum()), len(scored))
    else:
        out["pct_z"] = None

    # Beneish M
    if "m_score" in scored.columns:
        out["pct_m"] = _pct(int(scored["m_score"].notna().sum()), len(scored))
    else:
        out["pct_m"] = None

    # Labels (prefer scored; fall back to raw)
    lab_src = scored
    if raw is not None and {"fwd_return", "bench_fwd_return"}.issubset(raw.columns):
        lab_src = raw
    if {"fwd_return", "bench_fwd_return"}.issubset(lab_src.columns):
        both = lab_src["fwd_return"].notna() & lab_src["bench_fwd_return"].notna()
        out["n_labeled_rows"] = int(both.sum())
        if "ticker" in lab_src.columns:
            out["n_labeled_tickers"] = int(lab_src.loc[both, "ticker"].nunique())
        else:
            out["n_labeled_tickers"] = out["n_labeled_rows"]
        out["pct_labeled_tickers"] = _pct(
            out["n_labeled_tickers"], n if n else 1
        )
    else:
        out["n_labeled_rows"] = 0
        out["n_labeled_tickers"] = 0
        out["pct_labeled_tickers"] = 0.0

    # Governance: any non-null among GOV cols
    gov_present = [c for c in GOV_COLS if c in scored.columns]
    if gov_present:
        has_gov = scored[gov_present].notna().any(axis=1)
        out["pct_governance"] = _pct(int(has_gov.sum()), len(scored))
        out["n_governance"] = int(has_gov.sum())
    else:
        out["pct_governance"] = 0.0
        out["n_governance"] = 0

    # Quality distribution (helps explain screens)
    if "quality_score" in scored.columns:
        q = scored["quality_score"].dropna()
        out["quality_median"] = float(q.median()) if len(q) else None
        out["quality_p75"] = float(q.quantile(0.75)) if len(q) else None
        out["quality_max"] = float(q.max()) if len(q) else None
        out["n_q_ge_55"] = int((scored["quality_score"].fillna(-1) >= 55).sum())
        out["n_q_ge_65"] = int((scored["quality_score"].fillna(-1) >= 65).sum())
    else:
        out["quality_median"] = out["quality_p75"] = out["quality_max"] = None
        out["n_q_ge_55"] = out["n_q_ge_65"] = 0

    return out


def coverage_banner_text(cov: dict[str, Any]) -> str:
    """One-line summary for the main data banner."""
    parts = [f"**Universe:** {cov.get('n_tickers', 0):,} stocks"]
    if cov.get("avg_completeness") is not None:
        parts.append(f"avg fill {cov['avg_completeness']*100:.0f}%")
    if cov.get("pct_sparse") is not None:
        parts.append(f"sparse {cov['pct_sparse']:.0f}%")
    if cov.get("pct_f_full") is not None:
        parts.append(f"F 9/9 {cov['pct_f_full']:.0f}%")
    if cov.get("pct_z") is not None:
        parts.append(f"Z {cov['pct_z']:.0f}%")
    if cov.get("pct_m") is not None:
        parts.append(f"M {cov['pct_m']:.0f}%")
    if cov.get("pct_governance") is not None:
        parts.append(f"gov {cov['pct_governance']:.0f}%")
    if cov.get("n_labeled_tickers"):
        parts.append(f"labels {cov['n_labeled_tickers']:,}")
    return " · ".join(parts)


def coverage_detail_lines(cov: dict[str, Any]) -> list[str]:
    """Bullet lines for an expander / sidebar."""
    lines = []
    n = cov.get("n_tickers") or 0
    lines.append(f"**{n:,}** unique tickers in the scored universe.")
    if cov.get("avg_completeness") is not None:
        lines.append(
            f"Core-metric completeness: **{cov['avg_completeness']*100:.0f}%** average; "
            f"**{cov['pct_sparse']:.0f}%** of names sparse (<50% filled)."
        )
    if cov.get("n_data_warnings"):
        lines.append(
            f"Data warnings: **{cov['n_data_warnings']:,}** "
            f"({cov['pct_data_warnings']:.0f}% of universe)."
        )
    if cov.get("pct_f_any") is not None:
        lines.append(
            f"Piotroski: **{cov['pct_f_any']:.0f}%** with any tests; "
            f"**{cov['pct_f_full']:.0f}%** full 9/9 "
            f"(avg {cov['avg_f_tests']:.1f} tests used)."
        )
    if cov.get("pct_z") is not None:
        lines.append(f"Altman Z computable: **{cov['pct_z']:.0f}%**.")
    if cov.get("pct_m") is not None:
        lines.append(f"Beneish M computable: **{cov['pct_m']:.0f}%**.")
    lines.append(
        f"Governance fields present on **{cov.get('n_governance', 0):,}** names "
        f"({cov.get('pct_governance', 0):.0f}%)."
    )
    lines.append(
        f"ML labels: **{cov.get('n_labeled_tickers', 0):,}** tickers "
        f"({cov.get('n_labeled_rows', 0):,} rows)."
    )
    if cov.get("quality_median") is not None:
        lines.append(
            f"Quality distribution: median **{cov['quality_median']:.0f}**, "
            f"p75 **{cov['quality_p75']:.0f}**, max **{cov['quality_max']:.0f}** · "
            f"Q≥55: **{cov['n_q_ge_55']:,}** · Q≥65: **{cov['n_q_ge_65']:,}**."
        )
    return lines
