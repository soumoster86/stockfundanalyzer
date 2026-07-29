"""Train Model tab — primary actions first; docs collapsed."""

from __future__ import annotations

import io
import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

from src.enrich import FEATURE_COLS, RAW_FEATURE_COLS
from src.model import (
    HAS_LGBM,
    HAS_XGB,
    make_label,
    pack_model_bundle,
    predict_proba,
    train_outperformance_model,
)
from ui.theme import section

MODEL_PATH = os.path.join(tempfile.gettempdir(), "outperformance_model.joblib")


def _save_bundle(bundle: dict) -> None:
    joblib.dump(bundle, MODEL_PATH)
    st.session_state["model_bundle"] = bundle


def _load_bundle():
    if "model_bundle" in st.session_state:
        return st.session_state["model_bundle"]
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        st.session_state["model_bundle"] = bundle
        return bundle
    return None


def render_train(data: pd.DataFrame) -> None:
    section("Train global outperformance model")

    has_labels = {"fwd_return", "bench_fwd_return"}.issubset(data.columns)
    if has_labels:
        n_lab = int(
            (data["fwd_return"].notna() & data["bench_fwd_return"].notna()).sum()
        )
        st.caption(
            f"Labeled rows in current view: **{n_lab}** "
            f"(training uses the full multi-year panel when available)."
        )
    else:
        st.warning(
            "Current data has no label columns. "
            "Build labels offline (see **How to prepare labels** below)."
        )

    with st.expander("How to prepare labels", expanded=not has_labels):
        st.markdown(
            """
Needs `fwd_return` + `bench_fwd_return` (realized forward returns).

```bash
pip install yfinance
python -m src.build_labels --in fundamentals.csv --out labeled.csv \\
  --horizon-years 3 --benchmark ^NSEI
```

Then re-upload `labeled.csv`. Models are **session-scoped** on Cloud but you can
**download / upload** a joblib bundle to reuse them.
            """
        )

    algo_opts = ["randomforest"]
    if HAS_LGBM:
        algo_opts.insert(0, "lightgbm")
    if HAS_XGB:
        algo_opts.append("xgboost")

    c1, c2 = st.columns(2)
    with c1:
        kind = st.selectbox(
            "Algorithm",
            algo_opts,
            help="RandomForest is always available; LightGBM/XGBoost if installed.",
        )
    with c2:
        include_quality = st.checkbox(
            "Include quality_score as feature",
            value=False,
            help="Off by default — quality is a blend of the other fundamentals.",
        )
    calibrate = st.checkbox(
        "Calibrate probabilities (isotonic)",
        value=False,
        help="Better-aligned probabilities; needs enough labeled rows.",
    )

    if not (HAS_LGBM or HAS_XGB):
        st.caption(
            "Using scikit-learn RandomForest. Optional boosters: "
            "`pip install -r requirements-ml.txt`."
        )

    feats_base = list(RAW_FEATURE_COLS)
    if include_quality:
        feats_base = list(FEATURE_COLS)

    if st.button("Train", type="primary"):
        if not has_labels:
            st.error("Training needs `fwd_return` and `bench_fwd_return` columns.")
        else:
            train_src = st.session_state.get("raw_panel", data)
            train_df = make_label(train_src)
            feats = [c for c in feats_base if c in train_df.columns]
            try:
                model, report = train_outperformance_model(
                    train_df, feats, kind=kind, calibrate=calibrate
                )
            except ValueError as e:
                st.error(f"Cannot train: {e}")
            else:
                bundle = pack_model_bundle(model, feats, report)
                _save_bundle(bundle)
                st.success(
                    f"Model trained (n_train={report.get('n_train', '?')}"
                    f"{', calibrated' if report.get('calibrated') else ''}). "
                    "Download the bundle below to reuse later."
                )
                for split in ("valid", "test"):
                    if split not in report:
                        continue
                    r = report[split]
                    auc = r.get("auc")
                    if auc is None:
                        st.write(
                            f"**{split.upper()}** — AUC: n/a "
                            f"(n={r.get('n', 0)}"
                            + (f", {r['note']}" if r.get("note") else "")
                            + ")"
                        )
                    else:
                        edge = r.get("edge_vs_random", auc - 0.5)
                        st.write(
                            f"**{split.upper()}** — AUC: {auc:.3f} "
                            f"(edge vs random {edge:+.3f}; "
                            f"n={r['n']}, base rate={r['base_rate']:.2f})"
                        )
                if "feature_importance" in report:
                    with st.expander("Feature importance", expanded=True):
                        st.bar_chart(report["feature_importance"].head(15))

    st.markdown("**Model bundle**")
    bundle = _load_bundle()
    if bundle is not None:
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        b1, b2 = st.columns(2)
        with b1:
            st.download_button(
                "⬇️ Download model (.joblib)",
                data=buf.getvalue(),
                file_name="outperformance_model.joblib",
                mime="application/octet-stream",
            )
        with b2:
            if st.button("Score universe with saved model"):
                feats = bundle.get("features") or [
                    c for c in RAW_FEATURE_COLS if c in data.columns
                ]
                try:
                    probs = predict_proba(bundle["model"], data, feats)
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                else:
                    data["outperform_proba"] = probs
                    st.session_state["outperform_by_ticker"] = dict(
                        zip(data["ticker"], probs)
                    )
                    st.success(
                        "Universe scored. Open **Research → Universe Ranking** — "
                        "scores persist for this session."
                    )

    with st.expander("Upload a previously trained model", expanded=False):
        uploaded_model = st.file_uploader(
            "Model file (.joblib / .pkl)",
            type=["joblib", "pkl"],
            key="model_upload",
        )
        if uploaded_model is not None and st.button("Load uploaded model"):
            try:
                bundle = joblib.load(uploaded_model)
                if "model" not in bundle or "features" not in bundle:
                    st.error("Invalid bundle — expected keys `model` and `features`.")
                else:
                    _save_bundle(bundle)
                    st.success(
                        f"Loaded model with {len(bundle['features'])} features. "
                        "Click **Score universe** above."
                    )
            except Exception as e:
                st.error(f"Could not load model: {e}")
