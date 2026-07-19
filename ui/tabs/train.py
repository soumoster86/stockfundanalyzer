"""Train Model tab."""

from __future__ import annotations

import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

from src.enrich import FEATURE_COLS
from src.model import (
    HAS_LGBM,
    HAS_XGB,
    make_label,
    train_outperformance_model,
    predict_proba,
)

MODEL_PATH = os.path.join(tempfile.gettempdir(), "outperformance_model.joblib")


def render_train(data: pd.DataFrame) -> None:
    st.subheader("Train Global Outperformance Model")
    st.info(
        "⚠️ This needs `fwd_return` + `bench_fwd_return` columns (the realized "
        "forward returns). With current-dated fundamentals these labels don't yet "
        "exist, so the model can't be meaningfully trained until enough time has "
        "passed for forward windows to elapse. On the cloud, a trained model lives "
        "only for the current session (it isn't persisted across redeploys)."
    )

    algo_opts = ["randomforest"]
    if HAS_LGBM:
        algo_opts.insert(0, "lightgbm")
    if HAS_XGB:
        algo_opts.append("xgboost")
    kind = st.selectbox(
        "Algorithm",
        algo_opts,
        help="Which model to train. RandomForest is always available; "
             "LightGBM/XGBoost appear only if installed.",
    )
    if not (HAS_LGBM or HAS_XGB):
        st.caption(
            "ℹ️ LightGBM/XGBoost aren't installed in this deployment — using "
            "scikit-learn RandomForest. For local training install optional extras: "
            "`pip install -r requirements-ml.txt`."
        )

    if st.button("Train"):
        if not {"fwd_return", "bench_fwd_return"}.issubset(data.columns):
            st.error("Training needs `fwd_return` and `bench_fwd_return` columns.")
        else:
            train_df = make_label(data)
            feats = [c for c in FEATURE_COLS if c in train_df.columns]
            try:
                model, report = train_outperformance_model(train_df, feats, kind=kind)
            except ValueError as e:
                st.error(f"Cannot train: {e}")
            else:
                joblib.dump({"model": model, "features": feats}, MODEL_PATH)
                st.success(
                    f"Model trained and saved for this session "
                    f"(n_train={report.get('n_train', '?')})."
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
                        st.write(
                            f"**{split.upper()}** — AUC: {auc:.3f} "
                            f"(n={r['n']}, base={r['base_rate']:.2f})"
                        )
                if "feature_importance" in report:
                    st.bar_chart(report["feature_importance"].head(15))

    if os.path.exists(MODEL_PATH) and st.button("Score universe with saved model"):
        bundle = joblib.load(MODEL_PATH)
        probs = predict_proba(bundle["model"], data, bundle["features"])
        data["outperform_proba"] = probs
        st.session_state["outperform_by_ticker"] = dict(zip(data["ticker"], probs))
        st.success(
            "Universe scored. Switch to Ranking tab — scores persist for this session."
        )
