"""
Outperformance Model (Global Model)
------------------------------------
Trains ONE model across the whole universe to predict the probability that a
stock beats its benchmark over a forward horizon (e.g. 3-5 years).

Key correctness rules baked in:
  * Point-in-time features (caller must supply features as-of each date).
  * Time-based split (no random shuffling -> avoids look-ahead leakage).
  * Label = forward_return > benchmark_forward_return.

Models: LightGBM (default), XGBoost, RandomForest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Optional boosters
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def make_label(df, ticker_col="ticker", date_col="date",
               fwd_return_col="fwd_return", bench_return_col="bench_fwd_return"):
    """Binary outperformance label."""
    df = df.copy()
    df["target_outperform"] = (df[fwd_return_col] > df[bench_return_col]).astype(int)
    return df


def time_split(df, date_col="date", train_end=None, valid_end=None):
    """
    Split by date. Everything <= train_end -> train,
    (train_end, valid_end] -> valid, > valid_end -> test.
    """
    dates = pd.to_datetime(df[date_col])
    if train_end is None:
        q = dates.quantile([0.6, 0.8])
        train_end, valid_end = q.iloc[0], q.iloc[1]
    train = df[dates <= train_end]
    valid = df[(dates > train_end) & (dates <= valid_end)]
    test = df[dates > valid_end]
    return train, valid, test


def build_model(kind="lightgbm", **kw):
    kind = kind.lower()
    if kind == "lightgbm" and HAS_LGBM:
        return LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, **kw
        )
    if kind == "xgboost" and HAS_XGB:
        return XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, **kw
        )
    # fallback / RF
    return RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=20,
        n_jobs=-1, random_state=42, **kw
    )


def train_outperformance_model(df, feature_cols, kind="lightgbm",
                               date_col="date", label_col="target_outperform"):
    train, valid, test = time_split(df, date_col)

    Xtr, ytr = train[feature_cols], train[label_col]
    Xva, yva = valid[feature_cols], valid[label_col]
    Xte, yte = test[feature_cols], test[label_col]

    model = build_model(kind)
    model.fit(Xtr, ytr)

    report = {}
    for name, (X, y) in {"valid": (Xva, yva), "test": (Xte, yte)}.items():
        if len(y) and y.nunique() > 1:
            p = model.predict_proba(X)[:, 1]
            report[name] = {
                "auc": roc_auc_score(y, p),
                "n": len(y),
                "base_rate": float(y.mean()),
            }

    # feature importance
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        report["feature_importance"] = imp.sort_values(ascending=False)

    return model, report


def predict_proba(model, df, feature_cols):
    return model.predict_proba(df[feature_cols])[:, 1]
