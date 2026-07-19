"""
Outperformance Model (Global Model)
------------------------------------
Trains ONE model across the whole universe to predict the probability that a
stock beats its benchmark over a forward horizon (e.g. 3-5 years).

Key correctness rules baked in:
  * Point-in-time features (caller must supply features as-of each date).
  * Time-based split on unique dates (no random shuffle -> avoids look-ahead).
  * Label = forward_return > benchmark_forward_return.
  * Median imputation fitted on train only (no leakage into valid/test).
  * Rows with missing labels dropped; NaN features filled after split.

Models: LightGBM (default if installed), XGBoost, RandomForest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

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

MIN_TRAIN_ROWS = 20


def make_label(df, ticker_col="ticker", date_col="date",
               fwd_return_col="fwd_return", bench_return_col="bench_fwd_return"):
    """Binary outperformance label. Rows with missing returns become NaN labels."""
    df = df.copy()
    both = df[fwd_return_col].notna() & df[bench_return_col].notna()
    df["target_outperform"] = np.where(
        both,
        (df[fwd_return_col] > df[bench_return_col]).astype(float),
        np.nan,
    )
    return df


def time_split(df, date_col="date", train_end=None, valid_end=None):
    """
    Split by unique calendar dates (not row quantiles), so a panel of many
    tickers on the same dates does not leak future periods into train.

    Everything <= train_end -> train,
    (train_end, valid_end] -> valid, > valid_end -> test.
    """
    dates = pd.to_datetime(df[date_col])
    if train_end is None or valid_end is None:
        unique = pd.Series(sorted(dates.dropna().unique()))
        if len(unique) < 3:
            # Not enough distinct dates for a three-way split: put all labeled
            # history in train, leave valid/test empty (caller must handle).
            train_end = unique.iloc[-1] if len(unique) else pd.Timestamp.max
            valid_end = train_end
        else:
            # ~60% / 20% / 20% of unique dates
            i_train = max(0, int(np.floor(len(unique) * 0.6)) - 1)
            i_valid = max(i_train + 1, int(np.floor(len(unique) * 0.8)) - 1)
            i_valid = min(i_valid, len(unique) - 2)
            train_end = unique.iloc[i_train]
            valid_end = unique.iloc[i_valid]
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
    # Smaller defaults so tiny panels (tests / early history) still fit
    rf_kw = dict(n_estimators=200, max_depth=8, min_samples_leaf=2,
                 n_jobs=-1, random_state=42)
    rf_kw.update(kw)
    return RandomForestClassifier(**rf_kw)


def _labeled(frame, feature_cols, label_col):
    """Keep rows with a non-null label and at least one non-null feature."""
    y = frame[label_col]
    X = frame[feature_cols]
    mask = y.notna() & X.notna().any(axis=1)
    return X.loc[mask].astype(float), y.loc[mask].astype(int)


def train_outperformance_model(df, feature_cols, kind="lightgbm",
                               date_col="date", label_col="target_outperform"):
    """
    Train a classifier with train-only median imputation.

    Returns (pipeline, report). Raises ValueError when data is insufficient.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns present in the training frame.")

    work = df.dropna(subset=[date_col]).copy()
    train, valid, test = time_split(work, date_col)

    Xtr, ytr = _labeled(train, feature_cols, label_col)
    if len(ytr) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS} labeled training rows; got {len(ytr)}."
        )
    if ytr.nunique() < 2:
        raise ValueError("Training labels are all the same class — cannot fit a classifier.")

    clf = build_model(kind)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", clf),
    ])
    pipe.fit(Xtr, ytr)

    report = {
        "n_train": int(len(ytr)),
        "train_base_rate": float(ytr.mean()),
        "features": list(feature_cols),
    }

    for name, frame in {"valid": valid, "test": test}.items():
        X, y = _labeled(frame, feature_cols, label_col)
        if len(y) == 0:
            report[name] = {"auc": None, "n": 0, "base_rate": None, "note": "empty split"}
            continue
        if y.nunique() < 2:
            report[name] = {
                "auc": None, "n": int(len(y)), "base_rate": float(y.mean()),
                "note": "single class",
            }
            continue
        p = pipe.predict_proba(X)[:, 1]
        report[name] = {
            "auc": float(roc_auc_score(y, p)),
            "n": int(len(y)),
            "base_rate": float(y.mean()),
        }

    clf_step = pipe.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        imp = pd.Series(clf_step.feature_importances_, index=feature_cols)
        report["feature_importance"] = imp.sort_values(ascending=False)

    return pipe, report


def predict_proba(model, df, feature_cols):
    """Score rows; missing features are imputed by the fitted pipeline."""
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].astype(float)
    return model.predict_proba(X)[:, 1]
