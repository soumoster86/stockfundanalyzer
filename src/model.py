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

# Sparse India panels often have few realized multi-year forward returns.
# 15 is enough for a thin RandomForest; prefer more labels when possible.
MIN_TRAIN_ROWS = 15


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


def adaptive_time_split(
    df,
    feature_cols,
    date_col="date",
    label_col="target_outperform",
    min_train_rows: int = MIN_TRAIN_ROWS,
):
    """
    Time-based split that expands the train window when labels are sparse.

    Default 60/20/20-by-date often leaves almost all labels in valid/test when
    only early fiscal years have realized forward returns (e.g. 20 labeled rows
    total → 19 in train). Walk train_end later until min_train_rows labeled
    rows exist (still no future leakage into train). If still short, put all
    labeled history in train.
    """
    train, valid, test = time_split(df, date_col=date_col)
    Xtr, ytr = _labeled(train, feature_cols, label_col)
    if len(ytr) >= min_train_rows and ytr.nunique() >= 2:
        return train, valid, test, {"split": "default_60_20_20"}

    dates = pd.to_datetime(df[date_col])
    unique = pd.Series(sorted(dates.dropna().unique()))
    if unique.empty:
        return train, valid, test, {"split": "empty"}

    # Expand train_end date-by-date until enough labeled train rows
    for i in range(len(unique)):
        train_end = unique.iloc[i]
        tr = df[dates <= train_end]
        Xtr, ytr = _labeled(tr, feature_cols, label_col)
        if len(ytr) >= min_train_rows and ytr.nunique() >= 2:
            # Remainder → valid (no separate test when labels are scarce)
            va = df[dates > train_end]
            te = df.iloc[0:0].copy()
            return tr, va, te, {
                "split": "adaptive_expand_train",
                "train_end": str(pd.Timestamp(train_end).date()),
                "n_train_labeled": int(len(ytr)),
            }

    # Last resort: everything in train (still better than refusing to fit)
    Xall, yall = _labeled(df, feature_cols, label_col)
    empty = df.iloc[0:0].copy()
    return df, empty, empty, {
        "split": "all_in_train",
        "n_train_labeled": int(len(yall)),
        "n_labeled_total": int(len(yall)),
    }


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


def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce to float, replace ±inf with NaN, clip absurd magnitudes.

    Yahoo / fundamental ratios occasionally produce inf (e.g. PE with ~0 EPS).
    sklearn SimpleImputer + trees then fail at predict with float64 overflow.
    """
    out = X.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # Keep finite range well inside float64; still leaves room for large cash figures
    out = out.clip(lower=-1e15, upper=1e15)
    return out


def _labeled(frame, feature_cols, label_col):
    """Keep rows with a non-null label and at least one non-null feature."""
    y = frame[label_col]
    X = sanitize_features(frame[feature_cols])
    mask = y.notna() & X.notna().any(axis=1)
    return X.loc[mask].astype(float), y.loc[mask].astype(int)


def train_outperformance_model(
    df,
    feature_cols,
    kind="lightgbm",
    date_col="date",
    label_col="target_outperform",
    calibrate: bool = False,
):
    """
    Train a classifier with train-only median imputation.

    If calibrate=True and the validation split has both classes, wraps the
    classifier in sklearn CalibratedClassifierCV (isotonic, cv='prefit' style
    via a small holdout fit on validation when large enough).

    Returns (pipeline, report). Raises ValueError when data is insufficient.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns present in the training frame.")

    work = df.dropna(subset=[date_col]).copy()
    if label_col not in work.columns:
        raise ValueError(
            f"Missing label column `{label_col}`. Run make_label() first "
            "(needs fwd_return and bench_fwd_return)."
        )

    train, valid, test, split_info = adaptive_time_split(
        work, feature_cols, date_col=date_col, label_col=label_col
    )

    Xtr, ytr = _labeled(train, feature_cols, label_col)
    X_all, y_all = _labeled(work, feature_cols, label_col)
    if len(ytr) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS} labeled training rows; got {len(ytr)} "
            f"in train ({len(y_all)} labeled in the whole panel). "
            "Forward-return labels only exist for older fiscal years (need price "
            "history through the label horizon). Re-run "
            "`python -m src.build_labels` with more history, or a shorter "
            "`--horizon-years` (e.g. 1 or 2), then re-upload labeled.csv."
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
        "n_labeled_total": int(len(y_all)),
        "train_base_rate": float(ytr.mean()),
        "baseline_auc": 0.5,
        "features": list(feature_cols),
        "calibrated": False,
        "split": split_info,
    }


    # Optional probability calibration on the validation slice
    if calibrate:
        Xva, yva = _labeled(valid, feature_cols, label_col)
        if len(yva) >= 10 and yva.nunique() > 1:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                # Fit imputer+clf already done; calibrate using frozen base via cv=3 on train
                base = build_model(kind)
                cal_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3)),
                ])
                cal_pipe.fit(Xtr, ytr)
                pipe = cal_pipe
                report["calibrated"] = True
            except Exception as e:
                report["calibrate_error"] = str(e)

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
        auc = float(roc_auc_score(y, p))
        report[name] = {
            "auc": auc,
            "n": int(len(y)),
            "base_rate": float(y.mean()),
            "edge_vs_random": auc - 0.5,
        }

    clf_step = pipe.named_steps["clf"]
    # CalibratedClassifierCV nests the estimator
    inner = clf_step
    if hasattr(inner, "calibrated_classifiers_"):
        try:
            inner = inner.calibrated_classifiers_[0].estimator
        except Exception:
            inner = clf_step
    # Imputer may drop all-null columns → importances length < len(feature_cols)
    kept_features = list(feature_cols)
    imputer = pipe.named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "statistics_"):
        stats = np.asarray(imputer.statistics_)
        if len(stats) == len(feature_cols):
            kept_features = [
                f for f, s in zip(feature_cols, stats) if not (isinstance(s, float) and np.isnan(s))
            ]
            # sklearn keep_empty_features default False drops nan statistics cols
            mask = ~np.isnan(stats.astype(float, copy=False))
            if mask.shape[0] == len(feature_cols) and hasattr(inner, "feature_importances_"):
                if int(mask.sum()) == len(inner.feature_importances_):
                    kept_features = [f for f, m in zip(feature_cols, mask) if m]
    try:
        if hasattr(inner, "feature_importances_"):
            fi = inner.feature_importances_
            if len(fi) == len(kept_features):
                report["feature_importance"] = pd.Series(fi, index=kept_features).sort_values(
                    ascending=False
                )
            elif len(fi) == len(feature_cols):
                report["feature_importance"] = pd.Series(fi, index=feature_cols).sort_values(
                    ascending=False
                )
        elif hasattr(clf_step, "feature_importances_"):
            fi = clf_step.feature_importances_
            if len(fi) == len(kept_features):
                report["feature_importance"] = pd.Series(fi, index=kept_features).sort_values(
                    ascending=False
                )
    except Exception as e:
        report["feature_importance_error"] = str(e)

    return pipe, report


def predict_proba(model, df, feature_cols):
    """Score rows; missing features are imputed by the fitted pipeline."""
    feature_cols = [c for c in feature_cols if c in df.columns]
    # Align columns the model was trained on (missing → NaN → imputer)
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan
    X = sanitize_features(X[feature_cols])
    return model.predict_proba(X)[:, 1]


def pack_model_bundle(model, features, report=None) -> dict:
    """Serializable bundle for joblib download/upload."""
    return {
        "model": model,
        "features": list(features),
        "report": report or {},
        "version": 1,
    }

