"""Train and evaluate the failure classifier on the AI4I 2020 dataset.

Usage:
    python cnc_maintenance/train.py --data data/raw/ai4i2020.csv

Splits the data 60/20/20 (train/validation/test, stratified). The validation
set is used for early stopping and to pick the risk threshold; the test set is
only used once, for the final numbers. Writes the model and a metrics file to
models/.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import FEATURES, build_features

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=1,
)


def make_model(y_train):
    neg, pos = np.bincount(y_train)
    return xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)


def best_threshold(y_true, proba):
    """Threshold that maximises F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def scores(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fit(X_tr, y_tr, X_val, y_val):
    model = make_model(y_tr)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def cross_validate(X, y, folds=5):
    """Stratified k-fold: early stopping and threshold picked inside each fold."""
    f1s, pr_aucs = [], []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X.iloc[tr_idx], y[tr_idx], test_size=0.25, stratify=y[tr_idx], random_state=SEED
        )
        m = fit(X_tr, y_tr, X_val, y_val)
        thr = best_threshold(y_val, m.predict_proba(X_val)[:, 1])
        p = m.predict_proba(X.iloc[te_idx])[:, 1]
        f1s.append(f1_score(y[te_idx], (p >= thr).astype(int)))
        pr_aucs.append(average_precision_score(y[te_idx], p))
    return {
        "folds": folds,
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
        "pr_auc_std": round(float(np.std(pr_aucs)), 4),
    }


def single_row_latency_ms(booster, raw_rows, n=1000):
    """End-to-end latency for one raw row: feature engineering + prediction."""
    times = []
    for i in range(n):
        row = raw_rows.iloc[[i % len(raw_rows)]]
        t0 = time.perf_counter()
        booster.inplace_predict(build_features(row))
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times[50:])  # drop warm-up calls
    return {"p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "calls": int(len(times))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "raw", "ai4i2020.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    X = build_features(raw)
    y = raw["Machine failure"].to_numpy()
    print(f"Rows: {len(raw)}  failures: {y.sum()} ({y.mean():.2%})")

    idx = np.arange(len(raw))
    tr_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.25, stratify=y[tr_idx], random_state=SEED)

    model = fit(X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx])
    threshold = best_threshold(y[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1])
    test = scores(y[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1], threshold)
    cv = cross_validate(X, y)

    booster = model.get_booster()
    latency = single_row_latency_ms(booster, raw.iloc[test_idx])

    os.makedirs(args.out, exist_ok=True)
    booster.save_model(os.path.join(args.out, "model.json"))
    meta = {
        "features": FEATURES,
        "threshold": round(threshold, 4),
        "best_iteration": int(model.best_iteration),
        "split": {"train": len(tr_idx), "validation": len(val_idx), "test": len(test_idx),
                  "stratified": True, "seed": SEED},
        "test_metrics": test,
        "cross_validation": cv,
        "latency_single_row": latency,
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps({k: meta[k] for k in
                      ["threshold", "best_iteration", "test_metrics", "cross_validation", "latency_single_row"]},
                     indent=2))


if __name__ == "__main__":
    main()
