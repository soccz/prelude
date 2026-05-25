"""Train recommendation-quality meta-label model.

The base XGBoost heads generate candidates. This meta model learns from closed
paper/shadow ledger rows whether a candidate became net-positive under the
current TP5/EOD rule, then supplies a deployable confidence filter only if a
time holdout shows improvement.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.recommendation_quality import (
    CATEGORICAL_META_FEATURES,
    DEFAULT_META_MODEL_DIR,
    META_FEATURES,
    NUMERIC_META_FEATURES,
    build_meta_feature_frame,
)
from scripts.idea_validation_report import add_result_columns, load_candidate_ledger


MODEL_ID = "recommendation_quality_meta_label_v1"
MODEL_VERSION = "2026-05-25.1"


def _maybe_float(value, digits: int = 6):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, digits)


def _make_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_META_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_META_FEATURES),
    ])
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="liblinear",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf),
    ])


def _score_metrics(y_true, p, pnl) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    out = {
        "n": int(len(y)),
        "positive_rate_pct": _maybe_float(y.mean() * 100 if len(y) else np.nan),
        "brier": _maybe_float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else None,
        "auc": _maybe_float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "average_precision": _maybe_float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "net_pnl_sum_pct": _maybe_float(pd.to_numeric(pnl, errors="coerce").sum()),
        "avg_net_pnl_pct": _maybe_float(pd.to_numeric(pnl, errors="coerce").mean()),
    }
    return out


def _threshold_stats(p, y, pnl, threshold: float) -> dict:
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    pnl = pd.Series(pnl).astype(float).reset_index(drop=True)
    selected = p >= threshold
    n = int(selected.sum())
    if n == 0:
        return {
            "threshold": _maybe_float(threshold),
            "n_selected": 0,
            "selected_rate_pct": 0.0,
            "precision_pct": None,
            "net_pnl_sum_pct": 0.0,
            "avg_net_pnl_pct": None,
        }
    return {
        "threshold": _maybe_float(threshold),
        "n_selected": n,
        "selected_rate_pct": _maybe_float(n / len(p) * 100),
        "precision_pct": _maybe_float(y[selected].mean() * 100),
        "net_pnl_sum_pct": _maybe_float(pnl[selected].sum()),
        "avg_net_pnl_pct": _maybe_float(pnl[selected].mean()),
    }


def choose_threshold(p_train, y_train, pnl_train, min_selected: int) -> tuple[float, dict]:
    candidates = np.linspace(0.35, 0.75, 41)
    rows = []
    for threshold in candidates:
        stats = _threshold_stats(p_train, y_train, pnl_train, float(threshold))
        if stats["n_selected"] < min_selected:
            continue
        rows.append(stats)
    if not rows:
        return 0.5, _threshold_stats(p_train, y_train, pnl_train, 0.5)
    rows.sort(
        key=lambda r: (
            r["avg_net_pnl_pct"] if r["avg_net_pnl_pct"] is not None else -999,
            r["net_pnl_sum_pct"],
            r["n_selected"],
        ),
        reverse=True,
    )
    return float(rows[0]["threshold"]), rows[0]


def _time_split(df: pd.DataFrame, holdout_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date_dt"].dropna().dt.date.unique().tolist())
    if len(dates) < 4:
        cut = max(1, int(len(df) * (1 - holdout_frac)))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    n_holdout_dates = max(1, int(round(len(dates) * holdout_frac)))
    holdout_dates = set(dates[-n_holdout_dates:])
    train = df[~df["date_dt"].dt.date.isin(holdout_dates)].copy()
    holdout = df[df["date_dt"].dt.date.isin(holdout_dates)].copy()
    return train, holdout


def _feature_importance(pipe: Pipeline, top_n: int = 20) -> list[dict]:
    try:
        pre = pipe.named_steps["preprocessor"]
        names = pre.get_feature_names_out()
        coef = pipe.named_steps["model"].coef_[0]
    except Exception:
        return []
    rows = [
        {"feature": str(name), "coef": _maybe_float(c, 6), "abs_coef": abs(float(c))}
        for name, c in zip(names, coef)
    ]
    rows.sort(key=lambda r: r["abs_coef"], reverse=True)
    for row in rows:
        row.pop("abs_coef", None)
    return rows[:top_n]


def build_training_data(args) -> pd.DataFrame:
    candidates = load_candidate_ledger(args)
    enriched = add_result_columns(candidates)
    closed = enriched[enriched["net_pnl_pct"].notna()].copy()
    if len(closed) == 0:
        return closed
    closed["target_net_win"] = (closed["net_pnl_pct"] > 0).astype(int)
    closed = closed.sort_values(["date_dt", "channel", "coin"]).reset_index(drop=True)
    return closed


def train_and_write(args) -> dict:
    closed = build_training_data(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(closed) < args.min_samples:
        meta = {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "deployable": False,
            "reason": f"not enough closed samples: {len(closed)} < {args.min_samples}",
            "n_samples": int(len(closed)),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    train, holdout = _time_split(closed, args.holdout_frac)
    if len(train["target_net_win"].unique()) < 2 or len(holdout["target_net_win"].unique()) < 2:
        deployable = False
        split_reason = "train or holdout has one class"
    else:
        deployable = True
        split_reason = ""

    X_train = build_meta_feature_frame(train)
    y_train = train["target_net_win"].astype(int)
    X_holdout = build_meta_feature_frame(holdout)
    y_holdout = holdout["target_net_win"].astype(int)

    model = _make_pipeline()
    model.fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_holdout = model.predict_proba(X_holdout)[:, 1]

    min_selected = max(args.min_selected, int(len(train) * 0.15))
    threshold, train_threshold_stats = choose_threshold(
        p_train,
        y_train,
        train["net_pnl_pct"],
        min_selected=min_selected,
    )
    holdout_threshold_stats = _threshold_stats(
        p_holdout,
        y_holdout,
        holdout["net_pnl_pct"],
        threshold,
    )

    train_metrics = _score_metrics(y_train, p_train, train["net_pnl_pct"])
    holdout_metrics = _score_metrics(y_holdout, p_holdout, holdout["net_pnl_pct"])
    holdout_all_avg = float(holdout["net_pnl_pct"].mean()) if len(holdout) else np.nan
    selected_avg = holdout_threshold_stats["avg_net_pnl_pct"]
    selected_n = holdout_threshold_stats["n_selected"]
    selected_net = holdout_threshold_stats["net_pnl_sum_pct"]

    if split_reason:
        deployable = False
    if len(holdout) < args.min_holdout:
        deployable = False
        split_reason = f"holdout too small: {len(holdout)} < {args.min_holdout}"
    if selected_n < args.min_selected:
        deployable = False
        split_reason = f"selected holdout too small: {selected_n} < {args.min_selected}"
    if selected_avg is None or selected_avg < holdout_all_avg:
        deployable = False
        split_reason = "selected holdout avg does not improve over all holdout"
    if selected_net is None or selected_net <= 0:
        deployable = False
        split_reason = "selected holdout net pnl is not positive"

    meta = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "deployable": bool(deployable),
        "reason": split_reason or "time holdout gate passed",
        "target": "net_win",
        "threshold": _maybe_float(threshold),
        "n_samples": int(len(closed)),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "date_range": {
            "start": str(closed["date"].min()),
            "end": str(closed["date"].max()),
            "holdout_start": str(holdout["date"].min()) if len(holdout) else None,
            "holdout_end": str(holdout["date"].max()) if len(holdout) else None,
        },
        "features": META_FEATURES,
        "numeric_features": NUMERIC_META_FEATURES,
        "categorical_features": CATEGORICAL_META_FEATURES,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "train_threshold_stats": train_threshold_stats,
        "holdout_threshold_stats": holdout_threshold_stats,
        "top_coefficients": _feature_importance(model),
        "notes": [
            "Time holdout is used to reduce overfit risk.",
            "Artifact is used for live demotion only when deployable=true.",
            "Base candidate generators are unchanged.",
        ],
    }

    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    scored = closed[["date", "channel", "coin", "net_pnl_pct", "target_net_win"]].copy()
    scored["p_net_win"] = np.concatenate([p_train, p_holdout])
    scored["split"] = ["train"] * len(train) + ["holdout"] * len(holdout)
    scored["selected"] = scored["p_net_win"] >= threshold
    scored.to_csv(args.out_validation_csv, index=False)
    with open(args.out_validation_json, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument("--out-dir", default=DEFAULT_META_MODEL_DIR)
    parser.add_argument("--out-validation-csv", default="output/recommendation_meta_validation.csv")
    parser.add_argument("--out-validation-json", default="output/recommendation_meta_validation.json")
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--min-holdout", type=int, default=20)
    parser.add_argument("--min-selected", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    meta = train_and_write(args)
    print(json.dumps({
        "deployable": meta.get("deployable"),
        "reason": meta.get("reason"),
        "n_samples": meta.get("n_samples"),
        "threshold": meta.get("threshold"),
        "holdout": meta.get("holdout_threshold_stats"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
