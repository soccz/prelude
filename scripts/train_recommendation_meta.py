"""Train a candidate recommendation-quality meta-label model.

The base XGBoost heads generate candidates. This meta model learns from closed
paper/shadow ledger rows with complete ordered TP5/SL3/EOD paths whether a
candidate became net-positive. Automated runs only produce a validation
candidate; live use additionally requires explicit user-approved promotion.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
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
    DEFAULT_META_CANDIDATE_DIR,
    META_LEDGER_INPUT_KEYS,
    META_FEATURES,
    META_ARTIFACT_SCHEMA,
    META_TRAINING_GENERATOR_SOURCES,
    META_TRAINING_LINEAGE_SCHEMA,
    MODEL_ID,
    MODEL_VERSION,
    NUMERIC_META_FEATURES,
    PROJECT_ROOT,
    build_meta_feature_frame,
    meta_feature_schema_sha256,
    meta_runtime_versions,
    meta_training_row_schema_sha256,
)
from ledger.csv_store import ledger_lock
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_set_identity,
    payload_digest,
    resolve_identity_path,
    sha256_bytes,
    sha256_file,
    with_manifest_digest,
)
from scripts.idea_validation_report import add_result_columns, load_candidate_ledger


def _candidate_meta(payload: dict) -> dict:
    """Attach the immutable artifact contract to one non-active candidate."""
    meta = {
        "artifact_schema": META_ARTIFACT_SCHEMA,
        **payload,
        # Automated training may establish statistical candidacy, but cannot
        # satisfy the project's user-confirmation/promotion requirement.
        "deployable": False,
        "feature_schema_sha256": meta_feature_schema_sha256(),
        "runtime_versions": meta_runtime_versions(),
    }
    meta["artifact_sha256"] = payload_digest(
        meta,
        digest_key="artifact_sha256",
    )
    return meta


def _strict_training_rows(closed: pd.DataFrame) -> tuple[list[dict], dict]:
    features = build_meta_feature_frame(closed)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for idx, source in closed.iterrows():
        date = pd.Timestamp(source.get("date_dt"))
        date_text = str(source.get("date"))
        if (
            pd.isna(date)
            or date.time() != pd.Timestamp(0).time()
            or date_text != date.strftime("%Y-%m-%d")
        ):
            raise ValueError("training row has invalid or non-canonical date")
        channel = str(source.get("channel"))
        coin = str(source.get("coin"))
        if channel not in {"distribution", "preopen"} or not coin.startswith("KRW-"):
            raise ValueError("training row has invalid channel/coin key")
        key = (date_text, channel, coin)
        if key in seen:
            raise ValueError(f"duplicate recommendation training row: {key}")
        seen.add(key)

        net_pnl = float(source.get("net_pnl_pct"))
        target = source.get("target_net_win")
        promotion_eligible = source.get("promotion_eligible")
        if (
            not np.isfinite(net_pnl)
            or not isinstance(target, (int, np.integer))
            or isinstance(target, (bool, np.bool_))
            or int(target) not in {0, 1}
            or int(target) != int(net_pnl > 0)
            or not isinstance(promotion_eligible, (bool, np.bool_))
            or not bool(promotion_eligible)
            or source.get("outcome_contract")
            != "tp5_sl3_ordered_first_passage_net"
        ):
            raise ValueError("training row violates ordered outcome/target contract")
        row = {
            "date": date_text,
            "channel": channel,
            "coin": coin,
            "net_pnl_pct": net_pnl,
            "target_net_win": int(target),
        }
        for column in NUMERIC_META_FEATURES:
            value = features.at[idx, column]
            row[column] = None if pd.isna(value) else float(value)
            if row[column] is not None and not np.isfinite(row[column]):
                raise ValueError(f"training row has non-finite feature: {column}")
        for column in CATEGORICAL_META_FEATURES:
            row[column] = str(features.at[idx, column])
        rows.append(row)
    rows.sort(key=lambda row: (row["date"], row["channel"], row["coin"]))
    dates = sorted({str(row["date"]) for row in rows})
    contract = {
        "row_schema_sha256": meta_training_row_schema_sha256(),
        "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "n_rows": len(rows),
        "n_dates": len(dates),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
    }
    return rows, contract


def _training_source_paths(args) -> dict[str, Path]:
    values = {
        "paper_ledger_distribution": args.paper_ledger,
        "paper_ledger_preopen": args.paper_ledger_preopen,
        "shadow_ledger_distribution": args.shadow_ledger_distribution,
        "shadow_ledger_preopen": args.shadow_ledger_preopen,
    }
    if set(values) != set(META_LEDGER_INPUT_KEYS):
        raise RuntimeError("recommendation training ledger contract drift")
    return {
        name: resolve_identity_path(str(path), root=PROJECT_ROOT)
        for name, path in values.items()
    }


def _build_training_snapshot(args) -> tuple[pd.DataFrame, dict]:
    ledger_paths = _training_source_paths(args)
    generator_paths = {
        relative: PROJECT_ROOT / relative
        for relative in META_TRAINING_GENERATOR_SOURCES
    }
    all_sources = {
        **{f"ledger:{name}": path for name, path in ledger_paths.items()},
        **{
            f"generator:{relative}": path
            for relative, path in generator_paths.items()
        },
    }
    unique_ledgers = sorted(
        {Path(path).resolve() for path in ledger_paths.values()},
        key=str,
    )
    with ExitStack() as stack:
        for path in unique_ledgers:
            stack.enter_context(ledger_lock(path))
        before = file_set_identity(all_sources, root=PROJECT_ROOT)
        ledger_identities = {
            name: before[f"ledger:{name}"]
            for name in META_LEDGER_INPUT_KEYS
        }
        if any(not identity.get("exists") for identity in ledger_identities.values()):
            raise FileNotFoundError(
                "all four recommendation training ledgers must exist"
            )
        closed = build_training_data(args)
        after = file_set_identity(all_sources, root=PROJECT_ROOT)
    if before != after:
        raise ArtifactSourceChangedError(
            "recommendation training inputs changed during snapshot build"
        )

    _, training_rows = _strict_training_rows(closed)
    generator_identities = {
        relative: before[f"generator:{relative}"]
        for relative in META_TRAINING_GENERATOR_SOURCES
    }
    lineage = {
        "schema_version": META_TRAINING_LINEAGE_SCHEMA,
        "ledger_inputs": with_manifest_digest(
            {"files": ledger_identities},
            digest_key="bundle_sha256",
        ),
        "generator_sources": with_manifest_digest(
            {"files": generator_identities},
            digest_key="bundle_sha256",
        ),
        "training_rows": training_rows,
    }
    return closed, with_manifest_digest(lineage)


def _rejected_meta(
    closed: pd.DataFrame,
    reason: str,
    training_lineage: dict,
    *,
    n_train: int = 0,
    n_holdout: int = 0,
) -> dict:
    valid_dates = closed.get("date", pd.Series(dtype=str)).dropna().astype(str)
    return _candidate_meta({
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_status": "REJECTED",
        "validation_gate_passed": False,
        "promotion_status": "NOT_ELIGIBLE",
        "reason": reason,
        "target": "net_win",
        "threshold": None,
        "model_file": None,
        "model_sha256": None,
        "n_samples": int(len(closed)),
        "n_train": int(n_train),
        "n_holdout": int(n_holdout),
        "date_range": {
            "start": str(valid_dates.min()) if len(valid_dates) else None,
            "end": str(valid_dates.max()) if len(valid_dates) else None,
        },
        "features": META_FEATURES,
        "numeric_features": NUMERIC_META_FEATURES,
        "categorical_features": CATEGORICAL_META_FEATURES,
        "training_lineage": training_lineage,
    })


def _write_rejected_outputs(args, out_dir: Path, meta: dict) -> None:
    atomic_write_json(out_dir / "meta.json", meta)
    atomic_write_json(args.out_validation_json, meta)
    header = (
        "date,channel,coin,net_pnl_pct,target_net_win,"
        "p_net_win,split,selected\n"
    )
    atomic_write_bytes(args.out_validation_csv, header.encode("utf-8"))


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
    promotion_eligible = enriched.get(
        "promotion_eligible",
        pd.Series(False, index=enriched.index),
    ).map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))
    closed = enriched[
        enriched["net_pnl_pct"].notna()
        & promotion_eligible
        & enriched["date_dt"].notna()
    ].copy()
    if len(closed) == 0:
        return closed
    closed["target_net_win"] = (closed["net_pnl_pct"] > 0).astype(int)
    closed = closed.sort_values(["date_dt", "channel", "coin"]).reset_index(drop=True)
    return closed


def _train_and_write_locked(args, out_dir: Path) -> dict:
    if not 0.0 < args.holdout_frac < 1.0:
        raise ValueError("holdout_frac must be strictly between 0 and 1")
    if args.min_samples < 2 or args.min_holdout < 1 or args.min_selected < 1:
        raise ValueError("sample thresholds must be positive")

    closed, training_lineage = _build_training_snapshot(args)

    if len(closed) < args.min_samples:
        meta = _rejected_meta(
            closed,
            f"not enough closed samples: {len(closed)} < {args.min_samples}",
            training_lineage,
        )
        _write_rejected_outputs(args, out_dir, meta)
        return meta

    train, holdout = _time_split(closed, args.holdout_frac)
    if len(train["target_net_win"].unique()) < 2 or len(holdout["target_net_win"].unique()) < 2:
        meta = _rejected_meta(
            closed,
            "train or holdout has one class",
            training_lineage,
            n_train=len(train),
            n_holdout=len(holdout),
        )
        _write_rejected_outputs(args, out_dir, meta)
        return meta

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

    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    model_sha256 = sha256_bytes(model_bytes)
    model_name = f"model.{model_sha256}.pkl"
    meta = _candidate_meta({
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_status": "CANDIDATE" if deployable else "REJECTED",
        "validation_gate_passed": bool(deployable),
        "promotion_status": (
            "AWAITING_USER_APPROVAL" if deployable else "NOT_ELIGIBLE"
        ),
        "reason": split_reason or "time holdout gate passed",
        "target": "net_win",
        "threshold": _maybe_float(threshold),
        "model_file": model_name,
        "model_sha256": model_sha256,
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
        "training_lineage": training_lineage,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "train_threshold_stats": train_threshold_stats,
        "holdout_threshold_stats": holdout_threshold_stats,
        "top_coefficients": _feature_importance(model),
        "notes": [
            "Time holdout is used to reduce overfit risk.",
            "Automated training writes a non-active candidate only.",
            "Live demotion additionally requires explicit user-approved promotion.",
            "Base candidate generators are unchanged.",
        ],
    })

    # Publish immutable model bytes first, then atomically switch meta.json to
    # that content-addressed generation. A concurrent reader sees either the
    # complete previous generation or the complete new generation.
    model_path = out_dir / model_name
    if model_path.exists():
        if sha256_file(model_path) != model_sha256:
            raise RuntimeError(f"content-addressed model collision: {model_path}")
    else:
        atomic_write_bytes(model_path, model_bytes)
    atomic_write_json(out_dir / "meta.json", meta)

    scored = closed[["date", "channel", "coin", "net_pnl_pct", "target_net_win"]].copy()
    scored["p_net_win"] = np.concatenate([p_train, p_holdout])
    scored["split"] = ["train"] * len(train) + ["holdout"] * len(holdout)
    scored["selected"] = scored["p_net_win"] >= threshold
    atomic_write_bytes(
        args.out_validation_csv,
        scored.to_csv(index=False).encode("utf-8"),
    )
    atomic_write_json(args.out_validation_json, meta)
    return meta


def train_and_write(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The two daily close jobs and manual runs can overlap. Serialize the full
    # read/train/publish transaction for one candidate slot so validation
    # CSV/JSON and the meta pointer always describe the same winning run.
    with ledger_lock(out_dir / "meta.json"):
        return _train_and_write_locked(args, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_META_CANDIDATE_DIR,
        help=(
            "candidate-only artifact directory; live promotion requires "
            "explicit user approval, DEPLOYED metadata, and a source-pinned "
            "artifact digest (copying files alone never activates it)"
        ),
    )
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
