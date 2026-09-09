"""Fixed, offline R1 upper-head comparison; no I/O or deployment side effects.

A preserves delivered picks. B post-processes historical *forward* frozen
probabilities, not raw-head OOF scores. C changes both training sample and target
window, so its result cannot isolate an architecture improvement. Every fitted
object is ephemeral; the caller owns evidence loading and report publication.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from xgboost import XGBClassifier

from signals.recommend_training_data import (
    ALLOWED_FEATURES,
    OUTCOME_FIELDS,
    expanding_readiness_splits,
)

EXPERIMENT_CONFIG = {
    "schema": "recommend_upper_comparison.v1",
    "arms": {"A": "frozen_snapshot", "B": "historical_forward_score_intercept_only",
             "C": "fixed_xgb_inner_expanding_oof_intercept_only"},
    "feature_columns": list(ALLOWED_FEATURES),
    "slots": ["open", "preopen"], "target": "up10",
    "outer": {"min_train_dates": 10, "validation_dates": 5},
    "inner": {"min_train_dates": 5, "validation_dates": 2},
    "head": {
        "n_estimators": 80, "max_depth": 2, "learning_rate": 0.05,
        "min_child_weight": 10, "reg_lambda": 5, "subsample": 1.0,
        "colsample_bytree": 1.0, "scale_pos_weight": 1.0, "random_state": 42,
        "n_jobs": 1, "tree_method": "hist", "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "calibration": {
        "method": "fixed_slope_logit_intercept", "slope": 1.0, "clip": 1e-6,
        "bounds": [-12.0, 12.0], "boundary_tolerance": 1e-4,
        "xatol": 1e-9, "maxiter": 1000,
        "min_dates": 5, "min_positive": 30, "min_negative": 30,
    },
    "head_min_positive": 12, "head_min_negative": 12,
    "ranking": {"rr_eps": 0.001, "top_k": 3,
                "tie_break": ["p_dn10 ascending", "p_up10 descending",
                              "exp_downside descending", "rank ascending"]},
    "no_tuning": True, "no_early_stopping": True, "model_artifacts_saved": False,
}
_META_COLUMNS = (
    "date", "slot", "coin", "rank", "snapshot_id", "decision_started_at",
    "label_available_at", "outcome_end_at", "was_delivered", "label_status",
    "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio",
)
_OUTPUT_COLUMNS = ("fold", "p_up10_A", "p_up10_B", "p_up10_C", "p_up10_C_raw", "B_status", "C_status")
_TIME_COLUMNS = ("decision_started_at", "label_available_at", "outcome_end_at")
_FOLD_KEYS = ("fold", "status", "train_dates", "validation_dates", "purged_train_dates",
              "train_max_label_available_at", "validation_min_decision_started_at")


class UpperExperimentError(ValueError):
    """Evidence, chronology or baseline parity is unsafe: abort all fitting."""


class _ArmUnavailable(ValueError):
    """This predefined arm cannot be fitted safely; never substitute identity."""


def _aware(value, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise UpperExperimentError(f"invalid {field}") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise UpperExperimentError(f"{field} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise UpperExperimentError("frame must have unique columns")
    required = set(ALLOWED_FEATURES) | set(_META_COLUMNS) | set(OUTCOME_FIELDS)
    if missing := required - set(frame.columns):
        raise UpperExperimentError(f"missing columns: {sorted(missing)}")
    if set(_OUTPUT_COLUMNS) & set(frame.columns) or any(
        str(name).startswith("_upper_") for name in frame.columns
    ):
        raise UpperExperimentError("reserved experiment columns already exist")
    work = frame.copy(deep=True).reset_index(drop=True)
    if work.empty:
        raise UpperExperimentError("no evidence rows")
    for name in ("date", "slot", "coin", "snapshot_id", "label_status"):
        if not work[name].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise UpperExperimentError(f"invalid {name}")
    try:
        if any(date.fromisoformat(value).isoformat() != value for value in work["date"]):
            raise ValueError("noncanonical date")
    except ValueError as exc:
        raise UpperExperimentError("invalid date") from exc
    if not set(work["slot"]) <= set(EXPERIMENT_CONFIG["slots"]):
        raise UpperExperimentError("invalid slot")
    if not set(work["label_status"]) <= {"labeled", "halted_no_observations"}:
        raise UpperExperimentError("invalid label_status")
    if not work["was_delivered"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise UpperExperimentError("was_delivered must be boolean")
    numeric_types = (int, float, np.integer, np.floating)
    for name in (*ALLOWED_FEATURES, "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio", "rank"):
        nullable = name in ALLOWED_FEATURES
        for value in work[name]:
            if nullable and pd.isna(value):
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric_types)
                    or not np.isfinite(value)):
                raise UpperExperimentError(f"invalid numeric {name}")
        if name in {"p_up10", "p_dn5", "p_dn10"} and not work[name].between(0, 1).all():
            raise UpperExperimentError(f"probability out of range: {name}")
    if (work["rr_ratio"] < 0).any():
        raise UpperExperimentError("negative rr_ratio")
    labeled = work["label_status"].eq("labeled")
    if not work.loc[labeled, "up10"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise UpperExperimentError("labeled up10 must be boolean")
    if work.loc[~labeled, list(OUTCOME_FIELDS)].notna().any().any():
        raise UpperExperimentError("halted outcomes must be missing, not negative labels")
    for field in _TIME_COLUMNS:
        work[f"_upper_{field}"] = work[field].map(lambda value: _aware(value, field))
    if (work["_upper_label_available_at"] < work["_upper_outcome_end_at"]).any():
        raise UpperExperimentError("label availability precedes outcome end")
    if (work["_upper_outcome_end_at"] <= work["_upper_decision_started_at"]).any():
        raise UpperExperimentError("outcome end must follow decision")
    decision_dates = work["_upper_decision_started_at"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    if not decision_dates.eq(work["date"]).all():
        raise UpperExperimentError("decision date does not match canonical KST date")
    if work.duplicated(["date", "slot", "coin"]).any():
        raise UpperExperimentError("duplicate candidate identity")
    if (work.groupby(["date", "slot"])["snapshot_id"].nunique() != 1).any():
        raise UpperExperimentError("multiple snapshots for one date/slot")
    for _, group in work.groupby("snapshot_id", sort=True):
        if group["date"].nunique() != 1 or group["slot"].nunique() != 1:
            raise UpperExperimentError("snapshot spans multiple dates or slots")
        if any(group[f"_upper_{name}"].nunique() != 1 for name in _TIME_COLUMNS):
            raise UpperExperimentError("snapshot rows disagree on chronology")
        if sorted(group["rank"].tolist()) != list(range(1, len(group) + 1)):
            raise UpperExperimentError("snapshot ranks are not contiguous unique integers")
        delivered = group.loc[group["was_delivered"], "rank"].sort_values().tolist()
        if delivered != [1, 2, 3]:
            raise UpperExperimentError("snapshot must preserve exactly the delivered Top3")
    return work


def _parity(work: pd.DataFrame) -> dict:
    audit = []
    for snapshot_id, group in work.groupby("snapshot_id", sort=True):
        ranking = group.assign(_upper_ratio=group["p_up10"] / np.maximum(
            group["p_dn5"], EXPERIMENT_CONFIG["ranking"]["rr_eps"],
        )).sort_values(
            ["_upper_ratio", "p_dn10", "p_up10", "exp_downside", "rank"],
            ascending=[False, True, False, False, True], kind="stable",
        )
        actual = group.sort_values("rank")["coin"].tolist()
        recalculated = ranking["coin"].tolist()
        if recalculated[:3] != actual[:3]:
            raise UpperExperimentError(f"no-op Top3 parity failed: {snapshot_id}")
        audit.append({"snapshot_id": snapshot_id, "top3_passed": True,
                      "full_order_matches": actual == recalculated})
    return {"status": "passed", "snapshots": len(audit), "top3_passed": len(audit),
            "full_order_mismatches": sum(not item["full_order_matches"] for item in audit),
            "full_order_note": "lower-rank rounding differences allowed; Top3 order must match",
            "snapshots_audit": audit}


def _time_folds(work: pd.DataFrame, *, min_train_dates: int, validation_dates: int) -> list[dict]:
    # Reuse the readiness chronology contract, including global-date purging.
    rows = [{"metadata": {
        "date": row["date"], "slot": row["slot"], "was_delivered": row["was_delivered"],
        "decision_started_at": row["decision_started_at"],
        "label_available_at": row["label_available_at"],
    }, "outcomes": {name: row[name] for name in OUTCOME_FIELDS}} for row in work.to_dict("records")]
    return expanding_readiness_splits(
        rows, min_train_dates=min_train_dates, validation_dates=validation_dates,
    )


def _validate_folds(work: pd.DataFrame, supplied: list[dict]) -> list[dict]:
    expected = _time_folds(work, **EXPERIMENT_CONFIG["outer"])
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise UpperExperimentError("outer folds do not match frozen readiness split")
    for left, right in zip(supplied, expected):
        if not isinstance(left, dict) or any(left.get(key) != right[key] for key in _FOLD_KEYS):
            raise UpperExperimentError("outer folds do not match frozen readiness split")
    return expected


def _counts(rows: pd.DataFrame) -> dict:
    return {"rows": len(rows), "dates": rows["date"].nunique(),
            "positive": int(rows["up10"].eq(True).sum()),
            "negative": int(rows["up10"].eq(False).sum())}


def _check_probabilities(values, expected_length: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (expected_length,) or not np.isfinite(result).all() or (
        (result < 0).any() or (result > 1).any()
    ):
        raise _ArmUnavailable("invalid_model_probabilities")
    return result


def _fit_head(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    counts = _counts(train)
    if (counts["positive"] < EXPERIMENT_CONFIG["head_min_positive"]
            or counts["negative"] < EXPERIMENT_CONFIG["head_min_negative"]):
        raise _ArmUnavailable("insufficient_head_class_counts")
    features = list(ALLOWED_FEATURES)
    try:
        model = XGBClassifier(**EXPERIMENT_CONFIG["head"])
        model.fit(train[features].to_numpy(dtype=float), train["up10"].to_numpy(dtype=int))
        predictions = model.predict_proba(validation[features].to_numpy(dtype=float))[:, 1]
    except Exception as exc:
        raise _ArmUnavailable(f"head_fit_or_predict_failed: {type(exc).__name__}: {exc}") from exc
    return _check_probabilities(predictions, len(validation))


def _fit_intercept(raw: np.ndarray, rows: pd.DataFrame) -> dict:
    settings = EXPERIMENT_CONFIG["calibration"]
    counts = _counts(rows)
    if (counts["dates"] < settings["min_dates"] or counts["positive"] < settings["min_positive"]
            or counts["negative"] < settings["min_negative"]):
        raise _ArmUnavailable("insufficient_calibration_dates_or_classes")
    raw = _check_probabilities(raw, len(rows))
    logits = logit(np.clip(raw, settings["clip"], 1 - settings["clip"]))
    labels = rows["up10"].to_numpy(dtype=float)

    def loss(intercept):
        shifted = logits + intercept
        return float(np.mean(np.logaddexp(0, shifted) - labels * shifted))

    try:
        optimized = minimize_scalar(
            loss, bounds=tuple(settings["bounds"]), method="bounded",
            options={"xatol": settings["xatol"], "maxiter": settings["maxiter"]},
        )
        intercept, minimized = float(optimized.x), float(optimized.fun)
    except Exception as exc:
        raise _ArmUnavailable(f"calibration_optimizer_failed: {type(exc).__name__}: {exc}") from exc
    if not optimized.success or not np.isfinite([intercept, minimized]).all():
        raise _ArmUnavailable("calibration_optimizer_unsuccessful_or_nonfinite")
    lower, upper = settings["bounds"]
    if not lower + settings["boundary_tolerance"] < intercept < upper - settings["boundary_tolerance"]:
        raise _ArmUnavailable("calibration_optimizer_at_boundary")
    return {"intercept": intercept, "slope": 1.0, "counts": counts,
            "logloss_before": loss(0), "logloss_after": minimized}


def _apply_intercept(raw: np.ndarray, calibration: dict) -> np.ndarray:
    raw = _check_probabilities(raw, len(raw))
    clip = EXPERIMENT_CONFIG["calibration"]["clip"]
    return _check_probabilities(expit(logit(np.clip(raw, clip, 1 - clip)) + calibration["intercept"]), len(raw))


def _assert_separated(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train["date"]) & set(validation["date"]):
        raise UpperExperimentError("training and validation share a global date")
    if not train.empty and train["_upper_label_available_at"].max() >= validation["_upper_decision_started_at"].min():
        raise UpperExperimentError("training labels unavailable at validation decision")


def _upper_c(outer_train: pd.DataFrame, validation: pd.DataFrame, slot: str, audit: dict) -> tuple[np.ndarray, np.ndarray]:
    oof_rows, oof_predictions = [], []
    audit["inner_folds"] = []
    for fold in _time_folds(outer_train, **EXPERIMENT_CONFIG["inner"]):
        item = {key: fold[key] for key in _FOLD_KEYS}
        audit["inner_folds"].append(item)
        if fold["status"] != "split_ready":
            continue
        train = outer_train.loc[outer_train["date"].isin(fold["train_dates"])]
        held = outer_train.loc[outer_train["date"].isin(fold["validation_dates"])]
        _assert_separated(train, held)
        train = train.loc[train["slot"].eq(slot) & train["label_status"].eq("labeled")]
        held = held.loc[held["slot"].eq(slot) & held["label_status"].eq("labeled")]
        item["train_counts"], item["validation_counts"] = _counts(train), _counts(held)
        if held.empty:
            item["status"] = "no_labeled_validation_in_slot"
            continue
        raw = _fit_head(train, held)
        item["status"] = "oof_predicted"
        oof_rows.append(held)
        oof_predictions.append(raw)
    if not oof_rows:
        raise _ArmUnavailable("no_inner_oof_rows")
    combined = pd.concat(oof_rows)
    if not combined.index.is_unique:
        raise UpperExperimentError("inner OOF row predicted more than once")
    audit["oof_counts"] = _counts(combined)
    audit["oof_dates"] = sorted(combined["date"].unique().tolist())
    calibration = _fit_intercept(np.concatenate(oof_predictions), combined)
    audit["calibration"] = calibration
    train = outer_train.loc[outer_train["slot"].eq(slot) & outer_train["label_status"].eq("labeled")]
    raw = _fit_head(train, validation)
    return _apply_intercept(raw, calibration), raw


def run_upper_comparison(frame: pd.DataFrame, folds: list) -> dict:
    """Return ephemeral, genuinely held-out predictions for fixed outer folds.

    Unsafe input aborts *before* any fit. An unavailable B or C leaves its entire
    fold/slot NaN and an explicit reason; no identity or constant fallback exists.
    Halted validation rows remain candidates and receive predictions, but never
    supply negative training labels. Original frame columns/values are preserved.
    """
    work = _validate_frame(frame)
    parity = _parity(work)
    checked = _validate_folds(work, folds)
    original_columns = list(frame.columns)
    outputs, fold_audit = [], []
    for fold in checked:
        summary = {key: fold[key] for key in _FOLD_KEYS}
        summary["slots"] = []
        fold_audit.append(summary)
        if fold["status"] != "split_ready":
            continue
        train = work.loc[work["date"].isin(fold["train_dates"])]
        validation = work.loc[work["date"].isin(fold["validation_dates"])]
        _assert_separated(train, validation)
        for slot in EXPERIMENT_CONFIG["slots"]:
            held = validation.loc[validation["slot"].eq(slot)]
            fitting = train.loc[train["slot"].eq(slot) & train["label_status"].eq("labeled")]
            slot_audit = {"slot": slot, "train_counts": _counts(fitting),
                          "validation_candidates": len(held),
                          "validation_labeled": int(held["label_status"].eq("labeled").sum()),
                          "A": {"status": "frozen_snapshot"}, "B": {}, "C": {}}
            summary["slots"].append(slot_audit)
            if held.empty:
                slot_audit["status"] = "no_validation_candidates"
                continue
            output = held[original_columns].copy(deep=True)
            output["fold"] = fold["fold"]
            output["p_up10_A"] = held["p_up10"].to_numpy(dtype=float)
            output[["p_up10_B", "p_up10_C", "p_up10_C_raw"]] = np.nan
            output[["B_status", "C_status"]] = "unavailable"
            try:
                calibration = _fit_intercept(fitting["p_up10"].to_numpy(dtype=float), fitting)
                output["p_up10_B"] = _apply_intercept(held["p_up10"].to_numpy(dtype=float), calibration)
                output["B_status"] = "ok"
                slot_audit["B"] = {"status": "predicted", "calibration": calibration,
                                   "source": "historical_forward_scores_not_raw_head_oof"}
            except _ArmUnavailable as exc:
                slot_audit["B"] = {"status": "unavailable", "reason": str(exc)}
            try:
                calibrated, raw = _upper_c(train, held, slot, slot_audit["C"])
                output["p_up10_C"], output["p_up10_C_raw"] = calibrated, raw
                output["C_status"] = "ok"
                slot_audit["C"]["status"] = "predicted"
            except _ArmUnavailable as exc:
                slot_audit["C"].update(status="unavailable", reason=str(exc))
            outputs.append(output)
    predictions = (pd.concat(outputs).sort_values(["fold", "date", "slot", "rank"]).reset_index(drop=True)
                   if outputs else pd.DataFrame(columns=original_columns + list(_OUTPUT_COLUMNS)))
    return {"predictions": predictions, "fold_audit": fold_audit,
            "config": deepcopy(EXPERIMENT_CONFIG), "parity": parity}
