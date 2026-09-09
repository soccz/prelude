"""Pure, daily B/C target-window comparison; no storage or operational calls.

Both arms use one past-only training matrix and B-derived class weights. The
legacy calibration helper's name says OOF, but the supplied scores are fitted
training scores: this is deliberately resubstitution, not improved calibration.
The caller owns frozen data provenance, daily scheduling, and result evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
from numbers import Real

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ops.artifact_provenance import canonical_json_bytes
from scripts.downside_head_riskreward_v1 import _apply_calib, _oof_bucket_calib
from signals.recommend_training_data import ALLOWED_FEATURES


MODEL_CONFIG = {
    "version": "recommend_horizon_model.v1",
    "features": list(ALLOWED_FEATURES), "arms": ["B", "C"],
    "thresholds": {"p_up5": .05, "p_up10": .10, "p_up20": .20, "p_dn5": -.05, "p_dn10": -.10},
    "xgb_params": {
        "n_estimators": 180, "max_depth": 4, "learning_rate": .05,
        "subsample": .8, "colsample_bytree": .8, "min_child_weight": 5,
        "reg_lambda": 1.5, "n_jobs": 4, "eval_metric": "logloss",
        "tree_method": "hist", "random_state": 42, "objective": "binary:logistic",
    },
    "class_weight": "B_negative_over_positive_per_decision_head_shared_by_arms",
    "min_positive": 12, "min_classes": 2, "embargo_days": 5,
    "calibration": {
        "method": "same_fitted_score_bucket_resubstitution", "buckets": 10,
        "source": "scripts/downside_head_riskreward_v1.py",
        "helpers": ["_oof_bucket_calib", "_apply_calib"], "not_oof": True,
    },
    "expected_downside": "reuse_arm_dn5_raw_scores_rank_first_qcut10_mean_low",
    "preprocessing": "shared_train_only_median_inf_to_nan_all_nan_stays_missing",
    "failure_policy": "preflight_all_10_heads_before_fit; any_unavailable_returns_both_arms_missing_without_A_fallback",
    "fit_count_success": 10,
    "artifact_policy": "in_memory_booster_ubj_sha256_only_no_file",
}
_TRAIN_META = (
    "feature_date", "coin", "input_end_at", "B_target_end_at", "C_target_end_at",
    "C_target_present", "B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret",
)
_DECISION_FIELDS = {"date", "decision_started_at", "feature_row_date", "cutoff_exclusive"}


class HorizonModelError(ValueError):
    """Invalid evidence or model failure: no partial paired result is valid."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HorizonModelError(reason)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _matrix_digest(values: np.ndarray) -> str:
    """Canonical float64 little-endian, row-major bytes, with normalized NaN."""
    matrix = np.array(values, dtype="<f8", copy=True, order="C")
    matrix[np.isnan(matrix)] = np.nan
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _day(value: object, field: str) -> date:
    _require(isinstance(value, str), f"{field} must be an ISO date string")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise HorizonModelError(f"invalid {field}") from exc
    _require(result.isoformat() == value, f"noncanonical {field}")
    return result


def _time(value: object, field: str) -> pd.Timestamp:
    _require(isinstance(value, (str, datetime, pd.Timestamp)), f"{field} must be an aware timestamp")
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise HorizonModelError(f"invalid {field}") from exc
    _require(not pd.isna(result) and result.tzinfo is not None, f"{field} must be timezone-aware")
    return result.tz_convert("UTC")


def _numeric(series: pd.Series, field: str) -> pd.Series:
    _require(not pd.api.types.is_complex_dtype(series.dtype), f"invalid complex numeric field {field}")
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(float)
    _require(series.map(lambda value: value is None or value is pd.NA or (
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
    )).all(), f"invalid numeric field {field}")
    return series.astype(float)


def _prepare(training_frame: pd.DataFrame, prediction_features: pd.DataFrame, decision: dict) -> tuple:
    _require(isinstance(decision, dict) and set(decision) == _DECISION_FIELDS, "invalid decision fields")
    day = _day(decision["date"], "decision date")
    reference = _day(decision["feature_row_date"], "feature_row_date")
    cutoff = _day(decision["cutoff_exclusive"], "cutoff_exclusive")
    started = _time(decision["decision_started_at"], "decision_started_at")
    _require(reference == day - timedelta(days=1), "preopen feature reference must be D-1")
    _require(cutoff == reference - timedelta(days=MODEL_CONFIG["embargo_days"]), "invalid exclusive embargo cutoff")
    local = started.tz_convert("Asia/Seoul")
    _require(local.date() == day and local.hour == 8 and local.minute >= 45, "decision must be the preopen slot")
    _require(isinstance(training_frame, pd.DataFrame) and training_frame.columns.is_unique,
             "training frame must have unique columns")
    _require(set(training_frame.columns) == set(_TRAIN_META) | set(ALLOWED_FEATURES), "invalid training columns")
    _require(isinstance(prediction_features, pd.DataFrame) and prediction_features.columns.is_unique
             and list(prediction_features.columns) == list(ALLOWED_FEATURES), "prediction requires exactly the ordered 24 features")
    _require(prediction_features.index.is_unique, "prediction index must be unique")
    work = training_frame.copy(deep=True)
    work["feature_date"].map(lambda value: _day(value, "feature_date"))
    _require(work["coin"].map(lambda value: isinstance(value, str) and bool(value)).all(), "invalid coin identity")
    _require(not work.duplicated(["feature_date", "coin"]).any(), "duplicate training identity")
    _require(work["C_target_present"].map(lambda value: isinstance(value, (bool, np.bool_))).all(),
             "C_target_present must be boolean")
    work = work.sort_values(["feature_date", "coin"], kind="stable").reset_index(drop=True)
    start = pd.to_datetime(work["feature_date"]).dt.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    input_end = pd.to_datetime(work["input_end_at"].map(lambda value: _time(value, "input_end_at")), utc=True)
    b_end = pd.to_datetime(work["B_target_end_at"].map(lambda value: _time(value, "B_target_end_at")), utc=True)
    _require((input_end <= start).all(), "feature input ends after target start")
    _require((b_end == start + pd.Timedelta(days=1)).all(), "B target must end at exact calendar t+1 09:00")
    c_end_values = []
    for present, value, expected in zip(work.C_target_present, work.C_target_end_at,
                                        start + pd.Timedelta(days=2), strict=True):
        if not present and pd.isna(value):
            c_end_values.append(pd.NaT)
        else:
            stamp = _time(value, "C_target_end_at")
            _require(stamp == expected, "C target must end at exact calendar t+2 09:00")
            c_end_values.append(stamp)
    c_end = pd.Series(pd.to_datetime(c_end_values, utc=True), index=work.index)
    for field in (*ALLOWED_FEATURES, "B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret"):
        work[field] = _numeric(work[field], field)
    prediction = prediction_features.copy(deep=True)
    for field in ALLOWED_FEATURES:
        prediction[field] = _numeric(prediction[field], field)
    for arm in MODEL_CONFIG["arms"]:
        high, low = work[f"{arm}_high_ret"], work[f"{arm}_low_ret"]
        finite = np.isfinite(high) & np.isfinite(low)
        _require((high[finite] >= low[finite]).all(), f"{arm} high return below low return")
    eligible_date = work.feature_date.lt(cutoff.isoformat())
    present = work.C_target_present
    returns_finite = np.isfinite(work[["B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret"]]).all(axis=1)
    ended = b_end.lt(started) & c_end.lt(started)
    common = eligible_date & present & returns_finite & ended
    exclusions = {
        "at_or_after_exclusive_cutoff": int((~eligible_date).sum()),
        "missing_exact_next_calendar_target": int((eligible_date & ~present).sum()),
        "nonfinite_target_returns": int((eligible_date & present & ~returns_finite).sum()),
        "target_not_ended_before_decision": int((eligible_date & present & returns_finite & ~ended).sum()),
    }
    rejected = work.loc[eligible_date & ~common, ["feature_date", "coin"]].to_dict("records")
    selected = work.loc[common].copy()
    x = selected[list(ALLOWED_FEATURES)].replace([np.inf, -np.inf], np.nan)
    median = x.median()
    x_train = x.fillna(median).to_numpy(dtype=float)
    x_predict = prediction.replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(dtype=float)
    x_train.setflags(write=False)
    x_predict.setflags(write=False)
    preparation = {
        "input_rows": len(work), "training_rows": len(selected), "prediction_rows": len(prediction),
        "training_dates": int(selected.feature_date.nunique()), "exclusions": exclusions,
        "excluded_training_candidates": rejected,
        "training_keys_sha256": _digest(selected[["feature_date", "coin"]].to_dict("records")),
        "training_feature_date_min": selected.feature_date.min() if len(selected) else None,
        "training_feature_date_max": selected.feature_date.max() if len(selected) else None,
        "input_end_max": input_end.loc[common].max().isoformat() if common.any() else None,
        "B_target_end_max": b_end.loc[common].max().isoformat() if common.any() else None,
        "C_target_end_max": c_end.loc[common].max().isoformat() if common.any() else None,
        "train_medians": {name: None if pd.isna(value) else float(value) for name, value in median.items()},
        "all_nan_features": median.index[median.isna()].tolist(),
        "training_X_sha256": _matrix_digest(x_train), "prediction_X_sha256": _matrix_digest(x_predict),
        "matrix_hash_encoding": "float64_little_endian_C_order_normalized_nan; feature order is configuration.features",
    }
    return selected, x_train, x_predict, preparation


def _calibration_record(edges, hit_map, base: float) -> dict:
    return {"edges": None if edges is None else np.asarray(edges).astype(float).tolist(),
            "hit_map": None if hit_map is None else {str(key): float(value) for key, value in hit_map.items()},
            "base": float(base), "score_basis": "same_fitted_training_scores_not_oof"}


def _expected_downside(raw_train: np.ndarray, low: np.ndarray, raw_test: np.ndarray) -> tuple:
    frame = pd.DataFrame({"s": raw_train, "lr": low}).dropna()
    base = float(frame.lr.mean())
    try:
        frame["bk"] = pd.qcut(frame.s.rank(method="first"), MODEL_CONFIG["calibration"]["buckets"],
                              labels=False, duplicates="drop")
        grouped = frame.groupby("bk").agg(hi=("s", "max"), m=("lr", "mean"))
        edges, mapping = grouped.hi.to_numpy(), grouped.m.to_dict()
    except ValueError:
        edges, mapping = None, None
    return _apply_calib(raw_test, edges, mapping, base), _calibration_record(edges, mapping, base)


def fit_daily_pair(training_frame: pd.DataFrame, prediction_features: pd.DataFrame, *, decision: dict) -> dict:
    """Fit exactly ten heads, or return an explicit all-missing paired result.

    This function never reads/writes files, calls a DB/network, changes ranking,
    or substitutes frozen A when B/C cannot be trained. Only in-memory booster
    hashes are retained; no reusable model artifact is published.
    """
    selected, x_train, x_predict, preparation = _prepare(training_frame, prediction_features, decision)
    labels, heads, failures = {}, {}, []
    thresholds = MODEL_CONFIG["thresholds"]
    for arm in MODEL_CONFIG["arms"]:
        for head, threshold in thresholds.items():
            values = selected[f"{arm}_{'high' if threshold > 0 else 'low'}_ret"].to_numpy()
            y = ((values >= threshold) if threshold > 0 else (values <= threshold)).astype(np.uint8)
            labels[arm, head] = y
            positive, negative = int(y.sum()), int(len(y) - y.sum())
            classes = int(positive > 0) + int(negative > 0)
            ready = positive >= MODEL_CONFIG["min_positive"] and classes >= MODEL_CONFIG["min_classes"]
            heads[f"{arm}/{head}"] = {
                "positive": positive, "negative": negative, "classes": classes, "preflight_ready": ready,
                "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "label_hash_encoding": "uint8_in_shared_training_key_order",
            }
            if not ready:
                failures.append({"arm": arm, "head": head, "reason": "insufficient_positive_or_single_class"})
    columns = [name for arm in MODEL_CONFIG["arms"] for name in (
        *[f"{head}_{arm}" for head in thresholds], f"exp_downside_{arm}",
        *[f"{head}_{arm}_raw" for head in thresholds],
    )]
    scores = pd.DataFrame(np.nan, index=prediction_features.index.copy(), columns=columns)
    audit = {
        "schema": MODEL_CONFIG["version"], "configuration": deepcopy(MODEL_CONFIG), "decision": deepcopy(decision),
        **preparation, "heads": heads, "failures": failures, "fit_count": 0,
        "model_artifact_saved": False, "live_model_changed": False, "deployable": False,
        "calibration_is_oof": False, "expected_downside_reuses_dn5": True,
    }
    for head in thresholds:
        control = heads[f"B/{head}"]
        weight = control["negative"] / control["positive"] if control["positive"] else None
        for arm in MODEL_CONFIG["arms"]:
            heads[f"{arm}/{head}"]["effective_parameters"] = {**MODEL_CONFIG["xgb_params"], "scale_pos_weight": weight}
    if failures or not len(prediction_features):
        if not len(prediction_features):
            failures.append({"arm": "both", "head": None, "reason": "no_prediction_rows"})
        audit.update(status="unavailable", action="both_arms_missing_no_A_fallback",
                     B_status="unavailable", C_status="unavailable")
        scores["B_status"] = scores["C_status"] = "unavailable"
        return {"scores": scores, "audit": audit}
    for arm in MODEL_CONFIG["arms"]:
        for head in thresholds:
            detail = heads[f"{arm}/{head}"]
            model = XGBClassifier(**detail["effective_parameters"])
            try:
                model.fit(x_train, labels[arm, head])
                raw_train = np.asarray(model.predict_proba(x_train)[:, 1], dtype=float)
                raw_test = np.asarray(model.predict_proba(x_predict)[:, 1], dtype=float)
                _require(raw_train.shape == (len(x_train),) and raw_test.shape == (len(x_predict),)
                         and np.isfinite(raw_train).all() and np.isfinite(raw_test).all()
                         and ((raw_train >= 0) & (raw_train <= 1)).all()
                         and ((raw_test >= 0) & (raw_test <= 1)).all(), "invalid fitted probabilities")
                edges, hit_map, base = _oof_bucket_calib(raw_train, labels[arm, head], MODEL_CONFIG["calibration"]["buckets"])
                calibrated = _apply_calib(raw_test, edges, hit_map, base)
                _require(np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all(),
                         "invalid calibrated probabilities")
                scores[f"{head}_{arm}"] = calibrated
                scores[f"{head}_{arm}_raw"] = raw_test
                detail["calibration"] = _calibration_record(edges, hit_map, base)
                detail["booster_ubj_sha256"] = hashlib.sha256(model.get_booster().save_raw(raw_format="ubj")).hexdigest()
                audit["fit_count"] += 1
                if head == "p_dn5":
                    expected, record = _expected_downside(raw_train, selected[f"{arm}_low_ret"].to_numpy(), raw_test)
                    _require(np.isfinite(expected).all(), "invalid expected downside")
                    scores[f"exp_downside_{arm}"] = expected
                    audit[f"expected_downside_{arm}"] = record
            except Exception as exc:
                raise HorizonModelError(f"paired fit failed at {arm}/{head}: {exc}") from exc
    audit.update(status="fitted", action="paired_scores_only_not_deployable", B_status="fitted", C_status="fitted")
    scores["B_status"] = scores["C_status"] = "fitted"
    _require(audit["fit_count"] == MODEL_CONFIG["fit_count_success"], "unexpected fit count")
    canonical_json_bytes(audit)
    return {"scores": scores, "audit": audit}
