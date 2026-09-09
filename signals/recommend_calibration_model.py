"""Pure five-head kernels for one time-ordered calibration comparison.

Inner models stay fixed for a calendar block; later rows are predicted only
when their own features become available. Final models must reproduce the
frozen previous B result exactly. No files, DB, network or live models are used.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ops.artifact_provenance import canonical_json_bytes
from scripts.downside_head_riskreward_v1 import _apply_calib, _oof_bucket_calib
from signals import recommend_horizon_model as previous
from signals.recommend_training_data import ALLOWED_FEATURES

KERNEL_CONFIG = {
    "version": "recommend_calibration_model.v1",
    "features": list(ALLOWED_FEATURES),
    "thresholds": deepcopy(previous.MODEL_CONFIG["thresholds"]),
    "xgb_params": deepcopy(previous.MODEL_CONFIG["xgb_params"]),
    "class_weight": "B_negative_over_positive_from_each_kernel_training_set",
    "embargo_calendar_days": 5, "maximum_inner_block_calendar_days": 7,
    "minimum_inner_training_dates": 365, "minimum_positive": 12, "minimum_classes": 2,
    "preprocessing": "train_only_median_inf_to_nan_all_nan_stays_missing",
    "inner_prediction_timing": "frozen_block_model; own_row_features_available_at_or_before_row09KST",
    "final_reference_gate": "exact_X_keys_medians_y_weights_booster_raw_calibrated_and_expected_downside",
    "reference_calibration_buckets": 10,
    "model_artifacts": "memory_only_ubj_sha256; no_save_no_publish",
    "fits_per_available_kernel": 5,
}
HEADS = tuple(KERNEL_CONFIG["thresholds"])
RAW_COLUMNS = tuple(f"{head}_raw" for head in HEADS)
HELD_COLUMNS = ("feature_date", "coin", "input_end_at", "B_target_end_at", *ALLOWED_FEATURES)
TRAIN_COLUMNS = (*previous._TRAIN_META, *ALLOWED_FEATURES)


class CalibrationModelError(ValueError):
    """Invalid evidence, failed fit or any deviation from frozen reference B."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationModelError(message)


def _same(actual, expected, message: str) -> None:
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected), message)


def _schema(frame: pd.DataFrame, columns: tuple, name: str) -> None:
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
             and set(frame.columns) == set(columns), f"invalid {name} columns")
    _require(frame.index.is_unique and not frame.duplicated(["feature_date", "coin"]).any(),
             f"duplicate {name} identity/index")
    frame.feature_date.map(lambda value: previous._day(value, f"{name} feature_date"))
    _require(frame.coin.map(lambda value: isinstance(value, str) and bool(value)).all(), f"invalid {name} coin")


def _clock(frame: pd.DataFrame) -> tuple:
    start = pd.to_datetime(frame.feature_date).dt.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    inputs = pd.to_datetime(frame.input_end_at.map(lambda x: previous._time(x, "input_end_at")), utc=True)
    end = pd.to_datetime(frame.B_target_end_at.map(lambda x: previous._time(x, "B_target_end_at")), utc=True)
    _require(inputs.le(start).all(), "input ends after own row feature origin")
    _require(end.eq(start + pd.Timedelta(days=1)).all(), "invalid exact B target end")
    return start, inputs, end


def _matrix_pair(selected: pd.DataFrame, held: pd.DataFrame) -> tuple:
    train = selected[list(ALLOWED_FEATURES)].replace([np.inf, -np.inf], np.nan)
    medians = train.median()
    x_train = train.fillna(medians).to_numpy(dtype=float)
    x_test = held[list(ALLOWED_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)
    x_train.setflags(write=False)
    x_test.setflags(write=False)
    return x_train, x_test, medians


def _head_plan(selected: pd.DataFrame) -> tuple:
    labels, heads, failures = {}, {}, []
    for head, threshold in KERNEL_CONFIG["thresholds"].items():
        values = selected[f"B_{'high' if threshold > 0 else 'low'}_ret"].to_numpy()
        y = ((values >= threshold) if threshold > 0 else (values <= threshold)).astype(np.uint8)
        positive, negative = int(y.sum()), int(len(y) - y.sum())
        classes = int(positive > 0) + int(negative > 0)
        ready = positive >= KERNEL_CONFIG["minimum_positive"] and classes >= KERNEL_CONFIG["minimum_classes"]
        heads[head] = {"positive": positive, "negative": negative, "classes": classes,
                       "preflight_ready": ready, "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                       "label_hash_encoding": "uint8_in_shared_training_key_order",
                       "effective_parameters": {**KERNEL_CONFIG["xgb_params"],
                                                "scale_pos_weight": negative / positive if positive else None}}
        labels[head] = y
        if not ready:
            failures.append({"head": head, "reason": "insufficient_positive_or_single_class"})
    return labels, heads, failures


def _fit_five(x_train: np.ndarray, x_test: np.ndarray, labels: dict, heads: dict) -> tuple:
    train_scores, test_scores = {}, {}
    for head in HEADS:
        model = XGBClassifier(**heads[head]["effective_parameters"])
        try:
            model.fit(x_train, labels[head])
            for values, destination in ((x_train, train_scores), (x_test, test_scores)):
                raw = np.asarray(model.predict_proba(values)[:, 1], dtype=float)
                _require(raw.shape == (len(values),) and np.isfinite(raw).all()
                         and ((raw >= 0) & (raw <= 1)).all(), "invalid fitted probability")
                destination[f"{head}_raw"] = raw
            heads[head]["booster_ubj_sha256"] = hashlib.sha256(model.get_booster().save_raw(raw_format="ubj")).hexdigest()
        except Exception as exc:
            raise CalibrationModelError(f"five-head fit failed at {head}: {exc}") from exc
    return pd.DataFrame(train_scores), pd.DataFrame(test_scores)


def fit_inner_block(training_frame: pd.DataFrame, held_frame: pd.DataFrame, *,
                    block_start: str, block_end_exclusive: str) -> dict:
    """Fit one past-only model per head; held rows contain no outcomes."""
    start_day = previous._day(block_start, "block_start")
    end_day = previous._day(block_end_exclusive, "block_end_exclusive")
    _require(1 <= (end_day - start_day).days <= KERNEL_CONFIG["maximum_inner_block_calendar_days"],
             "inner block must span one to seven calendar days")
    origin = pd.Timestamp(start_day).tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    cutoff = (start_day - timedelta(days=KERNEL_CONFIG["embargo_calendar_days"])).isoformat()
    _schema(training_frame, TRAIN_COLUMNS, "training")
    _schema(held_frame, HELD_COLUMNS, "held")
    work = training_frame.copy(deep=True).sort_values(["feature_date", "coin"], kind="stable").reset_index(drop=True)
    held = held_frame.copy(deep=True)
    _require(held.feature_date.ge(block_start).all() and held.feature_date.lt(block_end_exclusive).all(),
             "held row outside the fixed calendar block")
    starts, inputs, b_end = _clock(work)
    held_starts, held_inputs, _ = _clock(held)
    _require(work.C_target_present.map(lambda x: isinstance(x, (bool, np.bool_))).all(), "C presence must be boolean")
    c_values = []
    for present, value, expected in zip(work.C_target_present, work.C_target_end_at,
                                        starts + pd.Timedelta(days=2), strict=True):
        if not present:
            _require(pd.isna(value), "absent C target has an end timestamp")
            c_values.append(pd.NaT)
        else:
            stamp = previous._time(value, "C_target_end_at")
            _require(stamp == expected, "invalid exact C target end")
            c_values.append(stamp)
    c_end = pd.Series(pd.to_datetime(c_values, utc=True), index=work.index)
    returns = ("B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret")
    for field in (*ALLOWED_FEATURES, *returns):
        work[field] = previous._numeric(work[field], field)
    for field in ALLOWED_FEATURES:
        held[field] = previous._numeric(held[field], field)
    for arm in ("B", "C"):
        high, low = work[f"{arm}_high_ret"], work[f"{arm}_low_ret"]
        finite_pair = np.isfinite(high) & np.isfinite(low)
        _require(high[finite_pair].ge(low[finite_pair]).all(), "target high below low")
    eligible = work.feature_date.lt(cutoff)
    finite = np.isfinite(work[list(returns)]).all(axis=1)
    mature = inputs.lt(origin) & b_end.lt(origin) & c_end.lt(origin)
    common = eligible & work.C_target_present & finite & mature
    selected = work.loc[common].copy()
    x_train, x_test, medians = _matrix_pair(selected, held)
    labels, heads, failures = _head_plan(selected)
    if selected.feature_date.nunique() < KERNEL_CONFIG["minimum_inner_training_dates"]:
        failures.append({"head": None, "reason": "fewer_than_365_training_dates"})
    if held.empty:
        failures.append({"head": None, "reason": "no_held_rows"})
    timing = [{"feature_date": day, "coin": coin,
               "raw_prediction_available_at": max(origin, row_start, input_end).tz_convert("UTC").isoformat()}
              for day, coin, row_start, input_end in zip(held.feature_date, held.coin, held_starts, held_inputs, strict=True)]
    audit = {"configuration": deepcopy(KERNEL_CONFIG), "status": "unavailable" if failures else "fitted",
             "fit_count": 0, "block_start": block_start, "block_end_exclusive": block_end_exclusive,
             "cutoff_exclusive": cutoff, "model_fit_origin_at": origin.tz_convert("UTC").isoformat(),
             "prediction_timing": timing, "training_rows": len(selected), "training_dates": int(selected.feature_date.nunique()),
             "held_rows": len(held), "training_keys_sha256": previous._digest(selected[["feature_date", "coin"]].to_dict("records")),
             "training_X_sha256": previous._matrix_digest(x_train), "prediction_X_sha256": previous._matrix_digest(x_test),
             "train_medians": {k: None if pd.isna(v) else float(v) for k, v in medians.items()},
             "excluded_before_cutoff": work.loc[eligible & ~common, ["feature_date", "coin"]].to_dict("records"),
             "heads": heads, "failures": failures, "historical_ingestion_proven": False,
             "model_artifact_saved": False, "live_model_changed": False, "deployable": False}
    if failures:
        return {"raw_scores": pd.DataFrame(np.nan, index=held.index.copy(), columns=RAW_COLUMNS), "audit": audit}
    _, raw_scores = _fit_five(x_train, x_test, labels, heads)
    raw_scores.index = held.index.copy()
    audit["fit_count"] = 5
    canonical_json_bytes(audit)
    return {"raw_scores": raw_scores, "audit": audit}


def fit_final_reference(training_frame: pd.DataFrame, prediction_features: pd.DataFrame, *,
                        decision: dict, reference_scores: pd.DataFrame, reference_audit: dict) -> dict:
    """Reproduce only the five B heads and fail on any frozen-reference drift."""
    _same(reference_audit["configuration"], previous.MODEL_CONFIG, "reference model configuration changed")
    _same(reference_audit["decision"], decision, "reference decision mismatch")
    _require(reference_audit["status"] == "fitted", "reference B was unavailable")
    _require(reference_scores.columns.is_unique and reference_scores.index.equals(prediction_features.index),
             "reference prediction index mismatch")
    expected_columns = {f"{head}_B{suffix}" for head in HEADS for suffix in ("", "_raw")} | {"exp_downside_B"}
    _require(expected_columns.issubset(reference_scores.columns), "missing reference score columns")
    selected, x_train, x_test, preparation = previous._prepare(training_frame, prediction_features, decision)
    for field, value in preparation.items():
        _same(value, reference_audit[field], f"reference preparation mismatch: {field}")
    labels, heads, failures = _head_plan(selected)
    _require(not failures and len(prediction_features) > 0, "final reference cannot be fitted")
    for head, detail in heads.items():
        for field, value in detail.items():
            _same(value, reference_audit["heads"][f"B/{head}"][field], f"reference {head}/{field} mismatch")
    raw_training, scores = _fit_five(x_train, x_test, labels, heads)
    scores.index = prediction_features.index.copy()
    for head in HEADS:
        raw = raw_training[f"{head}_raw"].to_numpy()
        edges, mapping, base = _oof_bucket_calib(raw, labels[head], KERNEL_CONFIG["reference_calibration_buckets"])
        scores[f"{head}_F"] = _apply_calib(scores[f"{head}_raw"].to_numpy(), edges, mapping, base)
        heads[head]["calibration"] = previous._calibration_record(edges, mapping, base)
        for field in ("calibration", "booster_ubj_sha256"):
            _same(heads[head][field], reference_audit["heads"][f"B/{head}"][field], f"reference {head}/{field} mismatch")
        for actual, expected in ((f"{head}_raw", f"{head}_B_raw"), (f"{head}_F", f"{head}_B")):
            _require(np.array_equal(scores[actual].to_numpy(), reference_scores[expected].to_numpy()),
                     f"reference score mismatch: {expected}")
    expected_downside, record = previous._expected_downside(raw_training.p_dn5_raw.to_numpy(),
                                                            selected.B_low_ret.to_numpy(), scores.p_dn5_raw.to_numpy())
    scores["exp_downside_F"] = expected_downside
    _same(record, reference_audit["expected_downside_B"], "reference expected downside calibration mismatch")
    _require(np.array_equal(expected_downside, reference_scores.exp_downside_B.to_numpy()), "reference expected downside mismatch")
    audit = {"configuration": deepcopy(KERNEL_CONFIG), "decision": deepcopy(decision), **preparation,
             "status": "fitted", "fit_count": 5, "heads": heads, "expected_downside_F": record,
             "reference_parity": "exact_all_checks_passed", "model_artifact_saved": False,
             "live_model_changed": False, "deployable": False}
    canonical_json_bytes(audit)
    return {"raw_training": raw_training, "training_keys": selected[["feature_date", "coin"]].reset_index(drop=True),
            "scores": scores, "audit": audit}
