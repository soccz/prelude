"""Synthetic-only tests of fixed inner OOT kernels and exact final-B replay."""
from __future__ import annotations

from copy import deepcopy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ops.artifact_provenance import canonical_json_bytes
from signals import recommend_calibration_model as kernel
from signals import recommend_horizon_model as previous


def make_case(days=400):
    index = np.arange(days * 2)
    dates = pd.date_range("2024-01-01", periods=days, tz="Asia/Seoul").repeat(2)
    frame = pd.DataFrame({
        "feature_date": dates.strftime("%Y-%m-%d"), "coin": [f"KRW-{i % 2}" for i in index],
        "input_end_at": (dates + pd.Timedelta(hours=9)).map(pd.Timestamp.isoformat),
        "B_target_end_at": (dates + pd.Timedelta(days=1, hours=9)).map(pd.Timestamp.isoformat),
        "C_target_end_at": (dates + pd.Timedelta(days=2, hours=9)).map(pd.Timestamp.isoformat),
        "C_target_present": True,
        "B_high_ret": np.take([.25, .12, .07, .02, .01], index % 5),
        "B_low_ret": np.take([-.12, -.07, -.02, -.01, -.03], index % 5),
        "C_high_ret": np.take([.12, .25, .02, .07, .01], index % 5),
        "C_low_ret": np.take([-.07, -.12, -.01, -.02, -.03], index % 5),
    })
    for j, feature in enumerate(kernel.ALLOWED_FEATURES):
        frame[feature] = np.sin(index / (j + 1)) + j
    frame[kernel.ALLOWED_FEATURES[-1]] = np.nan
    prediction = frame.loc[[11, 17, 23], list(kernel.ALLOWED_FEATURES)].copy()
    prediction.index = pd.Index([901, 77, 405], name="original")
    decision = {"date": "2026-09-07", "decision_started_at": "2026-09-07T08:50:01+09:00",
                "feature_row_date": "2026-09-06", "cutoff_exclusive": "2026-09-01"}
    return frame, prediction, decision


def held_rows(frame, start="2025-01-10", end="2025-01-17"):
    return frame.loc[frame.feature_date.ge(start) & frame.feature_date.lt(end), list(kernel.HELD_COLUMNS)].copy()


class FakeModel:
    def __init__(self, calls, **params):
        self.calls, self.params = calls, params

    def fit(self, x, y):
        self.x, self.y = np.array(x), np.array(y)
        self.calls.append(self)
        return self

    def predict_proba(self, x):
        raw = .03 + .94 / (1 + np.exp(-np.asarray(x)[:, 0] - .1 * self.y.mean()))
        return np.column_stack((1 - raw, raw))

    def get_booster(self):
        return self

    def save_raw(self, *, raw_format):
        assert raw_format == "ubj"
        return canonical_json_bytes({"params": self.params, "labels": self.y.tolist()})


@pytest.fixture
def fake(monkeypatch):
    calls = []
    def factory(**params):
        return FakeModel(calls, **params)
    monkeypatch.setattr(kernel, "XGBClassifier", factory)
    monkeypatch.setattr(previous, "XGBClassifier", factory)
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("no DB"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("no network"))
    return calls


def inner(frame, held=None, **kwargs):
    start, end = kwargs.get("block_start", "2025-01-10"), kwargs.get("block_end_exclusive", "2025-01-17")
    return kernel.fit_inner_block(frame, held_rows(frame, start, end) if held is None else held,
                                  block_start=start, block_end_exclusive=end)


def final(case, reference=None):
    frame, prediction, decision = case
    reference = previous.fit_daily_pair(frame, prediction, decision=decision) if reference is None else reference
    return kernel.fit_final_reference(frame, prediction, decision=decision,
                                      reference_scores=reference["scores"], reference_audit=reference["audit"])


def test_inner_five_heads_use_past_only_median_labels_and_class_weights(fake):
    frame, _, _ = make_case()
    held = held_rows(frame)
    result = inner(frame, held)
    audit = result["audit"]
    assert audit["fit_count"] == len(fake) == 5 and audit["status"] == "fitted"
    assert audit["cutoff_exclusive"] == "2025-01-05"
    assert audit["training_dates"] == 370 and audit["training_rows"] == 740
    assert list(result["raw_scores"].columns) == list(kernel.RAW_COLUMNS)
    assert result["raw_scores"].index.equals(held.index)
    selected = frame.loc[frame.feature_date.lt("2025-01-05")]
    for model, (head, threshold) in zip(fake, kernel.KERNEL_CONFIG["thresholds"].items(), strict=True):
        y = selected.B_high_ret.ge(threshold) if threshold > 0 else selected.B_low_ret.le(threshold)
        np.testing.assert_array_equal(model.y, y.astype(np.uint8))
        assert model.params == {**kernel.KERNEL_CONFIG["xgb_params"],
                                "scale_pos_weight": float((~y).sum() / y.sum())}
        assert audit["heads"][head]["positive"] == int(y.sum())
        assert np.isnan(model.x[:, -1]).all()


def test_model_origin_is_distinct_from_later_feature_and_prediction_availability(fake):
    frame, _, _ = make_case()
    audit = inner(frame)["audit"]
    assert audit["model_fit_origin_at"] == "2025-01-10T00:00:00+00:00"
    for timing in audit["prediction_timing"]:
        assert timing["raw_prediction_available_at"] == timing["feature_date"] + "T00:00:00+00:00"
    assert audit["prediction_timing"][-1]["raw_prediction_available_at"] > audit["model_fit_origin_at"]
    assert audit["historical_ingestion_proven"] is False


def test_future_training_values_and_held_labels_never_enter_fit(fake):
    frame, _, _ = make_case()
    held = held_rows(frame)
    before = inner(frame, held)
    future = frame.feature_date.ge("2025-01-05")
    frame.loc[future, list(kernel.ALLOWED_FEATURES)] = 1e90
    frame.loc[future, ["B_high_ret", "C_high_ret"]] = 1e6
    frame.loc[future, ["B_low_ret", "C_low_ret"]] = -.9
    after = inner(frame, held)
    assert before["audit"]["training_X_sha256"] == after["audit"]["training_X_sha256"]
    assert before["audit"]["heads"] == after["audit"]["heads"]
    pd.testing.assert_frame_equal(before["raw_scores"], after["raw_scores"])
    held["B_high_ret"] = 0.3
    with pytest.raises(kernel.CalibrationModelError, match="held columns"):
        inner(frame, held)


def test_input_immutability_and_training_held_shuffle_equivalence(fake):
    frame, _, _ = make_case()
    held = held_rows(frame)
    frame_before, held_before = frame.copy(deep=True), held.copy(deep=True)
    expected = inner(frame, held)
    actual = inner(frame.sample(frac=1, random_state=1), held.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(actual["raw_scores"].loc[held.index], expected["raw_scores"])
    assert actual["audit"]["training_keys_sha256"] == expected["audit"]["training_keys_sha256"]
    assert actual["audit"]["heads"] == expected["audit"]["heads"]
    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(held, held_before)


@pytest.mark.parametrize("reason", ["364_dates", "single_class", "11_positive", "empty_held"])
def test_inner_unavailable_never_fits_or_falls_back(fake, reason):
    frame, _, _ = make_case()
    held = held_rows(frame)
    if reason == "364_dates":
        frame = frame.loc[~frame.feature_date.isin(sorted(frame.feature_date.unique())[:6])]
    elif reason == "single_class":
        frame["B_low_ret"] = -.2
    elif reason == "11_positive":
        frame["B_high_ret"] = .01
        frame.loc[:10, "B_high_ret"] = .3
    else:
        held = held.iloc[:0]
    result = inner(frame, held)
    assert result["audit"]["status"] == "unavailable" and result["audit"]["fit_count"] == len(fake) == 0
    assert result["raw_scores"].isna().all().all() and result["audit"]["failures"]


def test_exactly_365_dates_and_12_positive_are_available(fake):
    frame, _, _ = make_case()
    frame["B_high_ret"] = .01
    frame.loc[:11, "B_high_ret"] = .3
    result = inner(frame, block_start="2025-01-05", block_end_exclusive="2025-01-12")
    assert result["audit"]["training_dates"] == 365 and result["audit"]["fit_count"] == 5
    assert result["audit"]["heads"]["p_up20"]["positive"] == 12


def test_missing_exact_C_and_nonfinite_returns_are_common_exclusions(fake):
    frame, _, _ = make_case()
    frame.loc[0, "C_target_present"] = False
    frame.loc[0, ["C_target_end_at", "C_high_ret", "C_low_ret"]] = [None, np.nan, np.nan]
    frame.loc[1, "B_low_ret"] = np.nan
    result = inner(frame)
    assert result["audit"]["training_rows"] == 738
    assert result["audit"]["excluded_before_cutoff"] == [
        {"feature_date": "2024-01-01", "coin": "KRW-0"},
        {"feature_date": "2024-01-01", "coin": "KRW-1"}]


@pytest.mark.parametrize("mutation", ["naive_held_input", "late_held_input", "wrong_held_end", "out_of_block",
                                       "too_long", "wrong_C_end", "absent_C_with_end", "duplicate_train",
                                       "duplicate_held", "bool_feature", "complex_feature", "numeric_string",
                                       "nonboolean_presence", "invalid_high_low"])
def test_invalid_inner_contract_fails_before_fit(fake, mutation):
    frame, _, _ = make_case()
    held = held_rows(frame)
    kwargs = {}
    first = held.index[0]
    if mutation == "naive_held_input":
        held.loc[first, "input_end_at"] = "2025-01-10T09:00:00"
    elif mutation == "late_held_input":
        held.loc[first, "input_end_at"] = "2025-01-10T09:00:01+09:00"
    elif mutation == "wrong_held_end":
        held.loc[first, "B_target_end_at"] = "2025-01-11T09:15:00+09:00"
    elif mutation == "out_of_block":
        held.loc[first, "feature_date"] = "2025-01-09"
    elif mutation == "too_long":
        kwargs["block_end_exclusive"] = "2025-01-18"
    elif mutation == "wrong_C_end":
        frame.loc[0, "C_target_end_at"] = "2024-01-04T09:00:00+09:00"
    elif mutation == "absent_C_with_end":
        frame.loc[0, "C_target_present"] = False
    elif mutation == "duplicate_train":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "duplicate_held":
        held = pd.concat([held, held.iloc[:1]])
    elif mutation in {"bool_feature", "complex_feature", "numeric_string"}:
        held[kernel.ALLOWED_FEATURES[0]] = {"bool_feature": True, "complex_feature": 1+1j,
                                           "numeric_string": "1"}[mutation]
    elif mutation == "nonboolean_presence":
        frame["C_target_present"] = 1
    else:
        frame.loc[0, "B_high_ret"] = -1
    with pytest.raises((kernel.CalibrationModelError, previous.HorizonModelError)):
        inner(frame, held, **kwargs)
    assert fake == []


def test_final_reproduces_every_reference_field_and_keeps_keys_raw_aligned(fake):
    case = make_case(80)
    frame, prediction, decision = case
    before = frame.copy(deep=True), prediction.copy(deep=True), deepcopy(decision)
    result = final(case)
    assert len(fake) == 15 and result["audit"]["fit_count"] == 5
    assert result["audit"]["reference_parity"] == "exact_all_checks_passed"
    assert list(result["raw_training"].columns) == list(kernel.RAW_COLUMNS)
    assert result["training_keys"].index.equals(result["raw_training"].index)
    assert result["scores"].index.equals(prediction.index)
    pd.testing.assert_frame_equal(result["training_keys"], frame[["feature_date", "coin"]])
    pd.testing.assert_frame_equal(frame, before[0])
    pd.testing.assert_frame_equal(prediction, before[1])
    assert decision == before[2]


@pytest.mark.parametrize("mutation", ["raw", "calibrated", "expected", "booster", "median", "X", "keys", "class_weight", "label", "index"])
def test_final_any_reference_drift_is_hard_failure(fake, mutation):
    case = make_case(80)
    frame, prediction, decision = case
    reference = previous.fit_daily_pair(frame, prediction, decision=decision)
    audit, scores = reference["audit"], reference["scores"]
    if mutation in {"raw", "calibrated", "expected"}:
        column = {"raw": "p_up10_B_raw", "calibrated": "p_up10_B", "expected": "exp_downside_B"}[mutation]
        scores.loc[scores.index[0], column] += .001
    elif mutation == "booster":
        audit["heads"]["B/p_up10"]["booster_ubj_sha256"] = "0" * 64
    elif mutation == "median":
        audit["train_medians"][kernel.ALLOWED_FEATURES[0]] += .01
    elif mutation == "X":
        audit["training_X_sha256"] = "0" * 64
    elif mutation == "keys":
        audit["training_keys_sha256"] = "0" * 64
    elif mutation == "class_weight":
        audit["heads"]["B/p_up10"]["effective_parameters"]["scale_pos_weight"] += 1
    elif mutation == "label":
        audit["heads"]["B/p_up10"]["label_sha256"] = "0" * 64
    else:
        reference["scores"] = scores.iloc[::-1]
    with pytest.raises(kernel.CalibrationModelError):
        final(case, reference)


def test_actual_fixed_XGB_final_matches_previous_B_all_outputs_and_boosters(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("no DB"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setattr(kernel.XGBClassifier, "save_model", lambda *a, **k: pytest.fail("no model file"))
    case = make_case(80)
    frame, prediction, decision = case
    frame.loc[0, kernel.ALLOWED_FEATURES[0]] = np.inf
    prediction.iloc[0, 0] = np.nan
    reference = previous.fit_daily_pair(frame, prediction, decision=decision)
    before = list(tmp_path.iterdir())
    result = final(case, reference)
    assert result["audit"]["reference_parity"] == "exact_all_checks_passed"
    assert result["audit"]["fit_count"] == 5
    assert list(tmp_path.iterdir()) == before
    for head in kernel.HEADS:
        np.testing.assert_array_equal(result["scores"][f"{head}_raw"], reference["scores"][f"{head}_B_raw"])
        np.testing.assert_array_equal(result["scores"][f"{head}_F"], reference["scores"][f"{head}_B"])
        assert result["audit"]["heads"][head]["booster_ubj_sha256"] == reference["audit"]["heads"][f"B/{head}"]["booster_ubj_sha256"]
    np.testing.assert_array_equal(result["scores"].exp_downside_F, reference["scores"].exp_downside_B)


def test_failed_inner_fit_never_returns_partial_scores(fake, monkeypatch):
    original_fit = FakeModel.fit
    def failing_fit(self, x, y):
        if len(self.calls) == 2:
            raise RuntimeError("synthetic failure")
        return original_fit(self, x, y)
    monkeypatch.setattr(FakeModel, "fit", failing_fit)
    frame, _, _ = make_case()
    with pytest.raises(kernel.CalibrationModelError, match="five-head fit failed at p_up20"):
        inner(frame)
    assert len(fake) == 2


def test_invalid_inner_probability_is_hard_failure(fake, monkeypatch):
    monkeypatch.setattr(FakeModel, "predict_proba", lambda self, x: np.full((len(x), 2), np.nan))
    frame, _, _ = make_case()
    with pytest.raises(kernel.CalibrationModelError, match="invalid fitted probability"):
        inner(frame)
    assert len(fake) == 1
