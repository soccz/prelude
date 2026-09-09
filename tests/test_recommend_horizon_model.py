"""Daily target-pair tests: synthetic inputs only, immutable and offline."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ops.artifact_provenance import canonical_json_bytes
from scripts.downside_head_riskreward_v1 import _apply_calib, _oof_bucket_calib
from signals import recommend_horizon_model as model


def _case(n=160):
    index = np.arange(n)
    days = pd.date_range("2026-01-01", periods=n // 4, tz="Asia/Seoul").repeat(4)
    frame = pd.DataFrame({
        "feature_date": days.strftime("%Y-%m-%d"), "coin": [f"KRW-{i % 4}" for i in index],
        "input_end_at": (days + pd.Timedelta(hours=9)).map(pd.Timestamp.isoformat),
        "B_target_end_at": (days + pd.Timedelta(days=1, hours=9)).map(pd.Timestamp.isoformat),
        "C_target_end_at": (days + pd.Timedelta(days=2, hours=9)).map(pd.Timestamp.isoformat),
        "C_target_present": True,
        "B_high_ret": np.take([.25, .12, .07, .02, .01], index % 5),
        "B_low_ret": np.take([-.12, -.07, -.02, -.01, -.03], index % 5),
        "C_high_ret": np.take([.02, .01, .25, .12, .07], index % 5),
        "C_low_ret": np.take([-.02, -.01, -.03, -.12, -.07], index % 5),
    })
    for j, feature in enumerate(model.ALLOWED_FEATURES):
        frame[feature] = np.sin(index / (j + 1)) + j
    frame[model.ALLOWED_FEATURES[-1]] = np.nan
    prediction = frame.loc[[13, 5, 24], list(model.ALLOWED_FEATURES)].copy()
    prediction.index = pd.Index([901, 77, 405], name="original_row")
    decision = {"date": "2026-09-07", "decision_started_at": "2026-09-07T08:50:01+09:00",
                "feature_row_date": "2026-09-06", "cutoff_exclusive": "2026-09-01"}
    return frame, prediction, decision


@pytest.fixture
def case():
    return _case()


class _FakeModel:
    def __init__(self, calls, **params):
        self.calls, self.params = calls, params

    def fit(self, x, y):
        self.x, self.y = np.array(x), np.array(y)
        self.calls.append(self)
        return self

    def predict_proba(self, x):
        raw = .05 + .85 / (1 + np.exp(-np.asarray(x)[:, 0]))
        return np.column_stack((1 - raw, raw))

    def get_booster(self):
        return self

    def save_raw(self, *, raw_format):
        assert raw_format == "ubj"
        return canonical_json_bytes({"params": self.params, "labels": self.y.tolist()})


@pytest.fixture
def fake(monkeypatch):
    calls = []
    monkeypatch.setattr(model, "XGBClassifier", lambda **params: _FakeModel(calls, **params))
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("no DB access"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("no network"))
    return calls


def _run(case):
    training, prediction, decision = case
    return model.fit_daily_pair(training, prediction, decision=decision)


def _set_day(frame, index, value):
    start = pd.Timestamp(value, tz="Asia/Seoul") + pd.Timedelta(hours=9)
    frame.loc[index, "feature_date"] = value
    frame.loc[index, "input_end_at"] = start.isoformat()
    frame.loc[index, "B_target_end_at"] = (start + pd.Timedelta(days=1)).isoformat()
    frame.loc[index, "C_target_end_at"] = (start + pd.Timedelta(days=2)).isoformat()


def test_ten_heads_share_one_training_X_and_B_weights_even_when_C_prevalence_differs(case, fake):
    frame, prediction, _ = case
    frame["C_high_ret"] = np.where(np.arange(len(frame)) % 2, .3, .01)
    result = _run(case)
    audit, scores = result["audit"], result["scores"]
    assert audit["fit_count"] == len(fake) == 10
    assert audit["status"] == "fitted" and audit["calibration_is_oof"] is False
    assert audit["model_artifact_saved"] is audit["live_model_changed"] is audit["deployable"] is False
    assert scores.index.equals(prediction.index) and scores.shape == (3, 24)
    for i, head in enumerate(model.MODEL_CONFIG["thresholds"]):
        left, right = fake[i], fake[i + 5]
        np.testing.assert_equal(left.x, right.x)
        weight = float((len(left.y) - left.y.sum()) / left.y.sum())
        assert left.params == right.params == {**model.MODEL_CONFIG["xgb_params"], "scale_pos_weight": weight}
        detail = audit["heads"][f"C/{head}"]
        assert detail["effective_parameters"]["scale_pos_weight"] == weight
        assert detail["label_sha256"] == hashlib.sha256(right.y.tobytes()).hexdigest()
        assert detail["booster_ubj_sha256"] == hashlib.sha256(right.save_raw(raw_format="ubj")).hexdigest()
    assert audit["heads"]["B/p_up20"]["positive"] != audit["heads"]["C/p_up20"]["positive"]
    assert audit["heads"]["C/p_up20"]["effective_parameters"]["scale_pos_weight"] == 4
    json.dumps(audit, allow_nan=False)


def test_exclusive_cutoff_missing_calendar_and_nonfinite_targets_are_shared_exclusions(case, fake):
    frame, _, _ = case
    _set_day(frame, 0, "2026-09-01")
    _set_day(frame, 1, "2026-09-04")
    frame.loc[2, "C_target_present"] = False
    frame.loc[2, ["C_target_end_at", "C_high_ret", "C_low_ret"]] = [None, np.nan, np.nan]
    frame.loc[3, "B_high_ret"] = np.inf
    frame.loc[4, "C_low_ret"] = np.nan
    result = _run(case)
    assert result["audit"]["training_rows"] == 155
    assert result["audit"]["exclusions"] == {
        "at_or_after_exclusive_cutoff": 2, "missing_exact_next_calendar_target": 1,
        "nonfinite_target_returns": 2, "target_not_ended_before_decision": 0,
    }
    assert len(result["audit"]["excluded_training_candidates"]) == 3
    assert all(len(fitted.y) == 155 for fitted in fake)


def test_prediction_extremes_never_change_train_median_or_fitted_labels(case, fake):
    frame, prediction, _ = case
    first, last = model.ALLOWED_FEATURES[0], model.ALLOWED_FEATURES[-1]
    frame.loc[0, first] = np.inf
    prediction[first] = np.nan
    expected = frame[first].replace([np.inf, -np.inf], np.nan).median()
    result = _run(case)
    assert result["audit"]["train_medians"][first] == expected
    assert result["audit"]["train_medians"][last] is None
    assert result["audit"]["all_nan_features"] == [last]
    assert np.isnan(fake[0].x[:, -1]).all()
    assert fake[0].x[0, 0] == expected
    prediction[first] = 1e12
    changed = _run(case)
    assert changed["audit"]["training_X_sha256"] == result["audit"]["training_X_sha256"]
    assert changed["audit"]["heads"] == result["audit"]["heads"]


def test_calibration_is_exact_existing_helper_and_dn5_expected_low_reuses_fit(case, fake):
    frame, prediction, _ = case
    result = _run(case)
    for arm, offset in (("B", 0), ("C", 5)):
        for i, head in enumerate(model.MODEL_CONFIG["thresholds"]):
            fitted = fake[offset + i]
            raw_train = fitted.predict_proba(fitted.x)[:, 1]
            raw_test = result["scores"][f"{head}_{arm}_raw"].to_numpy()
            edges, mapping, base = _oof_bucket_calib(raw_train, fitted.y, 10)
            np.testing.assert_array_equal(result["scores"][f"{head}_{arm}"], _apply_calib(raw_test, edges, mapping, base))
        dn5 = fake[offset + 3]
        low = frame.sort_values(["feature_date", "coin"])[f"{arm}_low_ret"].to_numpy()
        bucket = pd.DataFrame({"s": dn5.predict_proba(dn5.x)[:, 1], "low": low})
        bucket["bk"] = pd.qcut(bucket.s.rank(method="first"), 10, labels=False, duplicates="drop")
        grouped = bucket.groupby("bk").agg(hi=("s", "max"), mean=("low", "mean"))
        raw = dn5.predict_proba(prediction.fillna(0).to_numpy())[:, 1]
        expected = _apply_calib(raw, grouped.hi.to_numpy(), grouped["mean"].to_dict(), float(low.mean()))
        np.testing.assert_array_equal(result["scores"][f"exp_downside_{arm}"], expected)
    assert len(fake) == 10 and result["audit"]["expected_downside_reuses_dn5"] is True


def test_identical_targets_make_identical_paired_scores_and_hashes(case, fake):
    frame, _, _ = case
    frame[["C_high_ret", "C_low_ret"]] = frame[["B_high_ret", "B_low_ret"]].to_numpy()
    result = _run(case)
    for head in (*model.MODEL_CONFIG["thresholds"], "exp_downside"):
        np.testing.assert_array_equal(result["scores"][f"{head}_B"], result["scores"][f"{head}_C"])
    for head in model.MODEL_CONFIG["thresholds"]:
        assert result["audit"]["heads"][f"B/{head}"] == result["audit"]["heads"][f"C/{head}"]


@pytest.mark.parametrize("failure", ["rare_C", "single_class_B", "empty_training", "empty_prediction"])
def test_preflight_unavailable_never_fits_or_substitutes_A(case, fake, failure):
    frame, prediction, decision = case
    if failure == "rare_C":
        frame["C_high_ret"] = .01
        frame.loc[:10, "C_high_ret"] = .3
    elif failure == "single_class_B":
        frame["B_low_ret"] = -.2
    elif failure == "empty_training":
        frame = frame.iloc[:0]
    else:
        prediction = prediction.iloc[:0]
    result = model.fit_daily_pair(frame, prediction, decision=decision)
    assert result["audit"]["fit_count"] == len(fake) == 0
    assert result["audit"]["status"] == "unavailable" and result["audit"]["failures"]
    assert result["scores"].drop(columns=["B_status", "C_status"]).isna().all().all()
    assert set(result["scores"]["B_status"]) <= {"unavailable"}
    assert result["scores"].index.equals(prediction.index)
    json.dumps(result["audit"], allow_nan=False)


def test_input_frames_decision_and_row_permutations_preserve_pair(case, fake):
    frame, prediction, decision = case
    before, pred_before, dec_before = frame.copy(deep=True), prediction.copy(deep=True), deepcopy(decision)
    baseline = _run(case)
    changed = model.fit_daily_pair(frame.sample(frac=1, random_state=8), prediction.iloc[::-1], decision=decision)
    pd.testing.assert_frame_equal(changed["scores"].loc[prediction.index], baseline["scores"])
    assert changed["audit"]["training_keys_sha256"] == baseline["audit"]["training_keys_sha256"]
    assert changed["audit"]["heads"] == baseline["audit"]["heads"]
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(prediction, pred_before)
    assert decision == dec_before


@pytest.mark.parametrize("corruption", ["extra_label", "extra_coin", "reordered", "missing_feature", "duplicate_index", "numeric_string", "bool_feature"])
def test_prediction_input_never_accepts_labels_identity_or_schema_drift(case, fake, corruption):
    frame, prediction, decision = case
    if corruption == "extra_label":
        prediction["up10"] = True
    elif corruption == "extra_coin":
        prediction["coin"] = "KRW-A"
    elif corruption == "reordered":
        prediction = prediction[prediction.columns[::-1]]
    elif corruption == "missing_feature":
        prediction = prediction.iloc[:, :-1]
    elif corruption == "duplicate_index":
        prediction.index = [1, 1, 2]
    else:
        prediction[model.ALLOWED_FEATURES[0]] = "1" if corruption == "numeric_string" else True
    with pytest.raises(model.HorizonModelError):
        model.fit_daily_pair(frame, prediction, decision=decision)
    assert fake == []


@pytest.mark.parametrize("corruption", ["naive_input", "input_after_target", "wrong_B_end", "nearest_C_end", "nonboolean_presence", "duplicate_identity", "invalid_high_low", "wrong_reference", "wrong_cutoff", "open_slot"])
def test_identity_type_and_chronology_fail_before_any_fit(case, fake, corruption):
    frame, _, decision = case
    if corruption == "naive_input":
        frame.loc[0, "input_end_at"] = "2026-01-01T09:00:00"
    elif corruption == "input_after_target":
        frame.loc[0, "input_end_at"] = "2026-01-01T09:00:01+09:00"
    elif corruption == "wrong_B_end":
        frame.loc[0, "B_target_end_at"] = "2026-01-02T09:15:00+09:00"
    elif corruption == "nearest_C_end":
        frame.loc[0, "C_target_end_at"] = "2026-01-04T09:00:00+09:00"
    elif corruption == "nonboolean_presence":
        frame["C_target_present"] = 1
    elif corruption == "duplicate_identity":
        frame.loc[1, ["feature_date", "coin"]] = frame.loc[0, ["feature_date", "coin"]].to_numpy()
    elif corruption == "invalid_high_low":
        frame.loc[0, "B_high_ret"] = -.99
    elif corruption == "wrong_reference":
        decision["feature_row_date"] = "2026-09-07"
    elif corruption == "wrong_cutoff":
        decision["cutoff_exclusive"] = "2026-09-02"
    else:
        decision["decision_started_at"] = "2026-09-07T09:05:00+09:00"
    with pytest.raises(model.HorizonModelError):
        _run(case)
    assert fake == []


def test_terminal_missing_target_cannot_be_pretended_present(case, fake):
    frame, _, _ = case
    frame.loc[0, "C_target_end_at"] = None
    with pytest.raises(model.HorizonModelError, match="C_target_end_at"):
        _run(case)
    assert fake == []


def test_fit_exception_never_returns_partial_pair(case, fake, monkeypatch):
    original = _FakeModel.fit

    def fit(self, x, y):
        if len(self.calls) == 7:
            raise RuntimeError("synthetic learner failure")
        return original(self, x, y)

    monkeypatch.setattr(_FakeModel, "fit", fit)
    with pytest.raises(model.HorizonModelError, match="paired fit failed at C/p_up20"):
        _run(case)
    assert len(fake) == 7


def test_actual_tiny_xgb_repeats_exactly_without_IO_or_parameter_overrides(case, monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("no DB access"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setattr(model.XGBClassifier, "save_model", lambda *a, **k: pytest.fail("no model file"))
    before = list(tmp_path.iterdir())
    first, second = _run(case), _run(case)
    pd.testing.assert_frame_equal(first["scores"], second["scores"], check_exact=True)
    assert first["audit"] == second["audit"]
    assert first["audit"]["fit_count"] == 10
    assert first["scores"].drop(columns=["B_status", "C_status"]).notna().all().all()
    assert list(tmp_path.iterdir()) == before


def test_actual_legacy_rr_head_and_new_B_are_exactly_equal_including_reused_dn5(case, monkeypatch, tmp_path):
    """Existing helper performs six real fits; the pair reuses each arm's dn5.

    Only synthetic, already-eligible rows are passed to the legacy pure helper.
    No parameter overrides or fake learner mask differences between the paths.
    """
    from signals import recommend as legacy

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("no DB access"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setattr(model.XGBClassifier, "save_model", lambda *a, **k: pytest.fail("no model file"))
    for name in ("connect_readonly", "list_markets", "load_candles", "_build_panel", "score_candidates"):
        monkeypatch.setattr(legacy, name, lambda *a, **k: pytest.fail("pure legacy head only"))

    training, prediction, decision = case
    training = training.copy(deep=True)
    prediction = prediction.copy(deep=True)
    # Exercise shared train-only imputation as well as the existing all-NaN column.
    training.loc[0, model.ALLOWED_FEATURES[0]] = np.nan
    prediction.iloc[0, 0] = np.inf
    legacy_training = training.sort_values(["feature_date", "coin"], kind="stable").copy()
    legacy_training["intraday_high_ret"] = legacy_training.B_high_ret
    legacy_training["intraday_low_ret"] = legacy_training.B_low_ret
    legacy_training = legacy._rr_outcome_labels(legacy_training)
    features = list(model.ALLOWED_FEATURES)
    train_before, prediction_before = training.copy(deep=True), prediction.copy(deep=True)
    legacy_before = legacy_training.copy(deep=True)
    before_files = list(tmp_path.iterdir())

    fitted_models = []
    actual_fit = model.XGBClassifier.fit

    def recording_fit(self, *args, **kwargs):
        fitted_models.append(self)
        return actual_fit(self, *args, **kwargs)

    monkeypatch.setattr(model.XGBClassifier, "fit", recording_fit)
    expected = legacy._fit_rr_head(legacy_training, prediction, features)
    assert len(fitted_models) == 6  # Five binary heads plus the old duplicate dn5 fit.
    legacy_models = fitted_models.copy()
    actual = model.fit_daily_pair(training, prediction, decision=decision)
    assert len(fitted_models) == 16 and actual["audit"]["fit_count"] == 10
    for head in (*model.MODEL_CONFIG["thresholds"], "exp_downside"):
        np.testing.assert_array_equal(actual["scores"][f"{head}_B"].to_numpy(), expected[head],
                                      err_msg=f"legacy vs B exact parity: {head}")

    shared_x = legacy_training[features].replace([np.inf, -np.inf], np.nan)
    shared_x = shared_x.fillna(shared_x.median()).to_numpy()
    np.testing.assert_array_equal(legacy_models[3].predict_proba(shared_x),
                                  legacy_models[5].predict_proba(shared_x))
    assert legacy_models[3].get_params() == legacy_models[5].get_params()
    pd.testing.assert_frame_equal(training, train_before)
    pd.testing.assert_frame_equal(prediction, prediction_before)
    pd.testing.assert_frame_equal(legacy_training, legacy_before)
    assert list(tmp_path.iterdir()) == before_files


@pytest.mark.parametrize("location", ["prediction_feature", "training_feature", "training_return"])
def test_complex_numeric_fields_are_rejected_before_fit(case, fake, location):
    training, prediction, _ = case
    frame = prediction if location == "prediction_feature" else training
    field = "B_high_ret" if location == "training_return" else model.ALLOWED_FEATURES[0]
    frame[field] = frame[field].astype(complex) + 1j
    with pytest.raises(model.HorizonModelError):
        _run(case)
    assert fake == []


def test_exactly_twelve_positive_C_targets_pass_the_existing_minimum(case, fake):
    training, _, _ = case
    training["C_high_ret"] = .01
    training.loc[:11, "C_high_ret"] = .3
    result = _run(case)
    assert len(fake) == 10 and result["audit"]["status"] == "fitted"
    for head in ("p_up5", "p_up10", "p_up20"):
        assert result["audit"]["heads"][f"C/{head}"]["positive"] == 12


def test_values_after_cutoff_cannot_affect_training_medians_labels_or_models(case, fake):
    training, _, _ = case
    _set_day(training, 0, "2026-09-01")
    _set_day(training, 1, "2026-09-04")
    baseline = _run(case)
    training.loc[:1, list(model.ALLOWED_FEATURES)] = 1e90
    training.loc[:1, ["B_high_ret", "C_high_ret"]] = 1e6
    training.loc[:1, ["B_low_ret", "C_low_ret"]] = -.9
    changed = _run(case)
    for field in ("training_X_sha256", "training_keys_sha256", "train_medians", "heads"):
        assert changed["audit"][field] == baseline["audit"][field]
    pd.testing.assert_frame_equal(changed["scores"], baseline["scores"])
