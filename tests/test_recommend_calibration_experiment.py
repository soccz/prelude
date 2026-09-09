"""Synthetic, read/write-free orchestration contract tests."""
from __future__ import annotations

from copy import deepcopy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from signals import recommend_calibration_experiment as experiment
from signals import recommend_calibration_model as kernel
from signals import recommend_horizon_model as previous


def _raw(features):
    values = .05 + .85 / (1 + np.exp(-features.iloc[:, 0].to_numpy(float)))
    return pd.DataFrame({name: values.copy() for name in experiment.RAW_COLUMNS}, index=features.index)


def _data():
    days = pd.date_range("2024-01-01", "2025-12-31").tolist()
    days += list(pd.date_range("2026-01-23", periods=10).repeat(20))
    n = len(days)
    start = pd.DatetimeIndex(days).tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    i = np.arange(n)
    frame = pd.DataFrame({
        "feature_date": pd.DatetimeIndex(days).strftime("%Y-%m-%d"),
        "coin": [f"KRW-{number % 20:02d}" for number in range(n)],
        "input_end_at": start.map(pd.Timestamp.isoformat),
        "B_target_end_at": (start + pd.Timedelta(days=1)).map(pd.Timestamp.isoformat),
        "C_target_end_at": (start + pd.Timedelta(days=2)).map(pd.Timestamp.isoformat),
        "C_target_present": True,
        "B_high_ret": np.take([.25, .12, .07, .02, .01], i % 5),
        "B_low_ret": np.take([-.12, -.07, -.02, -.01, -.03], i % 5),
        "C_high_ret": np.take([.25, .12, .07, .02, .01], i % 5),
        "C_low_ret": np.take([-.12, -.07, -.02, -.01, -.03], i % 5),
    })
    for j, name in enumerate(experiment.ALLOWED_FEATURES):
        frame[name] = np.sin(i / (j + 1)) + j
    prediction_rows, metadata = [], {}
    for day in ("2026-07-28", "2026-07-29"):
        reference = pd.Timestamp(day) - pd.Timedelta(days=1)
        sid = f"snapshot-{day}"
        metadata[sid] = {"date": day, "decision_started_at": day+"T08:50:01+09:00",
                         "feature_row_date": str(reference.date()),
                         "cutoff_exclusive": str((reference-pd.Timedelta(days=5)).date())}
        for rank in range(1, 4):
            prediction_rows.append({**frame.loc[rank, list(experiment.ALLOWED_FEATURES)].to_dict(),
                                    "date": day, "slot": "preopen", "snapshot_id": sid,
                                    "coin": f"KRW-P{rank}", "rank": rank, "was_delivered": True,
                                    "mfe": .09999999999999998, "up10": False, "mae": -.02})
    prediction = pd.DataFrame(prediction_rows)
    refs = []
    for row in prediction.to_dict("records"):
        refs.append({**row, **{f"{head}_B": .2 for head in experiment.HEADS},
                     **{f"{head}_B_raw": .4 for head in experiment.HEADS}, "exp_downside_B": -.03})
    report = {"predictions": refs, "daily_fit_audit": [
        {"snapshot_id": sid, "decision": deepcopy(d)} for sid, d in metadata.items()]}
    return frame, prediction, metadata, report


@pytest.fixture
def case():
    return _data()


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def inner(training, held, *, block_start, block_end_exclusive):
        assert list(held.columns) == experiment.HELD_COLUMNS
        calls.append(("inner", block_start))
        origin = (pd.Timestamp(block_start, tz="Asia/Seoul") + pd.Timedelta(hours=9)).tz_convert("UTC")
        timing = [{"feature_date": day, "coin": coin,
                   "raw_prediction_available_at": (pd.Timestamp(day, tz="Asia/Seoul")+pd.Timedelta(hours=9)).tz_convert("UTC").isoformat()}
                  for day, coin in zip(held.feature_date, held.coin)]
        return {"raw_scores": _raw(held.loc[:, experiment.ALLOWED_FEATURES]), "audit": {
            "configuration": deepcopy(experiment.KERNEL_CONFIG), "status": "fitted", "fit_count": 5,
            "training_dates": 400, "model_fit_origin_at": origin.isoformat(),
            "prediction_timing": timing,
            "cutoff_exclusive": str((pd.Timestamp(block_start)-pd.Timedelta(days=5)).date()),
        }}

    def final(training, prediction, *, decision, reference_scores, reference_audit):
        calls.append(("outer", decision["date"]))
        assert list(prediction.columns) == list(experiment.ALLOWED_FEATURES)
        assert len(reference_scores.columns) == 11 and "mfe" not in reference_scores
        selected = experiment._common_training(training, decision)
        raw_training = _raw(selected.loc[:, experiment.ALLOWED_FEATURES])
        scores = _raw(prediction)
        for head in experiment.HEADS:
            scores[f"{head}_F"] = .2
        scores["exp_downside_F"] = -.03
        return {"raw_training": raw_training, "training_keys": selected[experiment.KEYS], "scores": scores,
                "audit": {"fit_count": 5, "status": "fitted", "reference_parity": "exact_all_checks_passed",
                          "configuration": deepcopy(experiment.KERNEL_CONFIG)}}

    monkeypatch.setattr(experiment, "fit_inner_block", inner)
    monkeypatch.setattr(experiment, "fit_final_reference", final)
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("DB forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    return calls


def test_inner_cache_precedes_daily_F_and_R_O_share_keys_labels_raw_and_downside(case, fake):
    result = experiment.run_calibration_experiment(*case)
    assert fake == [("inner", "2026-01-23"), ("inner", "2026-01-30"),
                    ("outer", "2026-07-28"), ("outer", "2026-07-29")]
    assert len(result["inner_audit"]) == 26 and len(result["outer_audit"]) == 2
    assert result["fit_count"] == 20 == sum(a["fit_count"] for a in result["inner_audit"]+result["outer_audit"])
    assert len(result["oof_cache"]) == 200 and len(result["resub_cache"]) == 380
    assert not result["oof_cache"].duplicated(experiment.KEYS).any()
    out = result["predictions"]
    for arm in ("F", "R", "O"):
        assert out[f"{arm}_status"].eq("fitted").all()
        np.testing.assert_array_equal(out[f"exp_downside_{arm}"], out.exp_downside_F)
        for head in experiment.HEADS:
            np.testing.assert_array_equal(out[f"{head}_{arm}_raw"], out[f"{head}_F_raw"])
    for audit, rows, days in zip(result["outer_audit"], (200, 180), (10, 9)):
        assert audit["calibration_rows"] == rows and audit["calibration_dates"] == days
        assert audit["missing_oof_rows"] == audit["nonfinite_oof_rows"] == 0
        assert audit["maps"]["R"] == audit["maps"]["O"]
    timing = result["oof_cache"]
    assert (pd.to_datetime(timing.prediction_available_at) >= pd.to_datetime(timing.fit_origin)).all()
    assert (pd.to_datetime(timing.target_end_at) > pd.to_datetime(timing.prediction_available_at)).all()


def test_all_inputs_unchanged_original_test_precision_and_shuffle_invariance(case, fake):
    training, prediction, meta, reference = case
    before = training.copy(deep=True), prediction.copy(deep=True), deepcopy(meta), deepcopy(reference)
    result = experiment.run_calibration_experiment(*case)
    changed = experiment.run_calibration_experiment(training.sample(frac=1, random_state=2),
                prediction.sample(frac=1, random_state=4), dict(reversed(list(meta.items()))), reference)
    pd.testing.assert_frame_equal(result["predictions"], changed["predictions"], check_exact=True)
    assert result["outer_audit"] == changed["outer_audit"]
    pd.testing.assert_frame_equal(training, before[0])
    pd.testing.assert_frame_equal(prediction, before[1])
    assert meta == before[2] and reference == before[3]
    pd.testing.assert_frame_equal(result["predictions"].loc[:, prediction.columns], prediction, check_exact=True)
    assert result["predictions"].mfe.iloc[0] < .1


def test_one_missing_oof_row_never_shrinks_matched_pool(case, fake, monkeypatch):
    original = pd.DataFrame.reindex

    def reindex(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if "block_id" in result and isinstance(result.index, pd.MultiIndex):
            result.iloc[0] = np.nan
        return result

    monkeypatch.setattr(pd.DataFrame, "reindex", reindex)
    result = experiment.run_calibration_experiment(*case)
    for a, rows in zip(result["outer_audit"], (200, 180)):
        assert a["calibration_rows"] == rows and a["missing_oof_rows"] == 1
        assert "missing_oof_rows" in a["failures"]
    assert len(result["resub_cache"]) == 380
    for arm in ("R", "O"):
        columns = [c for c in result["predictions"] if c.endswith(f"_{arm}") or c.endswith(f"_{arm}_raw")]
        assert result["predictions"][columns].isna().all().all()
        assert result["predictions"][f"{arm}_status"].eq("unavailable").all()
    assert result["predictions"].F_status.eq("fitted").all()


def test_unavailable_inner_block_does_not_trigger_resubstitution_fallback(case, fake, monkeypatch):
    original = experiment.fit_inner_block

    def inner(*args, **kwargs):
        result = original(*args, **kwargs)
        result["audit"].update(status="unavailable", fit_count=0)
        result["raw_scores"].loc[:, :] = np.nan
        return result

    monkeypatch.setattr(experiment, "fit_inner_block", inner)
    result = experiment.run_calibration_experiment(*case)
    assert result["fit_count"] == 10
    assert all(a["R_status"] == a["O_status"] == "unavailable" for a in result["outer_audit"])
    assert [a["nonfinite_oof_rows"] for a in result["outer_audit"]] == [200, 180]


@pytest.mark.parametrize("guard", ["rows", "dates", "positive", "negative"])
def test_matched_guard_is_joint_and_checked_before_maps(case, fake, monkeypatch, guard):
    training, prediction, meta, reference = case
    recent = training.feature_date.ge("2026-01-23")
    if guard == "rows":
        training = training.loc[~recent | training.index.isin(training.loc[recent].index[:140])].copy()
    elif guard == "dates":
        training = training.loc[~recent | training.feature_date.le("2026-01-26")].copy()
    else:
        training.loc[recent, "B_high_ret"] = .01 if guard == "positive" else .3
    monkeypatch.setattr(experiment, "_oof_bucket_calib", lambda *a: pytest.fail("preflight must precede every map"))
    result = experiment.run_calibration_experiment(training, prediction, meta, reference)
    assert all(a["R_status"] == a["O_status"] == "unavailable" for a in result["outer_audit"])


def test_none_map_is_unavailable_not_base_rate_fallback(case, fake, monkeypatch):
    monkeypatch.setattr(experiment, "_oof_bucket_calib", lambda *a: (None, None, .2))
    result = experiment.run_calibration_experiment(*case)
    assert all(a["failures"] and a["R_status"] == a["O_status"] == "unavailable" for a in result["outer_audit"])


def test_labels_are_created_once_per_outer_pool_and_shared_between_R_O(case, fake, monkeypatch):
    original = experiment._oof_bucket_calib
    calls = []

    def fit(raw, labels, buckets):
        calls.append((labels, raw.copy()))
        return original(raw, labels, buckets)

    monkeypatch.setattr(experiment, "_oof_bucket_calib", fit)
    experiment.run_calibration_experiment(*case)
    assert len(calls) == 20
    for start, rows in ((0, 200), (10, 180)):
        for head in range(5):
            assert calls[start+head][0] is calls[start+head+5][0]
            assert len(calls[start+head][0]) == rows


@pytest.mark.parametrize("corruption", ["reference_feature", "reference_label", "missing_reference", "duplicate_reference", "metadata", "prediction_key"])
def test_corrupt_frozen_inputs_fail_before_fit(case, fake, corruption):
    training, prediction, meta, reference = case
    if corruption in {"reference_feature", "reference_label"}:
        reference["predictions"][0][experiment.ALLOWED_FEATURES[0] if corruption.endswith("feature") else "mfe"] = .9
    elif corruption == "missing_reference":
        reference["predictions"].pop()
    elif corruption == "duplicate_reference":
        reference["predictions"].append(reference["predictions"][0])
    elif corruption == "metadata":
        meta[next(iter(meta))]["date"] = "2026-07-29"
    else:
        prediction.loc[1, "coin"] = prediction.loc[0, "coin"]
    with pytest.raises(experiment.CalibrationExperimentError):
        experiment.run_calibration_experiment(*case)
    assert fake == []


@pytest.mark.parametrize("corruption", ["keys", "raw_index", "partial_fit", "parity"])
def test_invalid_final_kernel_contract_is_hard_failure(case, fake, monkeypatch, corruption):
    original = experiment.fit_final_reference

    def final(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "keys":
            result["training_keys"].loc[0, "coin"] = "KRW-MISSING"
        elif corruption == "raw_index":
            result["raw_training"] = result["raw_training"].iloc[::-1]
        elif corruption == "partial_fit":
            result["audit"]["fit_count"] = 4
        else:
            result["audit"]["reference_parity"] = "not_checked"
        return result

    monkeypatch.setattr(experiment, "fit_final_reference", final)
    with pytest.raises(experiment.CalibrationExperimentError):
        experiment.run_calibration_experiment(*case)


@pytest.mark.parametrize("corruption", ["held_time", "held_keys", "raw_nonfinite", "training_mutation"])
def test_invalid_inner_evidence_is_not_a_legal_missing_oof(case, fake, monkeypatch, corruption):
    original = experiment.fit_inner_block

    def inner(training, held, **kwargs):
        result = original(training, held, **kwargs)
        if corruption == "held_time":
            result["audit"]["prediction_timing"][0]["raw_prediction_available_at"] = "2030-01-01T00:00:00+00:00"
        elif corruption == "held_keys":
            result["audit"]["prediction_timing"][0]["coin"] = "KRW-WRONG"
        elif corruption == "raw_nonfinite":
            result["raw_scores"].iloc[0, 0] = np.nan
        else:
            training.loc[0, experiment.ALLOWED_FEATURES[0]] = 999
        return result

    monkeypatch.setattr(experiment, "fit_inner_block", inner)
    original_training = case[0].copy(deep=True)
    with pytest.raises(experiment.CalibrationExperimentError):
        experiment.run_calibration_experiment(*case)
    pd.testing.assert_frame_equal(case[0], original_training)


def test_changing_OOF_raw_changes_only_maps_and_new_scores_not_original_inputs(case, fake, monkeypatch):
    original = experiment.fit_inner_block

    def inner(*args, **kwargs):
        result = original(*args, **kwargs)
        result["raw_scores"] = 1-result["raw_scores"]
        return result

    baseline = experiment.run_calibration_experiment(*case)
    monkeypatch.setattr(experiment, "fit_inner_block", inner)
    changed = experiment.run_calibration_experiment(*case)
    pd.testing.assert_frame_equal(changed["predictions"][case[1].columns], baseline["predictions"][case[1].columns], check_exact=True)
    assert changed["outer_audit"][0]["calibration_keys_sha256"] == baseline["outer_audit"][0]["calibration_keys_sha256"]
    assert changed["outer_audit"][0]["matched_head_counts"] == baseline["outer_audit"][0]["matched_head_counts"]
    assert changed["outer_audit"][0]["maps"]["O"] != baseline["outer_audit"][0]["maps"]["O"]


def test_real_kernel_integration_with_memory_only_estimator_and_reference_replay(case, monkeypatch):
    """Exercise actual preparation, parity and maps, mocking only the estimator."""
    fits = []

    class MemoryModel:
        def __init__(self, **params):
            self.params = params

        def fit(self, x, y):
            self.y = y.copy()
            fits.append(self)
            return self

        def predict_proba(self, x):
            raw = .03 + .94 / (1 + np.exp(-np.asarray(x)[:, 0] - .1*self.y.mean()))
            return np.column_stack((1-raw, raw))

        def get_booster(self):
            return self

        def save_raw(self, *, raw_format):
            assert raw_format == "ubj"
            return experiment.canonical_json_bytes({"params": self.params, "labels": self.y.tolist()})

    monkeypatch.setattr(kernel, "XGBClassifier", MemoryModel)
    monkeypatch.setattr(previous, "XGBClassifier", MemoryModel)
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("DB forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("file IO forbidden"))
    training, prediction, metadata, _ = case
    reference = {"predictions": [], "daily_fit_audit": []}
    for sid, group in prediction.groupby("snapshot_id", sort=False):
        baseline = previous.fit_daily_pair(training, group.loc[:, experiment.ALLOWED_FEATURES], decision=metadata[sid])
        reference["predictions"].extend(pd.concat([group, baseline["scores"]], axis=1).to_dict("records"))
        reference["daily_fit_audit"].append({"snapshot_id": sid, **baseline["audit"]})
    baseline_fits = len(fits)
    result = experiment.run_calibration_experiment(training, prediction, metadata, reference)
    assert baseline_fits == 20 and len(fits)-baseline_fits == result["fit_count"] == 20
    assert all(a["reference_fit"]["reference_parity"] == "exact_all_checks_passed" for a in result["outer_audit"])
    assert all(a["R_status"] == a["O_status"] == "fitted" for a in result["outer_audit"])
    assert all(a["training_dates"] >= 365 for a in result["inner_audit"] if a["fit_count"])
    pd.testing.assert_frame_equal(result["predictions"][prediction.columns], prediction, check_exact=True)
