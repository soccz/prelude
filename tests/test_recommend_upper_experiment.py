from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

import signals.recommend_upper_experiment as upper
from signals.recommend_training_data import ALLOWED_FEATURES, OUTCOME_FIELDS

KST = timezone(timedelta(hours=9))


def _frame(days=25, count=20):
    rows = []
    for day_index in range(days):
        day = date(2026, 7, 1) + timedelta(days=day_index)
        for slot in ("open", "preopen"):
            decision = datetime.combine(day, datetime.min.time(), KST).replace(
                hour=9 if slot == "open" else 8, minute=8 if slot == "open" else 53,
            )
            end = decision.replace(hour=9, minute=15 if slot == "open" else 0) + timedelta(days=1)
            for rank in range(1, count + 1):
                positive = rank % 2 == 0
                features = {name: float(rank % 2) + index / 50 for index, name in enumerate(ALLOWED_FEATURES)}
                features[ALLOWED_FEATURES[1]] = float(day_index)
                features[ALLOWED_FEATURES[-1]] = np.nan
                p_up10 = .8 - rank / 100
                row = {
                    **features, "date": day.isoformat(), "slot": slot, "coin": f"KRW-T{rank}",
                    "rank": rank, "snapshot_id": f"fixture-{day}-{slot}",
                    "decision_started_at": decision.astimezone(timezone.utc).isoformat(),
                    "outcome_end_at": end.astimezone(timezone.utc).isoformat(),
                    "label_available_at": (end + timedelta(minutes=10)).astimezone(timezone.utc).isoformat(),
                    "was_delivered": rank <= 3, "label_status": "labeled", "p_up10": p_up10,
                    "p_dn5": .2, "p_dn10": .05, "exp_downside": -.03, "rr_ratio": p_up10 / .2,
                    "actual_entry_open": 100., "mfe": .12 if positive else .03, "mae": -.02,
                    "eod_return_net": .0185, "up5": positive, "up10": positive, "up20": False,
                    "dn3": False, "dn5": False, "dn10": False,
                    "tp5_sl3_first_passage": "tp_first" if positive else "neither",
                    "tp5_before_sl3": positive, "tp5_sl3_return_net": .0485 if positive else .0185,
                    "diagnostic_future_target": 999.,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _folds(frame):
    return upper._time_folds(upper._validate_frame(frame), **upper.EXPERIMENT_CONFIG["outer"])


def _fake_head(train, validation):
    return .2 + .6 * validation[ALLOWED_FEATURES[0]].to_numpy(dtype=float)


def _no_fit(*args, **kwargs):
    pytest.fail("fit must not be reached")


def test_complete_contract_preserves_originals_and_has_no_identity_fallback(monkeypatch):
    frame = _frame()
    before = frame.copy(deep=True)
    folds = _folds(frame)
    folds_before = deepcopy(folds)
    monkeypatch.setattr(upper, "_fit_head", _fake_head)
    result = upper.run_upper_comparison(frame, folds)
    predictions = result["predictions"]
    pd.testing.assert_frame_equal(frame, before)
    assert folds == folds_before
    assert set(predictions["fold"]) == {2, 3}
    assert len(predictions) == 400
    assert predictions[["p_up10_A", "p_up10_B", "p_up10_C", "p_up10_C_raw"]].notna().all().all()
    assert predictions[["B_status", "C_status"]].eq("ok").all().all()
    assert np.array_equal(predictions["p_up10_A"], predictions["p_up10"])
    expected = frame.loc[frame["date"].isin(predictions["date"])].sort_values(["date", "slot", "rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[list(frame.columns)], expected)
    assert result["config"] == upper.EXPERIMENT_CONFIG
    result["config"]["head"]["max_depth"] = 99
    assert upper.EXPERIMENT_CONFIG["head"]["max_depth"] == 2
    assert result["parity"]["top3_passed"] == 50
    assert result["fold_audit"][0]["status"] == "insufficient_train_dates"


def test_noop_top3_parity_failure_blocks_every_fit(monkeypatch):
    frame = _frame()
    folds = _folds(frame)
    frame.loc[frame.index[19], "p_up10"] = .99
    monkeypatch.setattr(upper, "_fit_head", _no_fit)
    monkeypatch.setattr(upper, "minimize_scalar", _no_fit)
    with pytest.raises(upper.UpperExperimentError, match="Top3 parity"):
        upper.run_upper_comparison(frame, folds)


def test_lower_rank_rounding_order_difference_is_audited_not_blocked(monkeypatch):
    frame = _frame()
    frame.loc[[3, 4], "p_up10"] = frame.loc[[4, 3], "p_up10"].to_numpy()
    monkeypatch.setattr(upper, "_fit_head", _fake_head)
    result = upper.run_upper_comparison(frame, _folds(frame))
    assert result["parity"]["full_order_mismatches"] == 1
    assert result["parity"]["status"] == "passed"


@pytest.mark.parametrize("mutation", ["train", "validation", "status", "cutoff", "duplicate", "missing"])
def test_tampered_outer_folds_fail_before_fitting(monkeypatch, mutation):
    frame = _frame()
    folds = _folds(frame)
    if mutation == "train":
        folds[1]["train_dates"].append(folds[1]["validation_dates"][0])
    elif mutation == "validation":
        folds[1]["validation_dates"] = folds[1]["validation_dates"][1:]
    elif mutation == "status":
        folds[0]["status"] = "split_ready"
    elif mutation == "cutoff":
        folds[1]["train_max_label_available_at"] = folds[1]["validation_min_decision_started_at"]
    elif mutation == "duplicate":
        folds.append(deepcopy(folds[-1]))
    else:
        folds.pop()
    monkeypatch.setattr(upper, "_fit_head", _no_fit)
    monkeypatch.setattr(upper, "minimize_scalar", _no_fit)
    with pytest.raises(upper.UpperExperimentError, match="frozen readiness split"):
        upper.run_upper_comparison(frame, folds)


def test_label_availability_equality_is_purged_globally(monkeypatch):
    frame = _frame()
    stale = _folds(frame)
    affected = stale[1]["train_dates"][-1]
    frame.loc[frame["date"].eq(affected) & frame["slot"].eq("open"), "label_available_at"] = stale[1]["validation_min_decision_started_at"]
    current = _folds(frame)
    assert affected not in current[1]["train_dates"]
    assert affected in current[1]["purged_train_dates"]
    monkeypatch.setattr(upper, "_fit_head", _no_fit)
    monkeypatch.setattr(upper, "minimize_scalar", _no_fit)
    with pytest.raises(upper.UpperExperimentError, match="frozen readiness split"):
        upper.run_upper_comparison(frame, stale)


def test_all_inner_and_outer_fits_are_separated_and_halted_candidates_survive(monkeypatch):
    frame = _frame()
    halt = (frame["coin"].eq("KRW-T20") & frame["date"].isin(["2026-07-07", "2026-07-18"]))
    frame.loc[halt, "label_status"] = "halted_no_observations"
    for name in OUTCOME_FIELDS:
        frame[name] = frame[name].astype(object)
        frame.loc[halt, name] = np.nan
    calls = []

    def spy(train, validation):
        assert train["slot"].nunique() == validation["slot"].nunique() == 1
        assert train["slot"].iloc[0] == validation["slot"].iloc[0]
        assert train["label_status"].eq("labeled").all()
        assert not set(train["date"]) & set(validation["date"])
        assert pd.to_datetime(train["label_available_at"]).max() < pd.to_datetime(validation["decision_started_at"]).min()
        calls.append((set(train["date"]), set(validation["date"])))
        return _fake_head(train, validation)

    monkeypatch.setattr(upper, "_fit_head", spy)
    result = upper.run_upper_comparison(frame, _folds(frame))
    assert calls
    held_halts = result["predictions"].loc[lambda data: data["label_status"].ne("labeled")]
    assert len(held_halts) == 2
    assert held_halts[["p_up10_A", "p_up10_B", "p_up10_C"]].notna().all().all()
    assert held_halts[list(OUTCOME_FIELDS)].isna().all().all()
    c = result["fold_audit"][1]["slots"][0]["C"]
    assert c["oof_counts"]["dates"] == 7
    assert c["oof_dates"][0] > result["fold_audit"][1]["train_dates"][0]


def test_validation_outcomes_do_not_change_their_own_fold_predictions(monkeypatch):
    frame = _frame()
    folds = _folds(frame)
    changed = frame.copy(deep=True)
    held = changed["date"].isin(folds[1]["validation_dates"])
    changed.loc[held, "up10"] = ~changed.loc[held, "up10"]
    monkeypatch.setattr(upper, "_fit_head", _fake_head)
    before = upper.run_upper_comparison(frame, folds)["predictions"]
    after = upper.run_upper_comparison(changed, folds)["predictions"]
    columns = ["p_up10_A", "p_up10_B", "p_up10_C", "p_up10_C_raw"]
    pd.testing.assert_frame_equal(before.loc[before["fold"].eq(2), columns], after.loc[after["fold"].eq(2), columns])


def test_insufficient_classes_are_nan_with_reasons_not_fallback(monkeypatch):
    frame = _frame()
    frame["up10"] = False
    result = upper.run_upper_comparison(frame, _folds(frame))
    predictions = result["predictions"]
    assert predictions["p_up10_A"].notna().all()
    assert predictions[["p_up10_B", "p_up10_C", "p_up10_C_raw"]].isna().all().all()
    assert predictions[["B_status", "C_status"]].eq("unavailable").all().all()
    for fold in result["fold_audit"][1:]:
        for slot in fold["slots"]:
            assert slot["B"]["status"] == slot["C"]["status"] == "unavailable"
            assert "insufficient" in slot["B"]["reason"]
            assert "insufficient" in slot["C"]["reason"]


def test_intercept_has_fixed_unit_slope_and_recovers_constant_base_rate():
    rows = _frame(days=6)
    raw = np.full(len(rows), .2)
    fitted = upper._fit_intercept(raw, rows)
    assert fitted["intercept"] == pytest.approx(logit(.5) - logit(.2), abs=1e-6)
    assert fitted["slope"] == 1.
    scores = np.array([0., .1, .3, .9, 1.])
    actual = upper._apply_intercept(scores, fitted)
    np.testing.assert_allclose(actual, expit(logit(np.clip(scores, 1e-6, 1 - 1e-6)) + fitted["intercept"]))
    assert (np.diff(actual) > 0).all()


@pytest.mark.parametrize("result", [
    SimpleNamespace(success=False, x=0., fun=.5),
    SimpleNamespace(success=True, x=np.nan, fun=.5),
    SimpleNamespace(success=True, x=0., fun=np.inf),
    SimpleNamespace(success=True, x=-12., fun=.5),
    SimpleNamespace(success=True, x=11.99999, fun=.5),
])
def test_invalid_optimizer_result_is_unavailable(monkeypatch, result):
    monkeypatch.setattr(upper, "minimize_scalar", lambda *args, **kwargs: result)
    rows = _frame(days=6)
    with pytest.raises(upper._ArmUnavailable, match="optimizer"):
        upper._fit_intercept(rows["p_up10"].to_numpy(), rows)


def test_optimizer_failure_leaves_both_arms_missing(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(upper, "_fit_head", _fake_head)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(upper, "minimize_scalar", fail)
    result = upper.run_upper_comparison(frame, _folds(frame))
    assert result["predictions"][["p_up10_B", "p_up10_C", "p_up10_C_raw"]].isna().all().all()
    assert "synthetic failure" in result["fold_audit"][1]["slots"][0]["B"]["reason"]


def test_head_uses_only_24_raw_features_and_exact_parameters(monkeypatch):
    rows = _frame(days=6)
    captured = {}

    class Model:
        def __init__(self, **kwargs):
            captured["params"] = kwargs

        def fit(self, x, y):
            captured["x"], captured["y"] = x.copy(), y.copy()

        def predict_proba(self, x):
            captured["prediction_x"] = x.copy()
            return np.tile([.7, .3], (len(x), 1))

    monkeypatch.setattr(upper, "XGBClassifier", Model)
    result = upper._fit_head(rows.iloc[:120], rows.iloc[120:])
    assert captured["params"] == upper.EXPERIMENT_CONFIG["head"]
    assert captured["x"].shape == (120, 24)
    assert captured["prediction_x"].shape == (120, 24)
    assert np.isnan(captured["x"][:, -1]).all()
    np.testing.assert_equal(captured["x"], rows.iloc[:120][list(ALLOWED_FEATURES)].to_numpy(dtype=float))
    np.testing.assert_equal(captured["y"], rows.iloc[:120]["up10"].to_numpy(dtype=int))
    np.testing.assert_equal(result, np.full(120, .3))


def test_real_fixed_xgb_synthetic_fit_is_reproducible():
    rows = _frame(days=6)
    first = upper._fit_head(rows.iloc[:160], rows.iloc[160:])
    second = upper._fit_head(rows.iloc[:160], rows.iloc[160:])
    assert np.isfinite(first).all()
    assert ((first >= 0) & (first <= 1)).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("field,value", [
    ("p_up10", np.nan), ("p_dn5", 1.01), ("p_dn10", True), ("exp_downside", np.inf),
    ("rank", 0), ("up10", 1), ("was_delivered", 1), ("label_status", "partial"),
    ("date", "20260701"), ("decision_started_at", "2026-07-01T09:08:00"),
    ("label_available_at", "2026-07-01T00:00:00+00:00"),
    (ALLOWED_FEATURES[0], np.inf), (ALLOWED_FEATURES[0], True),
])
def test_invalid_frame_values_fail_before_any_fit(monkeypatch, field, value):
    frame = _frame()
    folds = _folds(frame)
    frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    monkeypatch.setattr(upper, "_fit_head", _no_fit)
    monkeypatch.setattr(upper, "minimize_scalar", _no_fit)
    with pytest.raises(upper.UpperExperimentError):
        upper.run_upper_comparison(frame, folds)


def test_missing_feature_duplicate_identity_and_halted_false_label_rejected():
    frame = _frame()
    with pytest.raises(upper.UpperExperimentError, match="missing columns"):
        upper._validate_frame(frame.drop(columns=ALLOWED_FEATURES[0]))
    with pytest.raises(upper.UpperExperimentError, match="duplicate candidate"):
        upper._validate_frame(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    frame.at[19, "label_status"] = "halted_no_observations"
    with pytest.raises(upper.UpperExperimentError, match="halted outcomes"):
        upper._validate_frame(frame)


def test_all_unready_folds_never_fit_and_return_empty(monkeypatch):
    frame = _frame(days=12)
    monkeypatch.setattr(upper, "_fit_head", _no_fit)
    monkeypatch.setattr(upper, "minimize_scalar", _no_fit)
    result = upper.run_upper_comparison(frame, _folds(frame))
    assert result["predictions"].empty
    assert result["fold_audit"][0]["status"] == "insufficient_train_dates"
