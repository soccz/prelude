"""Future outcomes, incomplete evidence and code changes cannot pick a policy."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from signals import recommend_regime_replay as replay


def _snapshot():
    return {
        "slot": "open", "model": {"id": "recommend_r1_open"},
        "code": {"score_source_sha256": "a" * 64}, "rule_version": "v1",
        "score_schema_version": 1, "btc_regime": "bull_volatile",
        "universe": [{"coin": f"C{i}", "feature_values": {
            "f_ret_1d": 0.1 if i < 2 else -0.1, "f_qv_ma7_vs_ma30": 1,
            "f_atr_pct_14": 0.03, "f_pos_in_20d_range": 0.7,
        }} for i in range(4)],
    }


def _metric(net=0.01, dn=1 / 3, up=1 / 3, mae=-0.04):
    return dict(up10=up, dn5=dn, whole_path_safe_up10=up,
                tp5_sl3_return_net=net, eod_return_net=net, mae=mae)


def _rows(n=16):
    records = []
    for i in range(n):
        decision = datetime(2026, 9, 1, 0, 10, tzinfo=timezone.utc) + timedelta(days=i)
        entry = decision + timedelta(minutes=5)
        records.append({
            "date": decision.date().isoformat(), "version": replay.version_from_snapshot(_snapshot()),
            "context": replay.context_from_snapshot(_snapshot()),
            "decision_at": decision.isoformat(), "entry_at": entry.isoformat(),
            "predictors_available_at": [decision.isoformat()],
            "label_available_at": (entry + timedelta(days=1, minutes=45)).isoformat(),
            "outcomes": {"control": _metric(), "challenger": _metric(net=0.02)},
            "changed_picks": 2,
        })
    return records


def test_context_thresholds_are_fixed_and_do_not_depend_on_outcomes():
    snapshot = _snapshot()
    assert replay.context_from_snapshot(snapshot)["state"] == "broad:expanding"
    snapshot["universe"][0]["feature_values"]["f_ret_1d"] = -1
    for row in snapshot["universe"]:
        row["feature_values"]["f_qv_ma7_vs_ma30"] = 0.9
    context = replay.context_from_snapshot(snapshot)
    assert context["state"] == "narrow:contracting"
    assert context["values"]["atr"] == 0.03


def test_missing_features_do_not_silently_change_context_universe():
    snapshot = _snapshot()
    snapshot["universe"][0]["feature_values"]["f_ret_1d"] = None
    context = replay.context_from_snapshot(snapshot)
    assert context["state"] is None
    assert context["missing"]["breadth"] == 1


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.2"])
def test_invalid_context_is_not_missing(bad):
    snapshot = _snapshot()
    snapshot["universe"][0]["feature_values"]["f_ret_1d"] = bad
    with pytest.raises(ValueError, match="invalid context"):
        replay.context_from_snapshot(snapshot)


def test_purge_t_minus_one_minimum_history_and_window_cap():
    plans = replay.selection_plans(_rows())
    assert all(plan["choices"]["recent"]["arm"] == "control" for plan in plans[:6])
    selected = plans[6]["choices"]["recent"]
    assert selected["arm"] == "challenger"
    assert selected["history_dates"] == [f"2026-09-{i:02d}" for i in range(1, 6)]
    assert len(plans[-1]["choices"]["recent"]["history_dates"]) == 10


def test_equal_time_and_delayed_labels_are_unavailable():
    rows = _rows(7)
    rows[0]["label_available_at"] = rows[6]["decision_at"]
    rows[1]["label_available_at"] = "2026-10-01T00:00:00+00:00"
    plan = replay.selection_plans(rows)[-1]["choices"]["recent"]
    assert len(plan["history_dates"]) == 3
    assert plan["arm"] == "control"


@pytest.mark.parametrize("field,value", [("eod_return_net", 0.01), ("dn5", 0.5), ("up10", 0), ("mae", -0.05)])
def test_net_downside_upside_and_mae_all_guard_switch(field, value):
    rows = _rows()
    for row in rows:
        row["outcomes"]["challenger"][field] = value
    result = replay.replay(rows, n_boot=10)
    assert result["policies"]["recent"]["challenger_dates"] == 0
    assert result["policies"]["same_context"]["challenger_dates"] == 0


def test_current_and_future_outcomes_cannot_change_earlier_plan():
    rows = _rows()
    reference = replay.selection_plans(rows)
    for row in rows[7:]:
        row["outcomes"] = {"control": _metric(net=-0.99), "challenger": _metric(net=10)}
    changed = replay.selection_plans(rows)
    assert changed[:8] == reference[:8]
    rows[7]["outcomes"] = None
    assert replay.selection_plans(rows)[7] == reference[7]


def test_appending_extreme_future_context_does_not_refit_thresholds():
    rows = _rows()
    expected = replay.selection_plans(rows[:8])
    for row in rows[8:]:
        row["context"]["state"] = "narrow:contracting"
        row["context"]["values"]["atr"] = 999
    assert replay.selection_plans(rows)[:8] == expected


def test_context_history_is_different_from_recent_history():
    rows = _rows(8)
    rows[-1]["context"]["state"] = "narrow:contracting"
    choices = replay.selection_plans(rows)[-1]["choices"]
    assert choices["recent"]["arm"] == "challenger"
    assert choices["same_context"]["reason"] == "insufficient_completed_history"


@pytest.mark.parametrize("version", [None, "different_model_slot_source_or_rule"])
def test_no_cross_version_training(version):
    rows = _rows(8)
    rows[-1]["version"] = version
    choices = replay.selection_plans(rows)[-1]["choices"]
    assert choices["recent"]["arm"] == "control"
    assert choices["recent"]["history_dates"] == []


def test_source_identity_contains_slot_rule_and_code():
    snapshot = _snapshot()
    reference = replay.version_from_snapshot(snapshot)
    for section, key in [(snapshot, "slot"), (snapshot, "rule_version"),
                         (snapshot["code"], "score_source_sha256")]:
        old = section[key]
        section[key] = "b" * 64
        assert replay.version_from_snapshot(snapshot) != reference
        section[key] = old
    snapshot["code"] = {}
    assert replay.version_from_snapshot(snapshot) is None


@pytest.mark.parametrize("fault", ["late_score", "future_predictor", "naive", "duplicate"])
def test_invalid_clock_and_duplicate_dates_fail(fault):
    rows = _rows(2)
    if fault == "late_score":
        rows[0]["entry_at"] = rows[0]["decision_at"]
    elif fault == "future_predictor":
        rows[0]["predictors_available_at"] = [rows[0]["entry_at"]]
    elif fault == "naive":
        rows[0]["decision_at"] = "2026-09-01T00:10:00"
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        replay.replay(rows, n_boot=10)


def test_replay_preserves_fixed_r1_cost_and_does_not_count_pending_as_zero():
    rows = _rows(8)
    original = deepcopy(rows)
    rows[-1]["outcomes"] = None
    result = replay.replay(rows, n_boot=10)
    fixed = result["policies"]["fixed_r1"]
    assert fixed["metrics"]["mean"]["eod_return_net"] == 0.01
    assert fixed["n_dates"] == 7
    assert result["pending_dates"] == ["2026-09-08"]
    assert fixed["minus_fixed_r1"]["mean"]["eod_return_net"] == 0
    assert result["policies"]["recent"]["changed_picks_evaluated"] == 2
    assert result["prospective_policy_records"] == 0
    assert not result["is_untouched_holdout"] and not result["deployable"]
    assert rows[:-1] == original[:-1]


def test_empty_replay_is_explicit_not_zero_performance():
    result = replay.replay([], n_boot=10)
    assert result["policies"]["recent"]["metrics"] is None
    assert result["plans"] == []


def test_premature_outcome_and_missing_predictor_clock_fail():
    rows = _rows(1)
    rows[0]["label_available_at"] = rows[0]["entry_at"]
    with pytest.raises(ValueError, match="path ends"):
        replay.selection_plans(rows)
    rows = _rows(1)
    rows[0]["predictors_available_at"] = []
    with pytest.raises(ValueError, match="predictor"):
        replay.selection_plans(rows)


def test_unknown_context_falls_back_for_both_policies():
    rows = _rows()
    rows[-1]["context"]["state"] = None
    choices = replay.selection_plans(rows)[-1]["choices"]
    assert choices["recent"]["reason"] == choices["same_context"]["reason"] == "unknown_context"


def test_failure_summary_separates_slot_version_and_context():
    rows = _rows(4)
    for row in rows:
        row["slot"] = "open"
        row["outcomes"]["universe"] = _metric(net=0)
    rows[1]["slot"] = "preopen"
    rows[2]["version"] = "different"
    rows[3]["context"]["state"] = "narrow:contracting"
    summary = replay.failure_summary(rows, n_boot=10)
    assert len(summary["groups"]) == 4
    assert all(group["n_dates"] == 1 for group in summary["groups"])
    assert all(group["r1_minus_universe"]["mean"]["eod_return_net"] == 0.01 for group in summary["groups"])
