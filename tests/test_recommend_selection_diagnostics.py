from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

import signals.recommend_selection_diagnostics as diagnosis
from signals.recommend_experiment_eval import OUTCOMES


def _frame(days=6, count=40):
    rows = []
    for index in range(days):
        day = (date(2026, 8, 1) + timedelta(days=index)).isoformat()
        for slot in ("open", "preopen"):
            for rank in range(1, count + 1):
                up = (rank + index) % 5 == 0
                down = (rank + index) % 7 == 0
                probability = .8 - rank / 100
                rows.append({"date": day, "slot": slot, "coin": f"KRW-T{rank}",
                             "snapshot_id": f"fixture-{day}-{slot}", "rank": rank,
                             "was_delivered": rank <= 3, "p_up10": probability, "p_dn5": .2,
                             "p_dn10": .03, "exp_downside": -.04, "rr_ratio": probability / .2,
                             "f_log_qv": float(rank), "f_atr_pct_14": rank / 1000,
                             "label_status": "labeled", "up10": up, "dn5": down,
                             "mfe": .12 if up else .04, "mae": -.06 if down else -.02,
                             "eod_return_net": rank / 1000 - .01 + index / 1000,
                             "tp5_sl3_return_net": -.0315 if down else .0485,
                             "unused_future_hint": 999.})
    return pd.DataFrame(rows)


def _run(frame, **kwargs):
    return diagnosis.analyze_selection(frame, n_boot=50, **kwargs)


def test_pure_deterministic_contract_and_input_shuffle_equivalence():
    frame = _frame()
    before = frame.copy(deep=True)
    first = _run(frame)
    second = _run(frame.sample(frac=1, random_state=79).set_axis(np.arange(len(frame)) * 2))
    pd.testing.assert_frame_equal(frame, before)
    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["input"] == {"candidate_rows": 480, "date_slots": 12, "dates": 6, "delivered_rows": 36}
    assert first["coverage"] == {"included_date_slots": 12, "excluded_date_slots": 0}
    assert first["model_fitted"] is first["deployable"] is first["automatic_adoption"] is False
    assert first["parity"]["top3_order_passed"] == 12
    assert first["config"] == diagnosis.SELECTION_CONFIG
    first["config"]["ranking"]["top_k"] = 99
    assert diagnosis.SELECTION_CONFIG["ranking"]["top_k"] == 3


def test_plans_matching_and_geometry_exist_before_any_outcome_access(monkeypatch):
    frame = _frame()
    stages = []
    original_plan, original_outcomes = diagnosis._build_plans, diagnosis._validate_outcomes

    def plans(scores):
        assert set(scores.columns) == set(diagnosis.PREDICTORS)
        assert not set(OUTCOMES) & set(scores.columns)
        assert "label_status" not in scores
        result = original_plan(scores)
        assert all(plan["matched"] and plan["geometry"]["candidates"] for plan in result)
        stages.append("plans_complete")
        return result

    def outcomes(data):
        assert stages == ["plans_complete"]
        stages.append("outcomes")
        original_outcomes(data)

    monkeypatch.setattr(diagnosis, "_build_plans", plans)
    monkeypatch.setattr(diagnosis, "_validate_outcomes", outcomes)
    _run(frame)
    assert stages == ["plans_complete", "outcomes"]


def test_changing_outcomes_never_changes_structural_plans_or_membership():
    frame = _frame()
    original = _run(frame)
    frame["up10"] = ~frame["up10"]
    frame["mfe"] = np.where(frame["up10"], .12, .04)
    frame["eod_return_net"] += .2
    changed = _run(frame)
    assert original["selection_audit"] == changed["selection_audit"]
    for slot in ("open", "preopen"):
        assert original["slots"][slot]["geometry"] == changed["slots"][slot]["geometry"]
        assert original["slots"][slot]["cohorts"] == changed["slots"][slot]["cohorts"]
        assert original["slots"][slot]["absolute"] != changed["slots"][slot]["absolute"]


def test_actual_full_matched_metrics_and_cohort_denominators_are_exact():
    frame = _frame()
    report = _run(frame)
    opened = report["slots"]["open"]
    expected = frame.loc[frame["slot"].eq("open") & frame["rank"].le(3), "eod_return_net"].mean()
    assert opened["absolute"]["actual_top3"]["mean"]["eod_return_net"] == pytest.approx(expected)
    assert opened["cohorts"]["actual_top3"]["denominators"]["candidate_memberships"] == 18
    assert opened["cohorts"]["full_universe"]["denominators"]["candidate_memberships"] == 240
    matched = opened["cohorts"]["matched_expectation"]["denominators"]
    assert matched["candidate_memberships"] == 180
    assert matched["distinct_candidate_rows"] == 60
    assert len({cohort["date_cohort_sha256"] for cohort in opened["cohorts"].values()}) == 1
    assert len({cohort["cohort_sha256"] for cohort in opened["cohorts"].values()}) == 3
    assert len({cohort["cohort_sha256"] for cohort in report["slots"]["preopen"]["cohorts"].values()}) == 3
    assert opened["cohorts"]["actual_top3"]["cohort_sha256"] != report["slots"]["preopen"]["cohorts"]["actual_top3"]["cohort_sha256"]
    for day in report["daily"]:
        expected_matched = frame.loc[frame["date"].eq(day["date"]) & frame["slot"].eq(day["slot"]) & frame["rank"].le(10), "eod_return_net"].mean()
        assert day["matched_expectation"]["eod_return_net"] == pytest.approx(expected_matched)
    assert "three-arm" in report["methodology"]["cohort"]


def test_daily_equal_weight_not_candidate_pooled_mean_and_lodo():
    frame = _frame(days=2)
    frame = frame.loc[~(frame["date"].eq("2026-08-02") & frame["rank"].gt(20))].copy()
    frame.loc[frame["date"].eq("2026-08-02"), "eod_return_net"] += .1
    report = _run(frame)
    for slot in ("open", "preopen"):
        data = frame.loc[frame["slot"].eq(slot)]
        expected = data.groupby("date")["eod_return_net"].mean().mean()
        actual = report["slots"][slot]["absolute"]["full_universe"]["mean"]["eod_return_net"]
        assert actual == pytest.approx(expected)
        assert actual != pytest.approx(data["eod_return_net"].mean())
        deltas = report["slots"][slot]["deltas"]["top3_minus_full_universe"]
        daily = [row for row in report["daily"] if row["slot"] == slot]
        assert deltas["leave_one_date_out"]["rows"][0]["mean_delta"]["eod_return_net"] == pytest.approx(
            daily[1]["actual_top3"]["eod_return_net"] - daily[1]["full_universe"]["eod_return_net"])
        assert deltas["ci_status"] == "insufficient_dates"


def test_rank_bands_are_fixed_descriptive_disjoint_and_cover_all_rows():
    report = _run(_frame())
    for slot in report["slots"].values():
        bands = slot["rank_bands"]
        assert list(bands) == ["1-3", "4-10", "11-30", "31+"]
        assert [row["candidate_rows"] for row in bands.values()] == [18, 42, 120, 60]
        assert all(row["descriptive_only"] and row["dates"] == 6 for row in bands.values())
        assert bands["1-3"]["day_equal_mean"] == slot["absolute"]["actual_top3"]["mean"]


def test_halted_candidate_is_not_preremoved_and_excludes_full_common_date_slot():
    frame = _frame()
    before = _run(frame)
    index = frame.index[39]
    frame.at[index, "label_status"] = "halted_no_observations"
    for field in OUTCOMES:
        frame[field] = frame[field].astype(object)
        frame.at[index, field] = np.nan
    report = _run(frame)
    assert len(report["selection_audit"]) == 12
    assert report["selection_audit"][0]["geometry"] == before["selection_audit"][0]["geometry"]
    assert report["excluded"][0]["reasons"] == [{"kind": "full_universe_label_unavailable", "coins": ["KRW-T40"]}]
    assert report["slots"]["open"]["structural_cohort"]["denominators"]["dates"] == 6
    assert report["slots"]["open"]["cohorts"]["actual_top3"]["denominators"]["dates"] == 5
    assert report["slots"]["preopen"]["cohorts"]["actual_top3"]["denominators"]["dates"] == 6
    assert report["slots"]["open"]["excluded"][0]["date"] == "2026-08-01"


def test_unsupported_matching_is_not_widened_or_silently_skipped():
    report = _run(_frame(days=2, count=8))
    assert report["status"] == "no_common_evaluable_dates"
    assert len(report["excluded"]) == 4
    assert all(reason["kind"] == "matched_cell_unsupported" and reason["pool_size"] == 2
               for entry in report["excluded"] for reason in entry["reasons"])
    assert report["slots"]["open"]["cohorts"]["actual_top3"]["denominators"]["dates"] == 0
    json.dumps(report, allow_nan=False)


def test_one_date_three_candidates_no_boundary_empty_bands_no_ci():
    frame = _frame(days=1, count=3)
    frame["f_log_qv"] = frame["f_atr_pct_14"] = 1.
    report = _run(frame)
    assert report["selection_audit"][0]["geometry"]["boundary"] == {"status": "no_unselected_candidate"}
    slot = report["slots"]["open"]
    assert slot["absolute"]["actual_top3"]["ci_status"] == "insufficient_dates"
    assert slot["deltas"]["top3_minus_matched_expectation"]["leave_one_date_out"]["status"] == "insufficient_dates"
    assert slot["rank_bands"]["4-10"]["status"] == "empty_band"
    assert slot["rank_bands"]["4-10"]["day_equal_mean"] is None
    json.dumps(report, allow_nan=False)


def test_log_decomposition_is_additive_and_zero_up_disables_entire_snapshot():
    frame = _frame()
    positive = _run(frame)
    for snapshot in positive["selection_audit"]:
        geometry = snapshot["geometry"]
        assert geometry["log_decomposition"]["status"] == "available"
        assert geometry["log_decomposition"]["max_additivity_residual"] < 1e-12
        for row in geometry["candidates"]:
            assert row["up_log_contribution"] + row["low_down_log_contribution"] == pytest.approx(row["centered_log_ratio"], abs=1e-12)
        assert sum(row["centered_log_ratio"] for row in geometry["candidates"]) == pytest.approx(0., abs=1e-12)
    frame.at[39, "p_up10"] = 0.
    report = _run(frame)
    geometry = report["selection_audit"][0]["geometry"]
    assert geometry["log_decomposition"]["status"] == "unavailable_zero_up_probability"
    assert all(row["centered_log_ratio"] is None for row in geometry["candidates"])
    assert all(row["up_percentile"] > 0 for row in geometry["candidates"])
    assert report["slots"]["open"]["geometry"]["log_cohort"]["denominators"]["dates"] == 5
    assert report["slots"]["open"]["cohorts"]["actual_top3"]["denominators"]["dates"] == 6


def test_zero_denominator_uses_only_approved_floor():
    frame = _frame(days=1)
    frame["p_dn5"] = 0.
    report = _run(frame)
    geometry = report["selection_audit"][0]["geometry"]
    assert geometry["downside_floor_rows"] == 40
    assert geometry["ties"]["effective_p_dn5"]["largest_tie"] == 40
    for row in geometry["candidates"]:
        assert row["effective_p_dn5"] == .001
        assert row["computed_ratio"] == pytest.approx(row["p_up10"] / .001)


def test_all_scores_tied_has_deterministic_original_rank_boundary():
    frame = _frame(days=1)
    frame["p_up10"] = .5
    frame["rr_ratio"] = 2.5
    frame["f_log_qv"] = frame["f_atr_pct_14"] = 1.
    report = _run(frame)
    geometry = report["selection_audit"][0]["geometry"]
    assert geometry["ties"]["computed_ratio"]["tied_rows"] == 40
    assert geometry["boundary"]["decisive_key"] == "original_rank"
    assert geometry["boundary"]["ratio_margin"] == 0
    assert geometry["boundary"]["nearest_unselected_coin"] == "KRW-T4"
    assert report["selection_audit"][0]["matched_pools"][0]["size"] == 40


def test_lower_rank_rounding_difference_reports_true_nearest_unselected():
    frame = _frame(days=1)
    frame.loc[[3, 4], "p_up10"] = frame.loc[[4, 3], "p_up10"].to_numpy()
    report = _run(frame)
    assert report["parity"]["full_order_mismatches"] == 1
    boundary = report["selection_audit"][0]["geometry"]["boundary"]
    assert boundary["original_rank4_coin"] == "KRW-T4"
    assert boundary["nearest_unselected_coin"] == "KRW-T5"


def test_top3_parity_failure_precedes_outcome_validation(monkeypatch):
    frame = _frame()
    frame.at[10, "p_up10"] = .99
    monkeypatch.setattr(diagnosis, "_validate_outcomes", lambda *args: pytest.fail("outcomes touched before parity"))
    with pytest.raises(diagnosis.SelectionDiagnosticsError, match="parity"):
        _run(frame)


@pytest.mark.parametrize("field,value", [("p_up10", np.nan), ("p_dn5", True), ("p_dn10", 1.01),
    ("rank", 0), ("rank", True), ("exp_downside", np.inf), ("f_atr_pct_14", np.nan),
    ("was_delivered", 1), ("slot", "other"), ("date", "20260801")])
def test_invalid_predictors_fail_closed(field, value):
    frame = _frame(days=1)
    frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    with pytest.raises(diagnosis.SelectionDiagnosticsError):
        _run(frame)


@pytest.mark.parametrize("field,value", [("up10", True), ("dn5", True), ("eod_return_net", True),
    ("mae", .01), ("mfe", np.nan), ("label_status", "partial")])
def test_inconsistent_or_invalid_outcome_blocks_complete_report(field, value):
    frame = _frame(days=1)
    frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    with pytest.raises(diagnosis.SelectionDiagnosticsError):
        _run(frame)


@pytest.mark.parametrize("kwargs", [{"n_boot": True}, {"n_boot": 0}, {"n_boot": 100001}, {"seed": True}, {"seed": -1}])
def test_invalid_bootstrap_arguments(kwargs):
    with pytest.raises(diagnosis.SelectionDiagnosticsError):
        diagnosis.analyze_selection(_frame(days=1), **kwargs)


def test_missing_duplicate_and_corrupt_halted_evidence_fail_closed():
    frame = _frame(days=1)
    with pytest.raises(diagnosis.SelectionDiagnosticsError, match="missing columns"):
        _run(frame.drop(columns="p_up10"))
    with pytest.raises(diagnosis.SelectionDiagnosticsError, match="duplicate candidate"):
        _run(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    frame.at[39, "label_status"] = "halted_no_observations"
    with pytest.raises(diagnosis.SelectionDiagnosticsError, match="halted row has outcomes"):
        _run(frame)


def test_config_is_json_serializable_and_has_no_fit_or_sweep():
    config = deepcopy(diagnosis.SELECTION_CONFIG)
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert not any(config[key] for key in ("model_fitted", "new_policy_evaluated", "threshold_sweep", "live_changes"))
