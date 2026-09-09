from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from signals.recommend_experiment_eval import ExperimentEvaluationError, evaluate_comparison


def _predictions(days=6, slots=("open",), coins=6):
    rows = []
    for d in range(days):
        day = (date(2026, 9, 1) + timedelta(days=d)).isoformat()
        for slot in slots:
            for c in range(coins):
                probability = 0.6 - c * 0.04
                up, down = c % 2 == 0, c % 3 == 0
                rows.append({"date": day, "slot": slot, "coin": f"KRW-C{c}", "rank": c + 1,
                             "fold": 2 if d < 3 else 3, "snapshot_id": f"s-{day}-{slot}",
                             "was_delivered": c < 3, "label_status": "labeled", "p_up10": probability,
                             "p_dn5": 0.1, "p_dn10": 0.02, "exp_downside": -0.03,
                             "rr_ratio": probability / 0.1, "p_up10_A": probability,
                             "p_up10_B": 0.95 if c == coins - 1 else probability,
                             "p_up10_C": 0.96 if c == coins - 2 else probability,
                             "f_log_qv": 10.0, "f_atr_pct_14": 0.02,
                             "up10": float(up), "dn5": float(down), "mfe": 0.15 if up else 0.03,
                             "mae": -0.06 if down else -0.01,
                             "eod_return_net": (c - 2) * 0.01 + d * 0.001 - 0.0015,
                             "tp5_sl3_return_net": 0.0485 if up else -0.0315})
    return pd.DataFrame(rows)


def test_common_cohort_absolute_cost_once_and_expected_matching():
    frame = _predictions(slots=("open", "preopen"))
    original = frame.copy(deep=True)
    report = evaluate_comparison(frame, n_boot=50)
    pd.testing.assert_frame_equal(frame, original)
    assert report["deployable"] is report["automatic_adoption"] is False
    assert report["coverage"] == {"included_date_slots": 12, "excluded_date_slots": 0}
    slot = report["slots"]["open"]
    assert slot["dates"] == 6 and slot["picks_per_arm"] == 18
    actual = frame[(frame.slot == "open") & frame.was_delivered]
    assert slot["absolute"]["A"]["mean"]["eod_return_net"] == pytest.approx(actual.eod_return_net.mean())
    assert slot["absolute"]["A"]["mean"]["tp5_sl3_return_net"] == pytest.approx(actual.tp5_sl3_return_net.mean())
    assert slot["matched"]["A"] == slot["full_universe"]
    assert slot["absolute"]["A"]["mean"]["whole_path_safe_up10"] == pytest.approx(1 / 3)
    assert slot["probability_diagnostics"]["A"]["all_candidates"]["up10"]["rows"] == 36
    assert slot["probability_diagnostics"]["A"]["selected_top3"]["up10"]["rows"] == 18
    assert len(slot["folds"]) == 2
    json.dumps(report, allow_nan=False)


def test_selection_is_independent_of_outcomes_and_halted_rows_are_not_removed():
    frame = _predictions()
    before = evaluate_comparison(frame, n_boot=10)
    row = (frame.date == "2026-09-01") & (frame.coin == "KRW-C5")
    frame.loc[row, "label_status"] = "halted_no_observations"
    frame.loc[row, ["up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"]] = np.nan
    report = evaluate_comparison(frame, n_boot=10)
    assert report["coverage"] == {"included_date_slots": 5, "excluded_date_slots": 1}
    assert before["selection_audit"][0]["selected_coins"] == report["selection_audit"][0]["selected_coins"]
    assert "KRW-C5" in report["selection_audit"][0]["selected_coins"]["B"]
    assert any("full_universe_label_unavailable" in r for r in report["excluded"][0]["reasons"])
    assert any("selected_label_unavailable:B" in r for r in report["excluded"][0]["reasons"])
    assert all(s["dates"] == 5 for s in report["slots"].values())


def test_unsupported_joint_cell_never_widens_after_outcomes():
    frame = _predictions(coins=8)
    frame["f_log_qv"] = frame["rank"].astype(float)
    frame["f_atr_pct_14"] = frame["rank"] / 100
    report = evaluate_comparison(frame, n_boot=10)
    assert report["status"] == "no_common_evaluable_dates"
    assert report["coverage"]["excluded_date_slots"] == 6
    assert all("matched_cell_unsupported" in reason for item in report["excluded"] for reason in item["reasons"])
    assert report["selection_audit"][0]["matched_pool_sizes"]["A"] == [2, 2, 2]


def test_date_paired_intervals_and_contributions_are_exact_for_constant_delta():
    report = evaluate_comparison(_predictions(), n_boot=100)
    slot = report["slots"]["open"]
    diff = slot["delta_vs_A"]["B"]
    expected = (0.05 - 0.02) / 3
    assert diff["mean"]["eod_return_net"] == pytest.approx(expected)
    for method in ("iid_date_ci95", "observed_date_block3_ci95"):
        assert diff[method]["eod_return_net"] == pytest.approx([expected, expected])
    sensitivity = slot["sensitivity_vs_A"]["B"]
    assert len(sensitivity["leave_one_date_out"]) == 6
    assert all(item["mean_delta"]["eod_return_net"] == pytest.approx(expected) for item in sensitivity["leave_one_date_out"])
    assert sensitivity["max_absolute_coin_contribution"]["eod_return_net"]["sum_all_coin_contributions"] == pytest.approx(expected)


def test_probability_diagnostics_match_manual_brier_and_single_class_auc():
    frame = _predictions()
    report = evaluate_comparison(frame, n_boot=10)
    metric = report["slots"]["open"]["probability_diagnostics"]["A"]["all_candidates"]["up10"]
    assert metric["brier"] == pytest.approx(np.mean((frame.p_up10 - frame.up10) ** 2))
    frame["up10"], frame["mfe"] = 0.0, 0.03
    second = evaluate_comparison(frame, n_boot=10)
    metric = second["slots"]["open"]["probability_diagnostics"]["C"]["all_candidates"]["up10"]
    assert metric["auc"] is None and metric["auc_status"] == "single_class"


def test_fewer_than_five_dates_has_point_estimates_but_no_ci():
    report = evaluate_comparison(_predictions(days=3), n_boot=10)
    summary = report["slots"]["open"]["delta_vs_A"]["C"]
    assert summary["ci_status"] == "insufficient_dates"
    assert summary["iid_date_ci95"] is summary["observed_date_block3_ci95"] is None


@pytest.mark.parametrize("field,value", [("p_up10_C", np.nan), ("p_up10_B", np.inf),
                                        ("p_dn5", -0.1), ("f_log_qv", None),
                                        ("was_delivered", 1), ("up10", 2.0),
                                        ("mfe", np.inf), ("label_status", "missing"),
                                        ("eod_return_net", np.nan)])
def test_invalid_values_fail_the_whole_report(field, value):
    frame = _predictions()
    if field == "was_delivered":
        frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    with pytest.raises(ExperimentEvaluationError):
        evaluate_comparison(frame, n_boot=10)


def test_duplicate_identity_cross_slot_fold_and_snapshot_reuse_are_blocked():
    frame = _predictions(slots=("open", "preopen"))
    with pytest.raises(ExperimentEvaluationError, match="duplicate"):
        evaluate_comparison(pd.concat([frame, frame.iloc[:1]]), n_boot=10)
    bad = frame.copy()
    bad.loc[(bad.date == "2026-09-01") & (bad.slot == "preopen"), "fold"] = 8
    with pytest.raises(ExperimentEvaluationError, match="cross-slot"):
        evaluate_comparison(bad, n_boot=10)
    bad = frame.copy()
    bad["snapshot_id"] = "same"
    with pytest.raises(ExperimentEvaluationError, match="snapshot reused"):
        evaluate_comparison(bad, n_boot=10)


def test_identity_transform_mismatch_is_fatal_not_an_excluded_day():
    frame = _predictions()
    frame.loc[5, ["p_up10", "p_up10_A"]] = 0.99
    with pytest.raises(ExperimentEvaluationError, match="identity Top3 parity"):
        evaluate_comparison(frame, n_boot=10)


def test_declared_unavailable_arm_excludes_whole_fold_slot_without_changing_A():
    frame = _predictions(slots=("open", "preopen"))
    frame["B_status"], frame["C_status"] = "ok", "ok"
    unavailable = frame.fold.eq(2) & frame.slot.eq("open")
    frame.loc[unavailable, "B_status"] = "unavailable"
    frame.loc[unavailable, "p_up10_B"] = np.nan
    report = evaluate_comparison(frame, n_boot=10)
    assert report["coverage"] == {"included_date_slots": 9, "excluded_date_slots": 3}
    assert report["input"]["date_slots"] == 12
    assert report["slots"]["open"]["dates"] == 3
    assert report["slots"]["preopen"]["dates"] == 6
    assert all(item["reasons"] == ["arm_unavailable:B"] for item in report["excluded"])
    for item in report["selection_audit"]:
        if item["fold"] == 2 and item["slot"] == "open":
            assert item["selected_coins"]["A"] == ["KRW-C0", "KRW-C1", "KRW-C2"]
            assert item["selected_coins"]["B"] is item["matched_pool_sizes"]["B"] is None
    bad = frame.copy()
    bad.loc[5, ["p_up10", "p_up10_A"]] = 0.99
    with pytest.raises(ExperimentEvaluationError, match="identity Top3 parity"):
        evaluate_comparison(bad, n_boot=10)


@pytest.mark.parametrize("mutation", ["finite_unavailable", "inf_unavailable", "mixed_status", "nan_ok"])
def test_arm_availability_is_explicit_and_internally_consistent(mutation):
    frame = _predictions()
    frame["B_status"] = "unavailable"
    frame["p_up10_B"] = np.nan
    if mutation == "finite_unavailable":
        frame.loc[0, "p_up10_B"] = 0.3
    elif mutation == "inf_unavailable":
        frame.loc[0, "p_up10_B"] = np.inf
    elif mutation == "mixed_status":
        frame.loc[0, "B_status"] = "ok"
        frame.loc[0, "p_up10_B"] = 0.3
    else:
        frame["B_status"] = "ok"
    with pytest.raises(ExperimentEvaluationError):
        evaluate_comparison(frame, n_boot=10)


def test_all_declared_unavailable_is_completed_but_insufficient():
    frame = _predictions()
    frame["C_status"], frame["p_up10_C"] = "unavailable", np.nan
    report = evaluate_comparison(frame, n_boot=10)
    assert report["status"] == "no_common_evaluable_dates"
    assert report["execution_status"] == "completed"
    assert report["data_availability"] == "insufficient_no_common_dates"
    assert report["coverage"]["excluded_date_slots"] == 6


def test_reproducible_and_input_row_order_independent():
    frame = _predictions(slots=("open", "preopen"))
    first = evaluate_comparison(frame, n_boot=20)
    second = evaluate_comparison(frame.sample(frac=1, random_state=21), n_boot=20)
    assert first == second
    changed = deepcopy(first)
    changed["deployable"] = True
    assert first["deployable"] is False


@pytest.mark.parametrize("arguments", [{"n_boot": 0}, {"n_boot": True}, {"seed": -1}, {"seed": True}])
def test_invalid_bootstrap_arguments(arguments):
    with pytest.raises(ExperimentEvaluationError):
        evaluate_comparison(_predictions(), **arguments)
