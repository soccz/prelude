"""Synthetic, hermetic tests of all-head selection and two fixed cohorts."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from signals.recommend_horizon_eval import (
    EVALUATION_CONFIG, HEADS, SCORES, HorizonEvaluationError, evaluate_horizon,
)


def predictions(days=6, coins=6):
    rows = []
    for d in range(days):
        day = (date(2026, 8, 1) + timedelta(days=d)).isoformat()
        for k in range(coins):
            up, down = k % 2 == 0, k % 3 == 0
            row = {"date": day, "slot": "preopen", "coin": f"KRW-C{k}", "rank": k + 1,
                   "snapshot_id": f"sid-{day}", "was_delivered": k < 3, "label_status": "labeled",
                   "p_up10": .6 - k * .04, "p_dn5": .1, "p_dn10": .02, "exp_downside": -.03,
                   "f_log_qv": 10., "f_atr_pct_14": .02,
                   "mfe": .15 if up else .03, "mae": -.06 if down else -.01,
                   "up5": up, "up10": up, "up20": False,
                   "dn5": down, "dn10": False,
                   "eod_return_net": (k - 2) * .01 + d * .001 - .0015,
                   "tp5_sl3_return_net": .0485 if up else -.0315,
                   "B_status": "fitted", "C_status": "fitted"}
            row["rr_ratio"] = row["p_up10"] / row["p_dn5"]
            for arm in ("B", "C"):
                values = {"p_up5": .7, "p_up10": row["p_up10"], "p_up20": .1,
                          "p_dn5": .1, "p_dn10": .02, "exp_downside": -.03}
                if (arm == "B" and k == coins - 1) or (arm == "C" and k == coins - 2):
                    values["p_up10"] = .95
                for name, value in values.items():
                    row[f"{name}_{arm}"] = value
                    if name.startswith("p_"):
                        row[f"{name}_{arm}_raw"] = value / 2
            rows.append(row)
    return pd.DataFrame(rows)


def halt(frame, *, day="2026-08-01", coin="KRW-C3"):
    mask = frame.date.eq(day) & frame.coin.eq(coin)
    frame.loc[mask, "label_status"] = "halted_no_observations"
    for column in (*HEADS, "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"):
        frame[column] = frame[column].astype(float)
        frame.loc[mask, column] = np.nan


def unavailable(frame, arm="B", day="2026-08-01"):
    mask = frame.date.eq(day)
    frame.loc[mask, f"{arm}_status"] = "unavailable"
    for name in SCORES:
        frame.loc[mask, f"{name}_{arm}"] = np.nan
        if name.startswith("p_"):
            frame.loc[mask, f"{name}_{arm}_raw"] = np.nan


def test_primary_and_supplement_are_distinct_fixed_cohorts_with_net_once():
    frame = predictions()
    before = frame.copy(deep=True)
    result = evaluate_horizon(frame, n_boot=30)
    pd.testing.assert_frame_equal(frame, before)
    assert result["configuration"] == EVALUATION_CONFIG
    assert not result["deployable"] and not result["automatic_promotion"]
    primary, supplement = result["primary"], result["supplemental"]
    assert primary["dates"] == supplement["dates"] == 6
    assert primary["picks_per_arm"] == 18
    assert primary["cohort_id"] != supplement["cohort_id"]
    actual = frame.loc[frame.was_delivered]
    assert primary["absolute"]["A"]["mean"]["eod_return_net"] == pytest.approx(actual.eod_return_net.mean())
    assert primary["absolute"]["A"]["mean"]["tp5_sl3_return_net"] == pytest.approx(actual.tp5_sl3_return_net.mean())
    assert primary["matched"]["A"] == primary["full_universe"]
    assert "matched" not in supplement
    assert len(primary["chronological_groups"]) == 2
    assert [g["dates"] for g in primary["chronological_groups"]] == [5, 1]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,new_value", [("p_dn5_B", .001), ("p_dn10_B", .001), ("exp_downside_B", -.001)])
def test_arm_specific_downside_tie_breaks_change_only_that_arm(field, new_value):
    frame = predictions(days=1)
    for arm in ("B", "C"):
        frame[f"p_up10_{arm}"] = .5
    before = evaluate_horizon(frame, n_boot=10)["selection_audit"]
    frame.at[5, field] = new_value
    after = evaluate_horizon(frame, n_boot=10)["selection_audit"]
    assert after[0]["selected_coins"]["B"] == ["KRW-C5", "KRW-C0", "KRW-C1"]
    assert before[0]["selected_coins"]["A"] == after[0]["selected_coins"]["A"]
    assert before[0]["selected_coins"]["C"] == after[0]["selected_coins"]["C"]


def test_outcome_changes_never_change_selection_or_matching():
    frame = predictions()
    before = evaluate_horizon(frame, n_boot=10)["selection_audit"]
    frame["up5"], frame["up10"], frame["up20"] = False, False, False
    frame["dn5"], frame["dn10"] = False, False
    frame["mfe"], frame["mae"] = .01, -.01
    frame["eod_return_net"], frame["tp5_sl3_return_net"] = -.0015, -.0015
    halt(frame)
    after = evaluate_horizon(frame, n_boot=10)["selection_audit"]
    assert before == after


def test_nonselected_halt_only_removes_primary_date_not_supplement():
    frame = predictions()
    halt(frame, coin="KRW-C3")
    result = evaluate_horizon(frame, n_boot=10)
    assert result["primary"]["dates"] == 5
    assert result["supplemental"]["dates"] == 6
    assert result["primary"]["excluded"][0]["reasons"] == ["full_universe_label_unavailable:KRW-C3"]
    prob = result["supplemental"]["probability_diagnostics"]["B"]["all_labeled_candidates"]
    assert (prob["candidate_rows"], prob["labeled_rows"], prob["unlabeled_rows"]) == (36, 35, 1)


def test_selected_halt_excludes_both_cohorts_without_removing_candidate():
    frame = predictions()
    halt(frame, coin="KRW-C5")
    result = evaluate_horizon(frame, n_boot=10)
    assert result["primary"]["dates"] == result["supplemental"]["dates"] == 5
    assert result["selection_audit"][0]["universe_rows"] == 6
    assert "KRW-C5" in result["selection_audit"][0]["selected_coins"]["B"]
    assert result["supplemental"]["excluded"][0]["reasons"] == ["selected_label_unavailable:B:KRW-C5"]


def test_small_matching_cells_only_remove_primary_without_widening():
    frame = predictions(coins=8)
    frame["f_log_qv"] = frame["rank"].astype(float)
    frame["f_atr_pct_14"] = frame["rank"] / 100
    result = evaluate_horizon(frame, n_boot=10)
    assert result["status"] == "no_primary_common_dates"
    assert result["data_availability"] == "insufficient_primary"
    assert result["primary"]["dates"] == 0 and result["supplemental"]["dates"] == 6
    assert result["selection_audit"][0]["matched_pool_sizes"]["A"] == [2, 2, 2]


def test_C_minus_B_paired_means_CI_LODO_and_coin_contributions():
    frame = predictions()
    result = evaluate_horizon(frame, n_boot=100)["primary"]
    expected = -.01 / 3
    contrast = result["contrasts"]["C_minus_B"]
    assert contrast["mean"]["eod_return_net"] == pytest.approx(expected)
    for name in ("iid_date_ci95", "observed_date_block3_ci95"):
        assert contrast[name]["eod_return_net"] == pytest.approx([expected, expected])
    sensitivity = result["sensitivity"]["C_minus_B"]
    assert len(sensitivity["leave_one_date_out"]) == 6
    assert all(row["mean_delta"]["eod_return_net"] == pytest.approx(expected) for row in sensitivity["leave_one_date_out"])
    assert sensitivity["sum_coin_contributions"]["eod_return_net"] == pytest.approx(expected)
    assert sum(row["eod_return_net"] for row in sensitivity["coin_contributions"].values()) == pytest.approx(expected)


def test_all_head_probability_diagnostics_use_arm_specific_predictions_and_raw():
    frame = predictions()
    frame["p_dn5_B"] = .4
    result = evaluate_horizon(frame, n_boot=10)["primary"]["probability_diagnostics"]
    diag = result["B"]["all_labeled_candidates"]
    assert set(diag["heads"]) == set(HEADS)
    assert diag["heads"]["dn5"]["brier"] == pytest.approx(np.mean((frame.p_dn5_B - frame.dn5.astype(float)) ** 2))
    assert diag["raw_heads"]["dn5"]["mean_prediction"] == pytest.approx(.05)
    assert result["A"]["all_labeled_candidates"]["heads"]["dn5"]["mean_prediction"] == pytest.approx(.1)
    assert diag["heads"]["up20"]["auc"] is None
    assert diag["heads"]["up20"]["auc_status"] == "single_class"
    assert result["C"]["selected_top3"]["heads"]["up10"]["rows"] == 18


def test_small_sample_has_point_estimate_without_CI():
    result = evaluate_horizon(predictions(days=3), n_boot=10)
    for name in ("primary", "supplemental"):
        summary = result[name]["contrasts"]["C_minus_B"]
        assert summary["ci_status"] == "insufficient_dates"
        assert summary["iid_date_ci95"] is summary["observed_date_block3_ci95"] is None


def test_single_day_has_no_LODO_and_identical_arms_have_zero_contributions():
    frame = predictions(days=1)
    for name in SCORES:
        frame[f"{name}_C"] = frame[f"{name}_B"]
    result = evaluate_horizon(frame, n_boot=10)["primary"]
    assert all(value == 0 for value in result["contrasts"]["C_minus_B"]["mean"].values())
    assert result["sensitivity"]["C_minus_B"]["leave_one_date_out"] == []


def test_input_order_does_not_change_any_report_field():
    frame = predictions()
    assert evaluate_horizon(frame, n_boot=15) == evaluate_horizon(frame.sample(frac=1, random_state=98), n_boot=15)


def test_declared_unavailable_day_has_no_selection_and_is_not_replaced_by_A():
    frame = predictions()
    unavailable(frame)
    result = evaluate_horizon(frame, n_boot=10)
    assert result["primary"]["dates"] == result["supplemental"]["dates"] == 5
    assert result["selection_audit"][0]["selected_coins"]["B"] is None
    assert result["supplemental"]["excluded"][0]["reasons"] == ["arm_unavailable:B"]


@pytest.mark.parametrize("kind", ["mixed_status", "finite_when_unavailable", "nan_when_ok", "infinite_raw", "partial_raw"])
def test_invalid_availability_or_partial_predictions_fail_closed(kind):
    frame = predictions()
    if kind == "mixed_status":
        frame.at[0, "B_status"] = "unavailable"
    elif kind == "finite_when_unavailable":
        unavailable(frame)
        frame.at[0, "p_dn10_B"] = .1
    elif kind == "nan_when_ok":
        frame.at[0, "p_dn5_C"] = np.nan
    elif kind == "infinite_raw":
        frame.at[0, "p_up5_B_raw"] = np.inf
    else:
        frame = frame.drop(columns="p_up5_C_raw")
    with pytest.raises(HorizonEvaluationError):
        evaluate_horizon(frame, n_boot=10)


def test_all_unavailable_is_completed_insufficient_and_A_parity_still_checked():
    frame = predictions(days=1)
    unavailable(frame)
    result = evaluate_horizon(frame, n_boot=10)
    assert result["execution_status"] == "completed"
    assert result["primary"]["dates"] == result["supplemental"]["dates"] == 0
    frame.at[5, "p_up10"] = .99
    with pytest.raises(HorizonEvaluationError, match="A identity"):
        evaluate_horizon(frame, n_boot=10)


@pytest.mark.parametrize("field,value", [("p_dn5_B", -.1), ("p_dn10_C", 1.1), ("exp_downside_B", .1),
                                        ("f_log_qv", np.nan), ("rank", 1.5), ("slot", "open"),
                                        ("up20", 1), ("dn10", 1), ("label_status", "missing")])
def test_invalid_predictors_outcomes_or_scope_block_report(field, value):
    frame = predictions()
    frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    with pytest.raises(ValueError):
        evaluate_horizon(frame, n_boot=10)


def test_duplicate_columns_rows_and_snapshot_reuse_fail():
    frame = predictions()
    with pytest.raises(ValueError):
        evaluate_horizon(pd.concat([frame, frame.iloc[:1]]), n_boot=10)
    with pytest.raises(ValueError):
        evaluate_horizon(pd.concat([frame, frame[["p_dn5"]]], axis=1), n_boot=10)
    frame["snapshot_id"] = "reused"
    with pytest.raises(ValueError):
        evaluate_horizon(frame, n_boot=10)


def test_bad_parameters_and_empty_input_fail():
    for kwargs in ({"n_boot": 0}, {"n_boot": True}, {"seed": -1}, {"seed": True}):
        with pytest.raises(ValueError):
            evaluate_horizon(predictions(), **kwargs)
    with pytest.raises(ValueError):
        evaluate_horizon(pd.DataFrame(), n_boot=10)
