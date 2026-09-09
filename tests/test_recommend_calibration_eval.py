"""Hermetic checks for fixed raw scores and distinct A/F/R/O cohort plans."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from signals.recommend_calibration_eval import (
    ARMS, CONTRASTS, EVALUATION_CONFIG, EXPERIMENT_ARMS, HEADS, SCORES,
    CalibrationEvaluationError, evaluate_calibration,
)


def predictions(days=6, coins=6):
    rows = []
    for day_index in range(days):
        day = (date(2026, 8, 1) + timedelta(days=day_index)).isoformat()
        for coin_index in range(coins):
            up, down = coin_index % 2 == 0, coin_index % 3 == 0
            row = {"date": day, "slot": "preopen", "coin": f"KRW-C{coin_index}", "rank": coin_index + 1,
                   "snapshot_id": f"sid-{day}", "was_delivered": coin_index < 3, "label_status": "labeled",
                   "p_up10": .6 - coin_index * .04, "p_dn5": .1, "p_dn10": .02, "exp_downside": -.03,
                   "f_log_qv": 10., "f_atr_pct_14": .02, "mfe": .15 if up else .03,
                   "mae": -.06 if down else -.01, "up5": up, "up10": up, "up20": False,
                   "dn5": down, "dn10": False, "eod_return_net": (coin_index - 2) * .01 + day_index * .001 - .0015,
                   "tp5_sl3_return_net": .0485 if up else -.0315}
            row["rr_ratio"] = row["p_up10"] / row["p_dn5"]
            for arm in EXPERIMENT_ARMS:
                row[f"{arm}_status"] = "fitted"
                values = {"p_up5": .7, "p_up10": row["p_up10"], "p_up20": .1,
                          "p_dn5": .1, "p_dn10": .02, "exp_downside": -.03}
                if (arm == "R" and coin_index == coins - 1) or (arm == "O" and coin_index == coins - 2):
                    values["p_up10"] = .95
                for name, value in values.items():
                    row[f"{name}_{arm}"] = value
                    if name.startswith("p_"):
                        row[f"{name}_{arm}_raw"] = .2 + coin_index * .01
            rows.append(row)
    return pd.DataFrame(rows)


def halt(frame, coin="KRW-C3", day="2026-08-01"):
    mask = frame.date.eq(day) & frame.coin.eq(coin)
    frame.loc[mask, "label_status"] = "halted_no_observations"
    for column in (*HEADS, "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"):
        frame[column] = frame[column].astype(float)
        frame.loc[mask, column] = np.nan


def unavailable(frame, arm="O", day="2026-08-01"):
    mask = frame.date.eq(day)
    frame.loc[mask, f"{arm}_status"] = "unavailable"
    for name in SCORES:
        frame.loc[mask, f"{name}_{arm}"] = np.nan
        if name.startswith("p_"):
            frame.loc[mask, f"{name}_{arm}_raw"] = np.nan


def test_real_arm_names_fixed_shared_raw_but_distinct_calibration_selection():
    frame = predictions()
    before = frame.copy(deep=True)
    result = evaluate_calibration(frame, n_boot=30)
    pd.testing.assert_frame_equal(frame, before)
    assert result["configuration"] == EVALUATION_CONFIG
    assert result["input"] == {"rows": 36, "dates": 6, "actual_delivered_rows": 18}
    picks = result["selection_audit"][0]["selected_coins"]
    assert picks == {"A": ["KRW-C0", "KRW-C1", "KRW-C2"], "F": ["KRW-C0", "KRW-C1", "KRW-C2"],
                     "R": ["KRW-C5", "KRW-C0", "KRW-C1"], "O": ["KRW-C4", "KRW-C0", "KRW-C1"]}
    primary = result["primary"]
    assert set(primary["absolute"]) == set(ARMS)
    assert set(primary["contrasts"]) == set(CONTRASTS)
    assert all(set(row["selected"]) == set(ARMS) for row in primary["membership"])
    assert primary["cohort_id"] != result["supplemental"]["cohort_id"]
    assert primary["dates"] == 6 and primary["picks_per_arm"] == 18
    actual = frame.loc[frame.was_delivered]
    for metric in ("tp5_sl3_return_net", "eod_return_net"):
        assert primary["absolute"]["A"]["mean"][metric] == pytest.approx(actual[metric].mean())
    assert primary["matched"]["A"] == primary["full_universe"]
    assert "matched" not in result["supplemental"]
    assert [group["dates"] for group in primary["chronological_groups"]] == [5, 1]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("p_up10_R", .99), ("p_dn5_R", .001), ("p_dn10_R", .001)])
def test_calibrated_up_down_heads_drive_own_rank_without_changing_raw(field, value):
    frame = predictions(days=1)
    for arm in EXPERIMENT_ARMS:
        frame[f"p_up10_{arm}"] = .5
    before = evaluate_calibration(frame, n_boot=10)["selection_audit"]
    frame.at[5, field] = value
    after = evaluate_calibration(frame, n_boot=10)["selection_audit"]
    assert after[0]["selected_coins"]["R"] == ["KRW-C5", "KRW-C0", "KRW-C1"]
    for arm in ("A", "F", "O"):
        assert after[0]["selected_coins"][arm] == before[0]["selected_coins"][arm]


def test_all_identical_calibrated_arms_have_zero_six_contrasts_and_no_promotion():
    frame = predictions()
    for arm in ("R", "O"):
        for name in SCORES:
            frame[f"{name}_{arm}"] = frame[f"{name}_F"]
    result = evaluate_calibration(frame, n_boot=40)
    assert result["deployable"] is result["automatic_promotion"] is False
    assert result["promotion_status"] == "NOT_EVALUATED"
    for cohort in ("primary", "supplemental"):
        for contrast in result[cohort]["contrasts"].values():
            assert all(value == 0 for value in contrast["mean"].values())
            assert all(limits == [0, 0] for limits in contrast["observed_date_block3_ci95"].values())


def test_probability_improvement_alone_never_approves_model():
    frame = predictions()
    frame["p_up10_O"] = frame["up10"].astype(float)  # Synthetic oracle to exercise no-verdict contract.
    result = evaluate_calibration(frame, n_boot=10)
    heads = result["primary"]["probability_diagnostics"]
    assert heads["O"]["all_labeled_candidates"]["heads"]["up10"]["brier"] == 0
    assert result["deployable"] is False and result["automatic_promotion"] is False
    assert result["promotion_status"] == "NOT_EVALUATED"
    assert not result["is_untouched_holdout"]


def test_outcome_changes_do_not_change_any_selection_or_matching_plan():
    frame = predictions()
    before = evaluate_calibration(frame, n_boot=10)["selection_audit"]
    for name in HEADS:
        frame[name] = False
    frame["mfe"], frame["mae"] = .01, -.01
    frame["eod_return_net"], frame["tp5_sl3_return_net"] = -.0015, -.0015
    halt(frame)
    after = evaluate_calibration(frame, n_boot=10)["selection_audit"]
    assert before == after


@pytest.mark.parametrize("coin,selected", [("KRW-C3", False), ("KRW-C5", True)])
def test_halt_checked_after_selection_and_cohorts_have_explicit_denominators(coin, selected):
    frame = predictions()
    halt(frame, coin=coin)
    result = evaluate_calibration(frame, n_boot=10)
    assert result["primary"]["dates"] == 5
    assert result["supplemental"]["dates"] == (5 if selected else 6)
    assert result["selection_audit"][0]["universe_rows"] == 6
    if selected:
        assert coin in result["selection_audit"][0]["selected_coins"]["R"]
        assert result["supplemental"]["excluded"][0]["reasons"] == [f"selected_label_unavailable:R:{coin}"]
    else:
        assert result["primary"]["excluded"][0]["reasons"] == [f"full_universe_label_unavailable:{coin}"]
        diag = result["supplemental"]["probability_diagnostics"]["O"]["all_labeled_candidates"]
        assert (diag["candidate_rows"], diag["labeled_rows"], diag["unlabeled_rows"]) == (36, 35, 1)


def test_unavailable_arm_removes_whole_common_date_without_A_fallback():
    frame = predictions()
    unavailable(frame)
    result = evaluate_calibration(frame, n_boot=10)
    assert result["primary"]["dates"] == result["supplemental"]["dates"] == 5
    assert result["selection_audit"][0]["selected_coins"]["O"] is None
    assert result["selection_audit"][0]["selected_coins"]["F"] is not None
    assert result["supplemental"]["excluded"][0]["reasons"] == ["arm_unavailable:O"]


def test_small_matched_cells_remove_primary_only_without_pool_expansion():
    frame = predictions(coins=8)
    frame["f_log_qv"] = frame["rank"].astype(float)
    frame["f_atr_pct_14"] = frame["rank"] / 100
    result = evaluate_calibration(frame, n_boot=10)
    assert result["status"] == "no_primary_common_dates"
    assert result["data_availability"] == "insufficient_primary"
    assert result["primary"]["dates"] == 0 and result["supplemental"]["dates"] == 6
    assert result["selection_audit"][0]["matched_pool_sizes"]["F"] == [2, 2, 2]


def test_paired_delta_bootstrap_lodo_conservation_and_group_means():
    result = evaluate_calibration(predictions(), n_boot=100)["primary"]
    expected = -.01 / 3
    summary = result["contrasts"]["O_minus_R"]
    assert summary["mean"]["eod_return_net"] == pytest.approx(expected)
    for name in ("iid_date_ci95", "observed_date_block3_ci95"):
        assert summary[name]["eod_return_net"] == pytest.approx([expected, expected])
    sensitivity = result["sensitivity"]["O_minus_R"]
    assert all(row["mean_delta"]["eod_return_net"] == pytest.approx(expected) for row in sensitivity["leave_one_date_out"])
    assert sum(row["eod_return_net"] for row in sensitivity["coin_contributions"].values()) == pytest.approx(expected)
    assert sensitivity["sum_coin_contributions"]["eod_return_net"] == pytest.approx(expected)
    assert all(group["contrasts"]["O_minus_R"]["eod_return_net"] == pytest.approx(expected) for group in result["chronological_groups"])


def test_head_diagnostics_use_calibrated_arm_scores_and_common_raw():
    frame = predictions()
    frame["p_dn5_R"] = .4
    result = evaluate_calibration(frame, n_boot=10)["primary"]["probability_diagnostics"]
    assert result["R"]["all_labeled_candidates"]["heads"]["dn5"]["brier"] == pytest.approx(np.mean((frame.p_dn5_R - frame.dn5.astype(float)) ** 2))
    assert result["O"]["all_labeled_candidates"]["raw_heads"] == result["R"]["all_labeled_candidates"]["raw_heads"] == result["F"]["all_labeled_candidates"]["raw_heads"]
    assert result["A"]["all_labeled_candidates"]["heads"]["up5"]["status"] == "not_recorded_or_no_labels"
    assert result["O"]["selected_top3"]["heads"]["up10"]["rows"] == 18


def test_shuffle_does_not_change_any_report_field():
    frame = predictions()
    assert evaluate_calibration(frame, n_boot=15) == evaluate_calibration(frame.sample(frac=1, random_state=98), n_boot=15)


@pytest.mark.parametrize("days", [1, 3])
def test_small_samples_have_no_CI_and_single_date_has_no_lodo(days):
    result = evaluate_calibration(predictions(days=days), n_boot=10)
    for name in ("primary", "supplemental"):
        assert result[name]["contrasts"]["O_minus_R"]["ci_status"] == "insufficient_dates"
        assert result[name]["contrasts"]["O_minus_R"]["iid_date_ci95"] is None
        if days == 1:
            assert result[name]["sensitivity"]["O_minus_R"]["leave_one_date_out"] == []


@pytest.mark.parametrize("field,value", [("exp_downside_O", -.029999999999), ("p_dn5_R_raw", .200000000001)])
def test_tiny_change_to_shared_predictor_is_hard_failure(field, value):
    frame = predictions()
    frame.at[0, field] = value
    with pytest.raises(CalibrationEvaluationError, match="shared predictor"):
        evaluate_calibration(frame, n_boot=10)


@pytest.mark.parametrize("kind", ["mixed_status", "finite_unavailable", "nan_fitted", "inf_raw", "missing_raw", "bad_status"])
def test_availability_and_raw_contract_fail_closed(kind):
    frame = predictions()
    if kind == "mixed_status":
        frame.at[0, "R_status"] = "unavailable"
    elif kind == "finite_unavailable":
        unavailable(frame)
        frame.at[0, "p_dn10_O"] = .1
    elif kind == "nan_fitted":
        frame.at[0, "p_up10_F"] = np.nan
    elif kind == "inf_raw":
        frame.at[0, "p_up5_O_raw"] = np.inf
    elif kind == "bad_status":
        frame.at[0, "F_status"] = "ok"
    else:
        frame = frame.drop(columns="p_up20_R_raw")
    with pytest.raises(CalibrationEvaluationError):
        evaluate_calibration(frame, n_boot=10)


def test_all_unavailable_completed_insufficient_but_A_identity_still_checked():
    frame = predictions(days=1)
    for arm in EXPERIMENT_ARMS:
        unavailable(frame, arm=arm)
    result = evaluate_calibration(frame, n_boot=10)
    assert result["execution_status"] == "completed" and result["data_availability"] == "insufficient_primary"
    assert result["primary"]["dates"] == result["supplemental"]["dates"] == 0
    frame.at[5, "p_up10"] = .99
    with pytest.raises(CalibrationEvaluationError, match="A identity"):
        evaluate_calibration(frame, n_boot=10)


@pytest.mark.parametrize("field,value", [("p_dn5_O", -.1), ("exp_downside_F", .1), ("f_log_qv", np.nan),
                                        ("rank", 1.5), ("slot", "open"), ("up20", 1), ("dn10", 1),
                                        ("label_status", "missing")])
def test_invalid_predictors_outcomes_and_scope(field, value):
    frame = predictions()
    frame[field] = frame[field].astype(object)
    frame.at[0, field] = value
    with pytest.raises(ValueError):
        evaluate_calibration(frame, n_boot=10)


def test_duplicate_rows_columns_snapshot_reuse_empty_input_and_bad_options():
    frame = predictions()
    invalid = [pd.concat([frame, frame.iloc[:1]]), pd.concat([frame, frame[["p_dn5"]]], axis=1), pd.DataFrame()]
    reused = frame.copy()
    reused["snapshot_id"] = "same"
    invalid.append(reused)
    for candidate in invalid:
        with pytest.raises(ValueError):
            evaluate_calibration(candidate, n_boot=10)
    for kwargs in ({"n_boot": True}, {"n_boot": 0}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError):
            evaluate_calibration(frame, **kwargs)

