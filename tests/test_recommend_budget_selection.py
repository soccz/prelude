from itertools import combinations
from math import fsum

import numpy as np
import pytest

from signals.recommend_budget_selection import evaluate_budget, select_budgeted
from test_recommend_experiment_eval import _predictions


def rows(n=7):
    return [
        {
            "coin": f"C{i}",
            "rank": i + 1,
            "p_up10_A": 0.5,
            "p_dn5": 0.1,
            "p_up10_C": i / 10,
        }
        for i in range(n)
    ]


def test_exact_optimum_with_feasible_original_and_no_weight_tuning():
    result = select_budgeted(rows())
    assert result["selected_coins"] == ["C4", "C5", "C6"]
    assert result["examined_combinations"] == result["feasible_combinations"] == 35
    assert result["selected_upper_sum"] >= result["original_upper_sum"]
    assert result["selected_down_sum"] <= result["original_down_sum"]


def test_all_ties_keep_original_and_other_ties_follow_original_ranks():
    data = rows()
    for r in data:
        r["p_up10_C"] = 0.2
    assert select_budgeted(data)["selected_coins"] == ["C0", "C1", "C2"]
    data[-1]["p_up10_C"] = 0.3
    assert select_budgeted(data)["selected_coins"] == ["C0", "C1", "C6"]


def test_exhaustive_independent_reference_including_float_boundaries():
    rng = np.random.default_rng(42)
    for _ in range(30):
        data = rows(8)
        for r in data:
            for field in ("p_up10_A", "p_dn5", "p_up10_C"):
                r[field] = float(
                    rng.choice(
                        [
                            0.0,
                            0.1,
                            0.2,
                            0.3,
                            0.30000000000000004,
                            0.3000000000000001,
                            0.5,
                            0.7,
                            1.0,
                        ]
                    )
                )
        allowed = []
        for combo in combinations(data, 3):
            if fsum(r["p_dn5"] for r in combo) <= fsum(
                r["p_dn5"] for r in data[:3]
            ) and fsum(r["p_up10_A"] for r in combo) >= fsum(
                r["p_up10_A"] for r in data[:3]
            ):
                allowed.append(combo)
        best = min(
            allowed,
            key=lambda combo: (
                -fsum(r["p_up10_C"] for r in combo),
                tuple(r["rank"] for r in combo),
            ),
        )
        actual = select_budgeted(data)
        assert actual["selected_coins"] == [r["coin"] for r in best]
        assert actual["feasible_combinations"] == len(allowed)


def test_order_and_outcome_invariance():
    data = rows()
    original = select_budgeted(data)
    for i, r in enumerate(data):
        r["eod_return_net"] = 100 * i
        r["up10"] = i % 2
        r["label_status"] = "halted"
    assert select_budgeted(list(reversed(data))) == original


@pytest.mark.parametrize(
    "bad", [None, True, float("nan"), float("inf"), -0.01, 1.01, "0.2"]
)
def test_bad_scores_never_fallback(bad):
    data = rows()
    data[3]["p_up10_C"] = bad
    with pytest.raises(ValueError):
        select_budgeted(data)


@pytest.mark.parametrize("mutation", ["small", "rank", "coin", "bool_rank"])
def test_bad_population_rejected(mutation):
    data = rows()
    if mutation == "small":
        data = data[:2]
    elif mutation == "rank":
        data[0]["rank"] = 9
    elif mutation == "coin":
        data[0]["coin"] = data[1]["coin"]
    else:
        data[0]["rank"] = True
    with pytest.raises(ValueError):
        select_budgeted(data)


def fixture_frame():
    data = _predictions(slots=("open", "preopen"))
    data["p_up10"] = data["p_up10_A"] = 0.5
    data["rr_ratio"] = 5.0
    return data


def test_evaluation_scores_selected_indices_not_fake_probabilities():
    frame = fixture_frame()
    report = evaluate_budget(frame, n_boot=20)
    assert report["coverage"] == {"included_date_slots": 12, "excluded_date_slots": 0}
    for slot, s in report["slots"].items():
        assert s["changed_dates"] == 6
        assert s["changed_picks"] == 6
        assert s["picks_per_arm"] == 18
        expected = frame[
            (frame.slot == slot) & frame.coin.isin(["KRW-C0", "KRW-C1", "KRW-C4"])
        ]
        assert s["absolute"]["E"]["mean"]["eod_return_net"] == pytest.approx(
            expected.eod_return_net.mean()
        )
        assert s["delta_vs_A"]["mean"]["eod_return_net"] == pytest.approx(0.02 / 3)
        assert s["delta_vs_A"]["observed_date_block3_ci95"][
            "eod_return_net"
        ] == pytest.approx([0.02 / 3, 0.02 / 3])
    assert all(
        a["budget"]["selected_down_sum"] <= a["budget"]["original_down_sum"]
        for a in report["selection_audit"]
    )


def test_halt_kept_in_selection_not_refilled():
    frame = fixture_frame()
    baseline = evaluate_budget(frame, n_boot=10)
    selected = (
        (frame.date == "2026-09-01") & (frame.coin == "KRW-C4") & (frame.slot == "open")
    )
    frame.loc[selected, "label_status"] = "halted_no_observations"
    frame.loc[
        selected, ["up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"]
    ] = np.nan
    alternative = evaluate_budget(frame, n_boot=10)
    assert alternative["coverage"]["excluded_date_slots"] == 1
    assert (
        baseline["selection_audit"][0]["selected_coins"]
        == alternative["selection_audit"][0]["selected_coins"]
    )


def test_declared_unavailable_c_has_no_synthetic_fallback():
    frame = fixture_frame()
    frame["C_status"] = "unavailable"
    frame["p_up10_C"] = np.nan
    result = evaluate_budget(frame, n_boot=10)
    assert result["coverage"]["included_date_slots"] == 0
    assert all(
        a["selected_coins"]["E"] is None and a["budget"] is None
        for a in result["selection_audit"]
    )


def test_future_outcomes_do_not_change_any_selected_coins():
    frame = fixture_frame()
    initial = evaluate_budget(frame, n_boot=10)
    frame["up10"] = 0.0
    frame["mfe"] = 0.02
    frame["dn5"] = 0.0
    frame["mae"] = -0.02
    frame["eod_return_net"] = frame["tp5_sl3_return_net"] = -0.0115
    changed = evaluate_budget(frame, n_boot=10)
    assert initial["selection_audit"] == changed["selection_audit"]
