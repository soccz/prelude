import copy

import numpy as np
import pytest

from signals import recommend_spread_diagnostics as spread


def rows():
    return [
        {
            "coin": f"COIN{i:03}",
            "rank": i + 1,
            "spread_fraction": (i + 1) / 1000,
            "book_event_age_seconds": 1.0,
            "f_log_qv": 10.0,
            "f_atr_pct_14": 0.1,
            "label_status": "labeled",
            "up10": i < 50,
            "dn5": i >= 50,
            "mfe": 0.12 if i < 50 else 0.01,
            "mae": -0.01 if i < 50 else -0.06,
            "eod_return_net": 0.02 if i < 50 else -0.03,
            "tp5_sl3_return_net": 0.0485 if i < 50 else -0.0315,
        }
        for i in range(100)
    ]


def test_pairs_outcome_blind_permutation_invariant_and_disjoint():
    source = rows()
    before = copy.deepcopy(source)
    plan = spread.pairing_plan(source)
    assert len(plan["pairs"]) == 50
    assert plan["pairs"][0] == {"narrow": "COIN000", "wide": "COIN099", "cell": [2, 2]}
    assert len({p[s] for p in plan["pairs"] for s in ("narrow", "wide")}) == 100
    for row in source:
        for field in spread.OUTCOMES:
            row[field] = object()
    assert spread.pairing_plan(list(reversed(source))) == plan
    assert (
        spread.pairing_plan(
            [{k: v for k, v in r.items() if k not in spread.OUTCOMES} for r in before]
        )
        == plan
    )


def test_ties_and_odd_cells_are_audited_not_arbitrarily_contrasted():
    source = rows()
    for row in source:
        row["spread_fraction"] = 0.001
        row["f_log_qv"] = row["rank"]
        row["f_atr_pct_14"] = row["rank"]
    plan = spread.pairing_plan(source)
    assert not plan["pairs"] and plan["equal_spread_pairs"] == 48
    assert plan["unpaired_middle_rows"] == 4
    result = spread.analyze([{"date": "2026-09-10", "rows": source}])
    assert result["excluded"][0]["reason"] == "no_unequal_spread_pairs"
    assert result["segments"]["all"]["dates"] == 0


def test_exact_net_delta_no_double_cost_equal_dates_and_separate_periods():
    source = rows()
    days = [
        {"date": "2026-09-10", "rows": source},
        {"date": "2026-09-20", "rows": copy.deepcopy(source)},
    ]
    before = copy.deepcopy(days)
    report = spread.analyze(days, n_boot=100)
    assert days == before
    assert len(report["excluded"]) == 18
    for segment in ("earlier", "later", "all"):
        delta = report["segments"][segment]["metrics"]["delta"]["mean"]
        assert delta == pytest.approx(
            dict(zip(spread.METRICS, [1, -1, 1, 0.08, 0.05, 0.05], strict=True))
        )
    assert report["segments"]["all"]["dates"] == 2
    assert report["segments"]["all"]["pairs"] == 100
    assert (
        report["segments"]["all"]["metrics"]["delta"]["ci_status"]
        == "insufficient_dates"
    )
    assert report["model_fitted"] is report["automatic_adoption"] is False
    for row in days[1]["rows"]:
        row["eod_return_net"] *= -1
    changed = spread.analyze(days)
    assert changed["daily"][0] == report["daily"][0]
    assert changed["daily"][1]["pairs"] == report["daily"][1]["pairs"]
    assert changed["segments"]["all"]["metrics"]["delta"]["mean"][
        "eod_return_net"
    ] == pytest.approx(0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("spread_fraction", -1),
        ("spread_fraction", 2),
        ("spread_fraction", float("nan")),
        ("book_event_age_seconds", 0),
        ("rank", True),
        ("rank", 2),
        ("f_atr_pct_14", -1),
        ("coin", "COIN001"),
    ],
)
def test_invalid_inputs_block(field, value):
    source = rows()
    source[0][field] = value
    with pytest.raises(ValueError):
        spread.pairing_plan(source)


def test_missing_or_unlabeled_full_date_not_row_selection():
    source = rows()
    source[99]["spread_fraction"] = None
    assert (
        spread.analyze([{"date": "2026-09-10", "rows": source}])["excluded"][0][
            "reason"
        ]
        == "missing_predictor"
    )
    source = rows()
    source[99]["label_status"] = "halted_no_observations"
    report = spread.analyze([{"date": "2026-09-10", "rows": source}])
    assert report["excluded"][0]["reason"] == "incomplete_full_universe_labels"
    assert len(report["excluded"][0]["plan"]["pairs"]) == 50
    with pytest.raises(ValueError, match="incomplete"):
        spread.pairing_plan(rows()[:-1])


def test_outcome_integrity_and_date_coverage():
    source = rows()
    source[0]["up10"] = False
    with pytest.raises(ValueError, match="mismatch"):
        spread.analyze([{"date": "2026-09-10", "rows": source}])
    with pytest.raises(ValueError, match="duplicate"):
        spread.analyze([{"date": "2026-09-10", "rows": rows()}] * 2)
    with pytest.raises(ValueError, match="outside"):
        spread.analyze([{"date": "2026-09-30", "rows": rows()}])


def test_date_cluster_ci_reproducible_not_row_independent():
    days = [{"date": f"2026-09-{d:02}", "rows": rows()} for d in range(10, 20)]
    for d, day in enumerate(days):
        for row in day["rows"]:
            row["eod_return_net"] *= d + 1
    result = spread.analyze(days)
    assert spread.analyze(list(reversed(days))) == result
    ci = result["segments"]["earlier"]["metrics"]["delta"]["observed_date_block3_ci95"][
        "eod_return_net"
    ]
    assert ci[1] - ci[0] > 0.05
    expected = np.mean(
        [d["metrics"]["delta"]["eod_return_net"] for d in result["daily"][1:]]
    )
    assert result["segments"]["earlier"]["leave_one_date_out"][0]["mean"][
        "eod_return_net"
    ] == pytest.approx(expected)


def test_dates_not_pairs_are_equal_weight_and_old_quotes_are_not_posthoc_filtered():
    first, second = rows(), rows()
    for r in second:
        r["spread_fraction"] = 0.002
        r["eod_return_net"] = 0
        r["book_event_age_seconds"] = 123.0
    second[0]["spread_fraction"] = 0.001
    second[99]["spread_fraction"] = 0.003
    second[0]["eod_return_net"] = -0.05
    report = spread.analyze(
        [{"date": "2026-09-10", "rows": first}, {"date": "2026-09-11", "rows": second}]
    )
    assert [d["n_pairs"] for d in report["daily"]] == [50, 1]
    assert report["daily"][1]["book_age_max_seconds"] == 123.0
    assert report["segments"]["all"]["metrics"]["delta"]["mean"][
        "eod_return_net"
    ] == pytest.approx(0)


def test_matching_uses_same_day_features_and_keeps_strata_separate():
    source = rows()
    for r in source:
        r["f_log_qv"] = r["rank"]
        r["f_atr_pct_14"] = 101 - r["rank"]
    plan = spread.pairing_plan(source)
    assert plan["observed_cells"] == 4 and len(plan["pairs"]) == 48
    by_coin = {r["coin"]: r for r in source}
    for p in plan["pairs"]:
        low, high = (by_coin[p[s]]["rank"] for s in ("narrow", "wide"))
        assert (low - 1) // 25 == (high - 1) // 25
