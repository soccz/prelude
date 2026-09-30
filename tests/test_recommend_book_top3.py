import copy
import json
from collections import Counter
from datetime import date, timedelta
from itertools import combinations

import numpy as np
import pytest

from signals import recommend_book_top3 as top
from test_recommend_book_pressure import rows


def day(date="2026-09-10", values=None):
    following = (top.pd.Timestamp(date)+top.pd.Timedelta(days=1)).date().isoformat()
    return {"date": date, "rows": rows() if values is None else values,
            "entry_at": date+"T09:15:00+09:00", "end_at": following+"T09:15:00+09:00"}


def test_fixed_rule_reaches_beyond_top10_without_outcomes():
    values = rows()
    before = copy.deepcopy(values)
    plan = top.select_top3(values)
    assert values == before
    assert plan["control"] == ["COIN000", "COIN001", "COIN002"]
    assert plan["challenger"] == ["COIN097", "COIN098", "COIN099"]
    assert plan["changed_picks"] == plan["selected_outside_top10"] == 3
    for r in values:
        for k in top.book.OUTCOMES:
            r[k] = object()
    assert top.select_top3(list(reversed(values))) == plan


def test_ties_keep_original_picks_without_fake_effect():
    values = rows()
    for r in values:
        r["pressure"] = .1
    result = top.evaluate([day(values=values)], n_boot=10)
    item = result["daily"][0]
    assert item["selection"]["challenger"] == item["selection"]["control"]
    assert item["replacement_metrics"] == {"added": None, "removed": None}
    assert set(item["metrics"]["challenger_minus_control"].values()) == {0.}
    assert result["segments"]["all"]["no_op_dates"] == 1


def test_cell_quota_exact_and_maximal_against_exhaustive_combinations():
    values = rows()
    # 25 members per diagonal cell; all three originals share the first cell.
    for r in values:
        r.update(f_log_qv=r["rank"], f_atr_pct_14=r["rank"]/100)
    plan = top.select_top3(values)
    assert plan["selected_original_ranks"] == [23,24,25]
    assert plan["pools"][0]["quota"] == 3 and len(plan["pools"][0]["coins"]) == 25
    indexed = {r["coin"]:r for r in values}
    objectives = [sum(indexed[c]["pressure"] for c in picks) for picks in combinations(plan["pools"][0]["coins"],3)]
    assert sum(indexed[c]["pressure"] for c in plan["challenger"]) == max(objectives)


def test_distinct_cell_quotas_and_singleton_not_excluded():
    values = rows()
    values[0]["f_log_qv"] = 1000  # Unique cell; its original pick cannot be changed.
    plan = top.select_top3(values)
    counts = Counter()
    for pool in plan["pools"]:
        counts[tuple(pool["cell"])] = sum(c in plan["challenger"] for c in pool["coins"])
        assert counts[tuple(pool["cell"])] == pool["quota"]
    assert "COIN000" in plan["challenger"]
    assert any(len(p["coins"]) == 1 for p in plan["pools"])
    assert len(top.evaluate([day(values=values)])["daily"]) == 1


def test_exact_random_expectation_cost_once_and_equal_day_mean():
    result = top.evaluate([day(),day("2026-09-20")], n_boot=10)
    item = result["daily"][0]["metrics"]
    assert item["control"]["eod_return_net"] == pytest.approx(.02)
    assert item["challenger"]["eod_return_net"] == pytest.approx(-.03)
    assert item["matched_random_expectation"]["eod_return_net"] == pytest.approx(-.005)
    assert item["challenger_minus_control"]["eod_return_net"] == pytest.approx(-.05)
    assert item["challenger_minus_matched_random_expectation"]["up10"] == pytest.approx(-.5)
    assert result["segments"]["all"]["picks_per_arm"] == 6
    assert result["segments"]["all"]["basket_proxy"]["control"]["status"] == "gaps_or_overlaps"
    assert result["segments"]["all"]["metrics"]["challenger_minus_control"]["observed_date_block3_ci95"] is None


def test_halted_selected_coin_not_replaced_by_labeled_coin():
    values = rows()
    values[-1]["label_status"] = "halted_no_observations"
    for k in top.book.OUTCOMES:
        values[-1][k] = None
    report = top.evaluate([day(values=values)])
    assert not report["daily"]
    assert "COIN099" in report["excluded"][0]["selection"]["challenger"]


@pytest.mark.parametrize("fault", ["input_missing", "nan", "bad_label", "missing_row", "duplicate_date", "future"])
def test_corrupt_inputs_or_missing_evidence_are_never_silently_reselected(fault):
    values = [day()]
    if fault == "input_missing":
        values[0]["rows"][-1]["pressure"] = None
        assert top.evaluate(values)["excluded"][0]["reason"] == "missing_predictor"
        return
    if fault == "nan":
        values[0]["rows"][0]["pressure"] = float("nan")
    elif fault == "bad_label":
        values[0]["rows"][0]["up10"] = False
    elif fault == "missing_row":
        values[0]["rows"].pop()
    elif fault == "duplicate_date":
        values *= 2
    else:
        values[0]["date"] = "2026-10-01"
    with pytest.raises(ValueError):
        top.evaluate(values)


def test_basket_includes_initial_capital_and_does_not_fill_unknown_days():
    days = [day((date(2026,9,10)+timedelta(days=i)).isoformat()) for i in range(5)]
    result = top.evaluate(days, n_boot=50)
    stats = result["segments"]["all"]["basket_proxy"]["challenger"]["metrics"]
    assert stats["cumulative_return"] == pytest.approx(.97**5-1)
    assert stats["max_drawdown"] == pytest.approx(.97**5-1)
    assert stats["positive_day_rate"] == 0 and stats["periods_per_year"] == 365
    intervals = result["segments"]["all"]["metrics"]["challenger_minus_control"]["observed_date_block3_ci95"]
    assert intervals["eod_return_net"] == pytest.approx([-.05,-.05])
    for name, value in (("entry_at", None),("end_at", "2026-09-11T09:30:00+09:00")):
        changed = copy.deepcopy(days)
        changed[0][name] = value
        assert top.evaluate(changed)["segments"]["all"]["basket_proxy"]["control"]["metrics"] is None


def test_later_outcomes_do_not_change_earlier_records_or_selection():
    days = [day(), day("2026-09-20")]
    original = top.evaluate(days)
    for row in days[1]["rows"]:
        row["eod_return_net"] *= 2
    altered = top.evaluate(days)
    assert altered["daily"][0] == original["daily"][0]
    assert altered["daily"][1]["selection"] == original["daily"][1]["selection"]


@pytest.mark.parametrize("seed,boot", [(True,100),(42,True),(42,0),(-1,100)])
def test_invalid_statistics_rejected(seed,boot):
    with pytest.raises(ValueError):
        top.evaluate([day()], seed=seed, n_boot=boot)


def test_cli_manifest_change_blocks_before_evaluation(tmp_path,monkeypatch):
    from scripts import compare_recommend_book_top3 as cli
    from ops.artifact_provenance import canonical_json_bytes
    design = {"schema":"recommend_book_top3_design.v1","endpoint_report":"source.json","configuration":top.CONFIG}
    path = tmp_path / "design.json"
    path.write_bytes(canonical_json_bytes(design))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "prepare_design", lambda p:{**design,"configuration":{}})
    monkeypatch.setattr(cli.selector, "evaluate", lambda *a,**k:pytest.fail("outcomes read after changed manifest"))
    with pytest.raises(ValueError,match="frozen selector design"):
        cli.run_comparison(path)


def test_random_expected_mean_is_combinatorial_expectation():
    vectors = np.arange(15).reshape(5,3)/100
    for quota in (1,2,3):
        sampled = np.array([vectors[list(c)].mean(0) for c in combinations(range(5),quota)])
        assert sampled.mean(0) == pytest.approx(vectors.mean(0))


def test_complete_report_is_strict_json_serializable():
    days = [day((date(2026,9,10)+timedelta(days=i)).isoformat()) for i in range(20)]
    result = top.evaluate(days, n_boot=10)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
