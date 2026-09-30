import copy

import pytest

from signals import recommend_book_pressure as book
from test_recommend_spread_diagnostics import rows as base_rows


def rows():
    return [{**r, "pressure": (i-49.5)/100, "l1_notional": 1000.} for i, r in enumerate(base_rows())]


def event(seq=1, *, event_ms=1, received=1_100_000, bid_size=3):
    return ({"market": "COIN", "event_at_ms": event_ms, "received_at_ns": received, "ingress_seq": seq},
            {"orderbook_units": [{"bid_price": 99., "ask_price": 101., "bid_size": bid_size, "ask_size": 1.}],
             "total_bid_size": 10**20, "total_ask_size": 0})


def test_endpoint_preserves_same_timestamp_updates_and_both_strict_clocks():
    records = [event(), event(2, bid_size=4), event(3, received=2_000_000), event(4, event_ms=2)]
    value = book.endpoint_quotes(records, cutoff=2_000_000, coins={"COIN"})["COIN"]
    assert value["ingress_seq"] == 2
    assert value["pressure"] == pytest.approx((396-101)/(396+101))
    assert value["l1_notional"] == 497  # full-book total_* deliberately ignored
    assert value["book_event_age_seconds"] == .001


def test_nanosecond_precision_is_not_float_coerced():
    cutoff = 1_790_725_680_000_000_001
    r = event(event_ms=1_790_725_679_999, received=cutoff-1)
    assert book.endpoint_quotes([r], cutoff=cutoff, coins={"COIN"})["COIN"]["received_at_ns"] == cutoff-1
    with pytest.raises(ValueError):
        book.endpoint_quotes([r], cutoff=float(cutoff), coins={"COIN"})


@pytest.mark.parametrize("fault", ["size", "depth", "locked", "clock", "overflow"])
def test_bad_quotes_fail_closed(fault):
    r, p = event()
    if fault == "size":
        p["orderbook_units"][0]["bid_size"] = -1
    elif fault == "depth":
        p["orderbook_units"] *= 2
    elif fault == "locked":
        p["orderbook_units"][0]["bid_price"] = 101
    elif fault == "clock":
        r["received_at_ns"] = True
    else:
        p["orderbook_units"][0]["bid_size"] = 1e308
    with pytest.raises(ValueError):
        book.endpoint_quotes([(r, p)], cutoff=2_000_000, coins={"COIN"})


def test_pairs_do_not_use_outcomes_and_do_not_reuse_coins():
    values = rows()
    plan = book.pairing_plan(values)
    assert len(plan["pairs"]) == 50
    assert plan["pairs"][0] == {"high": "COIN099", "low": "COIN000", "cell": [2, 2]}
    assert len({p[s] for p in plan["pairs"] for s in ("high", "low")}) == 100
    for r in values:
        for k in book.OUTCOMES:
            r[k] = object()
    assert book.pairing_plan(list(reversed(values))) == plan


def test_exact_cost_once_and_date_weighting():
    days = [{"date": "2026-09-10", "rows": rows()}, {"date": "2026-09-20", "rows": rows()}]
    before = copy.deepcopy(days)
    result = book.analyze(days, n_boot=10)
    assert days == before and len(result["excluded"]) == 18
    for s in result["segments"].values():
        assert s["metrics"]["delta"]["mean"] == pytest.approx(
            dict(zip(book.METRICS, [-1, 1, -1, -.08, -.05, -.05], strict=True)))
    assert result["segments"]["all"]["metrics"]["delta"]["observed_date_block3_ci95"] is None


def test_ties_odd_middle_missing_and_halted_not_reselected():
    values = rows()
    for r in values:
        r.update(pressure=0, f_log_qv=r["rank"], f_atr_pct_14=r["rank"])
    plan = book.pairing_plan(values)
    assert plan["equal_pressure_pairs"] == 48 and plan["unpaired_middle_rows"] == 4
    assert not plan["pairs"]
    values = rows()
    values[-1]["label_status"] = "halted_no_observations"
    result = book.analyze([{"date": "2026-09-10", "rows": values}])
    assert result["excluded"][0]["reason"] == "incomplete_labels"
    assert len(result["excluded"][0]["plan"]["pairs"]) == 50
    values[-1]["pressure"] = None
    assert book.analyze([{"date": "2026-09-10", "rows": values}])["excluded"][0]["reason"] == "missing_predictor"


@pytest.mark.parametrize("key,value", [("rank", True), ("pressure", float("nan")), ("pressure", 1),
                                      ("l1_notional", 0), ("f_atr_pct_14", -1), ("coin", "COIN001")])
def test_invalid_predictors_block(key, value):
    values = rows()
    values[0][key] = value
    with pytest.raises(ValueError):
        book.pairing_plan(values)


def test_no_target_redefinition_and_date_integrity():
    values = rows()
    values[0]["up10"] = False
    with pytest.raises(ValueError):
        book.analyze([{"date": "2026-09-10", "rows": values}])
    with pytest.raises(ValueError):
        book.analyze([{"date": "2026-10-01", "rows": rows()}])
    with pytest.raises(ValueError):
        book.analyze([{"date": "2026-09-10", "rows": rows()}]*2)
    with pytest.raises(ValueError):
        book.pairing_plan(rows()[:-1])


def test_cli_design_rejects_mutated_binding_and_existing_outputs(tmp_path, monkeypatch):
    from scripts import review_recommend_book_pressure as cli
    from ops.artifact_provenance import canonical_json_bytes, file_identity
    source = tmp_path / "source.json"
    source.write_bytes(canonical_json_bytes({"schema": "recommend_spread_design.v1", "configuration": cli.inputs.CONFIG,
                                           "input_files": [], "input_path": "unused"}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ())
    monkeypatch.setattr(cli.inputs, "_verify", lambda items: None)
    design = copy.deepcopy(cli.prepare_design(source))
    assert file_identity(source, root=tmp_path) in design["input_files"]
    output = tmp_path / "design.json"
    output.write_bytes(canonical_json_bytes(design))
    assert cli.main(["--prepare-design", str(source), "--output", str(output)]) == 2
    design["configuration"]["direction"] = "changed"
    output.write_bytes(canonical_json_bytes(design))
    # Deep-copy comparison: don't mutate the actual module CONFIG in this test.
    monkeypatch.setattr(cli, "prepare_design", lambda _: {**design, "configuration": {"direction": "high_minus_low"}})
    with pytest.raises(ValueError, match="frozen design"):
        cli.run_review(output)
