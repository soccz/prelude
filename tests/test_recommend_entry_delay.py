from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import scripts.evaluate_recommend_entry_delay as evaluator
from ledger.path_quality import assess_15m_window
from ops.recommendation_evidence import EvidenceError, EvidenceUnavailable
from signals.recommend_score_labels import _label_candidate


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("entry-delay analysis must never access the network")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)


def make_case(tmp_path, monkeypatch, *, day="2026-08-01", slot="open"):
    start = pd.Timestamp(f"{day}T09:15:00+09:00")
    db = tmp_path / "paths.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE candles (market TEXT,timestamp TEXT,open REAL,high REAL,low REAL,close REAL)")
    for market in ("KRW-BTC", "KRW-T1", "KRW-T2", "KRW-T3"):
        for index in range(-1, 99):
            o, h, low, c = 100.0, 102.0, 99.0, 101.0
            if market == "KRW-T1" and index == 2:
                h = 106.0
            if market == "KRW-T1" and index == 3:
                low = 94.0
            if market == "KRW-T2" and index == 0:
                h, low = 106.0, 96.0
            if market == "KRW-T3" and index >= 96:
                h, c = 111.0, 110.0
            timestamp = (start + pd.Timedelta(minutes=index*15)).strftime("%Y-%m-%d %H:%M:%S")
            con.execute("INSERT INTO candles VALUES (?,?,?,?,?,?)", (market, timestamp, o, h, low, c))
    con.commit()
    con.close()
    snapshot = {"asof": day, "slot": slot, "snapshot_id": f"snapshot-{day}-{slot}",
                "payload_sha256": "a"*64, "top3": []}
    rows = []
    for rank in range(1, 4):
        candidate = {"coin": f"KRW-T{rank}", "rank": rank}
        snapshot["top3"].append(candidate)
        assessment = assess_15m_window(candidate["coin"], start, db_path=db)
        rows.append(_label_candidate(candidate, assessment,
                                     snapshot_id=snapshot["snapshot_id"],
                                     snapshot_hash=snapshot["payload_sha256"],
                                     execution={"execution_start_at": start.isoformat()}))
    # Universe row four carries delivery_ok too, but was not actually sent.
    rows.append({"coin": "KRW-UNSENT", "rank": 4, "delivery_ok": True})
    evidence = {"snapshot": snapshot, "label": {"execution_start_at": start.isoformat(), "rows": rows},
                "receipt": {"delivery_ok": True}, "manifest": {"snapshot_id": snapshot["snapshot_id"]}}
    roots = {name: tmp_path / name for name in ("snapshots", "labels", "receipts")}
    for root in roots.values():
        root.mkdir()
    target = roots["snapshots"] / day
    target.mkdir()
    source = target / f"{slot}_r1.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluator, "load_recommendation_evidence", lambda *a, **kw: copy.deepcopy(evidence))
    options = dict(snapshot_root=roots["snapshots"], label_root=roots["labels"], receipt_root=roots["receipts"],
                   db_path=db, start_date=date.fromisoformat(day), end_date=date.fromisoformat(day),
                   now=datetime(2026, 8, 4, tzinfo=timezone.utc), n_boot=100, min_days=2, slots=(slot,))
    return options, evidence, db, start


def test_zero_parity_full_horizon_cost_and_only_actual_top3(tmp_path, monkeypatch):
    options, evidence, db, start = make_case(tmp_path, monkeypatch)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    before_evidence = copy.deepcopy(evidence)
    report = evaluator.evaluate_entry_delay(**options)
    assert report["status"] == "insufficient"
    assert not report["errors"]
    assert len(report["records"]) == 3
    assert all(r["canonical_parity"] == "passed" for r in report["records"])
    assert report["channels"]["open"]["primary"]["n_picks"] == 3
    for row in report["records"]:
        for delay, value in row["delays"].items():
            assert value["expected_bars"] == 96
            assert pd.Timestamp(value["start_at"]) == start + pd.Timedelta(minutes=int(delay))
            assert pd.Timestamp(value["end_at"]) - pd.Timestamp(value["start_at"]) == pd.Timedelta(days=1)
            assert len(value["raw_input"]["raw_input_sha256"]) == 64
    t1, t2, t3 = report["records"]
    assert t1["delays"]["0"]["outcomes"]["tp5_sl3_return_net"] == pytest.approx(0.0485)
    assert t1["delays"]["0"]["metrics"]["tp5_first"] == 1
    assert t1["delays"]["0"]["metrics"]["whole_path_safe_up10"] == 0
    assert t2["delays"]["0"]["outcomes"]["tp5_sl3_first_passage"] == "sl_first_same_bar"
    assert t2["delays"]["0"]["outcomes"]["tp5_sl3_return_net"] == pytest.approx(-0.0315)
    assert t3["delays"]["0"]["metrics"]["tp5_first"] == 0
    assert t3["delays"]["15"]["metrics"]["tp5_first"] == 1
    assert t3["delays"]["30"]["outcomes"]["eod_return_net"] == pytest.approx(0.0985)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert evidence == before_evidence
    assert json.loads(json.dumps(report, allow_nan=False))["schema"] == evaluator.REPORT_SCHEMA


def test_last_delay_maturity_excludes_entire_date(tmp_path, monkeypatch):
    options, _, _, start = make_case(tmp_path, monkeypatch)
    options["now"] = (start + pd.Timedelta(days=1, minutes=20)).to_pydatetime()
    report = evaluator.evaluate_entry_delay(**options)
    assert report["exclusion_counts"] == {"latest_delay_not_mature": 1}
    assert not report["records"]
    assert report["channels"]["open"]["primary"]["n_days"] == 0


def test_parity_mismatch_blocks_every_statistic(tmp_path, monkeypatch):
    options, evidence, _, _ = make_case(tmp_path, monkeypatch)
    evidence["label"]["rows"][0]["mfe"] = 0.99
    report = evaluator.evaluate_entry_delay(**options)
    assert report["status"] == "blocked_canonical_parity_or_path_integrity"
    assert report["errors"][0]["reason"] == "canonical_zero_delay_mismatch"
    assert report["channels"] == {}


def test_zero_db_damage_is_not_silently_dropped(tmp_path, monkeypatch):
    options, _, db, start = make_case(tmp_path, monkeypatch)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM candles WHERE market='KRW-BTC' AND timestamp=?",
                    (start.strftime("%Y-%m-%d %H:%M:%S"),))
    report = evaluator.evaluate_entry_delay(**options)
    assert report["status"].startswith("blocked")
    assert report["records"][0]["delays"]["0"]["path_quality"] == "benchmark_gap"


def test_only_delayed_btc_gap_excludes_common_rows(tmp_path, monkeypatch):
    options, _, db, start = make_case(tmp_path, monkeypatch)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM candles WHERE market='KRW-BTC' AND timestamp=?",
                    ((start + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),))
    report = evaluator.evaluate_entry_delay(**options)
    assert not report["errors"]
    assert all(r["canonical_parity"] == "passed" for r in report["records"])
    assert report["channels"]["open"]["primary"]["n_picks"] == 0
    assert report["channels"]["open"]["secondary"]["n_picks"] == 0


def test_structural_halt_never_becomes_flat_zero_return(tmp_path, monkeypatch):
    options, evidence, _, _ = make_case(tmp_path, monkeypatch)
    evidence["label"]["rows"][0].update(label_status="halted_no_observations", path_complete=False)
    report = evaluator.evaluate_entry_delay(**options)
    assert report["records"][0]["excluded_reason"] == "canonical_halted_no_observations"
    assert report["records"][0]["delays"] == {}
    assert report["channels"]["open"]["primary"]["n_picks"] == 0
    assert report["channels"]["open"]["secondary"]["n_picks"] == 2


def test_flat_fill_retained_with_input_hash_for_prior_close(tmp_path, monkeypatch):
    options, evidence, db, start = make_case(tmp_path, monkeypatch)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM candles WHERE market='KRW-T3' AND timestamp=?",
                    (start.strftime("%Y-%m-%d %H:%M:%S"),))
    assessment = assess_15m_window("KRW-T3", start, db_path=db)
    evidence["label"]["rows"][2] = _label_candidate(
        {"coin": "KRW-T3", "rank": 3}, assessment,
        snapshot_id=evidence["snapshot"]["snapshot_id"], snapshot_hash="a"*64,
        execution={"execution_start_at": start.isoformat()},
    )
    report = evaluator.evaluate_entry_delay(**options)
    zero = report["records"][2]["delays"]["0"]
    assert zero["path_quality"] == "flat_filled"
    assert zero["raw_input"]["prior_rows"] == 2
    assert zero["outcomes"]["actual_entry_open"] == 101.0
    assert report["channels"]["open"]["primary"]["n_picks"] == 3


@pytest.mark.parametrize("exception,blocked", [
    (EvidenceUnavailable("legacy_or_not_delivered"), False),
    (EvidenceError("receipt_snapshot_mismatch"), True),
])
def test_evidence_exclusion_vs_corruption(tmp_path, monkeypatch, exception, blocked):
    options, _, _, _ = make_case(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(evaluator, "load_recommendation_evidence", fail)
    report = evaluator.evaluate_entry_delay(**options)
    assert report["status"].startswith("blocked") is blocked
    assert not report["records"]
    assert bool(report["errors"]) is blocked
    assert bool(report["exclusions"]) is not blocked


def test_date_equal_not_row_equal_and_paired_ci():
    records = []
    for day, count, value in (("2026-08-01", 1, 1.0), ("2026-08-02", 3, 0.0)):
        for _ in range(count):
            records.append({"asof": day, "delays": {
                "0": {"metrics": {key: 0.0 for key in evaluator.METRICS}},
                "15": {"metrics": {key: value for key in evaluator.METRICS}},
            }})
    result = evaluator._summarize(records, (0, 15), n_boot=100, min_days=2, seed=42)
    for metric in result["delays"]["15"]["metrics"].values():
        assert metric["day_equal_mean"] == 0.5  # Row mean would be 0.25.
        assert metric["delta_vs_zero"] == 0.5
        assert metric["delta_ci95"] == [0.0, 1.0]
    assert result["delays"]["0"]["metrics"]["dn5"]["delta_ci95"] == [0.0, 0.0]


@pytest.mark.parametrize("delays", [(15, 30), (0, 5), (0, 15, 15), (0, -15), (0, 255), (False, 15)])
def test_invalid_diagnostic_grid(delays):
    with pytest.raises(ValueError):
        evaluator._validate_options(delays, 100, 2, 42)


@pytest.mark.parametrize("tree", ["recommend_snapshots", "recommend_score_labels", "recommend_receipts"])
def test_write_protects_default_inputs_even_when_custom_roots_are_supplied(tmp_path, monkeypatch, tree):
    monkeypatch.setattr(evaluator, "_ROOT", tmp_path)
    parent = tmp_path / "output" / tree / "2026-09-01"
    parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="canonical input"):
        evaluator.write_report_exclusive({}, parent / "open_r1.json", protected_roots=())
    assert list(parent.iterdir()) == []


def test_write_is_exclusive_and_protects_inputs(tmp_path):
    report = {"test": True}
    target = tmp_path / "report.json"
    evaluator.write_report_exclusive(report, target, protected_roots=())
    with pytest.raises(FileExistsError):
        evaluator.write_report_exclusive({"changed": True}, target, protected_roots=())
    assert json.loads(target.read_text()) == report
    with pytest.raises(ValueError):
        evaluator.write_report_exclusive(report, tmp_path / "new.json", protected_roots=(tmp_path,))
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        evaluator.write_report_exclusive(report, link, protected_roots=())


def test_missing_snapshot_root_is_explicit_unavailable(tmp_path):
    report = evaluator.evaluate_entry_delay(
        snapshot_root=tmp_path / "missing", label_root=tmp_path / "labels",
        receipt_root=tmp_path / "receipts", db_path=tmp_path / "must_not_create.db",
        start_date=date(2026, 8, 1), end_date=date(2026, 8, 2),
    )
    assert report["status"] == "insufficient"
    assert report["exclusion_counts"] == {"snapshot_root_missing": 1}
    assert not (tmp_path / "must_not_create.db").exists()


@pytest.mark.parametrize("clock,expected", [
    ("2026-08-02T09:35:00+09:00", 0),
    ("2026-08-02T09:35:00", 2),
])
def test_cli_analysis_clock(tmp_path, monkeypatch, capsys, clock, expected):
    options, _, _, _ = make_case(tmp_path, monkeypatch)
    args = ["evaluate_recommend_entry_delay.py", "--start", "2026-08-01",
            "--end", "2026-08-01", "--now", clock, "--slot", "open"]
    for option, key in (("--snapshot-root", "snapshot_root"), ("--label-root", "label_root"),
                        ("--receipt-root", "receipt_root"), ("--db", "db_path")):
        args.extend((option, str(options[key])))
    monkeypatch.setattr(evaluator.sys, "argv", args)
    assert evaluator.main() == expected
    captured = capsys.readouterr()
    if expected == 0:
        report = json.loads(captured.out)
        assert report["exclusion_counts"] == {"latest_delay_not_mature": 1}
    else:
        assert "timezone-aware" in captured.err
