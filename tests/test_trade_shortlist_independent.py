"""Independent native publication -> strict canonical-evidence integration.

Only synthetic files below pytest's temporary root are created. No production
document loader is mocked: receipt/label writers receive fake transport and a
pure OHLC assessor, with nonexistent DB paths and network/DB access forbidden.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

import test_recommend_microstructure_trial as raw_fixture
import test_recommend_score_labels as label_fixture
from notifier import delivery_receipt as receipts
from ops.artifact_provenance import canonical_json_bytes, file_identity
from ops.recommendation_evidence import load_recommendation_evidence
from scripts.evaluate_recommend_trade_shortlist_trial import main
from signals import recommend_score_labels as labels
from signals import recommend_trade_shortlist_eval as evaluation
from signals import recommend_trade_shortlist_trial as publisher

DAY = "2026-09-10"
OBSERVED = datetime.fromisoformat("2026-09-11T12:00:00+09:00")


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("independent audit attempted network, DB, or fitting metadata")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(raw_fixture.snapshots, name, forbidden)


def _clock(value):
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            stamp = datetime.fromisoformat(value)
            return stamp.astimezone(tz) if tz else stamp.replace(tzinfo=None)

    return Fixed


def _case(tmp_path, monkeypatch, *, changed=True, halt=None, durable="09:10:00"):
    paths = raw_fixture._files(
        tmp_path / "inputs" / DAY, monkeypatch, day=DAY, changed=changed
    )
    root, receipt_root, label_root = (
        tmp_path / name for name in ("trials", "receipts", "labels")
    )
    publication = publisher.record_trade_shortlist_trial(
        *paths, output_root=root,
        now_fn=lambda: datetime.fromisoformat(f"{DAY}T{durable}+09:00"),
    )
    snapshot = raw_fixture.snapshots.load_snapshot(paths[0])
    sent = f"{DAY}T09:08:02+09:00"
    with monkeypatch.context() as patch:
        patch.setattr(receipts, "datetime", _clock(f"{DAY}T09:08:03+09:00"))
        receipts.write_delivery_receipt(
            snapshot, delivery_ok=True,
            attempted_at=f"{DAY}T09:08:01+09:00", sent_at=sent,
            telegram_result=label_fixture._telegram_result(
                "synthetic", datetime.fromisoformat(sent).astimezone(timezone.utc).isoformat()
            ),
            message="synthetic", root=receipt_root,
        )
    ranks = {row["coin"]: row["rank"] for row in snapshot["universe"]}

    def assessor(coin, start_at, **kwargs):
        rank = ranks[coin]
        if rank == halt:
            return label_fixture._assessment(
                [], quality="target_no_observations", raw_bars=0,
                complete=False, start=start_at,
            )
        # Rank 3 loses 6%, rank 4 gains 11%, others remain flat. All three
        # branches use valid 96-bar paths and the canonical cost calculation.
        low, high, close = (
            (94.0, 100.0, 94.0) if rank == 3
            else (100.0, 111.0, 111.0) if rank == 4
            else (100.0, 100.0, 100.0)
        )
        bars = [(100.0, high, low, close)] + [(close, close, close, close)] * 95
        return label_fixture._assessment(bars, start=start_at)

    with monkeypatch.context() as patch:
        patch.setattr(labels, "datetime", _clock("2026-09-11T10:00:00+09:00"))
        label = labels.label_recommend_snapshot(
            paths[0], output_root=label_root, receipt_root=receipt_root,
            db_path=tmp_path / "nonexistent_assessor_only.db", now=OBSERVED,
            assessor=assessor, halt_prober=lambda _coin: True,
        )
    assert label["artifact_status"] == "complete"
    return paths, root, receipt_root, label_root, publication


def _evaluate(case, **kwargs):
    paths, root, receipt_root, label_root, _ = case
    return evaluation.evaluate_trade_shortlist_trials(
        root, label_root=label_root, receipt_root=receipt_root,
        now=kwargs.pop("now", OBSERVED), n_boot=20, **kwargs,
    )


@pytest.mark.parametrize(
    ("changed", "halt", "paired", "full", "status"),
    [
        (True, None, 1, 1, "prospective_comparable"),
        (False, None, 1, 1, "prospective_comparable"),
        (True, 3, 0, 0, "unavailable"),
        (True, 4, 0, 0, "unavailable"),
        (True, 100, 1, 0, "prospective_comparable"),
    ],
)
def test_native_full_chain_preserves_selection_cost_denominators_and_all_inputs(
    tmp_path, monkeypatch, changed, halt, paired, full, status,
):
    case = _case(tmp_path, monkeypatch, changed=changed, halt=halt)
    paths, root, receipt_root, label_root, _ = case
    before = {str(path): file_identity(path, root=tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    evidence = load_recommendation_evidence(
        paths[0], label_root=label_root, receipt_root=receipt_root, now=OBSERVED,
    )
    report = _evaluate(case)
    audit = report["dates"][0]
    plan = publisher.read_trade_shortlist_record(root, DAY, OBSERVED)["plan"]
    assert audit["status"] == status
    assert audit["control_top3"] == plan["control_top3"]
    assert audit["challenger_top3"] == plan["challenger_top3"]
    assert report["primary_paired"]["n_dates"] == paired
    assert report["full_universe_common"]["n_dates"] == full
    assert report["deployable"] is False
    if paired:
        daily = report["primary_paired"]["daily"][0]
        rows = {row["coin"]: row for row in evidence["label"]["rows"]}
        for arm in ("control", "challenger"):
            for metric in ("up10", "dn5", "tp5_sl3_return_net", "eod_return_net", "mae"):
                expected = sum(rows[coin][metric] for coin in plan[f"{arm}_top3"]) / 3
                assert daily[arm][metric] == pytest.approx(expected, abs=1e-15)
        if not changed:
            assert report["primary_paired"]["n_no_op_dates"] == 1
            assert set(daily["challenger_minus_control"].values()) == {0.0}
        else:
            assert daily["challenger_minus_control"]["eod_return_net"] == pytest.approx(0.17 / 3)
    assert before == {str(path): file_identity(path, root=tmp_path) for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("durable", ["09:14:59.999999", "09:15:00", "09:15:00.000001"])
def test_native_receipt_derived_entry_is_strict_not_nominal_0900(
    tmp_path, monkeypatch, durable,
):
    report = _evaluate(_case(tmp_path, monkeypatch, durable=durable))
    audit = report["dates"][0]
    entry = datetime.fromisoformat(audit["canonical_execution_start_at"])
    assert entry == datetime.fromisoformat(f"{DAY}T09:15:00+09:00")
    expected = "prospective_comparable" if durable < "09:15:00" else "late"
    assert audit["status"] == expected


def test_native_future_maturity_is_pending_even_when_label_file_exists(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    report = _evaluate(case, now=OBSERVED - timedelta(hours=3))
    assert report["dates"][0]["status"] == "pending"
    assert report["dates"][0]["reason"] == "label_not_available_as_of_now"
    assert report["coverage"]["paired_dates"] == 0


def test_cli_corrupt_canonical_label_cannot_publish_or_hide_behind_report_status(
    tmp_path, monkeypatch, capsys,
):
    paths, root, receipts_root, label_root, _ = _case(tmp_path, monkeypatch)
    label_path = label_root / DAY / paths[0].name
    document = json.loads(label_path.read_bytes())
    document["rows"][0]["eod_return_net"] = 99.0
    label_path.write_bytes(canonical_json_bytes(document))
    before = label_path.read_bytes()
    output = tmp_path / "must_not_publish.json"
    assert main([
        "--trial-root", str(root), "--receipt-root", str(receipts_root),
        "--label-root", str(label_root), "--now", OBSERVED.isoformat(),
        "--output", str(output), "--n-boot", "20",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["status"] == "blocked"
    assert not output.exists() and label_path.read_bytes() == before


def test_missing_commit_never_reads_a_corrupt_outcome_or_rebuilds_publication(
    tmp_path, monkeypatch,
):
    case = _case(tmp_path, monkeypatch)
    paths, root, _, label_root, publication = case
    Path(publication["commit_path"]).unlink()
    label_path = label_root / DAY / paths[0].name
    label_path.write_bytes(b"corrupt synthetic label")
    before = {str(path): path.read_bytes() for path in root.rglob("*.json")}
    report = _evaluate(case)
    assert report["dates"][0]["status"] == "uncertain"
    assert report["coverage"]["paired_dates"] == 0
    assert before == {str(path): path.read_bytes() for path in root.rglob("*.json")}
