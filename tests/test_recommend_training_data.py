from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_recommend_training_data as cli
import signals.recommend_training_data as training
from ops.recommendation_evidence import EvidenceError, EvidenceUnavailable

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _evidence(day: str = "2026-09-01", slot: str = "open", count: int = 4) -> dict:
    d = date.fromisoformat(day)
    feature_day = d if slot == "open" else d - timedelta(days=1)
    decision = f"{day}T{'09:08' if slot == 'open' else '08:53'}:00+09:00"
    start = f"{day}T{'09:15' if slot == 'open' else '09:00'}:00+09:00"
    end = datetime.fromisoformat(start) + timedelta(days=1)
    candidates = [{"coin": f"KRW-T{rank}", "rank": rank,
                   "feature_values": {name: (None if name == "f_rsi_14" else rank / 10)
                                      for name in training.ALLOWED_FEATURES}}
                  for rank in range(1, count + 1)]
    snapshot_id = f"recommend-{day}-{slot}"
    snapshot = {
        "asof": day, "slot": slot, "feature_asof": feature_day.isoformat(),
        "feature_columns": list(training.ALLOWED_FEATURES),
        "decision_started_at": decision, "decision_completed_at": decision,
        "snapshot_id": snapshot_id, "payload_sha256": "a" * 64,
        "model": {"id": f"recommend_r1_{slot}"},
        "universe": candidates, "top3": candidates[:3],
    }
    rows = [{**deepcopy(c), "label_status": "labeled", "path_quality": "complete",
             "raw_bars": 96, "flat_filled_bars": 0,
             "actual_entry_open": 100.0, "mfe": .12, "mae": -.02,
             "eod_return_net": .0185, "up5": True, "up10": True, "up20": False,
             "dn3": False, "dn5": False, "dn10": False,
             "tp5_sl3_first_passage": "tp_first", "tp5_before_sl3": True,
             "tp5_sl3_return_net": .0485} for c in candidates]
    label = {"rows": rows, "label_payload_sha256": "b" * 64,
             "path_window_end": end.isoformat(), "execution_start_at": start,
             "labeled_at": end.replace(hour=10, minute=0).isoformat()}
    return {"snapshot": snapshot, "label": label, "receipt": {"sent_at": decision},
            "manifest": {"snapshot_id": snapshot_id, "files": {"sha256": "c" * 64}}}


def _install(monkeypatch, documents: list[dict | Exception]):
    paths = [Path(f"fixture/{i}/open_r1.json") for i in range(len(documents))]
    mapping = dict(zip(paths, documents))
    monkeypatch.setattr(training, "discover_r1_snapshots", lambda *a, **k: paths)

    def load(path, **kwargs):
        result = mapping[path]
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)

    monkeypatch.setattr(training, "load_recommendation_evidence", load)


def _build(**overrides):
    return training.build_training_readiness(
        **{"snapshot_root": "fixtures", "label_root": "fixtures", "receipt_root": "fixtures",
           "start_date": date(2026, 9, 1), "end_date": date(2026, 9, 10), "now": NOW,
           "min_train_dates": 2, "validation_dates": 2, **overrides})


def test_full_universe_retains_raw_nulls_without_scores_or_outcomes_in_features(monkeypatch):
    document = _evidence()
    before = deepcopy(document)
    _install(monkeypatch, [document])
    report = _build()
    assert report["summary"]["rows"] == 4
    assert report["summary"]["delivered_rows"] == 3
    assert report["rows"][3]["metadata"]["was_delivered"] is False
    assert report["rows"][0]["model_features"] == document["snapshot"]["universe"][0]["feature_values"]
    assert set(report["rows"][0]["model_features"]) == set(training.ALLOWED_FEATURES)
    assert report["feature_missing_counts"]["f_rsi_14"] == 4
    assert report["rows"][0]["outcomes"]["tp5_sl3_return_net"] == .0485
    assert report["deployable"] is report["model_fitted"] is False
    assert report["methodology"]["is_untouched_holdout"] is False
    assert document == before


def test_cost_metadata_uses_decimal_contract_without_second_conversion(monkeypatch):
    from ledger.config import ROUND_TRIP_COST_PCT

    document = _evidence()
    _install(monkeypatch, [document])
    report = _build()
    assert report["methodology"]["round_trip_cost_fraction"] == ROUND_TRIP_COST_PCT
    assert report["methodology"]["round_trip_cost_fraction"] == pytest.approx(.0015)
    assert report["rows"][0]["outcomes"] == {
        name: document["label"]["rows"][0][name] for name in training.OUTCOME_FIELDS
    }


@pytest.mark.parametrize("mutation", ["new_feature", "missing_feature", "nan", "infinite", "boolean"])
def test_invalid_feature_contract_blocks_whole_export(monkeypatch, mutation):
    document = _evidence()
    values = document["snapshot"]["universe"][0]["feature_values"]
    if mutation == "new_feature":
        values["f_next_return"] = .9
        document["snapshot"]["feature_columns"].append("f_next_return")
    elif mutation == "missing_feature":
        values.pop("f_rsi_14")
    else:
        values["f_ret_1d"] = {"nan": float("nan"), "infinite": float("inf"), "boolean": True}[mutation]
    _install(monkeypatch, [document])
    with pytest.raises(training.TrainingDataError):
        _build()


def test_missing_excluded_but_corrupt_evidence_stops_whole_report(monkeypatch):
    _install(monkeypatch, [_evidence(), EvidenceUnavailable("receipt_missing_delivery_unknown")])
    report = _build()
    assert report["summary"]["snapshots_included"] == 1
    assert report["excluded"][0]["reason"] == "receipt_missing_delivery_unknown"
    _install(monkeypatch, [_evidence(), EvidenceError("checksum mismatch")])
    with pytest.raises(EvidenceError, match="checksum"):
        _build()


def test_halted_rows_are_audited_not_converted_into_negative_labels(monkeypatch):
    document = _evidence()
    document["label"]["rows"][-1]["label_status"] = "halted_no_observations"
    _install(monkeypatch, [document])
    report = _build()
    assert report["summary"]["rows"] == 3
    assert report["excluded"][0]["reason"] == "halted_no_observations"


def test_duplicate_date_slot_is_not_double_counted(monkeypatch):
    _install(monkeypatch, [_evidence(), _evidence()])
    with pytest.raises(training.TrainingDataError, match="duplicate"):
        _build()


def test_source_change_blocks_report_and_provenance_is_exported(monkeypatch):
    _install(monkeypatch, [_evidence()])
    report = _build()
    assert set(report["generator_sources"]["files"]) == set(training.GENERATOR_SOURCES)
    assert len(report["generator_sources"]["sha256"]) == 64
    assert report["summary"]["by_slot"]["open"]["dates"] == 1
    assert all(item["identical_rows"] == 4 for item in report["feature_alias_diagnostics"])
    versions = iter(({"sha256": "first"}, {"sha256": "changed"}))
    monkeypatch.setattr(training, "_source_manifest", lambda: next(versions))
    with pytest.raises(training.TrainingDataError, match="sources changed"):
        _build()


def test_export_identity_is_deterministic_for_frozen_inputs(monkeypatch):
    _install(monkeypatch, [_evidence()])
    first, second = _build(), _build()
    assert first == second
    digest = first.pop("report_payload_sha256")
    assert training._digest(first) == digest


def test_global_dates_prevent_cross_slot_and_coin_contamination(monkeypatch):
    documents = [_evidence(f"2026-09-{day:02d}", slot) for day in range(1, 8)
                 for slot in ("open", "preopen")]
    # A late regenerated preopen label also removes that day's open observations.
    documents[3]["label"]["labeled_at"] = "2026-09-15T10:00:00+09:00"
    _install(monkeypatch, documents)
    report = _build()
    assert report["summary"]["ready_folds"] >= 1
    for fold in report["folds"]:
        assert not set(fold["train_dates"]) & set(fold["validation_dates"])
        assert "2026-09-02" not in fold["train_dates"]
        assert fold["train"]["rows"] == len(fold["train_dates"]) * 8
        if fold["train_max_label_available_at"]:
            assert (datetime.fromisoformat(fold["train_max_label_available_at"])
                    < datetime.fromisoformat(fold["validation_min_decision_started_at"]))
        assert fold["is_untouched_holdout"] is False
    metadata = next(r["metadata"] for r in report["rows"] if r["metadata"]["slot"] == "preopen")
    assert metadata["nominal_shift1_input_bar_date"] == "2026-08-30"
    assert metadata["actual_latest_input_timestamp"] is None


def test_availability_equality_is_purged_not_treated_as_prior(monkeypatch):
    documents = [_evidence(f"2026-09-{day:02d}") for day in range(1, 4)]
    documents[0]["label"]["labeled_at"] = documents[2]["snapshot"]["decision_started_at"]
    _install(monkeypatch, documents)
    report = _build()
    assert "2026-09-01" in report["folds"][0]["purged_train_dates"]
    assert report["status"] == "insufficient_history"


@pytest.mark.parametrize("overrides", [{"now": datetime(2026, 9, 20)}, {"min_train_dates": 0},
                                      {"validation_dates": True}, {"slots": ("open", "open")}])
def test_invalid_arguments_fail_before_evidence_loading(monkeypatch, overrides):
    monkeypatch.setattr(training, "discover_r1_snapshots", lambda *a, **k: pytest.fail("must validate first"))
    with pytest.raises(training.TrainingDataError):
        _build(**overrides)


def test_new_export_is_exclusive_and_cannot_traverse_symlinks(tmp_path):
    output = tmp_path / "new.json"
    cli.write_new_report(output, {"rows": []}, root=tmp_path)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_new_report(output, {"overwritten": True}, root=tmp_path)
    assert output.read_bytes() == before
    directory = tmp_path / "destination"
    directory.mkdir()
    (tmp_path / "alias").symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError):
        cli.write_new_report(tmp_path / "alias/new.json", {}, root=tmp_path)
    assert not list(directory.iterdir())
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError):
        cli.write_new_report(tmp_path / "../escape.json", {}, root=tmp_path)


@pytest.mark.parametrize("tree", ["recommend_snapshots", "recommend_score_labels", "recommend_receipts"])
def test_new_export_cannot_pollute_missing_canonical_inputs(tmp_path, tree):
    parent = tmp_path / "output" / tree / "2026-09-01"
    parent.mkdir(parents=True)
    destination = parent / "open_r1.json"
    with pytest.raises(ValueError, match="canonical input"):
        cli.write_new_report(destination, {}, root=tmp_path)
    assert list(parent.iterdir()) == []


def test_cli_protects_custom_input_roots(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, [_evidence()])
    parent = tmp_path / "custom_labels"
    parent.mkdir()
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    original_writer = cli.write_new_report
    monkeypatch.setattr(cli, "write_new_report", lambda path, report, **kwargs:
                        original_writer(path, report, root=tmp_path, **kwargs))
    assert cli.main([
        "--start-date", "2026-09-01", "--end-date", "2026-09-10",
        "--now", NOW.isoformat(), "--label-root", str(parent),
        "--output", str(parent / "open_r1.json"),
    ]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert list(parent.iterdir()) == []


def test_cli_default_only_prints_summary_and_invalid_evidence_never_exports(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, [_evidence()])
    args = ["--start-date", "2026-09-01", "--end-date", "2026-09-10",
            "--now", NOW.isoformat()]
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: pytest.fail("unexpected write"))
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert "rows" not in result and result["summary"]["rows"] == 4
    _install(monkeypatch, [EvidenceError("corrupt")])
    assert cli.main(args + ["--output", str(tmp_path / "report.json")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_real_snapshot_receipt_label_export_is_read_only_and_never_fits(tmp_path, monkeypatch):
    import notifier.delivery_receipt as receipts
    import signals.recommend as scorer_module
    import signals.recommend_score_labels as labels
    import signals.recommend_snapshot as snapshots
    from ledger.path_quality import PathAssessment
    from notifier.telegram import TelegramSendResult, TelegramServerMessage

    class Clock(datetime):
        instant = datetime(2026, 9, 1, 0, 8, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.instant if tz is None else cls.instant.astimezone(tz)

    monkeypatch.setattr(snapshots, "datetime", Clock)
    monkeypatch.setattr(receipts, "datetime", Clock)
    monkeypatch.setattr(labels, "datetime", Clock)
    document = _evidence()
    universe = document["snapshot"]["universe"]
    for candidate in universe:
        candidate.update(score=.5, pump_prob=.02, pump_prob_pct="2.0%", rr_ratio=1.,
                         p_up5=.3, p_up10=.1, p_up20=.02, p_dn5=.1, p_dn10=.03,
                         exp_downside=-.04, dump_risk_flag=False, btc_regime="bear_quiet",
                         entry_open=100., sl=-.03, tp=.05)

    def fake_scorer(asof, **kwargs):
        return {"asof": asof, "slot": "open", "feature_date": asof, "btc_regime": "bear_quiet",
                "universe_n": 4, "calibration_source": "bucket_score_pump20",
                "rank_basis": "R1_riskreward(de-corr head)", "n_history_dates": 100,
                "ranking": "R1", "score_schema_version": "recommend_score.v2",
                "rule_version": "r1_riskreward_v1", "model_random_seed": 42,
                "feature_columns": list(training.ALLOWED_FEATURES),
                "training": {"start": "2025-01-01", "end": "2026-08-26",
                             "cutoff_exclusive": "2026-08-27", "embargo_days": 5,
                             "rows": 1000, "dates": 100},
                "universe": universe, "top3": universe[:3]}

    snapshot = snapshots.get_or_create_recommend_snapshot(
        "2026-09-01", slot="open", root=tmp_path / "snapshots", scorer=fake_scorer)
    Clock.instant = datetime(2026, 9, 1, 0, 8, 3, tzinfo=timezone.utc)
    digest = hashlib.sha256(b"test").hexdigest()
    transport = TelegramSendResult(
        delivery_ok=True, message_sha256=digest, chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
        chunk_count=1, telegram_messages=(TelegramServerMessage(
            message_id=1, server_date="2026-09-01T00:08:02+00:00", text_sha256=digest),), error=None)
    receipts.write_delivery_receipt(
        snapshot, delivery_ok=True, attempted_at="2026-09-01T00:08:01+00:00",
        sent_at="2026-09-01T00:08:02+00:00", root=tmp_path / "receipts",
        telegram_result=transport, message="test")
    Clock.instant = datetime(2026, 9, 2, 1, tzinfo=timezone.utc)

    def assess(market, start_at, **kwargs):
        return PathAssessment(bars=[(100., 111., 98., 101.)] * 96,
                              timestamps=tuple(pd.date_range(pd.Timestamp(start_at).tz_localize(None),
                                                             periods=96, freq="15min")),
                              path_complete=True, path_quality="complete", raw_bars=96,
                              expected_bars=96, benchmark_bars=96)

    labels.label_recommend_snapshot(snapshot["snapshot_path"], output_root=tmp_path / "labels",
                                    receipt_root=tmp_path / "receipts", db_path=tmp_path / "unused.db",
                                    now=Clock.instant, assessor=assess)
    monkeypatch.setattr(scorer_module, "score_candidates", lambda *a, **k: pytest.fail("fit invoked"))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = _build(snapshot_root=tmp_path / "snapshots", label_root=tmp_path / "labels",
                    receipt_root=tmp_path / "receipts")
    after = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert report["summary"]["rows"] == 4
    assert report["summary"]["delivered_rows"] == 3
    assert report["rows"][0]["outcomes"]["tp5_sl3_return_net"] == .0485
    assert report["input_manifests"][0]["snapshot_id"] == snapshot["snapshot_id"]
    assert before == after
