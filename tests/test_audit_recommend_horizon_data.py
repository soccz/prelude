"""Hermetic coverage-only audit: no fitting, operational IO, or target tuning."""
from __future__ import annotations

from copy import deepcopy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ops.artifact_provenance import file_set_identity, sha256_file
import scripts.audit_recommend_horizon_data as audit
from scripts.build_recommend_training_data import write_new_report
from test_recommend_slot_diagnostics import _case, _halt, _resync


def _frame_case():
    frame, metadata = _case(days=4)
    for index, name in enumerate(audit.ALLOWED_FEATURES):
        frame[name] = frame.coin.map(lambda coin: float(index) + ord(coin) / 100)
    frame[audit.ALLOWED_FEATURES[-1]] = np.nan
    return frame, metadata


def _fold(metadata):
    return [{
        "fold": 1, "status": "split_ready", "train_dates": ["2026-09-02"],
        "validation_dates": ["2026-09-04"],
        "validation_min_decision_started_at": metadata["snapshot-2026-09-04-preopen"]["decision_started_at"],
    }]


def _row(report, day="2026-09-02", coin="A"):
    return next(row for row in report["rows"] if row["date"] == day and row["coin"] == coin)


def _availability(frame, metadata, sid, stamp):
    value = pd.Timestamp(stamp).isoformat()
    metadata[sid]["label_created_at"] = value
    metadata[sid]["label_available_at"] = value
    frame.loc[frame.snapshot_id.eq(sid), "label_available_at"] = value


@pytest.fixture
def case():
    return _frame_case()


@pytest.fixture
def runtime(tmp_path, monkeypatch, case):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "SOURCES", ("generator.py",))
    paths = {name: tmp_path / filename for name, filename in (
        ("source", "generator.py"), ("evidence", "evidence.json"),
        ("input", "readiness.json"), ("design_path", "design.json"),
    )}
    paths["source"].write_text("# frozen generator\n")
    paths["evidence"].write_text('{"frozen": true}')
    paths["input"].write_text('{"readiness": true}')
    design = {
        "schema": audit.DESIGN_SCHEMA, "design_id": audit.DESIGN_ID,
        "scope": "read_only_horizon_coverage", "input_sha256": sha256_file(paths["input"]),
        "configuration": deepcopy(audit.CONFIG),
    }
    paths["design_path"].write_text(json.dumps(design))
    frame, metadata = case
    dataset = {
        "frame": frame, "folds": _fold(metadata), "summary": {"rows": len(frame)},
        "provenance": {"files": file_set_identity({"evidence": paths["evidence"]}, root=tmp_path)},
    }
    calls = []

    def load(path):
        assert path == paths["input"]
        calls.append("load_frozen_data")
        return dataset

    monkeypatch.setattr(audit, "load_experiment_data", load)
    monkeypatch.setattr(audit, "load_slot_metadata", lambda data: metadata)
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("DB access forbidden"))
    return {**paths, "design": design, "dataset": dataset, "metadata": metadata, "calls": calls}


def test_preserves_all_candidates_including_nonselected_and_current_halt(case):
    frame, metadata = case
    _halt(frame, frame.date.eq("2026-09-02") & frame.slot.eq("preopen") & frame.coin.eq("D"))
    report = audit.analyze_coverage(frame, metadata, [])
    assert report["candidate_rows"] == 16 and report["dates"] == 4
    assert report["current_labeled"] == 15 and report["paired_rows"] == 10
    assert sum(row["was_delivered"] for row in report["rows"]) == 12
    current = _row(report, coin="D")
    assert current["current_target_status"] == "halted_no_observations"
    assert current["control_status"] == "available"
    assert current["pair_available"] is False and current["pair_available_at"] is None
    assert report["training_ready"] is False
    json.dumps(report, allow_nan=False)


def test_exact_previous_calendar_day_never_uses_older_day_or_prior_open(case):
    frame, metadata = case
    removed = "snapshot-2026-09-02-preopen"
    frame = frame.loc[~frame.snapshot_id.eq(removed)].copy()
    metadata.pop(removed)
    report = audit.analyze_coverage(frame, metadata, [])
    target = _row(report, day="2026-09-03")
    assert target["control_date"] == "2026-09-02"
    assert target["control_status"] == "missing_prior_preopen_snapshot"
    assert target["control_snapshot_id"] is None and target["pair_available"] is False
    assert target["prior_open_candidate_exists"] is True
    assert target["prior_open_features_exactly_equal"] is True
    assert report["candidate_rows"] == 12


def test_absent_prior_coin_is_reported_without_replacement(case):
    frame, metadata = case
    sid = "snapshot-2026-09-01-preopen"
    frame.loc[frame.snapshot_id.eq(sid) & frame.coin.eq("D"), "coin"] = "X"
    _resync(frame, metadata, sid)
    report = audit.analyze_coverage(frame, metadata, [])
    target = _row(report, coin="D")
    assert target["control_status"] == "coin_absent_from_prior_preopen_universe"
    assert target["control_snapshot_id"] == sid and target["control_rank"] is None
    assert target["prior_open_candidate_exists"] is True
    assert target["pair_available"] is False and report["candidate_rows"] == 16


def test_halted_prior_targets_are_not_missing_snapshots_or_zero_outcomes(case):
    frame, metadata = case
    _halt(frame, frame.snapshot_id.eq("snapshot-2026-09-01-preopen"))
    report = audit.analyze_coverage(frame, metadata, [])
    day = next(row for row in report["daily"] if row["date"] == "2026-09-02")
    assert day["control_status_counts"] == {"prior_target_halted_no_observations": 4}
    assert day["control_available"] == day["pair_available"] == 0
    assert day["current_labeled"] == 4 and report["candidate_rows"] == 16


@pytest.mark.parametrize("field", ["execution_start_at", "outcome_end_at"])
def test_contradictory_prior_or_current_window_is_a_hard_failure(case, field):
    frame, metadata = case
    sid = "snapshot-2026-09-01-preopen"
    metadata[sid][field] = (pd.Timestamp(metadata[sid][field]) + pd.Timedelta(minutes=15)).isoformat()
    if field in frame:
        frame.loc[frame.snapshot_id.eq(sid), field] = metadata[sid][field]
    with pytest.raises(ValueError, match="entry time|horizon|window"):
        audit.analyze_coverage(frame, metadata, [])


def test_control_end_after_own_decision_is_not_training_lookahead(case):
    frame, metadata = case
    report = audit.analyze_coverage(frame, metadata, _fold(metadata))
    target = _row(report)
    own_decision = metadata[target["snapshot_id"]]["decision_started_at"]
    assert pd.Timestamp(target["control_window_end"]) > pd.Timestamp(own_decision)
    assert pd.Timestamp(target["pair_available_at"]) > pd.Timestamp(own_decision)
    assert target["control_window_start"] == "2026-09-01T09:00:00+09:00"
    assert target["control_window_end"] == "2026-09-02T09:00:00+09:00"
    assert report["folds"][0]["pair_available_before_validation"] == 4
    assert report["folds"][0]["validation_candidates_unchanged"] == 4


@pytest.mark.parametrize("seconds,eligible", [(-1, 4), (0, 0), (1, 0)])
def test_pair_availability_must_strictly_precede_validation(case, seconds, eligible):
    frame, metadata = case
    folds = _fold(metadata)
    boundary = pd.Timestamp(folds[0]["validation_min_decision_started_at"])
    _availability(frame, metadata, "snapshot-2026-09-02-preopen", boundary + pd.Timedelta(seconds=seconds))
    report = audit.analyze_coverage(frame, metadata, folds)
    assert report["folds"][0]["pair_available"] == 4
    assert report["folds"][0]["pair_available_before_validation"] == eligible
    assert report["folds"][0]["validation_candidates_unchanged"] == 4


def test_pair_availability_uses_later_of_both_labels(case):
    frame, metadata = case
    late = "2026-09-05T11:00:00+09:00"
    _availability(frame, metadata, "snapshot-2026-09-01-preopen", late)
    report = audit.analyze_coverage(frame, metadata, _fold(metadata))
    assert pd.Timestamp(_row(report)["pair_available_at"]) == pd.Timestamp(late)
    assert report["folds"][0]["pair_available_before_validation"] == 0


def test_exact_features_include_all_24_fields_and_normalize_nulls_only(case):
    frame, metadata = case
    assert len(audit.ALLOWED_FEATURES) == 24
    last = audit.ALLOWED_FEATURES[-1]
    frame[last] = frame[last].astype(object)
    current = frame.snapshot_id.eq("snapshot-2026-09-02-preopen") & frame.coin.eq("A")
    prior_open = frame.snapshot_id.eq("snapshot-2026-09-01-open") & frame.coin.eq("A")
    frame.loc[current, last] = None
    baseline = audit.analyze_coverage(frame, metadata, [])
    assert baseline["prior_open_exact_feature_rows"] == 12
    digest = _row(baseline)["feature_values_sha256"]
    for name in audit.ALLOWED_FEATURES:
        original = frame.loc[prior_open, name].iloc[0]
        frame.loc[prior_open, name] = 1.0 if pd.isna(original) else np.nextafter(float(original), np.inf)
        changed = audit.analyze_coverage(frame, metadata, [])
        assert _row(changed)["prior_open_features_exactly_equal"] is False, name
        assert _row(changed)["feature_values_sha256"] == digest
        frame.loc[prior_open, name] = original
    assert audit.analyze_coverage(frame, metadata, []) == baseline


@pytest.mark.parametrize("invalid", ["overlap", "training_decision_after_boundary", "invented_boundary"])
def test_invalid_folds_fail_closed(case, invalid):
    frame, metadata = case
    folds = _fold(metadata)
    if invalid == "overlap":
        folds[0]["validation_dates"].append("2026-09-02")
    elif invalid == "invented_boundary":
        folds[0]["validation_min_decision_started_at"] = metadata["snapshot-2026-09-02-preopen"]["decision_started_at"]
    else:
        folds[0]["train_dates"] = ["2026-09-04"]
        folds[0]["validation_dates"] = ["2026-09-02"]
        folds[0]["validation_min_decision_started_at"] = metadata["snapshot-2026-09-02-preopen"]["decision_started_at"]
    with pytest.raises(ValueError, match="overlap|not before validation|boundary differs"):
        audit.analyze_coverage(frame, metadata, folds)


@pytest.mark.parametrize("status,available", [("labeled", False), ("halted_no_observations", True)])
def test_label_availability_status_contradictions_fail_even_outside_preopen(case, status, available):
    frame, metadata = case
    mask = frame.snapshot_id.eq("snapshot-2026-09-04-open") & frame.coin.eq("A")
    frame.loc[mask, "label_status"] = status
    frame.loc[mask, "label_available"] = available
    with pytest.raises(ValueError, match="label availability contradicts status"):
        audit.analyze_coverage(frame, metadata, [])


def test_inputs_are_immutable_and_report_is_shuffle_invariant(case):
    frame, metadata = case
    folds = _fold(metadata)
    before, meta_before, folds_before = frame.copy(deep=True), deepcopy(metadata), deepcopy(folds)
    expected = audit.analyze_coverage(frame, metadata, folds)
    shuffled = frame.sample(frac=1, random_state=42)
    assert audit.analyze_coverage(shuffled, dict(reversed(list(metadata.items()))), folds) == expected
    pd.testing.assert_frame_equal(frame, before)
    assert metadata == meta_before and folds == folds_before


def test_valid_class_and_return_values_do_not_select_coverage(case):
    frame, metadata = case
    expected = audit.analyze_coverage(frame, metadata, _fold(metadata))
    frame["up10"], frame["dn5"] = False, False
    frame["mfe"], frame["mae"] = .01, -.01
    frame["eod_return_net"] = frame["tp5_sl3_return_net"] = -.0015
    assert audit.analyze_coverage(frame, metadata, _fold(metadata)) == expected
    frame.loc[frame.snapshot_id.eq("snapshot-2026-09-01-preopen"), "label_status"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported .*label status"):
        audit.analyze_coverage(frame, metadata, [])


def test_runtime_and_stdout_are_read_only_no_fit_db_or_network(runtime, tmp_path, monkeypatch, capsys):
    import xgboost

    for model_class in (xgboost.XGBModel, xgboost.XGBClassifier, xgboost.XGBRegressor):
        monkeypatch.setattr(model_class, "fit", lambda *a, **k: pytest.fail("fit forbidden"))
    monkeypatch.setattr(xgboost, "train", lambda *a, **k: pytest.fail("fit forbidden"))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    report = audit.run_audit(runtime["input"], runtime["design_path"])
    assert report["model_fitted"] is report["deployable"] is report["live_model_changed"] is False
    assert report["is_untouched_holdout"] is False
    assert report["promotion_status"] == "NOT_EVALUATED"
    checksum = report.pop("report_payload_sha256")
    assert audit._digest(report) == checksum
    assert audit.main(["--input", str(runtime["input"]), "--design", str(runtime["design_path"])]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["candidate_rows"] == 16 and summary["training_ready"] is False
    assert not {"rows", "daily", "folds", "limitations"} & summary.keys()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("field", ["schema", "input_sha256", "configuration"])
def test_unreviewed_design_blocks_before_loading(runtime, field):
    runtime["design"][field] = "unreviewed"
    runtime["design_path"].write_text(json.dumps(runtime["design"]))
    with pytest.raises(ValueError, match="unreviewed audit design"):
        audit.run_audit(runtime["input"], runtime["design_path"])
    assert runtime["calls"] == []


def test_evidence_already_changed_blocks_before_metadata_read(runtime, monkeypatch):
    runtime["evidence"].write_text("changed since frozen provenance")
    monkeypatch.setattr(audit, "load_slot_metadata", lambda data: pytest.fail("must fail before metadata"))
    with pytest.raises(ValueError, match="frozen evidence changed"):
        audit.run_audit(runtime["input"], runtime["design_path"])


@pytest.mark.parametrize("target", ["source", "input", "design_path", "evidence"])
def test_change_during_loader_blocks_before_metadata(runtime, monkeypatch, target):
    def load(path):
        runtime[target].write_text("changed during frozen data load")
        return runtime["dataset"]

    monkeypatch.setattr(audit, "load_experiment_data", load)
    monkeypatch.setattr(audit, "load_slot_metadata", lambda data: pytest.fail("must fail before metadata"))
    with pytest.raises(ValueError, match="changed"):
        audit.run_audit(runtime["input"], runtime["design_path"])


@pytest.mark.parametrize("target", ["source", "evidence"])
def test_change_during_final_serialization_blocks_return(runtime, monkeypatch, target):
    original = audit._digest

    def digest(value):
        if isinstance(value, dict) and "generator_files" in value:
            runtime[target].write_text("changed while assembling final report")
        return original(value)

    monkeypatch.setattr(audit, "_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        audit.run_audit(runtime["input"], runtime["design_path"])


@pytest.mark.parametrize("target", ["frame", "metadata", "folds"])
def test_runtime_input_mutation_cannot_be_exported(runtime, monkeypatch, target):
    if target in {"frame", "folds"}:
        def load(data):
            if target == "frame":
                data["frame"].loc[0, audit.ALLOWED_FEATURES[0]] = 123.0
            else:
                data["folds"][0]["train_dates"].append("2026-09-01")
            return runtime["metadata"]

        monkeypatch.setattr(audit, "load_slot_metadata", load)
    else:
        original = audit.analyze_coverage

        def analyze(frame, metadata, folds):
            result = original(frame, metadata, folds)
            metadata["snapshot-2026-09-01-preopen"]["model_id"] = "mutated"
            return result

        monkeypatch.setattr(audit, "analyze_coverage", analyze)
    with pytest.raises(ValueError, match=f"audit (input )?{target} changed"):
        audit.run_audit(runtime["input"], runtime["design_path"])


def test_cli_writes_only_new_file_and_protects_original_inputs(runtime, tmp_path, monkeypatch, capsys):
    calls = []

    def writer(path, report, *, protected_roots):
        calls.append(protected_roots)
        write_new_report(path, report, root=tmp_path, protected_roots=protected_roots)

    monkeypatch.setattr(audit, "write_new_report", writer)
    args = ["--input", str(runtime["input"]), "--design", str(runtime["design_path"])]
    target = tmp_path / "new-report.json"
    assert audit.main([*args, "--output", str(target)]) == 0
    before = target.read_bytes()
    assert json.loads(before)["audit"]["candidate_rows"] == 16
    assert calls == [(runtime["input"], runtime["design_path"])]
    assert audit.main([*args, "--output", str(target)]) == 2
    assert target.read_bytes() == before
    for original in (runtime["input"], runtime["design_path"]):
        original_bytes = original.read_bytes()
        assert audit.main([*args, "--output", str(original)]) == 2
        assert original.read_bytes() == original_bytes
    assert all(json.loads(line)["status"] == "blocked" for line in capsys.readouterr().err.splitlines())


def test_cli_failure_never_calls_writer(runtime, tmp_path, monkeypatch, capsys):
    runtime["input"].write_text("changed input")
    monkeypatch.setattr(audit, "write_new_report", lambda *a, **k: pytest.fail("publication forbidden"))
    target = tmp_path / "must-not-exist.json"
    assert audit.main(["--input", str(runtime["input"]), "--design", str(runtime["design_path"]),
                       "--output", str(target)]) == 2
    assert not target.exists()
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"
