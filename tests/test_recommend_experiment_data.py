"""Hermetic export rebinding tests; strict individual loaders have own suites."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from ops.artifact_provenance import file_set_identity
from ops.recommendation_evidence import EvidenceUnavailable
from signals import recommend_experiment_data as experiment
from signals import recommend_training_data as training

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _document(day: str, slot: str = "open") -> dict:
    start = datetime.fromisoformat(f"{day}T09:15:00+09:00")
    end = start + timedelta(days=1)
    decision = f"{day}T09:08:00+09:00"
    candidates = [{
        "coin": f"KRW-T{rank}", "rank": rank,
        "feature_values": {name: None if name == "f_rsi_14" else rank / 10
                           for name in training.ALLOWED_FEATURES},
        "p_up10": rank / 10, "p_dn5": rank / 20, "p_dn10": rank / 30,
        "exp_downside": -rank / 100, "rr_ratio": 2.0,
    } for rank in range(1, 5)]
    snapshot = {
        "asof": day, "slot": slot, "feature_asof": day,
        "feature_columns": list(training.ALLOWED_FEATURES),
        "decision_started_at": decision, "decision_completed_at": decision,
        "snapshot_id": f"recommend-{day}-{slot}", "payload_sha256": "a" * 64,
        "model": {"id": f"recommend_r1_{slot}"}, "universe": candidates,
        "top3": deepcopy(candidates[:3]),
    }
    rows = [{
        **deepcopy(candidate), "label_status": "labeled", "path_quality": "complete",
        "raw_bars": 96, "flat_filled_bars": 0, "actual_entry_open": 100.0,
        "mfe": .12, "mae": -.02, "eod_return_net": .0185,
        "up5": True, "up10": True, "up20": False,
        "dn3": False, "dn5": False, "dn10": False,
        "tp5_sl3_first_passage": "tp_first", "tp5_before_sl3": True,
        "tp5_sl3_return_net": .0485,
    } for candidate in candidates]
    return {
        "snapshot": snapshot,
        "label": {"rows": rows, "label_payload_sha256": "b" * 64,
                  "path_window_end": end.isoformat(), "execution_start_at": start.isoformat(),
                  "labeled_at": end.replace(hour=10, minute=0).isoformat()},
        "receipt": {"sent_at": decision},
    }


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    for name in training.GENERATOR_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture generator {name}\n", encoding="utf-8")
    documents = [_document(f"2026-09-{day:02d}") for day in (1, 3, 5)]
    documents[0]["label"]["rows"][-1]["label_status"] = "halted_no_observations"
    evidence_paths = []
    for document in documents:
        snapshot = document["snapshot"]
        paths = {role: tmp_path / role / snapshot["asof"] / f"{snapshot['slot']}_r1.json"
                 for role in ("snapshot", "label", "receipt")}
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document[role]), encoding="utf-8")
        evidence_paths.append(paths)
    calls = []

    def load(path, *, label_root, receipt_root, now):
        calls.append({"path": path, "now": now, "label_root": label_root, "receipt_root": receipt_root})
        paths = {"snapshot": path, "label": label_root / path.parent.name / path.name,
                 "receipt": receipt_root / path.parent.name / path.name}
        documents = {role: json.loads(path.read_text(encoding="utf-8")) for role, path in paths.items()}
        label, snapshot = documents["label"], documents["snapshot"]
        available = max(training._aware(label["path_window_end"], "end"),
                        training._aware(label["labeled_at"], "labeled"))
        documents["manifest"] = {
            "files": file_set_identity(paths, root=tmp_path),
            "snapshot_id": snapshot["snapshot_id"], "snapshot_payload_sha256": snapshot["payload_sha256"],
            "label_payload_sha256": label["label_payload_sha256"], "label_available_at": available.isoformat(),
        }
        return documents

    monkeypatch.setattr(training, "load_recommendation_evidence", load)
    monkeypatch.setattr(experiment, "load_recommendation_evidence", load)
    monkeypatch.setattr(training, "discover_r1_snapshots", lambda *a, **k:
                        [paths["snapshot"] for paths in evidence_paths])
    path = tmp_path / "readiness.json"

    def save(report):
        body = {key: value for key, value in report.items() if key != "report_payload_sha256"}
        report["report_payload_sha256"] = training._digest(body)
        path.write_text(json.dumps(report), encoding="utf-8")

    def rebuild():
        report = training.build_training_readiness(
            snapshot_root=tmp_path / "snapshot", label_root=tmp_path / "label",
            receipt_root=tmp_path / "receipt", start_date=date(2026, 9, 1),
            end_date=date(2026, 9, 10), now=NOW, min_train_dates=1, validation_dates=1,
        )
        save(report)
        calls.clear()
        return report

    return SimpleNamespace(root=tmp_path, path=path, paths=evidence_paths, calls=calls,
                           report=rebuild(), save=save, rebuild=rebuild)


def test_full_candidate_reconstruction_keeps_halted_rows_and_frozen_scores(archive, monkeypatch):
    monkeypatch.setattr(training, "discover_r1_snapshots", lambda *a, **k: pytest.fail("rediscovery forbidden"))
    before = {path: path.read_bytes() for path in archive.root.rglob("*") if path.is_file()}
    result = experiment.load_experiment_data(archive.path)
    frame = result["frame"]
    assert result["summary"]["rows"] == 12
    assert result["summary"]["labeled_rows"] == 11
    assert result["summary"]["unlabeled_rows"] == 1
    assert list(frame.columns) == [*training.ALLOWED_FEATURES, *experiment.METADATA_FIELDS,
                                    *experiment.BASELINE_FIELDS, *training.OUTCOME_FIELDS]
    assert set(training.ALLOWED_FEATURES).isdisjoint((*experiment.BASELINE_FIELDS,
                                                     *experiment.METADATA_FIELDS, *training.OUTCOME_FIELDS))
    halted = frame.loc[~frame.label_available].iloc[0]
    assert halted.coin == "KRW-T4" and halted.p_up10 == .4 and halted.p_dn5 == .2
    assert halted.label_status == "halted_no_observations"
    assert halted[list(training.OUTCOME_FIELDS)].isna().all()
    assert frame.f_rsi_14.isna().all()
    assert frame.was_delivered.sum() == 9
    assert result["folds"] == archive.report["folds"]
    assert result["as_of"] == NOW.isoformat()
    assert len(archive.calls) == 3 and all(call["now"] == NOW for call in archive.calls)
    assert result["provenance"]["files"]["readiness_report"]["sha256"] == result["provenance"]["report_file_sha256"]
    assert len(result["provenance"]["files"]) == 1 + len(training.GENERATOR_SOURCES) + 9
    assert before == {path: path.read_bytes() for path in archive.root.rglob("*") if path.is_file()}


def test_only_frozen_manifests_are_loaded_when_a_new_snapshot_appears(archive):
    extra = archive.root / "snapshot/2026-09-06/open_r1.json"
    extra.parent.mkdir()
    extra.write_text("not JSON", encoding="utf-8")
    assert len(experiment.load_experiment_data(archive.path)["frame"]) == 12
    assert all(call["path"] != extra for call in archive.calls)


@pytest.mark.parametrize("mutation", ["feature", "metadata", "bool_for_number", "missing_row", "duplicate_row",
                                       "fold", "summary", "halt_exclusion", "source_digest"])
def test_rehashed_but_semantically_changed_export_is_rejected(archive, mutation):
    report = archive.report
    if mutation == "feature":
        report["rows"][0]["model_features"]["f_ret_1d"] = .999
    elif mutation == "metadata":
        report["rows"][0]["metadata"]["label_available_at"] = "2026-09-01T00:00:00+00:00"
    elif mutation == "bool_for_number":
        report["rows"][0]["metadata"]["rank"] = True
    elif mutation == "missing_row":
        report["rows"].pop()
    elif mutation == "duplicate_row":
        report["rows"].append(deepcopy(report["rows"][0]))
    elif mutation == "fold":
        report["folds"][0]["train_dates"] = ["2026-09-05"]
    elif mutation == "summary":
        report["summary"]["rows"] = 12
    elif mutation == "halt_exclusion":
        report["excluded"] = []
    else:
        report["generator_sources"]["sha256"] = "0" * 64
    archive.save(report)
    with pytest.raises(experiment.ExperimentDataError):
        experiment.load_experiment_data(archive.path)


def test_unrehashed_report_mutation_fails_before_evidence_read(archive):
    archive.report["rows"].pop()
    archive.path.write_text(json.dumps(archive.report), encoding="utf-8")
    with pytest.raises(experiment.ExperimentDataError, match="checksum"):
        experiment.load_experiment_data(archive.path)
    assert not archive.calls


@pytest.mark.parametrize("role", ["snapshot", "label", "receipt"])
@pytest.mark.parametrize("mutation", ["change", "missing", "symlink"])
def test_changed_missing_or_symlinked_input_fails_closed(archive, role, mutation):
    path = archive.paths[0][role]
    if mutation == "change":
        document = json.loads(path.read_text(encoding="utf-8"))
        if role == "snapshot":
            document["universe"][0]["p_up10"] = .987
        else:
            document["changed"] = True
        path.write_text(json.dumps(document), encoding="utf-8")
    elif mutation == "missing":
        path.unlink()
    else:
        backup = archive.root / f"backup_{role}.json"
        backup.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(backup)
    with pytest.raises(experiment.ExperimentDataError):
        experiment.load_experiment_data(archive.path)


def test_generator_source_change_invalidates_export(archive):
    (archive.root / training.GENERATOR_SOURCES[0]).write_text("# changed", encoding="utf-8")
    with pytest.raises(experiment.ExperimentDataError, match="generator sources"):
        experiment.load_experiment_data(archive.path)
    assert not archive.calls


def test_frozen_cutoff_is_not_advanced_to_current_time(archive):
    archive.report["as_of"] = "2026-09-01T23:00:00+00:00"
    archive.save(archive.report)
    with pytest.raises(experiment.ExperimentDataError, match="future timestamps"):
        experiment.load_experiment_data(archive.path)
    assert archive.calls[0]["now"] == datetime.fromisoformat(archive.report["as_of"])


@pytest.mark.parametrize("kind", ["unavailable", "manifest_type_change"])
def test_strict_rebind_failures_are_fatal_not_new_exclusions(archive, monkeypatch, kind):
    original = experiment.load_recommendation_evidence

    def changed(*args, **kwargs):
        if kind == "unavailable":
            raise EvidenceUnavailable("label_not_available_as_of_now")
        evidence = original(*args, **kwargs)
        evidence["manifest"]["files"]["snapshot"]["exists"] = 1
        return evidence

    monkeypatch.setattr(experiment, "load_recommendation_evidence", changed)
    with pytest.raises(experiment.ExperimentDataError):
        experiment.load_experiment_data(archive.path)


@pytest.mark.parametrize("kind", ["label_missing", "duplicate_label", "duplicate_snapshot", "baseline_missing",
                                 "baseline_bool", "baseline_range"])
def test_reconstruction_rejects_broken_candidate_joins_and_scores(archive, monkeypatch, kind):
    original = experiment.load_recommendation_evidence

    def changed(*args, **kwargs):
        evidence = original(*args, **kwargs)
        if kind == "label_missing":
            evidence["label"]["rows"].pop()
        elif kind == "duplicate_label":
            evidence["label"]["rows"].append(deepcopy(evidence["label"]["rows"][0]))
        elif kind == "duplicate_snapshot":
            evidence["snapshot"]["universe"].append(deepcopy(evidence["snapshot"]["universe"][0]))
        elif kind == "baseline_missing":
            evidence["snapshot"]["universe"][0].pop("p_up10")
        elif kind == "baseline_bool":
            evidence["snapshot"]["universe"][0]["p_up10"] = True
        else:
            evidence["snapshot"]["universe"][0]["p_up10"] = 1.01
        return evidence

    monkeypatch.setattr(experiment, "load_recommendation_evidence", changed)
    with pytest.raises(experiment.ExperimentDataError):
        experiment.load_experiment_data(archive.path)


@pytest.mark.parametrize("target", ["source", "report", "receipt"])
def test_change_after_initial_binding_is_rejected_by_final_hash(archive, monkeypatch, target):
    original = experiment._frame_rows
    changed = False
    paths = {"source": archive.root / training.GENERATOR_SOURCES[0],
             "report": archive.path, "receipt": archive.paths[0]["receipt"]}

    def mutate(*args):
        nonlocal changed
        rows = original(*args)
        if not changed:
            path = paths[target]
            path.write_bytes(path.read_bytes() + b" ")
            changed = True
        return rows

    monkeypatch.setattr(experiment, "_frame_rows", mutate)
    with pytest.raises(experiment.ExperimentDataError, match="input files changed"):
        experiment.load_experiment_data(archive.path)


def test_empty_frozen_export_has_stable_frame_schema(archive):
    archive.paths.clear()
    archive.rebuild()
    result = experiment.load_experiment_data(archive.path)
    assert result["frame"].empty
    assert result["summary"]["rows"] == result["summary"]["labeled_rows"] == 0
    assert result["folds"] == []
    assert "p_up10" in result["frame"] and "label_available" in result["frame"]


def test_all_halted_snapshot_is_retained_without_false_training_negatives(archive):
    for paths in archive.paths:
        path = paths["label"]
        label = json.loads(path.read_text(encoding="utf-8"))
        for row in label["rows"]:
            row["label_status"] = "halted_no_observations"
        path.write_text(json.dumps(label), encoding="utf-8")
    archive.rebuild()
    result = experiment.load_experiment_data(archive.path)
    assert result["summary"]["rows"] == 12 and result["summary"]["labeled_rows"] == 0
    assert result["frame"][list(training.OUTCOME_FIELDS)].isna().all().all()
    assert not result["frame"].label_available.any()
    assert result["folds"] == []


def test_load_is_deterministic_for_frozen_files(archive):
    first = experiment.load_experiment_data(archive.path)
    second = experiment.load_experiment_data(archive.path)
    pd.testing.assert_frame_equal(first.pop("frame"), second.pop("frame"))
    assert first == second


def test_report_duplicate_json_keys_are_rejected(archive):
    raw = archive.path.read_text(encoding="utf-8")
    archive.path.write_text('{"schema":"duplicate",' + raw[1:], encoding="utf-8")
    with pytest.raises(experiment.ExperimentDataError, match="duplicate"):
        experiment.load_experiment_data(archive.path)
