from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.diagnose_recommend_selection as cli
from ops.artifact_provenance import file_set_identity, sha256_file


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("generator.py",))
    monkeypatch.setattr(cli, "SELECTION_CONFIG", copy.deepcopy(cli.SELECTION_CONFIG))
    monkeypatch.setattr(cli, "SLOT_CONFIG", copy.deepcopy(cli.SLOT_CONFIG))
    monkeypatch.setattr(cli, "STATISTICS", copy.deepcopy(cli.STATISTICS))
    source = tmp_path / "generator.py"
    source.write_text("# fixed source\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"evidence":true}')
    input_path = tmp_path / "readiness.json"
    input_path.write_text('{"input":true}')
    design_path = tmp_path / "design.json"
    design = {"schema": cli.DESIGN_SCHEMA, "design_id": cli.DESIGN_ID,
              "scope": cli.SCOPE, "input_sha256": sha256_file(input_path),
              "configuration": copy.deepcopy(cli._configuration())}
    design_path.write_text(json.dumps(design))
    frame = pd.DataFrame([{"date": "2026-09-01"}])
    dataset = {"frame": frame, "summary": {"rows": 1},
               "provenance": {"files": file_set_identity({"evidence": evidence}, root=tmp_path)}}
    metadata = {"snapshot-1": {"actual_latest_input_timestamp": None}}
    calls = []
    monkeypatch.setattr(cli, "load_experiment_data", lambda path: dataset)

    def load_metadata(data):
        assert data is dataset
        calls.append("metadata")
        return metadata

    def selection(data, **kwargs):
        assert data is frame
        assert kwargs == {"n_boot": 1000, "seed": 42}
        calls.append("selection")
        return {"status": "historical_diagnostics", "near_barrier": .09999999999999998}

    def slots(data, meta, **kwargs):
        assert data is frame and meta is metadata
        assert kwargs == {"n_boot": 1000, "seed": 42}
        calls.append("slots")
        return {"status": "historical_diagnostics"}

    monkeypatch.setattr(cli, "load_slot_metadata", load_metadata)
    monkeypatch.setattr(cli, "analyze_selection", selection)
    monkeypatch.setattr(cli, "analyze_slots", slots)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr("xgboost.XGBClassifier.fit", lambda *a, **k: pytest.fail("fitting forbidden"))
    return {"input": input_path, "design_path": design_path, "design": design,
            "source": source, "evidence": evidence, "dataset": dataset, "calls": calls}


def test_report_checksum_precision_no_fit_no_write(case, tmp_path, capsys):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = cli.run_diagnostics(case["input"], case["design_path"])
    assert report["model_fitted"] is report["deployable"] is report["live_model_changed"] is False
    assert report["is_untouched_holdout"] is False
    assert report["scope"] == "offline_diagnostics_only"
    assert case["calls"] == ["metadata", "selection", "slots"]
    digest = report.pop("report_payload_sha256")
    assert cli.sha256_bytes(cli.canonical_json_bytes(report)) == digest
    serialized = json.loads(cli.canonical_json_bytes(report))
    assert serialized["selection"]["near_barrier"] == .09999999999999998
    assert serialized["slot_metadata"]["snapshot-1"]["actual_latest_input_timestamp"] is None
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design_path"])]) == 0
    assert json.loads(capsys.readouterr().out)["promotion_status"] == "NOT_EVALUATED"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("field", ["schema", "design_id", "scope", "input_sha256", "configuration"])
def test_unreviewed_design_blocks_before_any_diagnostic(case, field):
    case["design"][field] = "unapproved"
    case["design_path"].write_text(json.dumps(case["design"]))
    with pytest.raises(ValueError):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("section", ["selection", "slots", "statistics"])
def test_configuration_type_and_value_are_exact(case, section):
    case["design"]["configuration"][section] = {"unapproved": True}
    case["design_path"].write_text(json.dumps(case["design"]))
    with pytest.raises(ValueError, match="configuration"):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["input", "design_path", "source", "evidence"])
def test_mutation_during_loader_blocks_before_metadata(case, monkeypatch, target):
    def load(path):
        case[target].write_text("changed during loader")
        return case["dataset"]

    monkeypatch.setattr(cli, "load_experiment_data", load)
    with pytest.raises(ValueError, match="before metadata loading"):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["input", "design_path", "source", "evidence"])
def test_mutation_during_metadata_blocks_before_analysis(case, monkeypatch, target):
    original = cli.load_slot_metadata

    def metadata(data):
        result = original(data)
        case[target].write_text("changed during metadata")
        return result

    monkeypatch.setattr(cli, "load_slot_metadata", metadata)
    with pytest.raises(ValueError, match="before analysis"):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == ["metadata"]


@pytest.mark.parametrize("target", ["input", "design_path", "source", "evidence"])
def test_mutation_during_analysis_never_returns_report(case, monkeypatch, target):
    def slots(*args, **kwargs):
        case[target].write_text("changed during analysis")
        return {"status": "completed"}

    monkeypatch.setattr(cli, "analyze_slots", slots)
    with pytest.raises(ValueError, match="during analysis or serialization"):
        cli.run_diagnostics(case["input"], case["design_path"])


@pytest.mark.parametrize("target", ["source", "evidence"])
def test_mutation_during_serialization_is_blocked(case, monkeypatch, target):
    def version(name):
        case[target].write_text("changed during serialization")
        return "fixture"

    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    with pytest.raises(ValueError, match="during analysis or serialization"):
        cli.run_diagnostics(case["input"], case["design_path"])


def test_runtime_config_mutation_blocks(case, monkeypatch):
    def slots(*args, **kwargs):
        cli.SELECTION_CONFIG["silent_change"] = 1
        return {}

    monkeypatch.setattr(cli, "analyze_slots", slots)
    with pytest.raises(ValueError, match="runtime configuration"):
        cli.run_diagnostics(case["input"], case["design_path"])


def test_existing_evidence_mutation_blocks(case):
    case["evidence"].write_text("changed before invocation")
    with pytest.raises(ValueError, match="frozen evidence"):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("stage", ["metadata", "selection", "slots"])
def test_frame_mutation_cannot_contaminate_later_diagnostics(case, monkeypatch, stage):
    name = {"metadata": "load_slot_metadata", "selection": "analyze_selection",
            "slots": "analyze_slots"}[stage]
    original = getattr(cli, name)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        case["dataset"]["frame"].loc[0, "date"] = "2026-09-02"
        return result

    monkeypatch.setattr(cli, name, mutate)
    with pytest.raises(ValueError, match="diagnostic frame changed"):
        cli.run_diagnostics(case["input"], case["design_path"])
    if stage != "slots":
        assert "slots" not in case["calls"]


def test_slot_metadata_mutation_cannot_be_exported(case, monkeypatch):
    def slots(frame, metadata, **kwargs):
        metadata["silent_change"] = True
        return {}

    monkeypatch.setattr(cli, "analyze_slots", slots)
    with pytest.raises(ValueError, match="slot metadata changed"):
        cli.run_diagnostics(case["input"], case["design_path"])


def test_cli_failure_does_not_publish(case, monkeypatch, tmp_path, capsys):
    case["input"].write_text("wrong input")
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: pytest.fail("must not publish"))
    result = cli.main(["--input", str(case["input"]), "--design", str(case["design_path"]),
                       "--output", str(tmp_path / "result.json")])
    assert result == 2
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"
    assert not (tmp_path / "result.json").exists()


def test_cli_writer_is_exclusive_and_protects_input_paths(case, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: calls.append((a, k)))
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design_path"]),
                     "--output", "_workspace/new-diagnostic.json"]) == 0
    assert calls[0][0][0] == Path("_workspace/new-diagnostic.json")
    assert calls[0][1]["protected_roots"] == (case["input"], case["design_path"])


def test_missing_source_blocks(case):
    case["source"].unlink()
    with pytest.raises(ValueError, match="missing"):
        cli.run_diagnostics(case["input"], case["design_path"])
    assert case["calls"] == []
