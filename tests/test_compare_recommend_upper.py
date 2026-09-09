from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.compare_recommend_upper as cli
from ops.artifact_provenance import file_set_identity, sha256_file


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("generator.py",))
    source = tmp_path / "generator.py"
    source.write_text("# fixed generator\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"fixed": true}')
    input_path = tmp_path / "readiness.json"
    input_path.write_text('{"input": true}')
    design_path = tmp_path / "design.json"
    design = {
        "schema": cli.DESIGN_SCHEMA, "design_id": cli.DESIGN_ID,
        "scope": "offline_comparison_only", "input_sha256": sha256_file(input_path),
        "model_config": copy.deepcopy(cli.EXPERIMENT_CONFIG),
        "statistics": {"n_boot": 1000, "seed": 42},
    }
    design_path.write_text(json.dumps(design))
    frame = pd.DataFrame([{"date": "2026-09-01", "p_up10_A": .1,
                           "p_up10_B": .2, "p_up10_C": float("nan")}])
    dataset = {
        "frame": frame, "folds": [], "summary": {"rows": 1},
        "provenance": {"files": file_set_identity({"evidence": evidence}, root=tmp_path)},
    }
    monkeypatch.setattr(cli, "load_experiment_data", lambda path: dataset)
    experiment = {"predictions": frame, "fold_audit": [], "parity": {"checked": 1},
                  "config": copy.deepcopy(cli.EXPERIMENT_CONFIG)}
    calls = []

    def run(frame, folds):
        calls.append("fit")
        return experiment

    monkeypatch.setattr(cli, "run_upper_comparison", run)
    monkeypatch.setattr(cli, "evaluate_comparison", lambda *a, **k: {"status": "insufficient"})
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    return {"input": input_path, "design_path": design_path, "design": design,
            "evidence": evidence, "source": source, "experiment": experiment, "calls": calls}


def test_runtime_is_not_deployable_and_default_does_not_write(case, tmp_path, capsys):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = cli.run_experiment(case["input"], case["design_path"])
    assert report["deployable"] is report["live_model_changed"] is False
    assert report["is_untouched_holdout"] is report["model_artifact_saved"] is False
    assert report["predictions"][0]["p_up10_C"] is None
    digest = report.pop("report_payload_sha256")
    assert cli.sha256_bytes(cli.canonical_json_bytes(report)) == digest
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design_path"])]) == 0
    assert json.loads(capsys.readouterr().out)["promotion_status"] == "NOT_EVALUATED"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_prediction_export_preserves_near_barrier_float_precision(case):
    value = .09999999999999998
    case["experiment"]["predictions"]["mfe"] = value
    report = cli.run_experiment(case["input"], case["design_path"])
    exported = json.loads(cli.canonical_json_bytes(report))
    assert exported["predictions"][0]["mfe"] == value
    assert exported["predictions"][0]["mfe"] < .10
    assert exported["predictions"][0]["p_up10_C"] is None


@pytest.mark.parametrize("mutation", ["input_hash", "model_config", "statistics", "scope", "schema"])
def test_unreviewed_design_is_rejected_before_fitting(case, mutation):
    design = case["design"]
    if mutation == "input_hash":
        design["input_sha256"] = "0" * 64
    elif mutation == "model_config":
        design["model_config"]["new_setting"] = True
    elif mutation == "statistics":
        design["statistics"]["seed"] = 123
    else:
        design[mutation] = "unapproved"
    case["design_path"].write_text(json.dumps(design))
    with pytest.raises(ValueError):
        cli.run_experiment(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["evidence", "source", "input", "design_path"])
def test_changes_during_fitting_block_the_report(case, monkeypatch, target):
    def change_during_fit(*args):
        case[target].write_text("changed during execution")
        return case["experiment"]

    monkeypatch.setattr(cli, "run_upper_comparison", change_during_fit)
    with pytest.raises(ValueError, match="changed"):
        cli.run_experiment(case["input"], case["design_path"])


def test_existing_evidence_change_blocks_before_fit(case):
    case["evidence"].write_text("changed before execution")
    with pytest.raises(ValueError, match="before fitting"):
        cli.run_experiment(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["source", "input", "design_path"])
def test_change_during_loading_is_rejected_before_fitting(case, monkeypatch, target):
    original = cli.load_experiment_data

    def change_during_load(path):
        dataset = original(path)
        case[target].write_text("changed during loading")
        return dataset

    monkeypatch.setattr(cli, "load_experiment_data", change_during_load)
    with pytest.raises(ValueError, match="before fitting"):
        cli.run_experiment(case["input"], case["design_path"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["source", "evidence"])
def test_change_during_final_serialization_is_rejected(case, monkeypatch, target):
    def version(name):
        case[target].write_text("changed during version serialization")
        return "fixture"

    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    with pytest.raises(ValueError, match="serialization"):
        cli.run_experiment(case["input"], case["design_path"])


def test_runtime_config_change_cannot_be_exported(case):
    case["experiment"]["config"]["silent_tuning"] = True
    with pytest.raises(ValueError, match="runtime model"):
        cli.run_experiment(case["input"], case["design_path"])


def test_cli_failure_never_publishes_a_report(case, monkeypatch, tmp_path, capsys):
    case["input"].write_text("mismatched input")
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: pytest.fail("must not export"))
    result = cli.main(["--input", str(case["input"]), "--design", str(case["design_path"]),
                       "--output", str(tmp_path / "result.json")])
    assert result == 2
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"
    assert not (tmp_path / "result.json").exists()


def test_cli_output_uses_exclusive_writer_and_protects_frozen_inputs(case, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: calls.append((a, k)))
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design_path"]),
                     "--output", "_workspace/new-report.json"]) == 0
    assert calls[0][0][0] == Path("_workspace/new-report.json")
    assert calls[0][1]["protected_roots"] == (case["input"], case["design_path"])
