from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.extend_recommend_upper as cli
from ops.artifact_provenance import file_set_identity, sha256_file


def _save(path, value):
    path.write_text(json.dumps(value))


def _seal(value):
    value.pop("report_payload_sha256", None)
    value["report_payload_sha256"] = cli.sha256_bytes(cli.canonical_json_bytes(value))
    return value


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "REFERENCE", "reference.json")
    monkeypatch.setattr(cli, "ORIGINAL_SOURCES", ("kernel.py",))
    monkeypatch.setattr(cli, "SOURCES", ("kernel.py", "extension.py"))
    source, adapter = tmp_path / "kernel.py", tmp_path / "extension.py"
    source.write_text("# old kernel\n")
    adapter.write_text("# new adapter\n")
    reference = tmp_path / "reference.json"
    _save(reference, _seal({
        "schema": "recommend_upper_comparison.v1", "design_id": "r1_upper_compare_20260907_v1",
        "configuration": copy.deepcopy(cli.EXPERIMENT_CONFIG),
        "generator_files": file_set_identity({"kernel.py": source}, root=tmp_path),
    }))
    input_path = tmp_path / "readiness.json"
    _save(input_path, {"requested": cli.SPEC["requested"]})
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"fixed": true}')
    frame = pd.DataFrame([
        {"date": str(day.date()), "slot": slot, "p_up10_A": .1,
         "p_up10_B": .2, "p_up10_C": float("nan"), "mfe": .09999999999999998}
        for day in pd.date_range("2026-09-01", "2026-09-30") for slot in ("open", "preopen")
    ])
    dataset = {
        "frame": frame, "folds": [], "summary": {"rows": len(frame)},
        "provenance": {"files": file_set_identity({"evidence": evidence}, root=tmp_path)},
    }
    experiment = {"predictions": frame, "fold_audit": [], "parity": {"checked": 60},
                  "config": copy.deepcopy(cli.EXPERIMENT_CONFIG)}
    calls = []
    evaluations = []

    def run(frame, folds):
        calls.append(frame.copy())
        return experiment

    def evaluate(frame, **kwargs):
        evaluations.append(frame.copy())
        return {"excluded": [], "coverage": {"included_date_slots": len(frame)}, "slots": {}}

    monkeypatch.setattr(cli, "load_experiment_data", lambda path: dataset)
    monkeypatch.setattr(cli, "run_upper_comparison", run)
    monkeypatch.setattr(cli, "evaluate_comparison", evaluate)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    design_path = tmp_path / "design.json"
    _save(design_path, cli.prepare_design(input_path))
    return {"input": input_path, "design": design_path, "evidence": evidence,
            "source": source, "adapter": adapter, "reference": reference,
            "dataset": dataset, "experiment": experiment,
            "calls": calls, "evaluations": evaluations}


def test_design_preparation_does_not_fit_or_load_candidates(case, monkeypatch):
    monkeypatch.setattr(cli, "load_experiment_data", lambda *a: pytest.fail("no fit at design stage"))
    report = cli.prepare_design(case["input"])
    assert report["frozen_bundle"]["reference"]["sha256"] == sha256_file(case["reference"])
    assert report["specification"]["primary_slot"] == "open"
    assert not case["calls"]


def test_only_fixed_new_window_is_evaluated_and_full_history_trains(case, tmp_path):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = cli.run_extension(case["input"], case["design"])
    assert len(case["calls"][0]) == 60
    evaluated = case["evaluations"][0]
    assert len(evaluated) == 44
    assert evaluated.date.min() == "2026-09-08"
    assert evaluated.date.max() == "2026-09-29"
    assert result["window_coverage"]["expected_date_slots"] == 44
    assert result["window_coverage"]["missing_evidence"] == []
    assert result["window_coverage"]["no_outer_prediction"] == []
    for key in ("deployable", "automatic_adoption", "is_untouched_holdout",
                "live_model_changed", "model_artifact_saved"):
        assert result[key] is False
    assert result["promotion_status"] == "NOT_EVALUATED"
    exported = json.loads(cli.canonical_json_bytes(result))
    assert exported["predictions"][0]["mfe"] == .09999999999999998
    assert exported["predictions"][0]["p_up10_C"] is None
    digest = result.pop("report_payload_sha256")
    assert digest == cli.sha256_bytes(cli.canonical_json_bytes(result))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_missing_dates_are_not_erased_or_filled_with_zero(case):
    frame = case["dataset"]["frame"]
    case["dataset"]["frame"] = frame[frame.date != "2026-09-08"].copy()
    case["experiment"]["predictions"] = frame[~frame.date.isin(["2026-09-08", "2026-09-09"])].copy()
    coverage = cli.run_extension(case["input"], case["design"])["window_coverage"]
    assert coverage["expected_date_slots"] == 44
    assert coverage["evidence_date_slots"] == 42
    assert coverage["predicted_date_slots"] == 40
    assert coverage["missing_evidence"] == [{"date": "2026-09-08", "slot": slot}
                                             for slot in ("open", "preopen")]
    assert coverage["no_outer_prediction"] == [{"date": "2026-09-09", "slot": slot}
                                               for slot in ("open", "preopen")]


@pytest.mark.parametrize("target", ["input", "source", "adapter", "reference"])
def test_changed_inputs_after_design_are_rejected_before_fit(case, target):
    case[target].write_text("changed after design")
    with pytest.raises((ValueError, KeyError)):
        cli.run_extension(case["input"], case["design"])
    assert not case["calls"]


@pytest.mark.parametrize("field", ["schema", "specification", "model_config", "frozen_bundle"])
def test_changed_design_fields_rejected_before_fit(case, field):
    design = json.loads(case["design"].read_text())
    design[field] = "unapproved"
    _save(case["design"], design)
    with pytest.raises(ValueError, match="frozen extension"):
        cli.run_extension(case["input"], case["design"])
    assert not case["calls"]


def test_original_source_changes_cannot_be_laundered_by_new_design(case):
    case["source"].write_text("changed kernel")
    with pytest.raises(ValueError, match="original study source changed"):
        cli.prepare_design(case["input"])
    assert not case["calls"]


@pytest.mark.parametrize("mutation", ["digest", "schema", "model", "history"])
def test_invalid_original_contract_or_history_blocks_preparation(case, mutation):
    if mutation == "history":
        _save(case["input"], {"requested": {**cli.SPEC["requested"], "start_date": "2026-09-01"}})
    else:
        reference = json.loads(case["reference"].read_text())
        if mutation == "digest":
            reference["report_payload_sha256"] = "0" * 64
        elif mutation == "schema":
            reference["schema"] = "other"
            _seal(reference)
        else:
            reference["configuration"]["head"]["n_estimators"] = 999
            _seal(reference)
        _save(case["reference"], reference)
    with pytest.raises(ValueError):
        cli.prepare_design(case["input"])


@pytest.mark.parametrize("target", ["input", "design", "source", "adapter", "reference", "evidence"])
@pytest.mark.parametrize("stage", ["load", "fit", "serialize"])
def test_changes_during_execution_never_publish(case, monkeypatch, target, stage):
    def change():
        case[target].write_text("changed during execution")

    if stage == "load":
        def load(path):
            change()
            return case["dataset"]
        monkeypatch.setattr(cli, "load_experiment_data", load)
    elif stage == "fit":
        def run(*args):
            change()
            return case["experiment"]
        monkeypatch.setattr(cli, "run_upper_comparison", run)
    else:
        def version(name):
            change()
            return "fixture"
        monkeypatch.setattr(cli.importlib.metadata, "version", version)
    with pytest.raises(ValueError, match="changed"):
        cli.run_extension(case["input"], case["design"])
    if stage == "load":
        assert not case["calls"]


def test_runtime_configuration_drift_rejected(case):
    case["experiment"]["config"]["tuned"] = True
    with pytest.raises(ValueError, match="runtime model"):
        cli.run_extension(case["input"], case["design"])


def test_cli_design_then_report_new_only_and_protected(case, tmp_path, capsys):
    prepared = tmp_path / "new-design.json"
    result = tmp_path / "new-result.json"
    assert cli.main(["--input", str(case["input"]), "--prepare-design", "--output", str(prepared)]) == 0
    assert cli.main(["--input", str(case["input"]), "--design", str(prepared), "--output", str(result)]) == 0
    result_bytes = result.read_bytes()
    assert cli.main(["--input", str(case["input"]), "--design", str(prepared), "--output", str(result)]) == 2
    assert result.read_bytes() == result_bytes
    assert cli.main(["--input", str(case["input"]), "--prepare-design"]) == 2
    assert "blocked" in capsys.readouterr().err


def test_cli_failure_writes_nothing(case, monkeypatch, tmp_path):
    case["input"].write_text("changed")
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: pytest.fail("no export"))
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design"]),
                     "--output", str(tmp_path / "result.json")]) == 2
    assert not (tmp_path / "result.json").exists()


def test_cli_writer_protects_original_evidence_and_sources(case, monkeypatch):
    writes = []
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: writes.append((a, k)))
    assert cli.main(["--input", str(case["input"]), "--design", str(case["design"]),
                     "--output", "_workspace/new.json"]) == 0
    protected = writes[0][1]["protected_roots"]
    assert all(case[key] in protected for key in ("input", "design", "reference", "source", "adapter"))
    assert writes[0][0][0] == Path("_workspace/new.json")
