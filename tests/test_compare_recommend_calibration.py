from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import scripts.compare_recommend_calibration as cli


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("generator.py",))
    source = tmp_path / "generator.py"
    source.write_text("# fixed\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"evidence":true}')
    paths = {key: tmp_path / f"{key}.json" for key in ("bundle", "reference", "design", "output")}
    paths["bundle"].write_text('{"bundle":true}')
    monkeypatch.setattr(cli, "BUNDLE_SHA256", cli.sha256_file(paths["bundle"]))
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "fixture")
    old_design = {"old_design": True}
    monkeypatch.setattr(cli, "reference_design", lambda: old_design)
    training = pd.DataFrame({"x": [np.nan, .09999999999999998], "coin": ["KRW-A", "KRW-B"]})
    prediction = pd.DataFrame({"date": ["2026-08-01", "2026-08-02"], "snapshot_id": ["a", "b"],
                               "rank": [1, 1], "slot": ["preopen", "preopen"],
                               "mfe": [.10000000000000009, .09999999999999998]})
    bound = cli._capture({"evidence": evidence})
    bundle = {"bundle_payload_sha256": "fixed_bundle_payload", "training": cli._encode_frame(training),
              "prediction": cli._encode_frame(prediction), "generator_manifest": bound, "data_provenance": bound,
              "data_audit": {"fixture": True}, "decision_metadata": {"a": {"date": "2026-08-01"}}}
    reference = cli._seal({
        "design": old_design, "generator_manifest": bound, "data_provenance": bound,
        "bundle_file_sha256": cli.BUNDLE_SHA256, "bundle_payload_sha256": bundle["bundle_payload_sha256"],
        "versions": {name: "fixture" for name in cli.PACKAGES}, "predictions": cli._records(prediction),
    }, "report_payload_sha256")
    paths["reference"].write_text(json.dumps(reference))
    monkeypatch.setattr(cli, "REFERENCE_SHA256", cli.sha256_file(paths["reference"]))
    paths["design"].write_text(json.dumps(cli.reviewed_design()))

    def load(path, digest, design):
        cli._same(cli.sha256_file(path), digest, "bundle checksum mismatch")
        assert design == old_design
        return copy.deepcopy(bundle), training.copy(deep=True), prediction.copy(deep=True)

    monkeypatch.setattr(cli, "load_bundle", load)
    calls = []

    def experiment(train, predict, metadata, ref, *, progress):
        calls.append("run")
        out = predict.copy(deep=True)
        out["p_up10_O"] = [.2, np.nan]
        return {"configuration": copy.deepcopy(cli.EXPERIMENT_CONFIG), "predictions": out,
                "fit_count": 15, "inner_audit": [{"fit_count": 5}],
                "outer_audit": [{"fit_count": 5}, {"fit_count": 5}],
                "oof_cache": pd.DataFrame({"raw": [.012345678901234567, np.nan]}),
                "resub_cache": pd.DataFrame({"raw": [.09999999999999998]})}

    monkeypatch.setattr(cli, "run_calibration_experiment", experiment)
    monkeypatch.setattr(cli, "evaluate_calibration", lambda *a, **k: {
        "configuration": copy.deepcopy(cli.EVALUATION_CONFIG),
        "primary": {"dates": 2}, "supplemental": {"dates": 2}})
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    return {**paths, "source": source, "evidence": evidence, "calls": calls,
            "bundle_value": bundle, "training": training, "prediction": prediction}


def run(case):
    return cli.run_experiment(case["bundle"], case["reference"], case["design"])


def test_fixed_local_inputs_report_precision_and_no_default_write(case):
    before = {path: path.read_bytes() for path in cli.ROOT.iterdir()}
    report = run(case)
    assert report["fit_count"] == 15
    assert report["deployable"] is report["model_artifact_saved"] is report["live_model_changed"] is False
    assert report["is_untouched_holdout"] is False
    cli._check_seal(report, "report_payload_sha256")
    saved = json.loads(cli.canonical_json_bytes(report))["predictions"]
    assert saved[0]["mfe"] > .1 and saved[1]["mfe"] < .1
    assert saved[1]["p_up10_O"] is None
    assert report["oof_cache"]["rows"] == 2 and report["resub_cache"]["rows"] == 1
    assert before == {path: path.read_bytes() for path in cli.ROOT.iterdir()}


@pytest.mark.parametrize("target", ["bundle", "reference", "design", "evidence"])
def test_corrupt_input_blocks_before_fitting(case, target):
    case[target].write_text("changed")
    with pytest.raises(ValueError):
        run(case)
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["bundle", "reference", "design", "source", "evidence"])
def test_file_changed_during_fitting_blocks_export(case, monkeypatch, target):
    original = cli.run_calibration_experiment

    def experiment(*args, **kwargs):
        result = original(*args, **kwargs)
        case[target].write_text("changed during fitting")
        return result

    monkeypatch.setattr(cli, "run_calibration_experiment", experiment)
    with pytest.raises(ValueError, match="changed"):
        run(case)


@pytest.mark.parametrize("target", ["source", "evidence", "design"])
def test_file_changed_during_deserialization_blocks_before_fit(case, monkeypatch, target):
    original = cli.load_bundle

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        case[target].write_text("changed during deserialize")
        return result

    monkeypatch.setattr(cli, "load_bundle", load)
    with pytest.raises(ValueError, match="changed"):
        run(case)
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["training", "prediction", "metadata", "reference", "output_original", "config", "fitcount", "partialcount"])
def test_mutation_or_invalid_execution_contract_blocks(case, monkeypatch, target):
    original = cli.run_calibration_experiment

    def experiment(train, predict, metadata, reference, **kwargs):
        result = original(train, predict, metadata, reference, **kwargs)
        if target == "training":
            train.loc[0, "x"] = 42.
        elif target == "prediction":
            predict.loc[0, "mfe"] = 42.
        elif target == "metadata":
            metadata["a"]["date"] = "2026-09-09"
        elif target == "reference":
            reference["predictions"][0]["mfe"] = 42.
        elif target == "output_original":
            result["predictions"].loc[0, "mfe"] = 42.
        elif target == "config":
            result["configuration"]["silent_tuning"] = True
        elif target == "fitcount":
            result["fit_count"] = 336
        else:
            result["fit_count"] = 10
        return result

    monkeypatch.setattr(cli, "run_calibration_experiment", experiment)
    with pytest.raises(ValueError):
        run(case)


def test_evaluator_cannot_change_saved_scores(case, monkeypatch):
    original = cli.evaluate_calibration

    def evaluate(frame, **kwargs):
        result = original(frame, **kwargs)
        frame.loc[0, "p_up10_O"] = .8
        return result

    monkeypatch.setattr(cli, "evaluate_calibration", evaluate)
    with pytest.raises(AssertionError):
        run(case)


def test_change_during_serialization_blocks(case, monkeypatch):
    original = cli._encode_frame

    def encode(frame):
        if list(frame.columns) == ["raw"]:
            case["evidence"].write_text("changed while serializing cache")
        return original(frame)

    monkeypatch.setattr(cli, "_encode_frame", encode)
    with pytest.raises(ValueError, match="changed"):
        run(case)


@pytest.mark.parametrize("kind", ["versions", "original_rows", "old_design", "bundle_payload", "seal"])
def test_rebound_reference_internal_contract_checked(case, monkeypatch, kind):
    ref = json.loads(case["reference"].read_text())
    ref.pop("report_payload_sha256")
    if kind == "versions":
        ref["versions"]["xgboost"] = "changed"
    elif kind == "original_rows":
        ref["predictions"][0]["mfe"] = 999.
    elif kind == "old_design":
        ref["design"] = {"different": True}
    elif kind == "bundle_payload":
        ref["bundle_payload_sha256"] = "changed"
    ref = cli._seal(ref, "report_payload_sha256")
    if kind == "seal":
        ref["report_payload_sha256"] = "changed"
    case["reference"].write_text(json.dumps(ref))
    monkeypatch.setattr(cli, "REFERENCE_SHA256", cli.sha256_file(case["reference"]))
    case["design"].write_text(json.dumps(cli.reviewed_design()))
    with pytest.raises(ValueError):
        run(case)
    assert case["calls"] == []


def test_cli_new_only_and_failure_no_publication(case, capsys):
    args = ["--bundle", str(case["bundle"]), "--reference", str(case["reference"]),
            "--design", str(case["design"]), "--output", str(case["output"])]
    assert cli.main(args) == 0
    digest = cli.sha256_file(case["output"])
    assert cli.main(args) == 2
    assert cli.sha256_file(case["output"]) == digest
    assert len(case["calls"]) == 1
    args[-1] = str(cli.ROOT / "new-output.json")
    case["reference"].write_text("broken")
    assert cli.main(args) == 2
    assert not (cli.ROOT / "new-output.json").exists()
