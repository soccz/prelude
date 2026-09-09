from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import scripts.compare_recommend_horizon as cli
from ops.artifact_provenance import sha256_file


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("generator.py",))
    source = tmp_path / "generator.py"
    source.write_text("# fixed generator\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"frozen":true}')
    input_path, audit_path = tmp_path / "readiness.json", tmp_path / "audit.json"
    input_path.write_text('{"input":true}')
    audit_path.write_text('{"audit":true}')
    design = copy.deepcopy(cli.reviewed_design())
    design.update(input_sha256=sha256_file(input_path), daily_decisions=2, maximum_fits=20)
    monkeypatch.setattr(cli, "reviewed_design", lambda: copy.deepcopy(design))
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    frame = pd.DataFrame({name: [1.0, np.nan] for name in cli.ALLOWED_FEATURES})
    frame["date"] = ["2026-09-01", "2026-09-02"]
    frame["snapshot_id"] = ["one", "two"]
    frame["slot"] = "preopen"
    frame["rank"] = 1
    frame["mfe"] = [.09999999999999998, .10000000000000009]
    frame["nullable"] = pd.Series([None, "text"], dtype=object)
    training = pd.DataFrame({"feature_date": ["2026-08-20"], "coin": ["KRW-FIX"],
                             "B_high_ret": [.10000000000000009], "C_target_present": [True]})
    meta = {sid: {"date": day} for sid, day in zip(frame.snapshot_id, frame.date)}
    dataset = {"training_frame": training, "prediction_frame": frame,
               "decision_metadata": meta, "data_audit": {"fixture": True},
               "provenance": cli._manifest({"evidence": evidence})}
    monkeypatch.setattr(cli, "prepare_horizon_data", lambda *args: dataset)
    calls = []

    def fit(train, predict, *, decision):
        assert list(predict.columns) == list(cli.ALLOWED_FEATURES)
        calls.append(decision["date"])
        return {"scores": pd.DataFrame({"p_up10_B": [.2], "p_up10_C": [np.nan]}, index=predict.index),
                "audit": {"configuration": copy.deepcopy(cli.MODEL_CONFIG), "fit_count": 10}}

    monkeypatch.setattr(cli, "fit_daily_pair", fit)
    monkeypatch.setattr(cli, "evaluate_horizon", lambda *args, **kwargs: {
        "configuration": copy.deepcopy(cli.EVALUATION_CONFIG), "status": "fixture"})
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    return {"input": input_path, "audit": audit_path, "design": design_path,
            "source": source, "evidence": evidence, "dataset": dataset, "calls": calls,
            "bundle": tmp_path / "bundle.json", "output": tmp_path / "result.json"}


def _prepare(case):
    bundle = cli.prepare_bundle(case["input"], case["audit"], case["design"])
    cli.write_new_report(case["bundle"], bundle, root=cli.ROOT)
    return sha256_file(case["bundle"])


def test_bundle_roundtrip_and_fixed_run_without_database(case, monkeypatch):
    digest = _prepare(case)
    assert case["calls"] == []
    monkeypatch.setattr(cli, "prepare_horizon_data", lambda *a: pytest.fail("run cannot read DB"))
    before = {path: path.read_bytes() for path in cli.ROOT.iterdir()}
    report = cli.run_experiment(case["bundle"], digest, case["design"])
    assert report["fit_count"] == 20
    assert case["calls"] == ["2026-09-01", "2026-09-02"]
    assert report["deployable"] is report["live_model_changed"] is False
    assert report["model_artifact_saved"] is report["is_untouched_holdout"] is False
    cli._check_seal(report, "report_payload_sha256")
    saved = json.loads(cli.canonical_json_bytes(report))["predictions"]
    assert saved[0]["mfe"] < .1 and saved[1]["mfe"] > .1
    assert saved[0]["p_up10_C"] is None
    assert before == {path: path.read_bytes() for path in cli.ROOT.iterdir()}


def test_frame_roundtrip_preserves_nan_nullable_order_and_precision():
    frame = pd.DataFrame({"value": np.array([.09999999999999998, np.nan, -.05000000000000007]),
                          "key": ["c", "a", "b"], "flag": [True, False, True],
                          "nullable": pd.Series([None, "2026-09-01", None], dtype=object)})
    pd.testing.assert_frame_equal(frame, cli._decode_frame(cli._encode_frame(frame)), check_exact=True)


@pytest.mark.parametrize("change", ["data", "sha256", "columns", "rows", "dtypes", "encoding"])
def test_frame_corruption_blocked(change):
    encoded = cli._encode_frame(pd.DataFrame({"x": [1.0]}))
    encoded[change] = "changed" if isinstance(encoded[change], str) else []
    with pytest.raises((ValueError, TypeError)):
        cli._decode_frame(encoded)


@pytest.mark.parametrize("target", ["source", "evidence", "input", "audit", "design", "bundle"])
def test_change_before_fit_is_blocked(case, target):
    digest = _prepare(case)
    case[target].write_text("changed")
    with pytest.raises((ValueError, KeyError)):
        cli.run_experiment(case["bundle"], digest, case["design"])
    assert case["calls"] == []


@pytest.mark.parametrize("target", ["source", "evidence", "input", "audit", "design", "bundle"])
def test_change_during_fit_is_blocked(case, monkeypatch, target):
    digest = _prepare(case)
    original = cli.fit_daily_pair

    def fit(*args, **kwargs):
        result = original(*args, **kwargs)
        case[target].write_text("changed during fit")
        return result

    monkeypatch.setattr(cli, "fit_daily_pair", fit)
    with pytest.raises(ValueError, match="changed"):
        cli.run_experiment(case["bundle"], digest, case["design"])


def test_changed_design_not_a_new_silent_experiment(case):
    design = json.loads(case["design"].read_text())
    design["model_config"]["xgb_params"]["max_depth"] = 99
    case["design"].write_text(json.dumps(design))
    with pytest.raises(ValueError, match="design differs"):
        cli.prepare_bundle(case["input"], case["audit"], case["design"])


def test_bundle_internal_checksum_and_external_anchor(case):
    digest = _prepare(case)
    bundle = json.loads(case["bundle"].read_text())
    bundle["decision_metadata"]["one"]["date"] = "2026-09-02"
    case["bundle"].write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="whole-file"):
        cli.run_experiment(case["bundle"], digest, case["design"])
    with pytest.raises(ValueError, match="payload_sha256"):
        cli.run_experiment(case["bundle"], sha256_file(case["bundle"]), case["design"])


@pytest.mark.parametrize("mutation", ["index", "collision", "fit_count", "configuration", "train", "metadata"])
def test_broken_model_contract_blocked(case, monkeypatch, mutation):
    digest = _prepare(case)
    original = cli.fit_daily_pair

    def fit(train, predict, **kwargs):
        result = original(train, predict, **kwargs)
        if mutation == "index":
            result["scores"].index = [999]
        elif mutation == "collision":
            result["scores"]["mfe"] = .2
        elif mutation == "fit_count":
            result["audit"]["fit_count"] = 11
        elif mutation == "configuration":
            result["audit"]["configuration"]["silent_tuning"] = True
        elif mutation == "metadata":
            kwargs["decision"]["date"] = "2026-09-09"
        else:
            train.loc[0, "B_high_ret"] = .9
        return result

    monkeypatch.setattr(cli, "fit_daily_pair", fit)
    with pytest.raises(ValueError):
        cli.run_experiment(case["bundle"], digest, case["design"])


def test_evaluator_cannot_mutate_exported_predictions(case, monkeypatch):
    digest = _prepare(case)

    def evaluate(frame, **kwargs):
        frame.loc[0, "mfe"] = .9
        return {"configuration": cli.EVALUATION_CONFIG}

    monkeypatch.setattr(cli, "evaluate_horizon", evaluate)
    with pytest.raises(AssertionError):
        cli.run_experiment(case["bundle"], digest, case["design"])


def test_evidence_change_during_decode_blocks_before_fit(case, monkeypatch):
    digest = _prepare(case)
    original = cli._decode_frame

    def decode(*args):
        result = original(*args)
        case["evidence"].write_text("changed during decode")
        return result

    monkeypatch.setattr(cli, "_decode_frame", decode)
    with pytest.raises(ValueError, match="changed"):
        cli.run_experiment(case["bundle"], digest, case["design"])
    assert case["calls"] == []


def test_prepare_mutation_and_final_serialization_mutation_blocked(case, monkeypatch):
    original = cli.prepare_horizon_data

    def load(*args):
        result = original(*args)
        case["source"].write_text("source changed during preparation")
        return result

    monkeypatch.setattr(cli, "prepare_horizon_data", load)
    with pytest.raises(ValueError, match="changed"):
        cli.prepare_bundle(case["input"], case["audit"], case["design"])
    monkeypatch.setattr(cli, "prepare_horizon_data", original)
    digest = _prepare(case)

    def version(name):
        case["evidence"].write_text("changed during serialization")
        return "fixture"

    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    with pytest.raises(ValueError, match="changed"):
        cli.run_experiment(case["bundle"], digest, case["design"])


def test_cli_exclusive_output_and_corruption_no_publication(case, capsys):
    args = ["prepare", "--input", str(case["input"]), "--audit", str(case["audit"]),
            "--design", str(case["design"]), "--output", str(case["bundle"])]
    assert cli.main(args) == 0
    digest = sha256_file(case["bundle"])
    assert cli.main(args) == 2
    assert sha256_file(case["bundle"]) == digest
    assert case["calls"] == []
    assert cli.main(["run", "--bundle", str(case["bundle"]), "--bundle-sha256", "0" * 64,
                     "--design", str(case["design"]), "--output", str(case["output"])]) == 2
    assert not case["output"].exists()


def test_run_cli_success_records_only_new_report(case, capsys):
    digest = _prepare(case)
    assert cli.main(["run", "--bundle", str(case["bundle"]), "--bundle-sha256", digest,
                     "--design", str(case["design"]), "--output", str(case["output"])]) == 0
    assert json.loads(case["output"].read_text())["fit_count"] == 20
    assert sha256_file(case["bundle"]) == digest
