import copy
import json

import pytest

import scripts.review_recommend_spread as cli
from test_recommend_spread_diagnostics import rows


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "input.json"
    source.write_text("frozen input")
    identity = cli.file_identity(source, root=tmp_path)
    days = [{"date": "2026-09-10", "rows": rows()}]

    def verify(items):
        for item in items:
            if cli.file_identity(tmp_path / item["path"], root=tmp_path) != item:
                raise ValueError("input changed")

    monkeypatch.setattr(cli, "_verify", verify)
    monkeypatch.setattr(
        cli, "load_inputs", lambda path: (copy.deepcopy(days), [identity])
    )
    design = tmp_path / "design.json"
    design.write_text(json.dumps(cli.prepare_design(source)))
    return source, design, days


def test_read_only_no_fit_no_trial_sealed_report(case, tmp_path):
    _source, design, _days = case
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = cli.run_review(design)
    digest = result.pop("report_payload_sha256")
    assert digest == cli.sha256_bytes(cli.canonical_json_bytes(result))
    assert (
        result["deployable"]
        is result["live_model_changed"]
        is result["new_trial_recorded"]
        is False
    )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("mutation", ["source", "config", "statistics", "during"])
def test_frozen_bytes_configuration_and_midrun_mutation(case, monkeypatch, mutation):
    source, design, _days = case
    if mutation == "source":
        source.write_text("tampered")
    elif mutation == "during":
        original = cli.analyze

        def changed(*a, **k):
            source.write_text("tampered during analysis")
            return original(*a, **k)

        monkeypatch.setattr(cli, "analyze", changed)
    else:
        contents = json.loads(design.read_text())
        if mutation == "config":
            contents["configuration"]["universe_size"] = 10
        else:
            contents["statistics"]["n_boot"] = 3
        design.write_text(json.dumps(contents))
    if mutation != "during":
        monkeypatch.setattr(
            cli,
            "analyze",
            lambda *a, **k: pytest.fail("invalid design must block before analysis"),
        )
    with pytest.raises(ValueError):
        cli.run_review(design)


def test_new_json_only_no_overwrite_or_path_escape(case, tmp_path, capsys):
    source, design, _days = case
    assert cli.main(["--prepare-design"]) == 2
    output = tmp_path / "result.json"
    assert cli.main(["--design", str(design), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert cli.main(["--design", str(design), "--output", str(output)]) == 2
    assert output.read_bytes() == saved
    assert cli.main(["--design", str(design), "--output", str(source)]) == 2
    assert source.read_text() == "frozen input"
    assert (
        cli.main(
            ["--design", str(design), "--output", str(tmp_path.parent / "outside.json")]
        )
        == 2
    )
    assert "blocked" in capsys.readouterr().err


@pytest.mark.parametrize(
    "damage", [None, "raw", "feature", "label_binding", "future_clock"]
)
def test_real_native_input_chain(tmp_path, monkeypatch, damage):
    import test_recommend_microstructure_trial as fixtures
    import test_recommend_trade_shortlist_trial as publication

    paths, root, _published = publication._publish(tmp_path, monkeypatch)
    record = publication._read(root)
    monkeypatch.setattr(cli.trial, "DEFAULT_TRIAL_ROOT", root)
    evidence = fixtures._evidence(paths[0])

    def canonical_loader(snapshot_path, *, label_root, receipt_root, now):
        assert snapshot_path == paths[0]
        assert label_root == cli.ROOT / "output/recommend_score_labels"
        assert receipt_root == cli.ROOT / "output/recommend_receipts"
        return copy.deepcopy(evidence)

    monkeypatch.setattr(cli, "load_recommendation_evidence", canonical_loader)
    generated = "2026-09-30T01:00:00+00:00"
    evaluation = cli.native._seal(
        {
            "schema": cli.evaluator.EVALUATION_SCHEMA,
            "config": cli.evaluator.TRIAL_CONFIG,
            "generator_sources": cli.evaluator._generator_sources(),
            "generated_at": generated,
            "inputs_unchanged": True,
            "deployable": False,
            "automatic_promotion": False,
            "trial_inputs": record["trial_artifacts"],
            "evidence_inputs": record["source_inputs"]
            + record["feature"]["provenance"]["files"],
            "dates": [
                {
                    "date": "2026-09-10",
                    "status": "prospective_comparable",
                    "snapshot_id": record["snapshot"]["snapshot_id"],
                    "evidence_manifest": copy.deepcopy(evidence["manifest"]),
                }
            ],
        }
    )
    if damage == "future_clock":
        generated = "2099-01-01T00:00:00+00:00"
    document = cli.native._seal(
        {
            "schema": cli.review.SCHEMA,
            "through_date": "2026-09-29",
            "generated_at": generated,
            "generator_sources": cli.review._sources(),
            "evaluations": {"shortlist": evaluation},
        }
    )
    path = tmp_path / "review.json"
    path.write_bytes(cli.canonical_json_bytes(document))
    if damage in ("raw", "feature"):
        target = paths[0].parent / "orderbook.jsonl.gz" if damage == "raw" else paths[1]
        target.write_bytes(b"tampered")
    elif damage == "label_binding":
        evidence["manifest"]["fixture"] = "changed"
    if damage:
        with pytest.raises(ValueError):
            cli.load_inputs(path)
    else:
        days, identities = cli.load_inputs(path)
        assert len(days) == 1 and len(days[0]["rows"]) == 100
        assert {r["coin"] for r in days[0]["rows"]} == {
            r["coin"] for r in record["snapshot"]["universe"]
        }
        assert any("orderbook.jsonl.gz" in item["path"] for item in identities)
