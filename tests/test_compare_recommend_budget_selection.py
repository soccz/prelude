import json

import pytest

import scripts.compare_recommend_budget_selection as cli
import test_compare_recommend_score_alignment as alignment_tests

alignment_case = alignment_tests.case


@pytest.fixture
def case(alignment_case, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("budget.py",))
    budget = tmp_path / "budget.py"
    budget.write_text("# budget selection\n")
    design = tmp_path / "budget-design.json"
    design.write_text(json.dumps(cli.prepare_design()))
    return {**alignment_case, "design": design, "budget": budget}


def test_no_writes_and_report_is_not_deployable(case, tmp_path):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = cli.run_comparison(case["design"])
    assert set(result["segments"]) == {"earlier", "later"}
    assert (
        result["deployable"]
        is result["model_fitted"]
        is result["automatic_adoption"]
        is False
    )
    assert all(v["original_A_parity"] == "passed" for v in result["segments"].values())
    digest = result.pop("report_payload_sha256")
    assert digest == cli.sha256_bytes(cli.canonical_json_bytes(result))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "key", ["earlier", "later", "design", "budget", "source", "kernel", "evidence"]
)
def test_frozen_design_and_input_change_rejected_before_selection(
    case, monkeypatch, key
):
    case[key].write_text("changed")
    monkeypatch.setattr(
        cli, "evaluate_budget", lambda *a, **k: pytest.fail("no evaluation")
    )
    with pytest.raises(ValueError):
        cli.run_comparison(case["design"])


@pytest.mark.parametrize(
    "key", ["earlier", "later", "design", "budget", "source", "kernel", "evidence"]
)
def test_change_during_evaluation_rejected(case, monkeypatch, key):
    original = cli.evaluate_budget

    def run(*a, **k):
        case[key].write_text("changed during evaluation")
        return original(*a, **k)

    monkeypatch.setattr(cli, "evaluate_budget", run)
    with pytest.raises(ValueError, match="changed during execution"):
        cli.run_comparison(case["design"])


def test_config_mutation_rejected(case):
    design = json.loads(case["design"].read_text())
    design["configuration"]["top_k"] = 4
    case["design"].write_text(json.dumps(design))
    with pytest.raises(ValueError, match="frozen budget design"):
        cli.run_comparison(case["design"])


def test_prepare_no_selection_and_writer_no_overwrite(
    case, monkeypatch, tmp_path, capsys
):
    original = cli.evaluate_budget
    monkeypatch.setattr(
        cli, "evaluate_budget", lambda *a, **k: pytest.fail("no evaluation at prepare")
    )
    design = tmp_path / "new-budget-design.json"
    assert cli.main(["--prepare-design", "--output", str(design)]) == 0
    monkeypatch.setattr(cli, "evaluate_budget", original)
    output = tmp_path / "result.json"
    assert cli.main(["--design", str(design), "--output", str(output)]) == 0
    before = output.read_bytes()
    assert cli.main(["--design", str(design), "--output", str(output)]) == 2
    assert output.read_bytes() == before
    assert cli.main(["--prepare-design"]) == 2
    assert "blocked" in capsys.readouterr().err
