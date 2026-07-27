from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from ops.code_lineage import python_code_lineage
import scripts.downside_veto_challenger_v1 as downside
import scripts.first_passage_head_challenger_v1 as first_passage
import scripts.safeup_head_challenger_v1 as safeup
import scripts.semivol_joint_challenger_v1 as semivol
import scripts.upside_head_challenger_v1 as upside


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _recorded_then_changed(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    recorded = module._code_lineage()
    current = json.loads(json.dumps(recorded))
    dependency = next(
        name
        for name in current["files"]
        if name != current["entrypoint"]
    )
    current["files"][dependency]["sha256"] = "0" * 64
    current["lineage_sha256"] = "0" * 64
    monkeypatch.setattr(module, "_code_lineage", lambda: current)
    return recorded


def test_python_code_lineage_is_transitive_and_content_only(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nested.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "direct.py").write_text(
        "from package import nested\n",
        encoding="utf-8",
    )
    (package / "lazy.py").write_text("VALUE = 2\n", encoding="utf-8")
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "from package import direct\n"
        "def load_later():\n"
        "    from package import lazy\n"
        "    return lazy.VALUE\n",
        encoding="utf-8",
    )

    before = python_code_lineage(entrypoint=entrypoint, root=tmp_path)
    assert set(before["files"]) == {
        "main.py",
        "package/__init__.py",
        "package/direct.py",
        "package/lazy.py",
        "package/nested.py",
    }
    assert "mtime" not in json.dumps(before)

    nested = package / "nested.py"
    stat = nested.stat()
    os.utime(
        nested,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    assert (
        python_code_lineage(entrypoint=entrypoint, root=tmp_path)
        == before
    )

    nested.write_text("VALUE = 3\n", encoding="utf-8")
    after = python_code_lineage(entrypoint=entrypoint, root=tmp_path)
    assert after != before
    assert (
        after["files"]["package/nested.py"]["sha256"]
        != before["files"]["package/nested.py"]["sha256"]
    )


@pytest.mark.parametrize(
    ("module", "required_dependency"),
    [
        (downside, "scripts/recommender_downside_exit_v1.py"),
        (upside, "scripts/binance_leadlag_v1.py"),
        (safeup, "scripts/downside_veto_challenger_v1.py"),
        (first_passage, "scripts/safeup_head_challenger_v1.py"),
        (semivol, "scripts/first_passage_head_challenger_v1.py"),
    ],
)
def test_challenger_lineage_contains_entrypoint_and_transitive_code(
    module,
    required_dependency: str,
) -> None:
    lineage = module._code_lineage()
    assert lineage["entrypoint"] in lineage["files"]
    assert required_dependency in lineage["files"]
    assert "ops/artifact_provenance.py" in lineage["files"]
    assert "mtime" not in json.dumps(lineage)


def test_downside_validator_rejects_changed_local_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "downside"
    coverage = Path(f"{prefix}_coverage.json")
    recorded = _recorded_then_changed(downside, monkeypatch)
    _write_json(
        coverage,
        {
            "schema": "downside_veto_challenger_v1",
            "script_sha256": downside._sha256(Path(downside.__file__)),
            "code_lineage": recorded,
        },
    )
    monkeypatch.setattr(downside, "_verify_manifest", lambda **_: {})

    with pytest.raises(RuntimeError, match="local code dependencies"):
        downside.validate_existing_artifacts(
            output_prefix=prefix,
            input_oos=tmp_path / "oos.csv",
            m15_db=tmp_path / "m15.db",
        )


def test_upside_validator_rejects_changed_local_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "upside"
    recorded = _recorded_then_changed(upside, monkeypatch)
    _write_json(
        Path(f"{prefix}_coverage.json"),
        {
            "schema": "upside_head_challenger_v1",
            "input_lineage": {
                "script_sha256": upside._sha256(Path(upside.__file__)),
                "code_lineage": recorded,
            },
        },
    )
    monkeypatch.setattr(upside, "_verify_manifest", lambda **_: {})

    with pytest.raises(RuntimeError, match="local code dependencies"):
        upside.validate_existing_artifacts(
            output_prefix=prefix,
            upbit_d1_db=tmp_path / "upbit.db",
            binance_d1_db=tmp_path / "binance.db",
            upbit_15m_db=tmp_path / "m15.db",
        )


def test_safeup_validators_reject_changed_local_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "safeup"
    predictions = Path(f"{prefix}_predictions.csv.gz")
    pd.DataFrame(
        {
            "scope": ["discovery_oof"],
            "date": ["2024-01-01"],
            "market": ["KRW-BTC"],
        }
    ).to_csv(predictions, index=False)
    recorded = _recorded_then_changed(safeup, monkeypatch)
    _write_json(
        Path(f"{prefix}_coverage.json"),
        {
            "schema": "safeup_head_challenger_v1",
            "locked_holdout": {},
            "prediction_scope_contract": {},
            "inputs": {
                "script_sha256": safeup._sha256(Path(safeup.__file__)),
                "code_lineage": recorded,
            },
        },
    )
    monkeypatch.setattr(safeup, "_verify_manifest", lambda **_: {})
    monkeypatch.setattr(
        safeup,
        "_prediction_scope_contract",
        lambda *_: {},
    )

    with pytest.raises(RuntimeError, match="local code dependencies"):
        safeup.validate_existing_artifacts(
            output_prefix=prefix,
            d1_db=tmp_path / "d1.db",
            m15_db=tmp_path / "m15.db",
        )

    schedule_prefix = tmp_path / "safeup_fp_schedule"
    _write_json(Path(f"{schedule_prefix}_contract.json"), {})
    _write_json(
        Path(f"{schedule_prefix}_coverage.json"),
        {
            "schema": "safeup_head_challenger_v1_fp_schedule",
            "split_schedule_sha256": "schedule",
            "historical_holdout_contaminated": True,
            "virgin_or_preregistered": False,
            "maximum_evidence_grade": "historical_comparison_only",
            "benchmark_axis": {
                "benchmark_complete_dates": 1,
                "candidate_dates": 1,
                "benchmark_incomplete_dates": 0,
            },
            "inputs": {
                "script_sha256": safeup._sha256(Path(safeup.__file__)),
                "code_lineage": recorded,
            },
        },
    )
    monkeypatch.setattr(
        safeup,
        "_verify_schedule_contract",
        lambda _: "schedule",
    )
    with pytest.raises(RuntimeError, match="local code dependencies"):
        safeup.validate_fp_schedule_artifacts(
            output_prefix=schedule_prefix,
            d1_db=tmp_path / "d1.db",
            m15_db=tmp_path / "m15.db",
        )


def test_first_passage_validator_rejects_changed_local_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "first_passage"
    recorded = _recorded_then_changed(first_passage, monkeypatch)
    _write_json(
        Path(f"{prefix}_coverage.json"),
        {
            "schema": "first_passage_head_challenger_v1",
            "input_lineage": {
                "script_sha256": first_passage._sha256(
                    Path(first_passage.__file__)
                ),
                "code_lineage": recorded,
            },
        },
    )
    _write_json(Path(f"{prefix}_path_panel_meta.json"), {})
    monkeypatch.setattr(
        first_passage,
        "_verify_manifest",
        lambda **_: {},
    )
    monkeypatch.setattr(
        first_passage,
        "_validate_safeup_baseline_lineage",
        lambda **_: {},
    )

    with pytest.raises(RuntimeError, match="local code dependencies"):
        first_passage.validate_existing_artifacts(
            output_prefix=prefix,
            d1_db=tmp_path / "d1.db",
            m15_db=tmp_path / "m15.db",
            baseline_predictions=tmp_path / "baseline.csv",
        )


def test_semivol_validator_rejects_changed_local_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "semivol"
    recorded = _recorded_then_changed(semivol, monkeypatch)
    _write_json(
        Path(f"{prefix}_coverage.json"),
        {
            "schema": "semivol_joint_challenger_v1",
            "inputs": {
                "script_sha256": semivol._sha256(Path(semivol.__file__)),
                "code_lineage": recorded,
            },
        },
    )
    monkeypatch.setattr(first_passage, "_verify_manifest", lambda **_: {})

    with pytest.raises(RuntimeError, match="local code dependencies"):
        semivol.validate_existing_artifacts(
            output_prefix=prefix,
            d1_db=tmp_path / "d1.db",
            m15_db=tmp_path / "m15.db",
            path_panel=tmp_path / "panel.csv.gz",
            path_panel_meta=tmp_path / "panel_meta.json",
            baseline_predictions=tmp_path / "baseline.csv.gz",
            core_reference=tmp_path / "core.csv.gz",
        )
