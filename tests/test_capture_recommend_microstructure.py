from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.collector_upbit_microstructure import CaptureConfig, CaptureResult
from scripts import capture_recommend_microstructure as runner


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "capture_storage_preflight", lambda *_args: [])
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    result = CaptureResult(
        manifest, {"complete": True, "cutoff_source": {"path": str(snapshot)}}
    )
    monkeypatch.setattr(runner.collector, "run_capture", lambda _config: result)
    calls = []
    feature = {
        "feature_evidence_valid": True,
        "rows": [],
        "effect_status": "not_evaluated",
    }
    monkeypatch.setattr(runner, "read_recommend_microstructure", lambda *_args: feature)

    def record(snapshot_path, feature_path, *, output_root):
        assert snapshot_path == snapshot
        assert json.loads(feature_path.read_text()) == feature
        calls.append((snapshot_path, feature_path, output_root))
        return {"score_path": str(output_root / "score.json"), "status": "committed"}

    monkeypatch.setattr(runner, "record_microstructure_trial", record)
    monkeypatch.setattr(
        runner,
        "evaluate_microstructure_trials",
        lambda *_args: {"effect_status": "not_evaluated"},
    )
    config = CaptureConfig(
        markets=("KRW-BTC",),
        cutoff_slot="open",
        output_root=tmp_path,
        asof=date(2026, 9, 8),
    )
    return config, result, feature, calls


def test_capture_feature_trial_order_and_immutable_inputs(session, tmp_path):
    config, result, _feature, calls = session
    report = runner.run_session(config, trial_root=tmp_path / "trials")
    assert report["feature_evidence_valid"] is True
    assert report["effect_status"] == "not_evaluated"
    assert len(calls) == 1
    assert result.manifest_path.read_text() == "{}"
    assert Path(result.manifest["cutoff_source"]["path"]).read_text() == "{}"
    assert (tmp_path / "recommend_features.json").stat().st_mode & 0o777 == 0o600


def test_missing_snapshot_does_not_build_or_record(session, monkeypatch):
    config, result, _feature, calls = session
    result.manifest["cutoff_source"] = {}
    monkeypatch.setattr(
        runner,
        "read_recommend_microstructure",
        lambda *_: pytest.fail("unexpected build"),
    )
    report = runner.run_session(config)
    assert report["unavailable_reason"] == "original_snapshot_not_available"
    assert not calls


def test_invalid_feature_is_retained_but_not_used_for_trial(session, tmp_path):
    config, result, feature, calls = session
    result.manifest["complete"] = False
    feature["feature_evidence_valid"] = False
    report = runner.run_session(config)
    assert report["unavailable_reason"] == "feature_evidence_invalid"
    assert json.loads((tmp_path / "recommend_features.json").read_text()) == feature
    assert not calls


def test_duplicate_feature_never_overwrites_or_replays_trial(session, tmp_path):
    config, _result, _feature, calls = session
    runner.run_session(config, trial_root=tmp_path / "trials")
    feature_path = tmp_path / "recommend_features.json"
    original = feature_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_session(config, trial_root=tmp_path / "trials")
    assert feature_path.read_bytes() == original
    assert len(calls) == 1


def test_changed_manifest_rejected_before_any_feature_write(
    session, monkeypatch, tmp_path
):
    config, result, feature, calls = session

    def changed(*_args):
        result.manifest_path.write_text('{"changed":true}')
        return feature

    monkeypatch.setattr(runner, "read_recommend_microstructure", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        runner.run_session(config)
    assert not (tmp_path / "recommend_features.json").exists()
    assert not calls


def test_canary_cannot_use_scheduled_feature_path(tmp_path):
    config = CaptureConfig(
        markets=("KRW-BTC",), duration_seconds=1, output_root=tmp_path
    )
    with pytest.raises(ValueError, match="real open snapshot"):
        runner.run_session(config)


def test_feature_writer_fsyncs_file_then_directory(monkeypatch, tmp_path):
    kinds = []
    real_fsync = os.fsync

    def fsync(descriptor):
        import stat

        kinds.append(
            "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        )
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    runner.write_feature_once(tmp_path / "feature.json", {"a": 1})
    file_index = kinds.index("file")
    assert "directory" in kinds[file_index + 1 :]


def test_feature_writer_never_follows_destination_symlink(tmp_path):
    original = tmp_path / "original"
    original.write_text("do not touch")
    link = tmp_path / "feature.json"
    link.symlink_to(original)
    with pytest.raises(FileExistsError):
        runner.write_feature_once(link, {"a": 1})
    assert original.read_text() == "do not touch"


def test_feature_write_failure_preserves_partial_and_does_not_record(
    session, monkeypatch, tmp_path
):
    config, _result, _feature, calls = session

    real_fsync = os.fsync

    def fail_fsync(descriptor):
        import stat

        if (
            stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and (tmp_path / "recommend_features.json").exists()
        ):
            raise OSError("injected durability failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="durability"):
        runner.run_session(config)
    assert (tmp_path / "recommend_features.json").exists()
    assert not calls


def test_feature_writer_rejects_parent_symlink_before_writing(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        runner.write_feature_once(linked / "feature.json", {"a": 1})
    assert not (actual / "feature.json").exists()


def test_evaluation_failure_preserves_recorded_trial(session, monkeypatch, tmp_path):
    config, _result, _feature, calls = session

    def fail_evaluation(*_args):
        assert len(calls) == 1  # Score was recorded first, never after evaluation.
        raise ValueError("bad historical evidence")

    monkeypatch.setattr(runner, "evaluate_microstructure_trials", fail_evaluation)
    with pytest.raises(ValueError, match="historical"):
        runner.run_session(config)
    assert len(calls) == 1
    assert (tmp_path / "recommend_features.json").exists()
    assert not (tmp_path / "trial_evaluation.json").exists()


def test_uncertain_trial_is_not_reported_as_success_or_evaluated(session, monkeypatch):
    config, _result, _feature, _calls = session
    monkeypatch.setattr(
        runner,
        "record_microstructure_trial",
        lambda *_args, **_kwargs: {"status": "uncertain"},
    )
    monkeypatch.setattr(
        runner,
        "evaluate_microstructure_trials",
        lambda *_args: pytest.fail("not committed"),
    )
    report = runner.run_session(config)
    assert report["unavailable_reason"] == "trial_publication_unconfirmed"
    assert report["evaluation_path"] is None


@pytest.mark.parametrize("trial_status, expected", [("committed", 0), ("uncertain", 1)])
def test_cli_requires_confirmed_publication(monkeypatch, trial_status, expected):
    monkeypatch.setattr(runner, "capture_storage_preflight", lambda *_args: [])
    monkeypatch.setattr(
        runner.collector,
        "_resolve_markets",
        lambda *_args: (("KRW-BTC",), "explicit", None, 1),
    )
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda *_args, **_kwargs: {
            "capture_complete": True,
            "feature_evidence_valid": True,
            "trial": {"status": trial_status},
            "evaluation_path": "evaluation.json",
        },
    )
    assert runner.main(["--asof", "2026-09-08"]) == expected


def test_storage_preflight_checks_both_filesystems_without_creating_paths(
    monkeypatch, tmp_path
):
    raw = tmp_path / "raw"
    trials = tmp_path / "trials"
    raw.mkdir()
    trials.mkdir()
    seen = []

    def statvfs(path):
        seen.append(Path(path))
        return SimpleNamespace(f_bavail=10 * 1024**2, f_frsize=1024)

    monkeypatch.setattr(runner.os, "statvfs", statvfs)
    report = runner.capture_storage_preflight(
        raw / "new", trials / "new", 512 * 1024**2
    )
    assert seen == [raw, trials]
    assert all(item["required_bytes"] == 5 * 1024**3 for item in report)
    assert all(item["available_bytes"] == 10 * 1024**3 for item in report)
    assert not (raw / "new").exists()
    assert not (trials / "new").exists()


@pytest.mark.parametrize("available", [0, -1, 5 * 1024**3 - 1])
def test_storage_preflight_rejects_low_space(monkeypatch, tmp_path, available):
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=available, f_frsize=1),
    )
    with pytest.raises(RuntimeError, match="insufficient capture storage"):
        runner.capture_storage_preflight(tmp_path, tmp_path, 512 * 1024**2)


def test_storage_preflight_accepts_exact_boundary_and_uses_unprivileged_space(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=5 * 1024**3, f_bfree=100 * 1024**3, f_frsize=1
        ),
    )
    report = runner.capture_storage_preflight(tmp_path, tmp_path, 512 * 1024**2)
    assert report[0]["available_bytes"] == report[0]["required_bytes"]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, float("inf")])
def test_storage_preflight_rejects_invalid_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="positive integer"):
        runner.capture_storage_preflight(tmp_path, tmp_path, budget)


@pytest.mark.parametrize(
    "available,unit", [(None, 1), (True, 1), (1, 0), (1, -1), (1, float("nan"))]
)
def test_storage_preflight_rejects_invalid_filesystem_response(
    monkeypatch, tmp_path, available, unit
):
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=available, f_frsize=unit),
    )
    with pytest.raises(RuntimeError, match="invalid filesystem space"):
        runner.capture_storage_preflight(tmp_path, tmp_path, 1)


def test_storage_preflight_rejects_file_parent(tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("preserve")
    with pytest.raises((OSError, ValueError), match="directory"):
        runner.capture_storage_preflight(occupied / "new", tmp_path, 1)
    assert occupied.read_text() == "preserve"


@pytest.mark.parametrize(
    "failure",
    [OSError("statvfs unavailable"), RuntimeError("insufficient capture storage")],
)
def test_cli_storage_failure_precedes_network_and_capture(monkeypatch, failure):
    def fail(*_args):
        raise failure

    monkeypatch.setattr(runner, "capture_storage_preflight", fail)
    monkeypatch.setattr(
        runner.collector,
        "_resolve_markets",
        lambda *_args: pytest.fail("must not connect"),
    )
    monkeypatch.setattr(
        runner, "run_session", lambda *_args, **_kwargs: pytest.fail("must not capture")
    )
    assert runner.main([]) == 2


def test_direct_session_checks_space_before_starting_collector(monkeypatch, tmp_path):
    def fail(*_args):
        raise RuntimeError("insufficient capture storage")

    monkeypatch.setattr(runner, "capture_storage_preflight", fail)
    monkeypatch.setattr(
        runner.collector, "run_capture", lambda *_args: pytest.fail("must not capture")
    )
    config = CaptureConfig(
        markets=("KRW-BTC",), cutoff_slot="open", output_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="insufficient capture storage"):
        runner.run_session(config)


def test_trial_volume_shortage_cannot_be_hidden_by_raw_volume(monkeypatch, tmp_path):
    raw, trials = tmp_path / "raw", tmp_path / "trials"
    raw.mkdir()
    trials.mkdir()
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_bavail=(10 if Path(path) == raw else 4) * 1024**3, f_frsize=1
        ),
    )
    with pytest.raises(RuntimeError, match="insufficient capture storage"):
        runner.capture_storage_preflight(raw, trials, 512 * 1024**2)


def test_cli_rechecks_storage_after_market_lookup_before_capture(monkeypatch, tmp_path):
    lookup_done = []
    checks = []

    def storage(*_args):
        checks.append(bool(lookup_done))
        if lookup_done:
            raise RuntimeError("insufficient capture storage")
        return []

    def markets(*_args):
        lookup_done.append(True)
        return ("KRW-BTC",), "explicit", None, 1

    monkeypatch.setattr(runner, "capture_storage_preflight", storage)
    monkeypatch.setattr(runner.collector, "_resolve_markets", markets)
    monkeypatch.setattr(
        runner.collector, "run_capture", lambda *_args: pytest.fail("must not capture")
    )
    assert (
        runner.main(
            [
                "--asof",
                "2026-09-08",
                "--output-root",
                str(tmp_path / "raw"),
                "--trial-root",
                str(tmp_path / "trials"),
            ]
        )
        == 2
    )
    assert checks == [False, True]
