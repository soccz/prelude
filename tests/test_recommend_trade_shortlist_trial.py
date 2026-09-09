"""Real-format synthetic files only; no actual source/evidence writes or fitting."""
from __future__ import annotations

import copy
import json
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
import requests

import test_recommend_microstructure as feature_fixtures
import test_recommend_microstructure_trial as fixtures
from ops.artifact_provenance import canonical_json_bytes, file_identity
from signals import recommend_microstructure_trial as old
from signals import recommend_snapshot as snapshots
from signals import recommend_trade_shortlist_trial as trial

DAY = "2026-09-10"


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("trial attempted network, DB or model metadata generation")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(snapshots, name, forbidden)


def _clock(minute=10, *, day=DAY):
    return datetime.fromisoformat(f"{day}T09:{minute:02}:00+09:00")


def _files(tmp_path, monkeypatch, **kwargs):
    return fixtures._files(tmp_path / "inputs", monkeypatch, day=DAY, **kwargs)


def _publish(tmp_path, monkeypatch, **kwargs):
    paths = _files(tmp_path, monkeypatch, **kwargs)
    root = tmp_path / "shortlist_trials"
    report = trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    return paths, root, report


def _read(root, minute=11):
    return trial.read_trade_shortlist_record(root, DAY, _clock(minute))


def _identities(paths):
    return [file_identity(path, root=trial.ROOT) for path in paths]


def test_native_evidence_publish_read_and_restart_are_bound_and_write_once(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    before = _identities(list(paths) + list(paths[0].parent.glob("*.gz")))
    root = tmp_path / "shortlist_trials"
    report = trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    record = _read(root)
    assert report["status"] == record["status"] == "committed"
    assert record["plan"] == record["score"]["plan"]
    assert record["plan"]["challenger_top3"] == [feature_fixtures.MARKETS[i] for i in (0, 1, 3)]
    assert record["score"]["trial_id"] == trial.TRIAL_ID != old.TRIAL_ID
    assert record["plan"]["prospective_prediction"] is False
    assert record["score"]["prospective_status"] == "pending_canonical_delivery_entry_check"
    assert record["score"]["deployable"] is False
    assert len(record["source_inputs"]) == len(record["trial_artifacts"]) == 2
    saved = _identities([Path(report["score_path"]), Path(report["commit_path"])])
    again = trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=lambda: _clock(11))
    assert again["reused"] is True and again["status"] == "committed"
    assert _identities([Path(report["score_path"]), Path(report["commit_path"])]) == saved
    assert _identities(list(paths) + list(paths[0].parent.glob("*.gz"))) == before
    assert trial.TRIAL_CONFIG["prospective_start_asof"] == DAY
    assert old.TRIAL_CONFIG["prospective_start_asof"] == "2026-09-08"


def test_two_namespaces_can_share_parent_without_modifying_old_records(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "shared_trials"
    original = old.record_microstructure_trial(*paths, output_root=root, now_fn=_clock)
    original_paths = [Path(original[key]) for key in ("score_path", "commit_path")]
    before = _identities(original_paths)
    new = trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert Path(new["score_path"]).parent != Path(original["score_path"]).parent
    assert before == _identities(original_paths)
    assert _read(root)["status"] == "committed"
    with old._directory(root, (DAY, old.TRIAL_ID), create=False) as directory:
        assert old._read_record(directory, now=_clock(11))["status"] == "committed"


def test_read_missing_is_read_only_and_does_not_create_parents(tmp_path):
    root = tmp_path / "absent" / "trial_root"
    result = _read(root)
    assert result["status"] == "missing" and result["score"] is None
    assert result["commit"] is result["snapshot"] is result["feature"] is result["plan"] is None
    assert result["source_inputs"] == [] and len(result["trial_artifacts"]) == 2
    assert not any(item["exists"] for item in result["trial_artifacts"])
    assert not (tmp_path / "absent").exists()


def test_unavailable_and_noop_are_committed_without_substitution(tmp_path, monkeypatch):
    _, root, report = _publish(tmp_path / "missing", monkeypatch, missing=True)
    missing = _read(root)
    assert report["plan_status"] == "unavailable" and missing["status"] == "committed"
    assert missing["plan"]["required_missing_features"] == [feature_fixtures.MARKETS[3]]
    assert missing["plan"]["challenger_top3"] is None
    _, root, report = _publish(tmp_path / "noop", monkeypatch, changed=False)
    noop = _read(root)
    assert noop["status"] == "committed" and noop["plan"]["is_no_op"] is True


def test_score_only_crash_is_uncertain_and_retry_never_creates_commit(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "trials"
    publish = old._publish_new

    def fail_commit(directory, name, document):
        if name == "commit.json":
            raise OSError("injected commit failure")
        publish(directory, name, document)

    with monkeypatch.context() as patch:
        patch.setattr(old, "_publish_new", fail_commit)
        with pytest.raises(OSError, match="injected"):
            trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    before = _identities(list(root.rglob("*.json")))
    assert _read(root)["status"] == "uncertain"
    report = trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert report["status"] == "uncertain" and report["reused"] is True
    assert before == _identities(list(root.rglob("*.json")))
    assert not Path(report["commit_path"]).exists()


def test_durable_clock_is_sampled_after_file_and_directory_fsync(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    events = []
    real_fsync = os.fsync

    def sync(fd):
        events.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    def clock():
        events.append("clock")
        return _clock()

    monkeypatch.setattr(os, "fsync", sync)
    trial.record_trade_shortlist_trial(*paths, output_root=tmp_path / "trials", now_fn=clock)
    samples = [i for i, value in enumerate(events) if value == "clock"]
    assert len(samples) == 2
    assert events[samples[1]-1] == "dir" and "file" in events[samples[0]:samples[1]]
    assert "file" in events[samples[1]:]


@pytest.mark.parametrize("phase", ["score_file", "score_directory"])
def test_fsync_failure_cannot_produce_commit(tmp_path, monkeypatch, phase):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "trials"
    sync, link = os.fsync, os.link
    linked = False

    def note_link(*args, **kwargs):
        nonlocal linked
        link(*args, **kwargs)
        linked = True

    def fail(fd):
        if (phase == "score_file" and stat.S_ISREG(os.fstat(fd).st_mode)) or (
            phase == "score_directory" and linked
        ):
            raise OSError("injected fsync failure")
        sync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", note_link)
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError):
            trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert not list(root.rglob("commit.json"))
    assert _read(root)["status"] == ("uncertain" if phase == "score_directory" else "missing")


def test_concurrent_first_publishers_cannot_overwrite_or_issue_two_commits(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "trials"
    publish = old._publish_new
    barrier = Barrier(2)

    def interleave(directory, name, document):
        if name == "score.json":
            barrier.wait(timeout=10)
        publish(directory, name, document)

    monkeypatch.setattr(old, "_publish_new", interleave)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: trial.record_trade_shortlist_trial(
            *paths, output_root=root, now_fn=_clock), range(2)))
    assert sorted(row["status"] for row in results) == ["committed", "uncertain"]
    assert _read(root)["status"] == "committed"
    assert sorted(path.name for path in (root / DAY / trial.TRIAL_ID).iterdir()) == ["commit.json", "score.json"]


@pytest.mark.parametrize("kind", ["checksum", "wrong_id", "config", "plan", "promotion", "source", "generator",
                                  "commit_binding", "commit_clock", "orphan"])
def test_corrupt_or_resealed_contradictions_fail_closed(tmp_path, monkeypatch, kind):
    _, root, report = _publish(tmp_path, monkeypatch)
    score_path, commit_path = Path(report["score_path"]), Path(report["commit_path"])
    score, commit = json.loads(score_path.read_bytes()), json.loads(commit_path.read_bytes())
    if kind == "orphan":
        score_path.unlink()
    elif kind in {"commit_binding", "commit_clock"}:
        if kind == "commit_binding":
            commit["snapshot_id"] = "other"
        else:
            commit["score_durable_observed_at"] = _clock(5).isoformat()
        commit_path.write_bytes(canonical_json_bytes(old._seal({k:v for k,v in commit.items() if k != "payload_sha256"})))
    else:
        if kind in {"checksum", "plan"}:
            score["plan"]["challenger_top3"] = score["plan"]["control_top3"]
        elif kind == "wrong_id":
            score["trial_id"] = old.TRIAL_ID
        elif kind == "config":
            score["config"]["prospective_start_asof"] = "2026-09-09"
        elif kind == "promotion":
            score["deployable"] = True
        elif kind == "source":
            score["source_inputs"][0]["sha256"] = "0"*64
        else:
            score["generator_sources"][0]["sha256"] = "0"*64
        if kind != "checksum":
            score = old._seal({k:v for k,v in score.items() if k != "payload_sha256"})
        score_path.write_bytes(canonical_json_bytes(score))
    with pytest.raises((ValueError, RuntimeError)):
        _read(root)


@pytest.mark.parametrize("name", ["trade.jsonl.gz", "orderbook.jsonl.gz", "manifest.json", "recommend_features.json"])
def test_raw_manifest_or_feature_mutation_is_detected_without_regeneration(tmp_path, monkeypatch, name):
    paths, root, _ = _publish(tmp_path, monkeypatch)
    path = paths[0].parent / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, RuntimeError)):
        _read(root)


@pytest.mark.parametrize("kind", ["root_symlink", "date_symlink", "score_symlink", "fifo"])
def test_output_symlinks_and_nonregular_files_cannot_escape_or_block(tmp_path, monkeypatch, kind):
    paths = _files(tmp_path, monkeypatch)
    root, outside = tmp_path / "trials", tmp_path / "outside"
    outside.mkdir()
    if kind == "root_symlink":
        root.symlink_to(outside, target_is_directory=True)
    elif kind == "date_symlink":
        root.mkdir()
        (root / DAY).symlink_to(outside, target_is_directory=True)
    else:
        folder = root / DAY / trial.TRIAL_ID
        folder.mkdir(parents=True)
        if kind == "score_symlink":
            (folder / "score.json").symlink_to(outside / "victim")
        else:
            os.mkfifo(folder / "score.json")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert list(outside.iterdir()) == []


def test_parent_alias_is_supported_without_following_leaf_symlinks(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    root = alias / "trials"
    trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert _read(root)["status"] == _read(physical / "trials")["status"] == "committed"


def test_backwards_clock_preserves_uncertain_score(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "trials"
    values = iter([_clock(), _clock()-timedelta(seconds=1)])
    with pytest.raises(trial.TradeShortlistTrialError, match="backwards"):
        trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=lambda: next(values))
    assert _read(root)["status"] == "uncertain"


def test_planning_cannot_precede_capture_completion(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    # Snapshot is complete near 09:05:01, while raw capture ends near 09:05:30.
    too_early = datetime.fromisoformat(DAY+"T09:05:10+09:00")
    root = tmp_path / "trials"
    with pytest.raises(trial.TradeShortlistTrialError, match="capture completion"):
        trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=lambda: too_early)
    assert not root.exists()


def test_source_change_during_publication_leaves_no_commit(tmp_path, monkeypatch):
    paths = _files(tmp_path, monkeypatch)
    root = tmp_path / "trials"
    original = trial._generator_sources
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        result = copy.deepcopy(original())
        if calls >= 3:
            result[0]["sha256"] = "0"*64
        return result

    with monkeypatch.context() as patch:
        patch.setattr(trial, "_generator_sources", changed)
        with pytest.raises(trial.TradeShortlistTrialError, match="generator changed"):
            trial.record_trade_shortlist_trial(*paths, output_root=root, now_fn=_clock)
    assert _read(root)["status"] == "uncertain"


@pytest.mark.parametrize("day", ["2026-9-10", "20260910", "../2026-09-10", "2026-09-10/other"])
def test_day_path_must_be_canonical_and_cannot_traverse(tmp_path, day):
    with pytest.raises(trial.TradeShortlistTrialError):
        trial.read_trade_shortlist_record(tmp_path, day, _clock())


def test_reader_requires_aware_now_and_does_not_evaluate_entry_or_labels(tmp_path, monkeypatch):
    _, root, _ = _publish(tmp_path, monkeypatch)
    with pytest.raises(trial.TradeShortlistTrialError):
        trial.read_trade_shortlist_record(root, DAY, datetime(2026, 9, 10, 10))
    # Publication may be after canonical entry: only the future evaluator can
    # classify it as late, without this reader looking at or changing outcomes.
    assert _read(root, 30)["status"] == "committed"


@pytest.mark.parametrize("state", ["committed", "uncertain"])
def test_existing_record_inspection_cannot_open_for_write_or_create_directories(tmp_path, monkeypatch, state):
    _, root, report = _publish(tmp_path, monkeypatch)
    if state == "uncertain":
        Path(report["commit_path"]).unlink()
    real_open = os.open

    def readonly_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        return real_open(path, flags, *args, **kwargs)

    def no_creation(*args, **kwargs):
        pytest.fail("read-only inspection attempted publication or directory creation")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", readonly_open)
        patch.setattr(os, "mkdir", no_creation)
        patch.setattr(old, "_publish_new", no_creation)
        assert _read(root)["status"] == state


def test_different_feature_file_cannot_replace_existing_daily_record(tmp_path, monkeypatch):
    paths, root, report = _publish(tmp_path, monkeypatch)
    saved = _identities([Path(report["score_path"]), Path(report["commit_path"])])
    copied = paths[1].with_name("different_feature.json")
    copied.write_bytes(paths[1].read_bytes())
    with pytest.raises(trial.TradeShortlistTrialError, match="different feature/snapshot"):
        trial.record_trade_shortlist_trial(paths[0], copied, output_root=root, now_fn=_clock)
    assert _identities([Path(report["score_path"]), Path(report["commit_path"])]) == saved
