"""Independent fault injection against the bounded operational status contract.

Only synthetic temporary artifacts are edited. Each diagnostic invocation is
read-only; these tests do not contact an exchange, load labels or fit a model.
"""

from __future__ import annotations

import builtins
import copy
import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
import requests

import test_recommend_microstructure_status as native_fixture
from data import collector_upbit_microstructure as collector
from ops import recommend_microstructure_status as status
from ops.artifact_provenance import canonical_json_bytes
from signals import recommend_microstructure as feature_contract
from signals import recommend_microstructure_trial as trial

DAY = native_fixture.DAY
NOW = native_fixture.NOW


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("independent test attempted an external request")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    path.write_bytes(canonical_json_bytes(value))


def _seal(value):
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    return {
        **payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def _refresh_identity(item, paths):
    path = trial._path(item["path"]).resolve()
    if path not in paths:
        return item
    content = path.read_bytes()
    return {
        **item,
        "exists": True,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reseal_downstream(capture, snapshot_path, trial_dir):
    """Repair every dependent document hash, leaving injected semantics intact."""
    manifest_path = capture / "manifest.json"
    feature_path = capture / "recommend_features.json"
    score_path, commit_path = trial_dir / "score.json", trial_dir / "commit.json"
    evaluation_path = capture / "trial_evaluation.json"
    paths = {
        path.resolve()
        for path in (
            snapshot_path,
            manifest_path,
            feature_path,
            score_path,
            commit_path,
        )
    }
    manifest = _read(manifest_path)
    manifest["cutoff_source"]["file_sha256"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    _write(manifest_path, manifest)
    feature = _read(feature_path)
    feature["provenance"]["capture_manifest_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    feature["provenance"]["raw_artifacts"] = {
        channel: copy.deepcopy(manifest["streams"][channel]["artifact"])
        for channel in ("trade", "orderbook")
    }
    feature["provenance"]["files"] = [
        _refresh_identity(item, paths) for item in feature["provenance"]["files"]
    ]
    _write(feature_path, feature)
    score = _read(score_path)
    score["source_inputs"] = [
        _refresh_identity(item, paths) for item in score["source_inputs"]
    ]
    score = _seal(score)
    _write(score_path, score)
    commit = _read(commit_path)
    commit["score_payload_sha256"] = score["payload_sha256"]
    commit = _seal(commit)
    _write(commit_path, commit)
    evaluation = _read(evaluation_path)
    evaluation["trial_inputs"] = [
        _refresh_identity(item, paths) for item in evaluation["trial_inputs"]
    ]
    evaluation["evidence_inputs"] = [
        _refresh_identity(item, paths) for item in evaluation["evidence_inputs"]
    ]
    for row in evaluation["dates"]:
        if row["date"] == DAY:
            row["score_durable_observed_at"] = commit["score_durable_observed_at"]
    _write(evaluation_path, _seal(evaluation))
    # A failure below must be semantic, not a stale payload checksum shortcut.
    for path in (score_path, commit_path, evaluation_path):
        value = _read(path)
        assert value == _seal(value)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("capture_before_snapshot", "capture ended before original snapshot"),
        ("score_before_capture", "score planned before capture ended"),
        ("commit_before_score", "commit binding/clock mismatch"),
        ("evaluation_before_commit", "session evaluation date/clock mismatch"),
    ],
)
def test_forward_clock_chain_rejects_fully_resealed_fault(
    tmp_path, monkeypatch, case, expected_reason
):
    roots, capture, snapshot_path, trial_dir = native_fixture._native(
        tmp_path, monkeypatch
    )
    if case == "capture_before_snapshot":
        path = capture / "manifest.json"
        value = _read(path)
        ended = feature_contract._ns(_read(snapshot_path)["decision_completed_at"]) - 1
        value.update(ended_at_ns=ended, ended_at=feature_contract._iso(ended))
    elif case == "score_before_capture":
        path = trial_dir / "score.json"
        value = _read(path)
        value["planned_at"] = DAY + "T09:05:02+09:00"
    elif case == "commit_before_score":
        path = trial_dir / "commit.json"
        value = _read(path)
        value["score_durable_observed_at"] = DAY + "T09:09:59+09:00"
    else:
        path = capture / "trial_evaluation.json"
        value = _read(path)
        value["generated_at"] = DAY + "T09:09:59+09:00"
    _write(path, value)
    _reseal_downstream(capture, snapshot_path, trial_dir)
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "evidence_invalid", result
    assert expected_reason in result["reason"], result
    assert result["attention_required"] is True


@pytest.mark.parametrize(
    "case", ["feature_only_two", "feature_missing_raw", "score_missing_generator"]
)
def test_full_graph_omission_rejected_after_all_downstream_hashes_resealed(
    tmp_path, monkeypatch, case
):
    roots, capture, snapshot_path, trial_dir = native_fixture._native(
        tmp_path, monkeypatch
    )
    if case.startswith("feature"):
        path = capture / "recommend_features.json"
        value = _read(path)
        value["provenance"]["files"] = value["provenance"]["files"][
            : 2 if case == "feature_only_two" else -1
        ]
    else:
        path = trial_dir / "score.json"
        value = _read(path)
        value["generator_sources"].pop()
    _write(path, value)
    _reseal_downstream(capture, snapshot_path, trial_dir)
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "evidence_invalid", result
    assert "source graph length mismatch" in result["reason"], result


def test_previous_due_uses_actual_observed_now_and_original_date(tmp_path, monkeypatch):
    roots, *_ = native_fixture._native(tmp_path, monkeypatch, changed=False)
    observed_now = "2026-09-10T09:00:00+09:00"
    result = status.inspect_daily_status(now=observed_now, **roots)
    previous = result["previous_due"]
    assert result["status"] == "waiting"
    assert previous["status"] == "complete_noop", result
    assert result["checked_at"] == previous["checked_at"] == observed_now
    assert previous["asof"] == DAY
    assert previous["due_at"] == DAY + "T10:00:00+09:00"
    assert result["attention_required"] is False
    assert result["research_start_asof"] == "2026-09-08"
    assert result["operational_start_asof"] == "2026-09-09"


def test_zero_observed_trades_can_be_operational_noop_not_feature_effect(
    tmp_path, monkeypatch
):
    original_sample = native_fixture.fixtures._sample

    def no_trades(*args, **kwargs):
        snapshot, manifest, records = original_sample(*args, **kwargs)
        records["trade"] = []
        manifest["streams"]["trade"]["artifact"].update(record_count=0, event_count=0)
        for connection in manifest["streams"]["trade"]["connections"]:
            connection.update(first_ingress_seq=None, last_ingress_seq=None)
        return snapshot, manifest, records

    monkeypatch.setattr(native_fixture.fixtures, "_sample", no_trades)
    roots, capture, *_ = native_fixture._native(
        tmp_path, monkeypatch, opportunity=False
    )
    feature = _read(capture / "recommend_features.json")
    assert feature["feature_evidence_valid"] is True
    assert feature["feature_available_rows"] == 0
    assert all(row[feature_contract.FEATURE] is None for row in feature["rows"])
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "complete_noop", result
    assert result["attention_required"] is False
    assert result["effect_status"] == "not_evaluated"
    assert result["prospective_status"] == "not_checked"


def test_same_size_raw_corruption_exposes_explicit_unchecked_content_boundary(
    tmp_path, monkeypatch
):
    roots, capture, *_ = native_fixture._native(tmp_path, monkeypatch)
    raw = capture / "trade.jsonl.gz"
    before = raw.read_bytes()
    damaged = bytes([before[0] ^ 1]) + before[1:]
    raw.write_bytes(damaged)
    declared = _read(capture / "manifest.json")["streams"]["trade"]["artifact"]
    assert len(damaged) == declared["size_bytes"]
    assert hashlib.sha256(damaged).hexdigest() != declared["sha256"]
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "complete_changed", result
    assert result["raw_content_hashes_checked"] is False
    assert result["scope"] == "bounded_metadata_bindings_and_raw_presence_size_only"
    assert result["prospective_status"] == "not_checked"


def test_diagnostic_forbids_raw_label_network_db_and_write_paths(tmp_path, monkeypatch):
    roots, *_ = native_fixture._native(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    original_os_open, original_open = os.open, builtins.open

    def forbidden(*args, **kwargs):
        pytest.fail("operational probe escaped its bounded read-only scope")

    def check_name(path):
        if not isinstance(path, int):
            text = os.fspath(path)
            assert not text.endswith(".gz")
            assert "recommend_score_labels" not in text
            assert "recommend_receipts" not in text

    def os_readonly(path, flags, *args, **kwargs):
        assert not flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        check_name(path)
        return original_os_open(path, flags, *args, **kwargs)

    def readonly(path, mode="r", *args, **kwargs):
        assert not set("wax+") & set(mode)
        check_name(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(os, "open", os_readonly)
        guard.setattr(builtins, "open", readonly)
        guard.setattr(Path, "read_bytes", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(gzip, "open", forbidden)
        guard.setattr(sqlite3, "connect", forbidden)
        guard.setattr(collector, "run_capture", forbidden)
        guard.setattr(feature_contract, "read_recommend_microstructure", forbidden)
        for name in (
            "_load_inputs",
            "load_recommendation_evidence",
            "record_microstructure_trial",
            "evaluate_microstructure_trials",
            "write_new_trial_report",
        ):
            guard.setattr(trial, name, forbidden)
        for name in (
            "mkdir",
            "makedirs",
            "remove",
            "unlink",
            "rename",
            "replace",
            "fsync",
        ):
            guard.setattr(os, name, forbidden)
        result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "complete_changed", result
    assert before == {path: path.read_bytes() for path in before}
