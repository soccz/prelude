"""Native synthetic artifact chains; no real DB, capture, network or alerts."""

import copy
import gzip
import json
import os

import pytest
import requests

import test_recommend_microstructure_trial as fixtures
from ops import recommend_microstructure_status as status
from ops.artifact_provenance import canonical_json_bytes, sha256_bytes
from ops.recommendation_evidence import EvidenceUnavailable
from signals import recommend_microstructure as features
from signals import recommend_microstructure_trial as trial

DAY = "2026-09-09"
NOW = DAY + "T10:30:00+09:00"


def _write(path, document):
    path.write_bytes(canonical_json_bytes(document))


def _read(path):
    return json.loads(path.read_bytes())


@pytest.fixture(autouse=True)
def no_external_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network, fit, or publication during status check")

    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def _native(tmp_path, monkeypatch, **kwargs):
    snapshot, manifest, records = fixtures._sample(monkeypatch, day=DAY, **kwargs)
    roots = dict(
        capture_root=tmp_path / "raw",
        snapshot_root=tmp_path / "snapshots",
        trial_root=tmp_path / "trials",
    )
    capture = roots["capture_root"].joinpath(*DAY.split("-"), manifest["capture_id"])
    capture.mkdir(parents=True)
    snapshot_path = roots["snapshot_root"] / DAY / "open_r1.json"
    snapshot_path.parent.mkdir(parents=True)
    _write(snapshot_path, snapshot)
    manifest["cutoff_source"].update(
        path=str(snapshot_path), file_sha256=sha256_bytes(snapshot_path.read_bytes())
    )
    for channel, values in records.items():
        path = capture / f"{channel}.jsonl.gz"
        path.write_bytes(
            gzip.compress(
                b"".join(canonical_json_bytes(row) + b"\n" for row in values), mtime=0
            )
        )
        manifest["streams"][channel]["artifact"].update(
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=sha256_bytes(path.read_bytes()),
        )
    _write(capture / "manifest.json", manifest)
    feature_path = capture / "recommend_features.json"
    _write(
        feature_path,
        features.read_recommend_microstructure(
            snapshot_path, capture / "manifest.json"
        ),
    )
    trial.record_microstructure_trial(
        snapshot_path,
        feature_path,
        output_root=roots["trial_root"],
        now_fn=lambda: DAY + "T09:10:00+09:00",
    )

    def immature(*args, **kwargs):
        raise EvidenceUnavailable("synthetic labels not mature")

    monkeypatch.setattr(trial, "load_recommendation_evidence", immature)
    evaluation = trial.evaluate_microstructure_trials(
        roots["trial_root"], now=DAY + "T09:11:00+09:00", n_boot=10
    )
    _write(capture / "trial_evaluation.json", evaluation)
    return roots, capture, snapshot_path, roots["trial_root"] / DAY / trial.TRIAL_ID


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, "complete_changed"),
        ({"opportunity": False}, "complete_noop"),
        ({"changed": False}, "complete_noop"),
        ({"missing": True}, "complete_unavailable"),
    ],
)
def test_native_success_and_unavailable_are_distinct(
    tmp_path, monkeypatch, options, expected
):
    roots, *_ = _native(tmp_path, monkeypatch, **options)
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == expected, result
    assert result["attention_required"] is False
    assert result["raw_content_hashes_checked"] is False
    assert result["prospective_status"] == "not_checked"
    assert result["deployable"] is False


@pytest.mark.parametrize(
    ("now", "expected", "attention"),
    [
        ("2026-09-08T18:00:00+09:00", "not_started", False),
        (DAY + "T09:59:59.999999+09:00", "waiting", False),
        (DAY + "T10:00:00+09:00", "capture_missing", True),
        (DAY + "T01:00:00+00:00", "capture_missing", True),
        ("2026-09-10T07:00:00+09:00", "waiting", True),
    ],
)
def test_calendar_missing_without_any_created_files(tmp_path, now, expected, attention):
    result = status.inspect_daily_status(
        now=now,
        capture_root=tmp_path / "raw",
        snapshot_root=tmp_path / "snap",
        trial_root=tmp_path / "trial",
    )
    assert result["status"] == expected
    assert result["attention_required"] is attention
    assert list(tmp_path.iterdir()) == []
    assert result["research_start_asof"] == "2026-09-08"
    if now.startswith("2026-09-10"):
        assert result["previous_due"]["status"] == "capture_missing"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("manifest.json", "capture_unfinished"),
        ("recommend_features.json", "feature_missing"),
        ("score.json", "publication_uncertain"),
        ("commit.json", "publication_uncertain"),
        ("trial_evaluation.json", "recorded_evaluation_missing"),
        ("snapshot", "upstream_snapshot_missing"),
        ("trade.jsonl.gz", "evidence_invalid"),
    ],
)
def test_missing_stage_never_reports_complete(tmp_path, monkeypatch, target, expected):
    roots, capture, snapshot, directory = _native(tmp_path, monkeypatch)
    path = (
        snapshot
        if target == "snapshot"
        else (directory if target in {"score.json", "commit.json"} else capture)
        / target
    )
    path.unlink()
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == expected, result
    assert result["attention_required"]


def test_missing_score_and_commit(tmp_path, monkeypatch):
    roots, _, _, directory = _native(tmp_path, monkeypatch)
    (directory / "score.json").unlink()
    (directory / "commit.json").unlink()
    assert status.inspect_daily_status(now=NOW, **roots)["status"] == "score_missing"


@pytest.mark.parametrize(
    "damage",
    [
        "complete",
        "quality",
        "capture_date",
        "capture_id",
        "snapshot_hash",
        "raw_size",
        "feature_hash",
        "feature_count",
        "feature_graph",
        "score_hash",
        "score_plan",
        "commit_binding",
        "commit_future",
        "evaluation_hash",
        "evaluation_date",
        "evaluation_inputs",
        "evaluation_audit",
        "evaluation_config",
    ],
)
def test_tampered_native_evidence_is_not_healthy(tmp_path, monkeypatch, damage):
    roots, capture, _, directory = _native(tmp_path, monkeypatch)
    target = "manifest.json"
    if damage.startswith("feature"):
        target = "recommend_features.json"
    elif damage.startswith("score"):
        target = "score.json"
    elif damage.startswith("commit"):
        target = "commit.json"
    elif damage.startswith("evaluation"):
        target = "trial_evaluation.json"
    path = (directory if target in {"score.json", "commit.json"} else capture) / target
    document = _read(path)
    if damage == "complete":
        document["complete"] = False
    elif damage == "quality":
        document["quality"]["warmup_ready"] = False
    elif damage == "capture_date":
        document["asof"] = "2026-09-08"
    elif damage == "capture_id":
        document["capture_id"] = "different"
    elif damage == "snapshot_hash":
        document["cutoff_source"]["file_sha256"] = "a" * 64
    elif damage == "raw_size":
        document["streams"]["trade"]["artifact"]["size_bytes"] += 1
    elif damage == "feature_hash":
        document["provenance"]["capture_manifest_payload_sha256"] = "a" * 64
    elif damage == "feature_count":
        document["feature_available_rows"] = 0
    elif damage == "feature_graph":
        document["provenance"]["files"] = document["provenance"]["files"][:2]
    elif damage == "score_hash":
        document["payload_sha256"] = "a" * 64
    elif damage == "score_plan":
        document["plan"]["changed_picks"] = 0
    elif damage == "commit_binding":
        document["score_payload_sha256"] = "a" * 64
    elif damage == "commit_future":
        document["score_durable_observed_at"] = "2027-01-01T10:00:00+09:00"
    elif damage == "evaluation_hash":
        document["payload_sha256"] = "a" * 64
    elif damage == "evaluation_date":
        document["generated_at"] = "2026-09-08T10:00:00+09:00"
    elif damage == "evaluation_inputs":
        document["trial_inputs"] = []
    elif damage == "evaluation_audit":
        document["dates"] = []
    elif damage == "evaluation_config":
        document["config"] = {}
    if target in {
        "score.json",
        "commit.json",
        "trial_evaluation.json",
    } and not damage.endswith("hash"):
        document = trial._seal(
            {key: value for key, value in document.items() if key != "payload_sha256"}
        )
    _write(path, document)
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["attention_required"], result


def test_multiple_captures_not_hidden_by_one_success(tmp_path, monkeypatch):
    roots, capture, *_ = _native(tmp_path, monkeypatch)
    (capture.parent / "second-incomplete").mkdir()
    assert (
        status.inspect_daily_status(now=NOW, **roots)["status"] == "capture_ambiguous"
    )


@pytest.mark.parametrize(
    "kind", ["symlink", "fifo", "oversize", "duplicate_json", "nan"]
)
def test_unsafe_document_fails_without_blocking(tmp_path, monkeypatch, kind):
    roots, capture, *_ = _native(tmp_path, monkeypatch)
    path = capture / "manifest.json"
    if kind == "symlink":
        path.rename(capture / "saved.json")
        path.symlink_to(capture / "saved.json")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "oversize":
        monkeypatch.setattr(status, "MAX_DOCUMENT_BYTES", 1)
    else:
        path.write_bytes(b'{"x":1,"x":2}' if kind == "duplicate_json" else b'{"x":NaN}')
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "evidence_invalid", result


def test_date_directory_symlink_rejected(tmp_path, monkeypatch):
    roots, capture, *_ = _native(tmp_path, monkeypatch)
    actual = capture.parent.with_name("saved")
    capture.parent.rename(actual)
    capture.parent.symlink_to(actual, target_is_directory=True)
    assert status.inspect_daily_status(now=NOW, **roots)["status"] == "evidence_invalid"


def test_no_writes_and_no_raw_content_reads(tmp_path, monkeypatch):
    roots, capture, *_ = _native(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original = os.open

    def readonly(path, flags, *args, **kwargs):
        assert flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC) == 0
        assert not str(path).endswith(".gz")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", readonly)
    monkeypatch.setattr(
        trial, "record_microstructure_trial", lambda *a, **k: pytest.fail("publication")
    )
    monkeypatch.setattr(
        trial,
        "evaluate_microstructure_trials",
        lambda *a, **k: pytest.fail("evaluation"),
    )
    result = status.inspect_daily_status(now=NOW, **roots)
    assert result["status"] == "complete_changed", result
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }


def test_group_mutation_rejected(tmp_path, monkeypatch):
    roots, capture, *_ = _native(tmp_path, monkeypatch)
    original = status._ReadSet.unchanged

    def mutate(reader):
        (capture / "trade.jsonl.gz").write_bytes(b"changed")
        original(reader)

    monkeypatch.setattr(status._ReadSet, "unchanged", mutate)
    assert status.inspect_daily_status(now=NOW, **roots)["status"] == "evidence_invalid"


def test_cli_reports_prior_due_failure_and_preserves_calendar(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(status, "ROOT", tmp_path)
    assert status.main(["--now", "2026-09-08T18:00:00+09:00"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["research_start_asof"] == "2026-09-08"
    assert status.main(["--now", "2026-09-10T07:00:00+09:00", "--format", "text"]) == 1
    assert "latest due 2026-09-09: capture_missing" in capsys.readouterr().out
    assert status.main(["--now", "2026-09-09T10:00:00"]) == 2
    assert "aware now required" in capsys.readouterr().out


def test_inputs_not_mutated(tmp_path, monkeypatch):
    roots, *_ = _native(tmp_path, monkeypatch)
    before = copy.deepcopy(roots)
    status.inspect_daily_status(now=NOW, **roots)
    assert roots == before
