"""Independent native-chain Top10 status faults; synthetic artifacts only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

import test_recommend_microstructure_trial as fixtures
from ops import recommend_trade_shortlist_status as status
from ops.artifact_provenance import canonical_json_bytes
from ops.recommendation_evidence import EvidenceUnavailable
from signals import recommend_microstructure_trial as native
from signals import recommend_trade_shortlist_eval as evaluate
from signals import recommend_trade_shortlist_trial as trial

DAY = "2026-09-10"
NOW = DAY + "T10:30:00+09:00"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("status attempted network or DB metadata collection")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(fixtures.snapshots, name, forbidden)


def _native(tmp_path, monkeypatch, **kwargs):
    paths = fixtures._files(tmp_path / "inputs", monkeypatch, day=DAY, **kwargs)
    root = tmp_path / "trials"
    trial.record_trade_shortlist_trial(
        *paths, output_root=root, now_fn=lambda: DAY + "T09:10:00+09:00"
    )

    def pending(*args, **kwargs):
        raise EvidenceUnavailable("label_missing")

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", pending)
    report = evaluate.evaluate_trade_shortlist_trials(
        root, now=DAY + "T09:11:00+09:00", n_boot=10
    )
    path = paths[1].parent / "trade_shortlist_evaluation.json"
    native.write_new_trial_report(path, report)
    return root, path, root / DAY / trial.TRIAL_ID


def _rewrite(path, mutate, *, seal=True):
    document = json.loads(path.read_bytes())
    mutate(document)
    if seal:
        document.pop("payload_sha256", None)
        document = native._seal(document)
    path.write_bytes(canonical_json_bytes(document))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "complete_changed"),
        ({"changed": False}, "complete_noop"),
        ({"missing": True}, "complete_unavailable"),
    ],
)
def test_native_publication_evaluation_binding_is_not_effect_or_eligibility(
    tmp_path, monkeypatch, kwargs, expected
):
    root, path, _ = _native(tmp_path, monkeypatch, **kwargs)
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == expected, report
    assert report["attention_required"] is False
    assert report["effect_status"] == "not_evaluated"
    assert report["prospective_status"] == "not_checked"
    assert report["deployable"] is False
    # Publication can be operationally healthy while outcome evidence is absent.
    assert json.loads(path.read_bytes())["coverage"]["paired_dates"] == 0


def test_before_launch_does_not_read_or_create_trial_root(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prelaunch probe inspected trial files")

    monkeypatch.setattr(trial, "read_trade_shortlist_record", forbidden)
    root = tmp_path / "absent"
    report = status.inspect_shortlist_status(
        now="2026-09-09T12:00:00+09:00", trial_root=root
    )
    assert report["status"] == "not_started" and not report["attention_required"]
    assert not root.exists()


@pytest.mark.parametrize(
    ("now", "expected", "attention"),
    [
        (DAY + "T09:59:59+09:00", "waiting", False),
        (DAY + "T10:00:00+09:00", "missing", True),
        ("2026-09-11T08:00:00+09:00", "waiting", True),
    ],
)
def test_missing_dates_and_completion_boundary(tmp_path, now, expected, attention):
    report = status.inspect_shortlist_status(now=now, trial_root=tmp_path / "absent")
    assert report["status"] == expected and report["attention_required"] is attention
    if now.startswith("2026-09-11"):
        assert report["previous_due"]["asof"] == DAY
        assert report["previous_due"]["status"] == "missing"
        assert report["previous_due"]["checked_at"] == now


def test_previous_due_preserves_actual_observation_time_and_native_success(
    tmp_path, monkeypatch
):
    root, _, _ = _native(tmp_path, monkeypatch)
    now = "2026-09-11T08:00:00+09:00"
    report = status.inspect_shortlist_status(now=now, trial_root=root)
    assert report["status"] == "waiting" and report["attention_required"] is False
    assert report["previous_due"]["status"] == "complete_changed"
    assert report["previous_due"]["checked_at"] == now
    # A past healthy result must not mask today's missing due publication.
    current = status.inspect_shortlist_status(
        now="2026-09-11T10:30:00+09:00", trial_root=root
    )
    assert current["status"] == "missing" and current["attention_required"]


@pytest.mark.parametrize(
    ("removed", "expected"),
    [
        ("evaluation", "recorded_evaluation_missing"),
        ("commit.json", "uncertain"),
        ("score.json", "evidence_invalid"),
    ],
)
def test_incomplete_publication_is_not_success(
    tmp_path, monkeypatch, removed, expected
):
    root, evaluation_path, record_path = _native(tmp_path, monkeypatch)
    (evaluation_path if removed == "evaluation" else record_path / removed).unlink()
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == expected and report["attention_required"]


@pytest.mark.parametrize(
    "fault",
    [
        "bad_payload",
        "wrong_schema",
        "wrong_trial",
        "wrong_config",
        "deployable",
        "auto_promotion",
        "inputs_changed",
        "before_durable",
        "future_generated",
        "wrong_day",
        "score_identity",
        "commit_identity",
        "missing_identity",
        "duplicate_identity",
        "missing_date",
        "duplicate_date",
        "wrong_snapshot",
        "wrong_plan_status",
        "control_top3",
        "challenger_top3",
        "changed_picks",
        "is_no_op",
        "feature_coverage",
        "score_durable_observed_at",
        "generator_missing",
        "generator_digest",
    ],
)
def test_resealed_evaluation_faults_never_claim_operational_completion(
    tmp_path, monkeypatch, fault
):
    root, path, _ = _native(tmp_path, monkeypatch)

    def mutate(document):
        audit = document["dates"][0]
        if fault == "bad_payload":
            document["seed"] = 99
        elif fault == "wrong_schema":
            document["schema"] = "recommend_microstructure_trial_evaluation.v1"
        elif fault == "wrong_trial":
            document["trial_id"] = "r1_boundary_trade_imbalance_v1"
        elif fault == "wrong_config":
            document["config"]["prospective_start_asof"] = "2026-09-09"
        elif fault in {"deployable", "auto_promotion", "inputs_changed"}:
            key = {
                "auto_promotion": "automatic_promotion",
                "inputs_changed": "inputs_unchanged",
            }.get(fault, fault)
            document[key] = key != "inputs_unchanged"
        elif fault in {"before_durable", "future_generated", "wrong_day"}:
            document["generated_at"] = {
                "before_durable": DAY + "T09:09:59+09:00",
                "future_generated": DAY + "T11:00:00+09:00",
                "wrong_day": "2026-09-11T09:11:00+09:00",
            }[fault]
        elif fault in {"score_identity", "commit_identity"}:
            suffix = "score.json" if fault == "score_identity" else "commit.json"
            next(
                item
                for item in document["trial_inputs"]
                if item["path"].endswith(suffix)
            )["sha256"] = "0" * 64
        elif fault == "missing_identity":
            document["trial_inputs"].pop()
        elif fault == "duplicate_identity":
            document["trial_inputs"].append(document["trial_inputs"][0])
        elif fault == "missing_date":
            document["dates"] = []
        elif fault == "duplicate_date":
            document["dates"].append(audit)
        elif fault == "wrong_snapshot":
            audit["snapshot_id"] = "recommend-wrong"
        elif fault == "wrong_plan_status":
            audit["plan_status"] = "unavailable"
        elif fault in {"control_top3", "challenger_top3"}:
            audit[fault] = list(reversed(audit[fault]))
        elif fault == "changed_picks":
            audit[fault] += 1
        elif fault == "is_no_op":
            audit[fault] = not audit[fault]
        elif fault == "feature_coverage":
            audit[fault]["shortlist_available_features"] = 9
        elif fault == "score_durable_observed_at":
            audit[fault] = DAY + "T09:09:59+09:00"
        elif fault == "generator_missing":
            document["generator_sources"].pop()
        else:
            document["generator_sources"][0]["sha256"] = "0" * 64

    _rewrite(path, mutate, seal=fault != "bad_payload")
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == "evidence_invalid", (fault, report)
    assert report["attention_required"]


def test_same_size_raw_corruption_fails_full_hash_native_probe(tmp_path, monkeypatch):
    root, evaluation_path, _ = _native(tmp_path, monkeypatch)
    raw_path = evaluation_path.parent / "trade.jsonl.gz"
    content = bytearray(raw_path.read_bytes())
    content[len(content) // 2] ^= 1
    raw_path.write_bytes(content)
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == "evidence_invalid" and report["attention_required"]


def test_symlink_evaluation_and_root_are_rejected(tmp_path, monkeypatch):
    root, evaluation_path, _ = _native(tmp_path, monkeypatch)
    moved = evaluation_path.with_name("original.json")
    evaluation_path.rename(moved)
    evaluation_path.symlink_to(moved)
    assert (
        status.inspect_shortlist_status(now=NOW, trial_root=root)["status"]
        == "evidence_invalid"
    )
    alias = tmp_path / "root_alias"
    alias.symlink_to(root, target_is_directory=True)
    assert (
        status.inspect_shortlist_status(now=NOW, trial_root=alias)["status"]
        == "evidence_invalid"
    )


def test_corrupt_large_or_duplicate_key_evaluation_is_bounded_failure(
    tmp_path, monkeypatch
):
    root, evaluation_path, _ = _native(tmp_path, monkeypatch)
    evaluation_path.write_bytes(b'{"x":1,"x":2}')
    assert (
        status.inspect_shortlist_status(now=NOW, trial_root=root)["status"]
        == "evidence_invalid"
    )
    evaluation_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    result = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert result["status"] == "evidence_invalid" and "bound" in result["reason"]


def test_diagnosis_never_writes_or_loads_canonical_labels(tmp_path, monkeypatch):
    root, _, _ = _native(tmp_path, monkeypatch)
    original_open = os.open

    def readonly_open(path, flags, *args, **kwargs):
        assert not flags & (
            os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND
        )
        return original_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("diagnosis wrote, published, evaluated, or joined outcomes")

    monkeypatch.setattr(os, "open", readonly_open)
    for name in ("write_bytes", "write_text", "mkdir", "unlink"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(native, "_publish_new", forbidden)
    monkeypatch.setattr(trial, "record_trade_shortlist_trial", forbidden)
    monkeypatch.setattr(evaluate, "load_recommendation_evidence", forbidden)
    monkeypatch.setattr(evaluate, "evaluate_trade_shortlist_trials", forbidden)
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == "complete_changed"


def test_evaluation_replaced_after_read_is_detected(tmp_path, monkeypatch):
    root, evaluation_path, _ = _native(tmp_path, monkeypatch)
    original = status._ReadSet.document

    def mutate(reader, path):
        document = original(reader, path)
        if Path(path) == evaluation_path:
            evaluation_path.write_bytes(evaluation_path.read_bytes() + b" ")
        return document

    monkeypatch.setattr(status._ReadSet, "document", mutate)
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == "evidence_invalid" and report["attention_required"]


def test_source_changes_during_probe_are_not_healthy_without_editing_sources(
    tmp_path, monkeypatch
):
    root, _, _ = _native(tmp_path, monkeypatch)
    original = evaluate._generator_sources
    calls = 0

    def changing_identity():
        nonlocal calls
        calls += 1
        sources = original()
        if calls > 1:
            sources[0]["sha256"] = "0" * 64
        return sources

    monkeypatch.setattr(evaluate, "_generator_sources", changing_identity)
    report = status.inspect_shortlist_status(now=NOW, trial_root=root)
    assert report["status"] == "evidence_invalid" and report["attention_required"]
    assert "source changed" in report["reason"]


def test_cli_exit_codes_and_explicit_scope(tmp_path, monkeypatch, capsys):
    root, _, _ = _native(tmp_path, monkeypatch)
    assert (
        status.main(["--trial-root", str(root), "--now", NOW, "--format", "text"]) == 0
    )
    text = capsys.readouterr().out
    assert "complete_changed" in text and "eligibility not checked" in text
    assert status.main(["--trial-root", str(tmp_path / "absent"), "--now", NOW]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "missing"
    assert status.main(["--trial-root", str(root), "--now", "naive"]) == 2
