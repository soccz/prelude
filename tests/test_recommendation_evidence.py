"""Cross-document checks; individual strict-loader contracts have own tests."""
from copy import deepcopy
from datetime import date, datetime, timezone

import pytest

import ops.recommendation_evidence as evidence


@pytest.fixture
def documents(tmp_path, monkeypatch):
    day = "2026-09-01"
    paths = {
        key: tmp_path / key / day / "open_r1.json"
        for key in ("snapshot", "label", "receipt")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    candidate = dict.fromkeys(evidence._CANDIDATE_KEYS)
    candidate.update(coin="KRW-TEST", rank=1, feature_values={"f_ret_1d": 0.1})
    snapshot = {
        "asof": day, "slot": "open", "ranking": "R1",
        "snapshot_schema": evidence.SNAPSHOT_SCHEMA_VERSION,
        "score_schema_version": "recommend_score.v2",
        "rank_basis": "R1_riskreward(de-corr head)",
        "snapshot_id": "recommend-test", "payload_sha256": "snapshot-hash",
        "snapshot_path": str(paths["snapshot"]), "feature_asof": day,
        "model": {"ranking": "R1", "id": "recommend_r1_open"},
        "rule": {"version": "r1_riskreward_v1"}, "code": {}, "data": {},
        "decision_started_at": f"{day}T09:07:00+09:00",
        "decision_completed_at": f"{day}T09:08:00+09:00",
        "created_at": f"{day}T09:08:00+09:00",
        "universe": [candidate], "top3": [candidate],
    }
    receipt = {
        "delivery_ok": True, "telegram_messages": [{"message_id": 1}],
        "attempted_at": f"{day}T09:08:01+09:00",
        "sent_at": f"{day}T09:08:02+09:00",
        "recorded_at": f"{day}T09:08:03+09:00",
    }
    label = {
        "artifact_status": "complete", "label_code": {},
        "round_trip_cost_fraction": 0.0015,
        "provenance_cohort": "forward_observed", "forward_eligible": True,
        "asof": day, "slot": "open", "ranking": "R1",
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_payload_sha256": snapshot["payload_sha256"],
        "snapshot_model": snapshot["model"], "snapshot_rule": snapshot["rule"],
        "snapshot_code": {}, "snapshot_data": {}, "feature_asof": day,
        "snapshot_path": str(paths["snapshot"]), "receipt_path": str(paths["receipt"]),
        "delivery_ok": True, "execution_time_basis": "delivery_sent_at",
        "execution_at": receipt["sent_at"],
        "execution_start_at": f"{day}T09:15:00+09:00",
        "path_window_start": f"{day}T09:15:00+09:00",
        "path_window_end": "2026-09-02T09:15:00+09:00",
        "labeled_at": "2026-09-02T10:00:00+09:00",
        "label_payload_sha256": "label-hash", "rows": [deepcopy(candidate)],
    }
    monkeypatch.setattr(evidence, "load_snapshot", lambda *a, **k: deepcopy(snapshot))
    monkeypatch.setattr(evidence, "read_delivery_receipt", lambda *a, **k: deepcopy(receipt))
    monkeypatch.setattr(evidence, "load_label_artifact", lambda *a, **k: deepcopy(label))
    kwargs = {
        "label_root": tmp_path / "label", "receipt_root": tmp_path / "receipt",
        "now": datetime(2026, 9, 3, tzinfo=timezone.utc),
    }
    return paths, snapshot, receipt, label, kwargs


def test_bound_evidence_records_source_hashes_and_availability(documents):
    paths, _, _, _, kwargs = documents
    before = {key: path.read_bytes() for key, path in paths.items()}
    report = evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)
    assert report["manifest"]["label_available_at"] == "2026-09-02T10:00:00+09:00"
    assert all(len(item["sha256"]) == 64 for item in report["manifest"]["files"].values())
    assert before == {key: path.read_bytes() for key, path in paths.items()}
    assert not list(paths["snapshot"].parents[2].rglob("*.lock"))


@pytest.mark.parametrize("key,value", [
    ("snapshot_payload_sha256", "other"), ("slot", "preopen"),
    ("snapshot_code", {"changed": True}), ("round_trip_cost_fraction", 0.003),
    ("execution_start_at", "2026-09-01T09:30:00+09:00"),
    ("path_window_end", "2026-09-02T09:30:00+09:00"),
    ("labeled_at", "2026-09-02T09:14:00+09:00"),
])
def test_cross_document_contradiction_is_not_a_normal_exclusion(documents, key, value):
    paths, _, _, label, kwargs = documents
    label[key] = value
    with pytest.raises(evidence.EvidenceError) as error:
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)
    assert not isinstance(error.value, evidence.EvidenceUnavailable)


def test_future_label_is_unavailable_not_training_evidence(documents):
    paths, _, _, _, kwargs = documents
    kwargs["now"] = datetime.fromisoformat("2026-09-02T09:20:00+09:00")
    with pytest.raises(evidence.EvidenceUnavailable, match="label_not_available"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_candidate_values_are_bound_not_only_ids(documents):
    paths, _, _, label, kwargs = documents
    label["rows"][0]["feature_values"]["f_ret_1d"] = 0.2
    with pytest.raises(evidence.EvidenceError, match="candidate_mismatch"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_missing_receipt_is_explicitly_unknown(documents):
    paths, _, _, _, kwargs = documents
    paths["receipt"].unlink()
    with pytest.raises(evidence.EvidenceUnavailable, match="delivery_unknown"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_delivery_cannot_precede_decision_completion(documents):
    paths, snapshot, _, _, kwargs = documents
    snapshot["decision_completed_at"] = "2026-09-01T09:10:00+09:00"
    snapshot["created_at"] = snapshot["decision_completed_at"]
    with pytest.raises(evidence.EvidenceError, match="decision_delivery_chronology"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_unapproved_score_schema_is_rejected(documents):
    paths, snapshot, _, _, kwargs = documents
    snapshot["score_schema_version"] = "recommend_score.v1"
    with pytest.raises(evidence.EvidenceError, match="unapproved_r1_ranking"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_corrupt_snapshot_not_hidden_by_missing_label(documents, monkeypatch):
    paths, _, _, _, kwargs = documents
    paths["label"].unlink()

    def invalid_loader(*args, **kw):
        raise ValueError("corrupt snapshot")

    monkeypatch.setattr(evidence, "load_snapshot", invalid_loader)
    with pytest.raises(evidence.EvidenceError, match="evidence_invalid") as error:
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)
    assert not isinstance(error.value, evidence.EvidenceUnavailable)


def test_same_named_legacy_label_is_excluded(documents):
    paths, _, _, label, kwargs = documents
    del label["round_trip_cost_fraction"]
    with pytest.raises(evidence.EvidenceUnavailable, match="legacy_label"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_replay_is_excluded(documents):
    paths, _, _, label, kwargs = documents
    label.update(provenance_cohort="scheduled_replay", forward_eligible=False)
    with pytest.raises(evidence.EvidenceUnavailable, match="not_forward"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_changed_source_is_rejected(documents, monkeypatch):
    paths, _, _, label, kwargs = documents

    def changing_loader(*args, **kw):
        paths["label"].write_text('{"changed":true}', encoding="utf-8")
        return deepcopy(label)

    monkeypatch.setattr(evidence, "load_label_artifact", changing_loader)
    with pytest.raises(evidence.EvidenceError, match="evidence_changed"):
        evidence.load_recommendation_evidence(paths["snapshot"], **kwargs)


def test_symlink_input_is_rejected(documents, tmp_path):
    paths, _, _, _, kwargs = documents
    alias = tmp_path / "alias.json"
    alias.symlink_to(paths["snapshot"])
    with pytest.raises(evidence.EvidenceError, match="symlink"):
        evidence.load_recommendation_evidence(alias, **kwargs)


def test_discovery_excludes_noncanonical_files(documents):
    paths, _, _, _, _ = documents
    directory = paths["snapshot"].parent
    for name in ("open_r2.json", "open_r1.limit2.json", "copy.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    assert evidence.discover_r1_snapshots(
        directory.parent, start_date=date(2026, 9, 1), end_date=date(2026, 9, 2)
    ) == [paths["snapshot"]]


@pytest.mark.parametrize("value", ["2026-09-01", datetime(2026, 9, 1), None, 0])
def test_naive_or_invalid_asof_time_is_rejected(value):
    with pytest.raises(evidence.EvidenceError, match="timezone-aware"):
        evidence.aware_time(value)
