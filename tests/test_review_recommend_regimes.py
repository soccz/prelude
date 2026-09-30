"""Canonical-evidence adapter and new-report-only CLI isolation."""

from copy import deepcopy
from datetime import date, datetime, timezone

import pytest

from ops import recommend_trial_review as daily_review
from ops.artifact_provenance import atomic_write_json, file_identity
from ops.recommendation_evidence import EvidenceError, EvidenceUnavailable
from scripts import review_recommend_regimes as review
from test_recommend_training_data import _evidence


def _install(tmp_path, monkeypatch, documents=None):
    document = _evidence()
    document["snapshot"].update({"created_at": document["snapshot"]["decision_completed_at"],
                                 "code": {"score_source_sha256": "a" * 64},
                                 "rule_version": "v1", "score_schema_version": 1})
    source = tmp_path / "input.json"
    atomic_write_json(source, {"evidence": True})
    document["manifest"]["files"] = {"snapshot": file_identity(source, root=review.ROOT)}
    documents = documents or [document]
    paths = [tmp_path / f"2026-09-{i + 1:02d}" / "open_r1.json" for i in range(len(documents))]
    mapping = dict(zip(paths, documents, strict=True))
    monkeypatch.setattr(review, "discover_r1_snapshots", lambda *a, **kw: paths)

    def load(path, **kwargs):
        value = mapping[path]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    monkeypatch.setattr(review, "load_recommendation_evidence", load)
    return document, source


def _build():
    return review.build_review(start_date=date(2026, 9, 1), end_date=date(2026, 9, 3),
                              slots=("open",), now=datetime(2026, 9, 4, tzinfo=timezone.utc), n_boot=10)


def test_review_does_not_mutate_inputs_and_reports_missing(tmp_path, monkeypatch):
    document, source = _install(tmp_path, monkeypatch)
    before, original = source.read_bytes(), deepcopy(document)
    result = _build()
    assert result["summary"]["snapshots_included"] == 1
    assert result["summary"]["source_versions"] == 1
    assert len(result["missing_snapshots"]) == 2
    assert result["diagnostics"]["groups"][0]["r1"]["mean"]["eod_return_net"] == pytest.approx(.0185)
    assert not result["deployable"]
    assert document == original and source.read_bytes() == before


def test_unavailable_is_excluded_not_zero_but_corrupt_is_fatal(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, [EvidenceUnavailable("legacy_snapshot")])
    report = _build()
    assert report["excluded"][0]["reason"] == "legacy_snapshot"
    assert report["diagnostics"]["groups"] == []
    _install(tmp_path, monkeypatch, [EvidenceError("bad digest")])
    with pytest.raises(EvidenceError, match="bad digest"):
        _build()


def test_source_change_during_build_fails(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    values = iter([[{"sha": "before"}], [{"sha": "after"}]])
    monkeypatch.setattr(review, "_sources", lambda: next(values))
    with pytest.raises(EvidenceError, match="sources changed"):
        _build()


def test_evidence_change_during_build_fails(tmp_path, monkeypatch):
    _, source = _install(tmp_path, monkeypatch)
    original = review.failure_summary

    def mutate(records, **kwargs):
        atomic_write_json(source, {"evidence": "changed"})
        return original(records, **kwargs)

    monkeypatch.setattr(review, "failure_summary", mutate)
    with pytest.raises(EvidenceError, match="inputs changed"):
        _build()


def test_delivered_less_than_three_is_not_an_imputed_full_cohort():
    evidence = _evidence(count=2)
    with pytest.raises(EvidenceUnavailable, match="three"):
        review._record(evidence)


def test_halted_selected_excluded_and_halted_nonselected_disclosed():
    evidence = _evidence()
    halted = evidence["label"]["rows"][-1]
    halted["label_status"] = "halted_no_observations"
    for key in ("up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"):
        halted[key] = None
    record = review._record(evidence)
    assert record["labeled_universe_size"] == 3
    assert record["halted_universe_size"] == 1
    evidence["snapshot"]["top3"][-1] = evidence["snapshot"]["universe"][-1]
    with pytest.raises(EvidenceUnavailable, match="selected_outcome"):
        review._record(evidence)


def test_cli_writes_new_report_only_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    _install(tmp_path, monkeypatch)
    output = tmp_path / "new.json"
    native_write = review.write_new_report
    monkeypatch.setattr(review, "write_new_report", lambda p, r: native_write(p, r, root=tmp_path))
    args = ["--start-date", "2026-09-01", "--end-date", "2026-09-03", "--output", str(output)]
    assert review.main(args) == 0
    before = output.read_bytes()
    assert review.main(args) == 1
    assert output.read_bytes() == before
    assert "FileExistsError" in capsys.readouterr().err


def test_native_adapter_uses_later_score_durability_not_original_alert(tmp_path, monkeypatch):
    evidence, _ = _install(tmp_path, monkeypatch)
    snapshot, label = tmp_path / "snapshot.json", tmp_path / "label.json"
    atomic_write_json(snapshot, evidence["snapshot"])
    atomic_write_json(label, evidence["label"])
    metric = review._record(evidence)["outcomes"]["control"]
    audit = {"date": "2026-09-01", "status": "prospective_comparable",
             "score_durable_observed_at": "2026-09-01T09:10:00+09:00",
             "canonical_execution_start_at": "2026-09-01T09:15:00+09:00",
             "evidence_manifest": {"files": {"snapshot": {"path": str(snapshot)}, "label": {"path": str(label)}}}}
    evaluation = {"trial_id": "test", "dates": [audit], "primary_paired": {"daily": [
        {"date": "2026-09-01", "changed_picks": 1, "control": metric, "challenger": metric},
    ]}}
    result = daily_review.adaptive_replay(evaluation, n_boot=10)
    assert result["plans"][0]["decision_at"] == audit["score_durable_observed_at"]
    assert result["plans"][0]["context"]["state"] == "broad:contracting"
    assert result["policies"]["fixed_r1"]["n_dates"] == 1
    audit["score_durable_observed_at"] = audit["canonical_execution_start_at"]
    with pytest.raises(ValueError, match="before common entry"):
        daily_review.adaptive_replay(evaluation, n_boot=10)
