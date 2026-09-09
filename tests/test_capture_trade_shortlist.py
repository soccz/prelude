"""Independent synthetic integration checks of the added Top10 wiring only."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest
import requests

from scripts import capture_recommend_microstructure as runner
from test_capture_recommend_microstructure import session as session

_LAZY_RECORD = runner.record_trade_shortlist


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic capture wiring test attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def wired(session, monkeypatch, tmp_path):
    config, capture, feature, old_calls = session
    config = replace(config, asof=date(2026, 9, 10))
    events, new_roots = [], []
    original_record = runner.record_microstructure_trial

    def old_record(*args, **kwargs):
        result = original_record(*args, **kwargs)
        events.append("old_commit")
        return result

    def new_record(snapshot_path, feature_path, *, output_root):
        assert len(old_calls) == 1
        assert feature_path.exists() and json.loads(feature_path.read_text()) == feature
        new_roots.append(Path(output_root))
        events.append("new_commit")
        return {"status": "committed", "score_path": str(output_root / "new_score.json")}

    def old_evaluate(*args):
        assert len(old_calls) == 1
        events.append("old_evaluation")
        return {"effect_status": "not_evaluated", "fixture": "original"}

    def new_evaluate(root):
        assert len(old_calls) == 1 and new_roots == [Path(root)]
        events.append("new_evaluation")
        return {"effect_status": "not_evaluated", "fixture": "shortlist"}

    monkeypatch.setattr(runner, "record_microstructure_trial", old_record)
    monkeypatch.setattr(runner, "record_trade_shortlist", new_record)
    monkeypatch.setattr(runner, "evaluate_microstructure_trials", old_evaluate)
    monkeypatch.setattr(runner, "evaluate_trade_shortlist", new_evaluate)
    return config, capture, feature, old_calls, events, new_roots


@pytest.mark.parametrize("day", [date(2026, 9, 8), date(2026, 9, 9)])
def test_before_launch_no_new_storage_publication_evaluation_or_output(wired, monkeypatch, tmp_path, day):
    config, _, _, old_calls, events, new_roots = wired
    storage_roots = []
    monkeypatch.setattr(runner, "capture_storage_preflight", lambda raw, trial, budget: storage_roots.append(trial) or [])
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    report = runner.run_session(replace(config, asof=day), trial_root=old_root, shortlist_root=new_root)
    assert events == ["old_commit", "old_evaluation"] and len(old_calls) == 1
    assert storage_roots == [old_root] and new_roots == []
    assert report["shortlist_due"] is False and report["shortlist_trial"] is None
    assert report["shortlist_evaluation_path"] is None and report["research_errors"] == []
    assert not new_root.exists() and not (tmp_path / "trade_shortlist_evaluation.json").exists()


@pytest.mark.parametrize("custom_root", [False, True])
def test_launch_order_records_both_predictions_before_either_cumulative_evaluation(wired, tmp_path, custom_root):
    config, capture, _, old_calls, events, new_roots = wired
    old_root = tmp_path / "old"
    selected_root = tmp_path / "custom_shortlist" if custom_root else None
    report = runner.run_session(config, trial_root=old_root, shortlist_root=selected_root)
    assert events == ["old_commit", "new_commit", "old_evaluation", "new_evaluation"]
    assert len(old_calls) == 1
    assert new_roots == [selected_root or tmp_path / "recommend_trade_shortlist_trials"]
    assert report["trial"]["status"] == report["shortlist_trial"]["status"] == "committed"
    assert report["shortlist_due"] is True and report["research_errors"] == []
    assert json.loads(Path(report["evaluation_path"]).read_text())["fixture"] == "original"
    assert json.loads(Path(report["shortlist_evaluation_path"]).read_text())["fixture"] == "shortlist"
    assert capture.manifest_path.read_text() == "{}"
    assert Path(capture.manifest["cutoff_source"]["path"]).read_text() == "{}"


@pytest.mark.parametrize("instant,due", [("2026-09-09T14:59:59+00:00", False), ("2026-09-09T15:00:00+00:00", True)])
def test_default_asof_uses_kst_calendar_at_launch_boundary(wired, monkeypatch, tmp_path, instant, due):
    config, _, _, _, events, _ = wired

    class Clock:
        @staticmethod
        def now(tz):
            assert tz.key == "Asia/Seoul"
            return datetime.fromisoformat(instant).astimezone(tz)

    monkeypatch.setattr(runner, "datetime", Clock)
    report = runner.run_session(replace(config, asof=None), trial_root=tmp_path / "old")
    assert report["shortlist_due"] is due
    assert ("new_commit" in events) is due


@pytest.mark.parametrize("failure", [OSError("statvfs failed"), RuntimeError("insufficient space"), ValueError("bad volume")])
def test_new_storage_failure_preserves_old_commit_and_evaluation(wired, monkeypatch, tmp_path, failure):
    config, _, _, old_calls, events, new_roots = wired
    old_root, new_root = tmp_path / "old", tmp_path / "new"

    def storage(raw, trial, budget):
        if trial == new_root:
            raise failure
        return []

    monkeypatch.setattr(runner, "capture_storage_preflight", storage)
    report = runner.run_session(config, trial_root=old_root, shortlist_root=new_root)
    assert events == ["old_commit", "old_evaluation"] and len(old_calls) == 1
    assert report["trial"]["status"] == "committed" and Path(report["evaluation_path"]).exists()
    assert new_roots == [] and report["shortlist_trial"] is None
    assert len(report["research_errors"]) == 1 and report["research_errors"][0].startswith("shortlist_storage:")


@pytest.mark.parametrize("failure", [OSError("disk"), ValueError("bad input"), ImportError("new dependency missing")])
def test_new_publisher_exception_does_not_suppress_old_evaluation(wired, monkeypatch, tmp_path, failure):
    config, _, _, old_calls, events, _ = wired

    def fail(*args, **kwargs):
        events.append("new_failure")
        raise failure

    monkeypatch.setattr(runner, "record_trade_shortlist", fail)
    report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == ["old_commit", "new_failure", "old_evaluation"]
    assert len(old_calls) == 1 and report["trial"]["status"] == "committed"
    assert Path(report["evaluation_path"]).exists() and report["shortlist_evaluation_path"] is None
    assert report["research_errors"][0].startswith("shortlist_publication:")


@pytest.mark.parametrize("status", ["uncertain", "missing"])
def test_new_unconfirmed_publication_is_not_evaluated_or_reported_success(wired, monkeypatch, tmp_path, status):
    config, _, _, old_calls, events, _ = wired
    monkeypatch.setattr(runner, "record_trade_shortlist", lambda *args, **kwargs: {"status": status})
    report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == ["old_commit", "old_evaluation"] and len(old_calls) == 1
    assert report["shortlist_trial"]["status"] == status
    assert report["research_errors"] == ["shortlist_publication_unconfirmed"]
    assert Path(report["evaluation_path"]).exists() and report["shortlist_evaluation_path"] is None


@pytest.mark.parametrize("at_write", [False, True])
def test_old_evaluation_failure_still_attempts_new_evaluation(wired, monkeypatch, tmp_path, at_write):
    config, _, _, old_calls, events, _ = wired
    if at_write:
        writer = runner.write_feature_once

        def write(path, document):
            if Path(path).name == "trial_evaluation.json":
                raise OSError("old evaluation persistence failed")
            writer(path, document)

        monkeypatch.setattr(runner, "write_feature_once", write)
    else:
        def fail(*args):
            events.append("old_evaluation")
            raise RuntimeError("bad original history")

        monkeypatch.setattr(runner, "evaluate_microstructure_trials", fail)
    report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == ["old_commit", "new_commit", "old_evaluation", "new_evaluation"]
    assert len(old_calls) == 1 and report["trial"]["status"] == report["shortlist_trial"]["status"] == "committed"
    assert report["evaluation_path"] is None and Path(report["shortlist_evaluation_path"]).exists()
    assert report["research_errors"][0].startswith("original_evaluation:")


@pytest.mark.parametrize("at_write", [False, True])
def test_new_evaluation_failure_keeps_old_result_and_both_publications(wired, monkeypatch, tmp_path, at_write):
    config, _, _, old_calls, events, _ = wired
    if at_write:
        writer = runner.write_feature_once

        def write(path, document):
            if Path(path).name == "trade_shortlist_evaluation.json":
                raise OSError("new evaluation persistence failed")
            writer(path, document)

        monkeypatch.setattr(runner, "write_feature_once", write)
    else:
        def fail(*args):
            events.append("new_evaluation")
            raise RuntimeError("bad new history")

        monkeypatch.setattr(runner, "evaluate_trade_shortlist", fail)
    report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == ["old_commit", "new_commit", "old_evaluation", "new_evaluation"]
    assert len(old_calls) == 1 and report["trial"]["status"] == report["shortlist_trial"]["status"] == "committed"
    assert Path(report["evaluation_path"]).exists() and report["shortlist_evaluation_path"] is None
    assert report["research_errors"][0].startswith("shortlist_evaluation:")


def test_old_unconfirmed_publication_stops_both_evaluations_and_new_publication(wired, monkeypatch, tmp_path):
    config, _, _, _, events, new_roots = wired
    monkeypatch.setattr(runner, "record_microstructure_trial", lambda *args, **kwargs: {"status": "uncertain"})
    report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == [] and new_roots == []
    assert report["unavailable_reason"] == "trial_publication_unconfirmed"
    assert report["shortlist_trial"] is report["evaluation_path"] is report["shortlist_evaluation_path"] is None


def _complete_report():
    return {
        "capture_complete": True, "feature_evidence_valid": True, "trial": {"status": "committed"},
        "evaluation_path": "old_evaluation.json", "shortlist_due": True,
        "shortlist_trial": {"status": "committed"}, "shortlist_evaluation_path": "new_evaluation.json",
        "research_errors": [], "effect_status": "not_evaluated",
    }


@pytest.mark.parametrize("field,value,expected", [
    (None, None, 0), ("capture_complete", False, 1), ("feature_evidence_valid", False, 1),
    ("trial", None, 1), ("trial", {"status": "uncertain"}, 1), ("evaluation_path", None, 1),
    ("shortlist_trial", None, 1), ("shortlist_trial", {"status": "uncertain"}, 1),
    ("shortlist_evaluation_path", None, 1), ("research_errors", ["new_storage_failure"], 1),
])
def test_cli_success_requires_every_due_path_and_zero_errors(monkeypatch, field, value, expected):
    report = _complete_report()
    if field:
        report[field] = value
    before = copy.deepcopy(report)
    monkeypatch.setattr(runner, "capture_storage_preflight", lambda *args: [])
    monkeypatch.setattr(runner.collector, "_resolve_markets", lambda *args: (("KRW-BTC",), "explicit", None, 1))
    monkeypatch.setattr(runner, "run_session", lambda *args, **kwargs: report)
    assert runner.main(["--asof", "2026-09-10"]) == expected
    assert report == before


def test_cli_new_storage_failure_is_rechecked_and_reported_without_blocking_old_capture(wired, monkeypatch, tmp_path):
    _, _, _, old_calls, events, new_roots = wired
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    storage_calls = []

    def storage(raw, trial, budget):
        storage_calls.append(Path(trial))
        if Path(trial) == new_root:
            raise RuntimeError("new trial volume full")
        return []

    monkeypatch.setattr(runner, "capture_storage_preflight", storage)
    monkeypatch.setattr(runner.collector, "_resolve_markets", lambda *args: (("KRW-BTC",), "explicit", None, 1))
    code = runner.main(["--asof", "2026-09-10", "--output-root", str(tmp_path),
                        "--trial-root", str(old_root), "--shortlist-root", str(new_root)])
    assert code == 1 and storage_calls == [old_root, new_root, old_root, new_root]
    assert events == ["old_commit", "old_evaluation"] and len(old_calls) == 1 and new_roots == []
    assert (tmp_path / "trial_evaluation.json").exists()


def test_launch_storage_recheck_after_lookup_blocks_capture_without_calendar_dependent_expectations(monkeypatch, tmp_path):
    looked_up, checks = [], []

    def storage(*args):
        checks.append(bool(looked_up))
        if looked_up:
            raise RuntimeError("old trial volume became full")
        return []

    def markets(*args):
        looked_up.append(True)
        return ("KRW-BTC",), "explicit", None, 1

    monkeypatch.setattr(runner, "capture_storage_preflight", storage)
    monkeypatch.setattr(runner.collector, "_resolve_markets", markets)
    monkeypatch.setattr(runner.collector, "run_capture", lambda *args: pytest.fail("capture must not start"))
    assert runner.main(["--asof", "2026-09-10", "--output-root", str(tmp_path),
                        "--trial-root", str(tmp_path / "old")]) == 2
    assert checks == [False, False, True]


def test_lazy_new_import_failure_occurs_only_after_old_score(wired, monkeypatch, tmp_path):
    config, _, _, old_calls, events, _ = wired
    import builtins

    original_import = builtins.__import__

    def fail_new_import(name, *args, **kwargs):
        if name == "signals.recommend_trade_shortlist_trial":
            assert len(old_calls) == 1 and events == ["old_commit"]
            raise ImportError("broken new-only trial import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runner, "record_trade_shortlist", _LAZY_RECORD)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", fail_new_import)
        report = runner.run_session(config, trial_root=tmp_path / "old")
    assert events == ["old_commit", "old_evaluation"]
    assert report["research_errors"][0].startswith("shortlist_publication:ImportError:")
    assert Path(report["evaluation_path"]).exists()
