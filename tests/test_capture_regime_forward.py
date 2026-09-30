"""New pre-entry recorder is downstream of both original immutable trials."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from scripts import capture_recommend_microstructure as runner
from ops import recommend_regime_forward as forward
from test_capture_trade_shortlist import wired as wired, no_network as no_network, _complete_report
from test_capture_recommend_microstructure import session as session  # noqa: F401


def test_prelaunch_never_calls_new_publisher(wired, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "record_regime_forward", lambda *a, **kw: pytest.fail("prelaunch publication"))
    report = runner.run_session(replace(wired[0], asof=date(2026, 9, 30)), trial_root=tmp_path / "old")
    assert not report["regime_due"] and report["regime_trial"] is None
    assert not report["research_errors"]
    assert runner.REGIME_START == forward.START


def test_new_stage_receives_durable_morning_evaluation_after_originals(wired, monkeypatch, tmp_path):
    config, _, _, _, events, _ = wired
    destination = tmp_path / "custom_forward"

    def publish(path, **kwargs):
        assert path.is_file() and events == ["old_commit", "new_commit", "old_evaluation", "new_evaluation"]
        assert kwargs["output_root"] == destination
        events.append("forward")
        return {"status": "committed", "eligibility": "ready"}

    monkeypatch.setattr(runner, "record_regime_forward", publish)
    report = runner.run_session(replace(config, asof=forward.START), trial_root=tmp_path / "old", regime_root=destination)
    assert events[-1] == "forward"
    assert report["regime_due"] and report["research_errors"] == []


@pytest.mark.parametrize("fault", [OSError("disk"), ImportError("dependency"), ValueError("bad history")])
def test_new_failure_cannot_erase_original_results(wired, monkeypatch, tmp_path, fault):
    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(runner, "record_regime_forward", fail)
    report = runner.run_session(replace(wired[0], asof=forward.START), trial_root=tmp_path / "old")
    assert report["trial"]["status"] == report["shortlist_trial"]["status"] == "committed"
    assert Path(report["evaluation_path"]).exists() and Path(report["shortlist_evaluation_path"]).exists()
    assert report["research_errors"][0].startswith("regime_publication:")


@pytest.mark.parametrize("status, eligibility, error", [("committed", "late", True),
                                                       ("uncertain", "uncertain", True),
                                                       ("committed", "unavailable", False)])
def test_late_uncertain_and_structural_unavailable_are_distinct(wired, monkeypatch, tmp_path, status, eligibility, error):
    monkeypatch.setattr(runner, "record_regime_forward", lambda *a, **kw: {"status": status, "eligibility": eligibility})
    report = runner.run_session(replace(wired[0], asof=forward.START), trial_root=tmp_path / "old")
    assert bool(report["research_errors"]) is error


def test_missing_native_evaluation_never_constructs_a_forward_score(wired, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("native unavailable")

    monkeypatch.setattr(runner, "evaluate_trade_shortlist", fail)
    monkeypatch.setattr(runner, "record_regime_forward", lambda *a, **kw: pytest.fail("missing native input"))
    report = runner.run_session(replace(wired[0], asof=forward.START), trial_root=tmp_path / "old")
    assert report["trial"]["status"] == "committed"
    assert "regime_missing_native_evaluation" in report["research_errors"]


@pytest.mark.parametrize("trial,expected", [(None, 1), ({"status": "committed", "eligibility": "late"}, 1),
                                          ({"status": "committed", "eligibility": "ready"}, 0),
                                          ({"status": "committed", "eligibility": "unavailable"}, 0)])
def test_cli_cannot_report_success_without_due_forward_record(monkeypatch, capsys, trial, expected):
    report = _complete_report()
    report.update(regime_due=True, regime_trial=trial)
    monkeypatch.setattr(runner, "capture_storage_preflight", lambda *a, **kw: [])
    monkeypatch.setattr(runner.collector, "_resolve_markets", lambda *a: (("KRW-BTC",), "explicit", None, 1))
    monkeypatch.setattr(runner, "run_session", lambda *a, **kw: report)
    assert runner.main(["--asof", "2026-10-01"]) == expected
