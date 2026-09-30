from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from ops import dashboard_research as research
from ops import recommend_trial_review as review

NOW = datetime.fromisoformat("2026-09-30T11:00:00+09:00")
ASOF = "2026-09-30"


@pytest.fixture
def source():
    control = dict(zip(research.METRICS, (.01, .3, .2), strict=True))
    challenger = dict(zip(research.METRICS, (.005, .4, .1), strict=True))
    diff = {key: challenger[key] - control[key] for key in control}
    row = {
        "paired_dates": 20, "changed_dates": 10, "no_op_dates": 10,
        "changed_picks": 13, "picks_per_arm": 60,
        "historical_exclusions": [{"reason": "/private/token"}],
        "awaiting_outcomes": [ASOF],
        "metrics": {name: {"mean": value, "observed_date_block3_ci95": {"eod_return_net": [-.02, .01]}}
                    for name, value in (("control", control), ("challenger", challenger),
                                        ("challenger_minus_control", diff))},
    }
    return {
        "status": "evaluated", "attention_required": False, "deployable": False,
        "checked_at": NOW.isoformat(), "generated_at": "2026-09-30T10:20:00+09:00",
        "through_date": "2026-09-29", "reason": "PRIVATE", "path": "/private/snapshot",
        "summary": {name: copy.deepcopy(row) for name in ("boundary", "shortlist")},
        "forward": {"prospective_policy_records": 0, "paired_dates": 0,
                    "historical_exclusions": [], "changed_picks": dict.fromkeys(research.POLICIES, 0)},
    }


def project(source):
    return research.project_review(source, asof=ASOF, now=NOW)


def test_aggregates_only_no_private_details_or_live_verdict(source):
    before = copy.deepcopy(source)
    result = project(source)
    assert source == before
    assert result["trials"]["shortlist"]["excluded_dates"] == 1
    assert result["trials"]["shortlist"]["pending_dates"] == 1
    assert result["forward"]["ready_records"] == 0
    assert result["automatic_promotion"] is False
    assert "PRIVATE" not in json.dumps(result) and "/private" not in json.dumps(result)


@pytest.mark.parametrize("n", [0, 1, 4, 5])
def test_empty_and_small_samples_do_not_fake_zero_returns_or_ci(source, n):
    for row in source["summary"].values():
        row.update(paired_dates=n, changed_dates=0, no_op_dates=n, changed_picks=0, picks_per_arm=3*n)
        if n == 0:
            row["metrics"] = None
        elif n < 5:
            row["metrics"]["challenger_minus_control"]["observed_date_block3_ci95"] = None
    result = project(source)
    metrics = result["trials"]["shortlist"]["metrics"]
    if n == 0:
        assert metrics is None
    else:
        assert (metrics["net_difference_block_ci95"] is None) == (n < 5)


def test_missing_report_is_unknown_not_zero(source):
    source.update(status="unavailable", attention_required=True)
    result = project(source)
    assert result["trials"] == {} and result["forward"] is None
    assert result["through_date"] is None and result["attention_required"] is True


def test_incomplete_review_retains_available_cohort_with_warning(source):
    source.update(status="incomplete", attention_required=True)
    result = project(source)
    assert result["attention_required"] is True
    assert result["trials"]["shortlist"]["paired_dates"] == 20


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(private_path="/private"),
    lambda p: p.update(automatic_promotion=True),
    lambda p: p.update(attention_required=True),
    lambda p: p.update(observed_at=(NOW + timedelta(seconds=1)).isoformat()),
    lambda p: p.update(observed_at=(NOW - timedelta(hours=7)).isoformat()),
    lambda p: p.update(report_generated_at="2026-09-29T11:00:00+09:00", through_date="2026-09-28"),
    lambda p: p.update(through_date=ASOF),
    lambda p: p["trials"]["shortlist"].update(paired_dates=True),
    lambda p: p["trials"]["shortlist"].update(changed_picks=31),
    lambda p: p["trials"]["shortlist"].update(no_op_dates=11),
    lambda p: p["trials"]["shortlist"].update(picks_per_arm=59),
    lambda p: p["trials"]["shortlist"]["metrics"]["control"].update(dn5=1.01),
    lambda p: p["trials"]["shortlist"]["metrics"]["control"].update(eod_return_net=float("nan")),
    lambda p: p["trials"]["shortlist"]["metrics"]["challenger_minus_control"].update(up10=.9),
    lambda p: p["trials"]["shortlist"]["metrics"].update(net_difference_block_ci95=[.1, -.1]),
    lambda p: p["forward"].update(ready_records=1),
    lambda p: p["forward"].update(paired_dates=1),
    lambda p: p["forward"]["changed_picks"].update(recent=1),
    lambda p: p["forward"]["changed_picks"].update(fixed_r1=1),
])
def test_publication_rejects_contract_contradictions(source, mutate):
    result = project(source)
    mutate(result)
    with pytest.raises((ValueError, TypeError)):
        research.validate_research_progress(result, asof=ASOF, now=NOW)


def test_historical_asof_never_runs_current_probe(monkeypatch):
    monkeypatch.setattr(research.subprocess, "run", lambda *a, **k: pytest.fail("probe called"))
    result = research.build_research_progress(asof="2026-09-29", now=NOW)
    assert result["status"] == "historical_not_observed"
    research.validate_research_progress(result, asof="2026-09-29", now=NOW)


@pytest.mark.parametrize("fault", ["timeout", "exit", "json", "oversize", "invalid"])
def test_probe_failure_is_bounded_redacted_and_nonblocking(monkeypatch, fault):
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 20 and kwargs["cwd"] == research.ROOT
        assert "--refresh" not in args[0]
        if fault == "timeout":
            raise subprocess.TimeoutExpired("PRIVATE", 20)
        return SimpleNamespace(returncode=2 if fault == "exit" else 0,
                               stdout=b"x" * 32769 if fault == "oversize" else b'{"secret":"PRIVATE"}')

    monkeypatch.setattr(research.subprocess, "run", run)
    result = research.build_research_progress(asof=ASOF, now=NOW)
    assert result["status"] == "unavailable" and "PRIVATE" not in json.dumps(result)


def test_child_uses_native_inspector_without_refresh(source, monkeypatch, capsys):
    monkeypatch.setattr(review, "inspect_review", lambda **kw: source)
    monkeypatch.setattr(review, "build_review", lambda **kw: pytest.fail("refresh forbidden"))
    assert research.main(["--probe", "--now", NOW.isoformat()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == project(source)


def test_october_records_are_counted_separately_from_mature_comparisons(source):
    now = NOW.replace(month=10, day=3)
    source.update(checked_at=now.isoformat(), generated_at="2026-10-03T10:20:00+09:00", through_date="2026-10-02")
    source["forward"].update(prospective_policy_records=3, paired_dates=2)
    source["forward"]["changed_picks"]["recent"] = 3
    result = research.project_review(source, asof="2026-10-03", now=now)
    assert result["forward"]["ready_records"] == 3
    assert result["forward"]["paired_dates"] == 2
    assert result["forward"]["changed_picks"]["recent"] == 3


def test_native_due_clock_uses_kst_even_if_timestamp_has_utc_offset(source):
    now = datetime.fromisoformat("2026-09-30T00:59:00+00:00")
    source.update(checked_at=now.isoformat(), generated_at="2026-09-29T02:00:00+00:00",
                  through_date="2026-09-28")
    result = research.project_review(source, asof=ASOF, now=now)
    assert result["status"] == "evaluated"
    result["observed_at"] = NOW.isoformat()
    with pytest.raises(ValueError):
        research.validate_research_progress(result, asof=ASOF, now=NOW)


def test_build_and_asset_validator_are_wired():
    import ast
    from scripts import build_dashboard, validate_dashboard_assets
    from pathlib import Path

    for module, name in ((build_dashboard, "build_research_progress"),
                         (validate_dashboard_assets, "validate_research_progress")):
        tree = ast.parse(Path(module.__file__).read_text())
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == name for node in ast.walk(tree))
