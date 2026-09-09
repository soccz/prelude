"""Static service graph/resource contracts; never starts a systemd service."""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest


DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SELFTEST = "prelude-selftest.service"
RECOMMENDATIONS = ("prelude-preopen.service", "prelude-distribution.service")
CLOSES = ("prelude-close.service", "prelude-preopen-close.service")


def _unit(path):
    document = configparser.ConfigParser(interpolation=None)
    document.read_string(path.read_text())
    return document


def _graph():
    graph = {}
    for path in DEPLOY.glob("prelude-*.service"):
        unit = _unit(path)
        graph.setdefault(path.name, set())
        for later in unit["Unit"].get("Before", "").split():
            graph[path.name].add(later)
        for earlier in unit["Unit"].get("After", "").split():
            graph.setdefault(earlier, set()).add(path.name)
    return graph


def _has_path(graph, start, finish):
    pending, visited = [start], set()
    while pending:
        current = pending.pop()
        if current == finish:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(graph.get(current, ()))
    return False


@pytest.mark.parametrize("recommendation", RECOMMENDATIONS)
def test_legacy_ordering_reproduces_late_boot_blocking_path(recommendation):
    graph = _graph()
    # Reconstruct only the retired ordering edges, not a running scheduler.
    graph.setdefault(SELFTEST, set()).update(RECOMMENDATIONS)
    for close in CLOSES:
        graph.setdefault(close, set()).add(SELFTEST)
    assert _has_path(graph, SELFTEST, recommendation)
    for close in CLOSES:
        assert _has_path(graph, close, recommendation)


@pytest.mark.parametrize("recommendation", RECOMMENDATIONS)
def test_current_selftest_has_no_direct_or_transitive_recommendation_order(recommendation):
    graph = _graph()
    assert not _has_path(graph, SELFTEST, recommendation)
    unit = _unit(DEPLOY / recommendation)
    for kind in ("Requires", "Wants", "Requisite", "BindsTo"):
        assert SELFTEST not in unit["Unit"].get(kind, "").split()


@pytest.mark.parametrize("close", CLOSES)
def test_selftest_does_not_wait_for_close_catchup(close):
    assert not _has_path(_graph(), close, SELFTEST)


def test_resource_limits_replace_ordering_not_test_execution():
    unit = _unit(DEPLOY / SELFTEST)
    service = unit["Service"]
    assert service["ExecStart"] == (
        "/home/soccz/22tb/prelude/venv/bin/python -m pytest -q -p no:cacheprovider"
    )
    assert service["Nice"] == "19"
    assert service["CPUQuota"] == "50%"
    assert service["CPUWeight"] == "10"
    assert service["IOSchedulingClass"] == "idle"
    assert service["IOWeight"] == "10"
    assert service["MemoryMax"] == "4G"
    assert service["TimeoutStartSec"] == "2700"
    assert service["User"] == "soccz"
    assert service["NoNewPrivileges"] == "true"
    assert service["UMask"] == "0077"
    assert service["StandardOutput"] == service["StandardError"] == "journal"
    assert unit["Unit"]["OnFailure"] == "prelude-failure-alert@%n.service"
    assert unit["Unit"]["After"].split() == ["network-online.target"]
    assert not unit["Unit"].get("Before", "")
    for name in ("ExecCondition", "ExecStartPre", "SuccessExitStatus"):
        assert name not in service
    assert not any(name.lower().startswith("condition") for name in unit["Unit"])


def test_thread_and_notification_guards_reach_pytest_children():
    environment = _unit(DEPLOY / SELFTEST)["Service"]["Environment"].split()
    assert len(environment) == len(set(environment))
    expected = {
        "TZ=Asia/Seoul", "TMPDIR=/home/soccz/22tb/tmp",
        "PRELUDE_FORBID_TELEGRAM=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    }
    assert set(environment) == expected


def test_daily_timer_and_persistent_catchup_are_unchanged():
    timer = _unit(DEPLOY / "prelude-selftest.timer")
    assert dict(timer["Timer"]) == {
        "oncalendar": "*-*-* 07:30:00 Asia/Seoul",
        "persistent": "true",
        "randomizeddelaysec": "60",
    }
    assert timer["Install"]["WantedBy"] == "timers.target"
