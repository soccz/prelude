"""Selftest status evidence only: no real unit, test suite, or Telegram call."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ops import selftest_status as status

NOW = datetime.fromisoformat("2026-09-10T10:30:00+09:00")


def _properties(**changes):
    values = {
        "Id": status.UNIT,
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "Result": "success",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "ExecMainStartTimestamp": "Wed 2026-09-09 22:30:29 UTC",
        "ExecMainExitTimestamp": "Wed 2026-09-09 22:48:12 UTC",
    }
    values.update(changes)
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def _stub(monkeypatch, output=None, *, returncode=0, error=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if error is not None:
            raise error
        return SimpleNamespace(
            returncode=returncode,
            stdout=_properties() if output is None else output,
            stderr="private diagnostic must never enter the report",
        )

    monkeypatch.setattr(status.subprocess, "run", run)
    return calls


def test_success_uses_kst_date_and_a_fixed_readonly_bounded_command(monkeypatch):
    calls = _stub(monkeypatch)
    report = status.inspect_selftest(now=NOW)
    assert report["state"] == "passed"
    assert report["attention_required"] is False
    assert report["started_at"] == "2026-09-10T07:30:29+09:00"
    assert report["completed_at"] == "2026-09-10T07:48:12+09:00"
    command, options = calls[0]
    assert len(calls) == 1
    assert command[:4] == ["systemctl", "show", status.UNIT, "--no-pager"]
    assert command[4::2] == ["--property"] * len(status.PROPERTIES)
    assert command[5::2] == list(status.PROPERTIES)
    assert options["timeout"] == 8
    assert options["check"] is False
    assert options["env"]["LC_ALL"] == "C"
    assert options["env"]["TZ"] == "UTC"


@pytest.mark.parametrize(
    "changes,expected",
    [
        (
            {
                "ActiveState": "failed",
                "SubState": "failed",
                "Result": "exit-code",
                "ExecMainStatus": "1",
            },
            "failed",
        ),
        ({"Result": "timeout"}, "failed"),
        ({"ActiveState": "activating", "SubState": "start"}, "in_progress"),
        ({"ActiveState": "active", "SubState": "running"}, "in_progress"),
        ({"ActiveState": "deactivating", "SubState": "stop"}, "in_progress"),
        (
            {
                "ExecMainStartTimestamp": "",
                "ExecMainExitTimestamp": "",
                "ExecMainCode": "0",
            },
            "not_run",
        ),
        (
            {
                "ExecMainStartTimestamp": "Tue 2026-09-08 22:30:00 UTC",
                "ExecMainExitTimestamp": "Tue 2026-09-08 22:48:00 UTC",
            },
            "stale",
        ),
        ({"ExecMainStartTimestamp": "Wed 2026-09-09 14:59:00 UTC"}, "stale"),
        ({"ExecMainStatus": "1"}, "invalid_state"),
        ({"ExecMainCode": "2"}, "invalid_state"),
        ({"SubState": "failed"}, "invalid_state"),
        ({"LoadState": "not-found"}, "invalid_state"),
        ({"Id": "unrelated.service"}, "invalid_state"),
        ({"ActiveState": "unknown"}, "invalid_state"),
        ({"Result": ""}, "invalid_state"),
        ({"ExecMainStatus": ""}, "invalid_state"),
        ({"ExecMainStatus": "zero"}, "invalid_state"),
        ({"ExecMainCode": ""}, "invalid_state"),
        ({"ExecMainExitTimestamp": ""}, "invalid_state"),
        ({"ExecMainExitTimestamp": "Wed 2026-09-09 22:29:00 UTC"}, "invalid_state"),
        ({"ExecMainExitTimestamp": "Thu 2026-09-10 02:00:00 UTC"}, "invalid_state"),
        ({"ExecMainExitTimestamp": "2026-09-09 22:48:12"}, "invalid_state"),
        ({"ExecMainExitTimestamp": "Wed 2026-09-09 22:48:12 KST"}, "invalid_state"),
        ({"ExecMainExitTimestamp": "Tue 2026-09-09 22:48:12 UTC"}, "invalid_state"),
    ],
)
def test_non_successful_or_inconsistent_evidence_cannot_pass(
    monkeypatch, changes, expected
):
    _stub(monkeypatch, _properties(**changes))
    report = status.inspect_selftest(now=NOW)
    assert report["state"] == expected
    assert report["attention_required"] is True


@pytest.mark.parametrize(
    "output",
    [
        "",
        _properties().replace("Result=success\n", ""),
        _properties() + "Result=success\n",
        _properties() + "Unexpected=0\n",
        _properties() + "malformed\n",
    ],
)
def test_missing_duplicate_or_malformed_properties_fail_closed(monkeypatch, output):
    _stub(monkeypatch, output)
    report = status.inspect_selftest(now=NOW)
    assert report["state"] == "probe_error"
    assert report["attention_required"] is True


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("private path"),
        PermissionError("private path"),
        subprocess.TimeoutExpired("systemctl", 8),
    ],
)
def test_missing_systemctl_permissions_or_timeout_are_attention(monkeypatch, error):
    _stub(monkeypatch, error=error)
    report = status.inspect_selftest(now=NOW)
    assert report["state"] == "probe_error"
    assert report["attention_required"] is True
    assert "private" not in json.dumps(report)


def test_nonzero_exit_cannot_be_overridden_by_success_stdout(monkeypatch):
    _stub(monkeypatch, returncode=1)
    report = status.inspect_selftest(now=NOW)
    assert report["state"] == "probe_error"
    assert report["attention_required"] is True
    assert "private" not in json.dumps(report)


def test_real_cli_without_systemctl_is_a_failed_probe():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "ops.selftest_status", "--format", "json"],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PATH": "",
            "PRELUDE_FORBID_TELEGRAM": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["state"] == "probe_error"
    assert report["attention_required"] is True
    assert "FileNotFoundError" in report["reason"]


def test_same_day_successful_rerun_checks_latest_execution_not_old_failure(monkeypatch):
    _stub(
        monkeypatch,
        _properties(
            ExecMainStartTimestamp="Thu 2026-09-10 00:00:00 UTC",
            ExecMainExitTimestamp="Thu 2026-09-10 00:20:00 UTC",
        ),
    )
    assert status.inspect_selftest(now=NOW)["state"] == "passed"


def test_naive_clock_is_rejected_before_any_process(monkeypatch):
    calls = _stub(monkeypatch)
    with pytest.raises(ValueError, match="timezone-aware"):
        status.inspect_selftest(now=datetime(2026, 9, 10))
    assert not calls


def test_equivalent_utc_clock_keeps_kst_date(monkeypatch):
    _stub(monkeypatch)
    assert (
        status.inspect_selftest(now=NOW.astimezone(timezone.utc))["asof"]
        == "2026-09-10"
    )


@pytest.mark.parametrize("attention", [False, True])
def test_cli_exit_is_based_on_evidence(monkeypatch, capsys, attention):
    monkeypatch.setattr(
        status,
        "inspect_selftest",
        lambda: {
            "attention_required": attention,
            "asof": "2026-09-10",
            "state": "failed" if attention else "passed",
            "reason": "verified fixture",
        },
    )
    assert status.main(["--format", "text"]) == int(attention)
    assert "selftest 2026-09-10:" in capsys.readouterr().out
    assert status.main(["--format", "json"]) == int(attention)
    assert json.loads(capsys.readouterr().out)["attention_required"] is attention
