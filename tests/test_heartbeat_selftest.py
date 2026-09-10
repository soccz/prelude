"""Selftest warning integration in an isolated fake heartbeat environment."""

import pytest

from test_heartbeat_microstructure import ROOT, SELFTEST_TIMEOUT, _run

PROBE = "-B -m ops.selftest_status --format text"
WARNING = "selftest daily completion probe FAIL"


def test_today_passed_selftest_remains_silent(tmp_path):
    result, calls, log, message = _run(tmp_path, probe=PROBE, status="passed")
    assert result.returncode == 0
    assert calls.count(PROBE) == 1
    assert "selftest daily completion checked" in log
    assert "[ok] all checks pass" in log
    assert message is None


@pytest.mark.parametrize("probe_rc", [1, 2, 7, 124, 127, 137])
def test_selftest_fault_warns_once_and_preserves_other_checks(tmp_path, probe_rc):
    result, calls, log, message = _run(
        tmp_path, probe=PROBE, status="passed", probe_rc=probe_rc
    )
    assert result.returncode == 0
    assert calls.count(PROBE) == 1
    assert "selftest daily completion checked" not in log
    assert "microstructure daily status checked" in log
    assert "shortlist daily status checked" in log
    assert "backup manifest/archive/checksum provenance ok" in log
    assert "scripts/v2_scoreboard.py" in calls
    assert log.count(WARNING) == 1
    assert message is not None and message.count(WARNING) == 1
    assert "[ok] all checks pass" not in log
    assert "[alert sent] 1 issue" in log


@pytest.mark.parametrize("mode,expected", [("term", 124), ("kill", 137)])
def test_stuck_selftest_status_is_bounded_and_later_probes_continue(
    tmp_path, mode, expected
):
    result, _, log, message = _run(tmp_path, probe=PROBE, mode=mode)
    assert result.returncode == 0
    assert f"exit={expected}; details logged" in log
    assert "shortlist daily status checked" in log
    assert WARNING in message


def test_failed_alert_delivery_keeps_generic_onfailure_fallback(tmp_path):
    result, _, log, message = _run(tmp_path, probe=PROBE, probe_rc=1, delivery_rc=19)
    assert result.returncode == 1
    assert WARNING in message
    assert "heartbeat alert delivery FAIL (exit=19)" in log


def test_selftest_probe_has_separate_fixed_bound_and_no_mutations():
    source = (ROOT / "scripts/heartbeat.sh").read_text()
    hook = source.split("# 2s)", 1)[1].split("# 2e)", 1)[0]
    assert hook.count(SELFTEST_TIMEOUT) == hook.count(PROBE) == 1
    assert '>>"$LOG" 2>&1' in hook
    for forbidden in (
        "systemctl ",
        "reset-failed",
        "send_telegram",
        "pytest",
        "$(python",
    ):
        assert forbidden not in hook
