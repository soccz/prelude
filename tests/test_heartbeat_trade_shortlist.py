"""New-trial heartbeat isolation; only fake local processes are exercised."""

import pytest

from test_heartbeat_microstructure import PROBE as OLD_PROBE, ROOT, TIMEOUT, _run

PROBE = "-m ops.recommend_trade_shortlist_status --format text"
WARNING = "shortlist scheduled evidence probe FAIL"


@pytest.mark.parametrize(
    "status",
    [
        "not_started",
        "waiting",
        "complete_noop",
        "complete_changed",
        "complete_unavailable",
    ],
)
def test_operational_success_never_claims_effect_or_sends_alert(tmp_path, status):
    result, calls, log, message = _run(tmp_path, status=status, probe=PROBE)
    assert result.returncode == 0
    assert calls.count(PROBE) == calls.count(OLD_PROBE) == 1
    assert "shortlist daily status checked" in log
    assert WARNING not in log
    assert message is None


@pytest.mark.parametrize("probe_rc", [1, 2, 7, 124, 127, 137])
def test_new_trial_fault_preserves_other_checks_and_warns_once(tmp_path, probe_rc):
    result, calls, log, message = _run(tmp_path, probe_rc=probe_rc, probe=PROBE)
    assert result.returncode == 0
    assert calls.count(PROBE) == calls.count(OLD_PROBE) == 1
    assert "microstructure daily status checked" in log
    assert "shortlist daily status checked" not in log
    assert "backup manifest/archive/checksum provenance ok" in log
    assert "scripts/v2_scoreboard.py" in calls
    assert log.count(WARNING) == 1
    assert f"exit={probe_rc}; details logged" in log
    assert message is not None and message.count(WARNING) == 1


@pytest.mark.parametrize("mode,expected", [("term", 124), ("kill", 137)])
def test_native_hash_probe_is_bounded_and_still_reaches_backup(
    tmp_path, mode, expected
):
    result, _, log, message = _run(tmp_path, mode=mode, probe=PROBE)
    assert result.returncode == 0
    assert f"exit={expected}; details logged" in log
    assert "backup manifest/archive/checksum provenance ok" in log
    assert WARNING in message


def test_new_warning_delivery_failure_is_nonzero(tmp_path):
    result, _, log, message = _run(tmp_path, probe_rc=1, delivery_rc=19, probe=PROBE)
    assert result.returncode == 1
    assert WARNING in message
    assert "heartbeat alert delivery FAIL (exit=19)" in log


def test_new_hook_is_readonly_and_separately_bounded():
    source = (ROOT / "scripts/heartbeat.sh").read_text()
    assert source.count(PROBE) == 1
    assert source.count(OLD_PROBE) == 1
    assert source.count(TIMEOUT) == 5
    hook = source.split("# 3s)", 1)[1].split("# 3r)", 1)[0]
    assert hook.count(PROBE) == hook.count(TIMEOUT) == 1
    assert '>>"$LOG" 2>&1' in hook
    for forbidden in ("systemctl", "run_capture", "send_telegram", "$(python"):
        assert forbidden not in hook
