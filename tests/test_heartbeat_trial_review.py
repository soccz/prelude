"""Only isolated fake shell work, never live heartbeat or Telegram."""

import pytest

from test_heartbeat_microstructure import REVIEW_TIMEOUT, ROOT, _run

PROBE = "-m ops.recommend_trial_review --format text"
WARNING = "post-label trial review probe FAIL"


def test_completed_review_does_not_alert_or_promote(tmp_path):
    result, calls, log, message = _run(tmp_path, probe=PROBE)
    assert result.returncode == 0
    assert calls.count(PROBE) == 1
    assert "post-label trial review checked" in log
    assert WARNING not in log
    assert message is None


@pytest.mark.parametrize("rc", [1, 2, 7, 124, 137])
def test_review_failure_warns_but_does_not_block_other_checks(tmp_path, rc):
    result, calls, log, message = _run(tmp_path, probe=PROBE, probe_rc=rc)
    assert result.returncode == 0
    assert "shortlist daily status checked" in log
    assert "backup manifest/archive/checksum provenance ok" in log
    assert message.count(WARNING) == 1
    assert f"exit={rc}" in message


@pytest.mark.parametrize("mode, rc", [("term", 124), ("kill", 137)])
def test_review_probe_deadline_is_enforced(tmp_path, mode, rc):
    result, _, log, message = _run(tmp_path, probe=PROBE, mode=mode)
    assert result.returncode == 0
    assert f"exit={rc}" in message
    assert "backup manifest/archive/checksum provenance ok" in log


def test_hook_only_reads_separately_bounded_report():
    source = (ROOT / "scripts/heartbeat.sh").read_text()
    hook = source.split("# 3r)", 1)[1].split("# 3f)", 1)[0]
    assert hook.count(PROBE) == hook.count(REVIEW_TIMEOUT) == 1
    assert "--refresh" not in hook
