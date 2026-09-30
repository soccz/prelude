"""Only isolated shell fixtures: never execute the real heartbeat or send."""

import pytest

from test_heartbeat_microstructure import ROOT, TIMEOUT, _run

PROBE = "-m ops.recommend_regime_forward --format text"
WARNING = "regime forward publication probe FAIL"


def test_complete_or_prelaunch_publication_is_silent(tmp_path):
    result, calls, log, message = _run(tmp_path, probe=PROBE)
    assert result.returncode == 0 and calls.count(PROBE) == 1
    assert "regime forward publication checked" in log and message is None


@pytest.mark.parametrize("rc", [1, 2, 124, 137])
def test_forward_failure_warns_without_stopping_other_checks(tmp_path, rc):
    result, _, log, message = _run(tmp_path, probe=PROBE, probe_rc=rc)
    assert result.returncode == 0
    assert message.count(WARNING) == 1 and f"exit={rc}" in message
    assert "backup manifest/archive/checksum provenance ok" in log


@pytest.mark.parametrize("mode,rc", [("term", 124), ("kill", 137)])
def test_probe_is_bounded(tmp_path, mode, rc):
    result, _, _, message = _run(tmp_path, probe=PROBE, mode=mode)
    assert result.returncode == 0 and f"exit={rc}" in message


def test_probe_has_no_publish_refresh_or_repair_mode():
    source = (ROOT / "scripts/heartbeat.sh").read_text().split("# 3f)", 1)[1].split("# 3a)", 1)[0]
    assert source.count(TIMEOUT) == source.count(PROBE) == 1
    assert "--refresh" not in source
