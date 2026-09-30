"""Fake copies only; no real sends, training, collectors, units or other terminals."""

import pytest

from test_close_research_isolation import _run as close
from test_heartbeat_microstructure import ROOT, TIMEOUT, _run as heartbeat
from test_shell_failure_propagation import _run_with_fake_python as distribution

PROBE = "-m ops.recommend_book_forward --format text"
RECORD = "-m ops.recommend_book_forward --record --format text"


@pytest.mark.parametrize("rc", [0, 1, 2, 124, 137])
def test_record_only_after_r1_and_failure_does_not_change_core_exit(tmp_path, rc):
    result, calls, log = distribution(
        tmp_path, "daily_run_distribution.sh", fail_exact=RECORD, fail_rc=rc
    )
    assert result.returncode == 0
    assert calls.index("scripts/recommend_send.py --slot open") < calls.index(RECORD)
    assert calls.index("scripts/recommend_today.py --require-receipt") < calls.index(
        RECORD
    )
    assert calls.index(RECORD) < calls.index("-m data.collector_4h --all --days 2")
    assert ("[DEGRADED] L1 pre-entry record failed" in log) == (rc != 0)


def test_record_skipped_when_primary_data_gate_failed(tmp_path):
    result, calls, _ = distribution(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact="-m data.collector_d1 --update",
        fail_rc=7,
    )
    assert result.returncode == 7 and RECORD not in calls


@pytest.mark.parametrize("failure", ["none", "error", "term", "kill"])
def test_forward_review_failure_never_blocks_core_and_old_research(tmp_path, failure):
    command = "-m ops.recommend_book_forward"
    options = (
        {"fail_command": command}
        if failure == "error"
        else (
            {"hang_command": command, "ignore_term": failure == "kill"}
            if failure != "none"
            else {}
        )
    )
    result, calls, log, _, _ = close(tmp_path, "daily_close_preopen.sh", **options)
    assert result.returncode == 0
    where = calls.index("CALL:" + command + " --refresh")
    assert (
        calls.index("DONE:labels")
        < where
        < calls.index("CALL:scripts/train_recommendation_meta.py")
    )
    assert ("[DEGRADED] L1 forward review failed" in log) == (failure != "none")


@pytest.mark.parametrize("rc", [0, 1, 2, 124, 137])
def test_probe_has_one_warning_or_silent_success(tmp_path, rc):
    result, calls, log, message = heartbeat(tmp_path, probe=PROBE, probe_rc=rc)
    assert result.returncode == 0 and calls.count(PROBE) == 1
    if rc:
        assert message.count("L1 forward readiness probe FAIL") == 1
    else:
        assert message is None and "L1 forward readiness checked" in log


def test_budget_and_backup_and_no_unit_change():
    source = (
        (ROOT / "scripts/heartbeat.sh")
        .read_text()
        .split("# 3g)", 1)[1]
        .split("# 3a)", 1)[0]
    )
    assert source.count(TIMEOUT) == source.count(PROBE) == 1
    assert "--refresh" not in source and "--record" not in source
    assert (
        "output/recommend_book_forward \\\n"
        in (ROOT / "scripts/backup_db.sh").read_text()
    )
    assert (
        "/usr/bin/timeout --signal=TERM --kill-after=10s 120s"
        in (ROOT / "scripts/daily_run_distribution.sh").read_text()
    )
