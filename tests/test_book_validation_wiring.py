"""Only isolated shell copies/fake Python; no real services or notifications."""

import pytest

from test_close_research_isolation import _run as close
from test_heartbeat_microstructure import ROOT, TIMEOUT, _run as heartbeat

PROBE = "-m ops.recommend_book_validation --format text"
REFRESH = "-m ops.recommend_book_validation"


def test_waiting_probe_is_silent(tmp_path):
    result, calls, log, message = heartbeat(tmp_path, probe=PROBE)
    assert result.returncode == 0 and calls.count(PROBE) == 1
    assert "fixed L1 validation checked" in log and message is None


@pytest.mark.parametrize("rc", [1, 2, 124, 137])
def test_missing_invalid_incomplete_warn_but_other_checks_continue(tmp_path, rc):
    result, _, log, message = heartbeat(tmp_path, probe=PROBE, probe_rc=rc)
    assert result.returncode == 0
    assert (
        message.count("fixed L1 validation probe FAIL") == 1 and f"exit={rc}" in message
    )
    assert "backup manifest/archive/checksum provenance ok" in log


@pytest.mark.parametrize("mode,rc", [("term", 124), ("kill", 137)])
def test_read_probe_is_bounded(tmp_path, mode, rc):
    result, _, _, message = heartbeat(tmp_path, probe=PROBE, mode=mode)
    assert result.returncode == 0 and f"exit={rc}" in message


@pytest.mark.parametrize("failure", ["none", "error", "term", "kill"])
def test_new_research_failure_cannot_block_core_or_existing_reports(tmp_path, failure):
    options = {}
    if failure == "error":
        options = {"fail_command": REFRESH}
    elif failure != "none":
        options = {"hang_command": REFRESH, "ignore_term": failure == "kill"}
    result, calls, log, _, _ = close(tmp_path, "daily_close_preopen.sh", **options)
    assert result.returncode == 0
    position = calls.index("CALL:" + REFRESH + " --refresh")
    assert calls.index("DONE:labels") < position
    assert calls.index("CALL:scripts/train_recommendation_meta.py") > position
    assert calls.index("CALL:scripts/build_idea_validation_html.py") > position
    assert ("[DEGRADED] fixed L1 validation failed" in log) == (failure != "none")


def test_read_probe_never_refreshes_or_initializes_and_backup_includes_namespace():
    source = (
        (ROOT / "scripts/heartbeat.sh")
        .read_text()
        .split("# 3b) Fixed L1", 1)[1]
        .split("# 3f)", 1)[0]
    )
    assert source.count(TIMEOUT) == source.count(PROBE) == 1
    assert "--refresh" not in source and "--initialize" not in source
    assert (
        "    output/recommend_book_validation \\\n"
        in (ROOT / "scripts/backup_db.sh").read_text()
    )
