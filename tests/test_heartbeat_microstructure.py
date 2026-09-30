"""Isolated heartbeat wiring tests: all Python/DB/notification work is fake."""

from __future__ import annotations

import os
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = "/usr/bin/timeout --signal=TERM --kill-after=5s 30s"
SELFTEST_TIMEOUT = "/usr/bin/timeout --signal=TERM --kill-after=5s 15s"
REVIEW_TIMEOUT = "/usr/bin/timeout --signal=TERM --kill-after=5s 20s"
PROBE = "-m ops.recommend_microstructure_status --format text"
WARNING = "microstructure scheduled evidence probe FAIL"


def _run(
    tmp_path,
    *,
    status="complete",
    probe_rc=0,
    mode="normal",
    delivery_rc=0,
    probe=PROBE,
):
    repo = tmp_path / "repo"
    scripts, output, data = (repo / part for part in ("scripts", "output", "data"))
    fake_bin, backup = tmp_path / "bin", tmp_path / "backup"
    for path in (scripts, output, data, fake_bin, backup):
        path.mkdir(parents=True)
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    yesterday = today - timedelta(days=1)
    (output / "paper_ledger.csv").write_text(
        f"date,coin,status\n{yesterday},KRW-TEST,closed\n"
    )
    (output / "policy_competition_summary.json").write_text("{}")
    (output / "cron_publish.log").write_text(
        f"=== prelude publish dashboard {today} 10:10:00 ===\n"
        "[done] 10:15:00 committed + pushed\n"
    )
    for name in ("upbit_d1.db", "policy_competition.db"):
        (data / name).write_bytes(b"fixture-only; never opened as SQLite")
    source = (ROOT / "scripts/heartbeat.sh").read_text()
    assert source.count(TIMEOUT) == 5
    assert source.count(SELFTEST_TIMEOUT) == 1
    # Production deadline stays fixed; shorten only this copied fixture.
    source = source.replace(TIMEOUT, TIMEOUT.replace("5s 30s", "0.2s 0.3s"))
    source = source.replace(
        SELFTEST_TIMEOUT, SELFTEST_TIMEOUT.replace("5s 15s", "0.2s 0.3s")
    )
    assert source.count(REVIEW_TIMEOUT) == 1
    source = source.replace(REVIEW_TIMEOUT, REVIEW_TIMEOUT.replace("5s 20s", "0.2s 0.3s"))
    script = scripts / "heartbeat.sh"
    script.write_text(source)
    commands = {
        "python": r"""#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALL_LOG"
if [ "$*" = "$SELECTED_PROBE" ]; then
    if [ "$PROBE_MODE" != "empty" ]; then
        printf 'status=%s\n' "$PROBE_STATUS"
        printf 'probe diagnostic stderr\n' >&2
    fi
    if [ "$PROBE_MODE" = "large" ]; then
        /usr/bin/head -c 2097152 /dev/zero | /usr/bin/tr '\0' x
        printf '\nend-of-large-probe\n'
    fi
    if [ "$PROBE_MODE" = "term" ] || [ "$PROBE_MODE" = "kill" ]; then
        if [ "$PROBE_MODE" = "kill" ]; then trap '' TERM; else trap 'exit 0' TERM; fi
        while :; do /usr/bin/sleep 1; done
    fi
    exit "$PROBE_RC"
fi
if [ "$*" = "-m ops.recommend_microstructure_status --format text" ] || \
   [ "$*" = "-m ops.recommend_trade_shortlist_status --format text" ] || \
   [ "$*" = "-m ops.recommend_trial_review --format text" ] || \
   [ "$*" = "-m ops.recommend_regime_forward --format text" ] || \
   [ "$*" = "-m ops.recommend_book_validation --format text" ] || \
   [ "$*" = "-m ops.recommend_book_forward --format text" ] || \
   [ "$*" = "-B -m ops.selftest_status --format text" ]; then
    exit 0
fi
if [ "${1:-}" = "-" ]; then
    /usr/bin/cat >/dev/null
    printf 'ok\n'
    exit 0
fi
if [ "${1:-}" = "scripts/v2_scoreboard.py" ]; then
    printf '[v2-scoreboard] terminal KILL\n'
    exit 21
fi
if [ "${1:-}" = "-c" ]; then
    printf '%s' "$HEARTBEAT_MESSAGE" > "$MESSAGE_FILE"
    exit "$DELIVERY_RC"
fi
if [ "$*" = "-m ops.backup_manifest --backup-dir $PRELUDE_BACKUP_DIR --date $BACKUP_DATE --wait-seconds 0 --poll-seconds 10" ]; then
    exit 0
fi
printf 'unexpected fake Python command\n' >&2
exit 99
""",
        "sqlite3": "#!/bin/bash\nprintf 'ok\\n'\n",
        "df": (
            "#!/bin/bash\n"
            "printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\\n'\n"
            "printf '/dev/mock 100 10 90 10%% /mock\\n'\n"
        ),
    }
    for name, body in commands.items():
        executable = fake_bin / name
        executable.write_text(body)
        executable.chmod(0o755)
    call_log, message_file = tmp_path / "calls.log", tmp_path / "message.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "PRELUDE_FORBID_TELEGRAM": "1",
        "PRELUDE_BACKUP_DIR": str(backup),
        "PRELUDE_BACKUP_WAIT_SECONDS": "0",
        "BACKUP_DATE": today.strftime("%Y%m%d"),
        "CALL_LOG": str(call_log),
        "MESSAGE_FILE": str(message_file),
        "PROBE_STATUS": status,
        "SELECTED_PROBE": probe,
        "PROBE_RC": str(probe_rc),
        "PROBE_MODE": mode,
        "DELIVERY_RC": str(delivery_rc),
    }
    process = subprocess.Popen(
        ["/bin/bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("isolated heartbeat probe exceeded fixture deadline")
    result = subprocess.CompletedProcess(
        process.args, process.returncode, stdout, stderr
    )
    calls = call_log.read_text().splitlines()
    log = (output / "cron_heartbeat.log").read_text()
    message = message_file.read_text() if message_file.exists() else None
    return result, calls, log, message


@pytest.mark.parametrize(
    "status", ["not_started", "waiting", "complete", "complete_unavailable"]
)
def test_successful_probe_is_checked_without_alerting(tmp_path, status):
    result, calls, log, message = _run(tmp_path, status=status)
    assert result.returncode == 0
    assert calls.count(PROBE) == 1
    assert f"status={status}" in log
    assert "probe diagnostic stderr" in log
    assert "microstructure daily status checked" in log
    assert WARNING not in log
    assert message is None
    assert "[ok] all checks pass" in log


@pytest.mark.parametrize("probe_rc", [1, 2, 7, 124, 127, 137])
def test_any_nonzero_probe_warns_once_without_trusting_ok_stdout(tmp_path, probe_rc):
    result, calls, log, message = _run(tmp_path, status="ok", probe_rc=probe_rc)
    # Detailed delivery succeeded: preserve the existing no-double-alert exit.
    assert result.returncode == 0
    assert calls.count(PROBE) == 1
    assert "status=ok" in log
    assert log.count(WARNING) == 1
    assert f"exit={probe_rc}; details logged" in log
    assert "microstructure daily status checked" not in log
    assert "scripts/v2_scoreboard.py" in calls
    assert "backup manifest/archive/checksum provenance ok" in log
    assert "[alert sent] 1 issue" in log
    assert message is not None and message.count(WARNING) == 1
    assert "status=ok" not in message
    assert "probe diagnostic stderr" not in message


def test_empty_failed_probe_output_still_warns(tmp_path):
    result, _, log, message = _run(tmp_path, mode="empty", probe_rc=2)
    assert result.returncode == 0
    assert "status=" not in log
    assert "[alert sent] 1 issue" in log
    assert message is not None and WARNING in message


@pytest.mark.parametrize(("mode", "expected_rc"), [("term", 124), ("kill", 137)])
def test_real_timeout_and_forced_kill_remain_visible(tmp_path, mode, expected_rc):
    result, calls, log, message = _run(tmp_path, mode=mode)
    assert result.returncode == 0
    assert f"exit={expected_rc}; details logged" in log
    assert "scripts/v2_scoreboard.py" in calls
    assert "[alert sent] 1 issue" in log
    assert message is not None and WARNING in message


@pytest.mark.parametrize("probe_rc", [0, 1])
def test_large_probe_output_is_logged_without_pipe_or_message_pollution(
    tmp_path, probe_rc
):
    result, _, log, message = _run(tmp_path, mode="large", probe_rc=probe_rc)
    assert result.returncode == 0
    assert len(log) > 2 * 1024 * 1024
    assert "end-of-large-probe" in log
    if probe_rc:
        assert message is not None and "end-of-large-probe" not in message
        assert "[alert sent] 1 issue" in log
    else:
        assert message is None


def test_alert_delivery_failure_preserves_onfailure_fallback(tmp_path):
    result, _, log, message = _run(tmp_path, probe_rc=1, delivery_rc=19)
    assert result.returncode == 1
    assert WARNING in message
    assert "heartbeat alert delivery FAIL (exit=19)" in log
    assert "[alert sent]" not in log


def test_hook_contract_uses_readonly_cli_once_and_fixed_bound():
    source = (ROOT / "scripts/heartbeat.sh").read_text()
    hook = source.split("# 2e) Scheduled microstructure", 1)[1].split("# 3) disk", 1)[0]
    assert hook.count(TIMEOUT) == 1
    assert hook.count(PROBE) == 1
    assert '>>"$LOG" 2>&1' in hook
    assert "micro_probe_rc=$?" in hook
    assert "$(python" not in hook
    assert "systemctl" not in hook
    assert "run_capture" not in hook
    assert "send_telegram" not in hook
    assert "2026-09-08" not in hook and "2026-09-09" not in hook
