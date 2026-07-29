from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest


def _copy_radar_verdict_validator(repo: Path) -> None:
    destination = repo / "ops"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "__init__.py",
        "artifact_provenance.py",
        "file_lock.py",
        "radar_verdict.py",
    ):
        shutil.copy2(Path("ops") / name, destination / name)


def _backup_script_source(repo: Path) -> str:
    deploy = repo / "deploy"
    deploy.mkdir(parents=True, exist_ok=True)
    shutil.copy2("deploy/lock_exec.py", deploy / "lock_exec.py")
    return Path("scripts/backup_db.sh").read_text().replace(
        'BACKUP_DIR="/home/soccz/22tb/backup/prelude_db"',
        'BACKUP_DIR="$PROJ_ROOT/backup"',
    )


def _valid_terminal_verdict_bytes() -> bytes:
    from ops.radar_verdict import terminal_candidate

    scorecard = {
        "status": "early_kill",
        "closed_n": 1,
        "mean_net_pct": -0.5,
        "ci95": None,
        "per_day_t": None,
        "regimes": ["bear_quiet"],
        "criteria": {
            "n>=200": False,
            "mean>0": False,
            "CI_0_제외": False,
            "2레짐_or_t>=2": False,
        },
        "criteria_met": 0,
        "early_kill_breached": True,
        "terminal_metric_values": {
            "mean_net_pct": -0.5,
            "ci95": None,
            "per_day_t": None,
        },
    }
    candidate = terminal_candidate(
        scorecard,
        asof=date(2026, 7, 1),
        recorded_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    assert candidate is not None
    return (
        json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _run_with_fake_python(
    tmp_path: Path,
    script_name: str,
    *,
    fail_exact: str,
    fail_rc: int,
    close_plan_fault: str = "",
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    """운영 shell 을 격리 복사하고 모든 Python command 를 빠른 fake 로 대체."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(Path("scripts") / script_name, scripts / script_name)

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALL_LOG"
if [[ "$*" == "-m ops.close_input_gate --through-asof $(date -d yesterday +%F) --cohort "* ]] &&
   [[ "$*" == *" --output-format nul" ]]; then
    target=$(date -d yesterday +%F)
    older=$(date -d "2 days ago" +%F)
    future=$(date -d tomorrow +%F)
    case "${CLOSE_PLAN_FAULT:-}" in
        empty)
            exit 0
            ;;
        missing_sentinel)
            printf '%s\\0close\\0' "$target"
            exit 0
            ;;
        bad_mode)
            printf '%s\\0invalid\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$target"
            exit 0
            ;;
        duplicate)
            printf '%s\\0close\\0%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' \
                "$target" "$target"
            exit 0
            ;;
        unsorted)
            printf '%s\\0close\\0%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' \
                "$target" "$older"
            exit 0
            ;;
        future)
            printf '%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$future"
            exit 0
            ;;
        missing_target)
            printf '%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$older"
            exit 0
            ;;
        odd)
            printf '%s\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$target"
            exit 0
            ;;
        valid_output_nonzero)
            printf '%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$target"
            exit 7
            ;;
    esac
    printf '%s\\0close\\0__PRELUDE_CLOSE_PLAN_V1_OK__\\0' "$target"
    exit 0
fi
ARGS="$*"
NORMALIZED="${ARGS% --decision-date ????-??-??}"
NORMALIZED="${NORMALIZED%% --cohort *}"
if [ "$NORMALIZED" = "$FAIL_EXACT" ]; then
    exit "$FAIL_RC"
fi
exit 0
"""
    )
    fake_python.chmod(0o755)

    call_log = tmp_path / "calls.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CALL_LOG": str(call_log),
            "FAIL_EXACT": fail_exact,
            "FAIL_RC": str(fail_rc),
            "CLOSE_PLAN_FAULT": close_plan_fault,
        }
    )
    result = subprocess.run(
        ["bash", str(scripts / script_name)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = call_log.read_text() if call_log.exists() else ""
    logs = "\n".join(
        path.read_text() for path in sorted((repo / "output").glob("*.log"))
    )
    return result, calls, logs


def _run_auxiliary_shell_with_fake_python(
    tmp_path: Path,
    script_name: str,
    fake_python_body: str,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(Path("scripts") / script_name, scripts / script_name)

    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        f"{fake_python_body}\n"
    )
    fake_python.chmod(0o755)

    call_log = tmp_path / "calls.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CALL_LOG": str(call_log),
        }
    )
    result = subprocess.run(
        ["bash", str(scripts / script_name)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = call_log.read_text() if call_log.exists() else ""
    logs = "\n".join(
        path.read_text() for path in sorted((repo / "output").glob("*.log"))
    )
    return result, calls, logs


def test_measure_failure_propagates_original_exit_code(tmp_path):
    result, calls, logs = _run_auxiliary_shell_with_fake_python(
        tmp_path,
        "measure_run.sh",
        "exit 31",
    )

    assert result.returncode == 31
    assert calls
    assert "measure failed (exit=31)" in logs
    assert "done " not in logs


def test_retrain_data_failure_blocks_training_and_preserves_first_exit(tmp_path):
    result, calls, logs = _run_auxiliary_shell_with_fake_python(
        tmp_path,
        "retrain_run.sh",
        """
if [ "$*" = "-m data.collector_d1 --update" ]; then
    exit 27
fi
exit 0
""",
    )

    assert result.returncode == 27
    assert "-m data.collector_d1 --update" in calls
    assert "-m data.collector_4h --update" in calls
    assert "-m data.collector_binance_d1 --all --days 1095" in calls
    assert "-m signals.retrain --n-trials 30" not in calls
    assert "retrain skipped" in logs


def test_retrain_notification_failure_is_not_reported_as_success(tmp_path):
    result, calls, logs = _run_auxiliary_shell_with_fake_python(
        tmp_path,
        "retrain_run.sh",
        """
if [ "$1" = "-c" ]; then
    exit 42
fi
exit 0
""",
    )

    assert result.returncode == 42
    assert "-m signals.retrain --n-trials 30" in calls
    assert "retrain notification failed (exit=42)" in logs
    assert "done " not in logs


@pytest.mark.parametrize(
    ("command", "rc"),
    [
        ("scripts/recommend_send.py --slot open", 7),
        ("scripts/recommend_today.py --require-receipt", 8),
        ("scripts/pump_detector_v2_today.py --send-telegram", 9),
    ],
)
def test_distribution_critical_failure_propagates_after_followups(
    tmp_path, command, rc
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert "scripts/recommend_today.py --ranking R2" in calls
    assert "scripts/pump_detector_v2_today.py --send-telegram" in calls
    assert "[critical]" in logs
    assert f"exit={rc}" in logs


@pytest.mark.parametrize(
    ("command", "rc", "message"),
    [
        (
            "scripts/predict_today_distribution.py --universe top100 --top-k 10",
            16,
            "distribution record-only prediction",
        ),
        ("scripts/recommend_today.py --ranking R2", 17, "R2 record-only ledger"),
        ("scripts/recommend_today.py --ranking A1", 18, "A1 record-only ledger"),
        ("scripts/pump_detector_today.py", 19, "PUMP hunter record-only ledger"),
        ("-m data.collector_binance_d1 --all --days 3", 20, "Binance D1 refresh"),
    ],
)
def test_distribution_auxiliary_failure_is_visible_and_propagates(
    tmp_path,
    command,
    rc,
    message,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert "scripts/pump_detector_v2_today.py --send-telegram" in calls
    assert message in logs
    assert "[critical]" in logs


def test_distribution_partial_d1_update_blocks_all_signal_generation(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact="-m data.collector_d1 --update",
        fail_rc=25,
    )

    assert result.returncode == 25
    assert "scripts/recommend_send.py --slot open" not in calls
    assert "scripts/predict_today_distribution.py" not in calls
    assert "-m data.collector_4h --all --days 2" in calls
    assert "partial/failed D1 update" in logs


def test_distribution_runs_r1_before_4h_and_legacy_distribution(tmp_path):
    result, calls, _ = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact="never",
        fail_rc=99,
    )

    assert result.returncode == 0
    commands = calls.splitlines()
    assert commands.index("scripts/recommend_send.py --slot open") < commands.index(
        "-m data.collector_4h --all --days 2"
    )
    assert commands.index("-m data.collector_4h --all --days 2") < commands.index(
        "scripts/predict_today_distribution.py --universe top100 --top-k 10"
    )


def test_recommend_health_failure_skips_stale_signals_but_runs_maintenance(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact="scripts/health_check.py --channel recommend --no-telegram",
        fail_rc=23,
    )

    assert result.returncode == 23
    assert "scripts/recommend_send.py --slot open" not in calls
    assert "scripts/recommend_today.py --ranking R2" not in calls
    assert "scripts/pump_detector_v2_today.py --send-telegram" not in calls
    assert "-m data.collector_4h --all --days 2" in calls
    assert "-m data.collector_binance_d1 --all --days 3" in calls
    assert "stale D1" in logs


def test_distribution_health_failure_skips_only_legacy_distribution(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_distribution.sh",
        fail_exact="scripts/health_check.py --channel distribution --no-telegram",
        fail_rc=24,
    )

    assert result.returncode == 24
    assert "scripts/recommend_send.py --slot open" in calls
    assert "scripts/predict_today_distribution.py" not in calls
    assert "scripts/recommend_today.py --ranking R2" in calls
    assert "scripts/pump_detector_v2_today.py --send-telegram" in calls
    assert "distribution record skipped" in logs


def test_preopen_r1_send_failure_propagates(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact="scripts/recommend_send.py --slot preopen",
        fail_rc=11,
    )

    assert result.returncode == 11
    assert (
        "scripts/recommend_today.py --slot preopen --require-receipt"
        in calls
    )
    assert "R1 preopen send failed" in logs


def test_preopen_r1_ledger_failure_propagates(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact="scripts/recommend_today.py --slot preopen --require-receipt",
        fail_rc=18,
    )

    assert result.returncode == 18
    assert "scripts/recommend_send.py --slot preopen" in calls
    assert "R1 preopen ledger failed" in logs


def test_preopen_15m_failure_blocks_only_legacy_record(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact="-m data.collector_15m_upbit --all --days 1",
        fail_rc=26,
    )

    assert result.returncode == 26
    assert "scripts/health_check.py --channel preopen" not in calls
    assert "scripts/predict_preopen_trigger.py" not in calls
    assert "scripts/recommend_send.py --slot preopen" in calls
    assert "legacy preopen record skipped" in logs


def test_preopen_d1_failure_blocks_all_signal_generation(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact="-m data.collector_d1 --update",
        fail_rc=27,
    )

    assert result.returncode == 27
    assert "scripts/health_check.py --channel recommend-preopen" not in calls
    assert "scripts/recommend_send.py --slot preopen" not in calls
    assert "-m data.collector_15m_upbit --all --days 1" in calls
    assert "scripts/health_check.py --channel preopen" not in calls
    assert "scripts/predict_preopen_trigger.py" not in calls
    assert "partial/failed D1 update" in logs


def test_preopen_d1_health_failure_blocks_r1_but_runs_legacy_maintenance(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact=(
            "scripts/health_check.py --channel "
            "recommend-preopen --no-telegram"
        ),
        fail_rc=28,
    )

    assert result.returncode == 28
    assert "scripts/recommend_send.py --slot preopen" not in calls
    assert "-m data.collector_15m_upbit --all --days 1" in calls
    assert "scripts/health_check.py --channel preopen --no-telegram" in calls
    assert "scripts/predict_preopen_trigger.py" in calls
    assert "stale D1" in logs


def test_preopen_record_only_prediction_failure_propagates_after_r1_send(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_run_preopen.sh",
        fail_exact=(
            "scripts/predict_preopen_trigger.py --top-k 8 "
            "--universe top100 --no-telegram"
        ),
        fail_rc=29,
    )

    assert result.returncode == 29
    assert "scripts/recommend_send.py --slot preopen" in calls
    assert calls.index(
        "scripts/recommend_send.py --slot preopen"
    ) < calls.index("scripts/predict_preopen_trigger.py")
    assert "preopen record-only prediction" in logs


@pytest.mark.parametrize(
    ("command", "rc"),
    [
        ("scripts/close_recommend_ledger.py", 12),
        (
            "scripts/close_recommend_ledger.py "
            "--ledger output/shadow_ledger_pump_hunter_v2.csv",
            13,
        ),
        ("-m ops.champion_selector", 14),
    ],
)
def test_distribution_close_critical_failure_runs_later_steps(
    tmp_path, command, rc
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_distribution.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert "-m ops.champion_selector" in calls
    assert "scripts/idea_validation_report.py" in calls
    assert "[critical]" in logs


def test_distribution_close_meta_failure_propagates_but_followups_run(tmp_path):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_distribution.sh",
        fail_exact="scripts/train_recommendation_meta.py",
        fail_rc=15,
    )

    assert result.returncode == 15
    assert "scripts/idea_validation_report.py" in calls
    assert "recommendation meta train failed" in logs


@pytest.mark.parametrize(
    ("command", "rc", "message"),
    [
        (
            "scripts/close_recommend_ledger.py "
            "--ledger output/shadow_ledger_recommend_r2.csv",
            33,
            "R2 recommend close",
        ),
        (
            "scripts/close_recommend_ledger.py "
            "--ledger output/shadow_ledger_recommend_sustain.csv",
            34,
            "A1 recommend close",
        ),
        (
            "scripts/close_recommend_ledger.py "
            "--ledger output/shadow_ledger_pump_hunter.csv",
            35,
            "PUMP hunter close",
        ),
        ("-m ops.policy_competition", 36, "policy_competition"),
        ("scripts/idea_validation_report.py", 37, "idea validation report"),
        ("scripts/build_idea_validation_html.py", 38, "idea validation HTML"),
    ],
)
def test_distribution_close_audit_failures_propagate(
    tmp_path,
    command,
    rc,
    message,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_distribution.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert message in logs
    if command == "scripts/idea_validation_report.py":
        assert "scripts/build_idea_validation_html.py" not in calls
        assert "stale input forbidden" in logs


def test_distribution_close_15m_update_failure_still_runs_strict_path_closes(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_distribution.sh",
        fail_exact="-m data.collector_15m_upbit --all --days 1",
        fail_rc=27,
    )

    assert result.returncode == 27
    assert "scripts/close_paper_ledger.py" in calls
    assert "scripts/close_recommend_ledger.py" in calls
    assert "-m ops.champion_selector" in calls
    assert "close 15m universe update failed" in logs


def test_close_shells_skip_precontract_cohorts_without_marking_forward_valid():
    for script in (
        "scripts/daily_close_distribution.sh",
        "scripts/daily_close_preopen.sh",
    ):
        source = Path(script).read_text()
        assert 'skip-legacy-unverifiable' in source
        assert "never forward-valid" in source


def test_close_shells_bind_each_closer_to_a_validated_decision_date(tmp_path):
    result, calls, _logs = _run_with_fake_python(
        tmp_path,
        "daily_close_distribution.sh",
        fail_exact="never",
        fail_rc=99,
    )

    assert result.returncode == 0
    assert "-m ops.close_input_gate --through-asof" in calls
    close_calls = [
        line
        for line in calls.splitlines()
        if line.startswith("scripts/close_recommend_ledger.py")
    ]
    assert len(close_calls) == 5
    assert all(" --decision-date " in line for line in close_calls)


@pytest.mark.parametrize(
    "script_name",
    [
        "daily_close_distribution.sh",
        "daily_close_preopen.sh",
    ],
)
@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "missing_sentinel",
        "bad_mode",
        "duplicate",
        "unsorted",
        "future",
        "missing_target",
        "odd",
        "valid_output_nonzero",
    ],
)
def test_close_shell_rejects_malformed_or_incomplete_gate_plan(
    tmp_path,
    script_name,
    fault,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        script_name,
        fail_exact="never",
        fail_rc=99,
        close_plan_fault=fault,
    )

    assert result.returncode == (7 if fault == "valid_output_nonzero" else 2)
    assert "scripts/close_recommend_ledger.py" not in calls
    assert "evidence gate" in logs
    assert "[critical]" in logs


@pytest.mark.parametrize(
    ("command", "rc"),
    [
        ("scripts/close_preopen_ledger.py", 16),
        (
            "scripts/close_recommend_ledger.py "
            "--ledger output/shadow_ledger_recommend_preopen.csv",
            20,
        ),
    ],
)
def test_preopen_close_failure_runs_optional_followups_and_propagates(
    tmp_path, command, rc
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_preopen.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert "scripts/train_recommendation_meta.py" in calls
    assert "scripts/idea_validation_report.py" in calls
    assert "[critical]" in logs


def test_preopen_close_15m_update_failure_runs_strict_downstream_and_propagates(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_preopen.sh",
        fail_exact="-m data.collector_15m_upbit --all --days 1",
        fail_rc=27,
    )

    assert result.returncode == 27
    assert "scripts/label_recommend_snapshots.py" in calls
    assert "scripts/evaluate_recommend_score_labels.py" in calls
    assert "scripts/close_preopen_ledger.py" in calls
    assert "scripts/close_recommend_ledger.py" in calls
    assert "scripts/train_recommendation_meta.py" in calls
    assert "preopen close 15m universe update failed" in logs


def test_preopen_close_label_failure_propagates_but_partial_retries_as_warning(tmp_path):
    yesterday = subprocess.run(
        ["date", "-d", "yesterday", "+%F"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = f"scripts/label_recommend_snapshots.py --through-date {yesterday}"

    failed, calls, failed_logs = _run_with_fake_python(
        tmp_path / "failed",
        "daily_close_preopen.sh",
        fail_exact=command,
        fail_rc=31,
    )
    assert failed.returncode == 31
    assert command in calls
    assert "scripts/train_recommendation_meta.py" in calls
    assert "full-universe score labeling" in failed_logs

    partial, _, partial_logs = _run_with_fake_python(
        tmp_path / "partial",
        "daily_close_preopen.sh",
        fail_exact=command,
        fail_rc=2,
    )
    assert partial.returncode == 2
    assert "score label partial" in partial_logs


def test_preopen_close_evaluator_failure_is_critical_and_followups_continue(
    tmp_path,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_preopen.sh",
        fail_exact="scripts/evaluate_recommend_score_labels.py",
        fail_rc=32,
    )

    assert result.returncode == 32
    assert "scripts/evaluate_recommend_score_labels.py" in calls
    assert "scripts/train_recommendation_meta.py" in calls
    assert "full-universe score evaluation" in logs


@pytest.mark.parametrize(
    ("command", "rc", "message"),
    [
        ("scripts/train_recommendation_meta.py", 41, "recommendation meta train"),
        ("-m ops.policy_competition", 42, "policy_competition"),
        ("scripts/idea_validation_report.py", 43, "idea validation report"),
        ("scripts/build_idea_validation_html.py", 44, "idea validation HTML"),
    ],
)
def test_preopen_close_audit_failures_propagate(
    tmp_path,
    command,
    rc,
    message,
):
    result, calls, logs = _run_with_fake_python(
        tmp_path,
        "daily_close_preopen.sh",
        fail_exact=command,
        fail_rc=rc,
    )

    assert result.returncode == rc
    assert message in logs
    if command == "scripts/idea_validation_report.py":
        assert "scripts/build_idea_validation_html.py" not in calls
        assert "stale input forbidden" in logs


def test_backup_and_heartbeat_finish_nonzero_on_detected_failures():
    backup = Path("scripts/backup_db.sh").read_text()
    heartbeat = Path("scripts/heartbeat.sh").read_text()

    assert 'if [ "$N_FAIL" -gt 0 ]; then' in backup
    assert "exit 1" in backup
    assert "--busy-exit 75" in backup
    assert "--wait" in backup
    assert "/usr/bin/timeout --signal=TERM --kill-after=30s 3000s" in backup
    assert 'EXIT=1' in heartbeat
    assert 'exit "$EXIT"' in heartbeat
    assert "send_telegram(os.environ['HEARTBEAT_MESSAGE']) else 1" in heartbeat
    assert "heartbeat alert delivery FAIL" in heartbeat


def test_backup_lock_wait_is_bounded_and_fails_loud(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    backup = repo / "backup"
    scripts.mkdir(parents=True)
    backup.mkdir()
    source = _backup_script_source(repo).replace(
        "--kill-after=30s 3000s",
        "--kill-after=1s 1s",
        1,
    )
    backup_script = scripts / "backup_db.sh"
    backup_script.write_text(source, encoding="utf-8")
    lock_file = backup / ".backup.lock"
    holder_code = (
        "import fcntl, pathlib, sys, time\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "with path.open('a+') as handle:\n"
        "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
        "    print('locked', flush=True)\n"
        "    time.sleep(10)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_file)],
        cwd=repo,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        started = time.monotonic()
        result = subprocess.run(
            ["bash", str(backup_script)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        elapsed = time.monotonic() - started
    finally:
        holder.terminate()
        holder.wait(timeout=5)
        if holder.stdout is not None:
            holder.stdout.close()

    assert result.returncode == 124
    assert elapsed < 4


def test_backup_failure_continues_to_ledger_tar_then_exits_nonzero(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "data").mkdir()
    (repo / "output").mkdir()
    (repo / "data" / "upbit_d1.db").write_bytes(b"not-a-real-db")
    (repo / "output" / "paper_ledger.csv").write_text(
        "date,coin,status\n2026-07-24,KRW-BTC,CLOSED\n"
    )

    source = _backup_script_source(repo)
    backup_script = scripts / "backup_db.sh"
    backup_script.write_text(source)

    for name, body in {
        "sqlite3": "#!/usr/bin/env bash\nexit 7\n",
        "python": "#!/usr/bin/env bash\nexit 0\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body)
        executable.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    result = subprocess.run(
        ["bash", str(backup_script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (repo / "output" / "cron_backup.log").read_text()

    assert result.returncode == 1
    assert ".backup FAIL" in log
    assert "evidence tar:" in log


def test_backup_refreshes_expired_last_copy_before_retention(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    scripts.mkdir(parents=True)
    data.mkdir()
    output.mkdir()
    backup.mkdir()

    source_db = data / "archive.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('recoverable')")
    old_source_time = time.time() - 30 * 24 * 3600
    os.utime(source_db, (old_source_time, old_source_time))

    expired = backup / "archive_20260701.db"
    shutil.copy2(source_db, expired)
    expired_time = time.time() - 15 * 24 * 3600
    os.utime(expired, (expired_time, expired_time))
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    remaining = sorted(backup.glob("archive_*.db"))
    assert result.returncode == 0
    assert remaining
    with sqlite3.connect(remaining[-1]) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == (
            "recoverable"
        )
    manifest_path = backup / f"ledgers_{time.strftime('%Y%m%d')}.manifest"
    assert manifest_path.exists()
    manifest = dict(
        line.split("=", 1)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    )
    assert manifest["schema"] == "prelude_evidence_backup.v1"
    assert manifest["terminal_verdict_pair"] == "absent"
    archive = backup / manifest["archive"]
    checksum_path = backup / manifest["checksum"]
    assert archive.exists()
    assert checksum_path.exists()
    assert archive.stat().st_mode & 0o077 == 0
    assert checksum_path.stat().st_mode & 0o077 == 0
    assert manifest_path.stat().st_mode & 0o077 == 0
    assert remaining[-1].stat().st_mode & 0o077 == 0
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    checksum = subprocess.run(
        [
            "sha256sum",
            "-c",
            checksum_path.name,
        ],
        cwd=backup,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checksum.returncode == 0


def test_backup_captures_terminal_verdict_state_and_anchor_in_one_generation(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    for path in (scripts, data, output, backup):
        path.mkdir(parents=True)

    source_db = data / "upbit_d1.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('terminal')")
    state = output / "radar_terminal_verdict.json"
    anchor = data / "radar_terminal_verdict.anchor.json"
    legacy_champion = output / (
        "champion_state.legacy."
        f"{'a' * 64}.json"
    )
    state_bytes = _valid_terminal_verdict_bytes()
    anchor_bytes = state_bytes
    legacy_champion_bytes = b'{"schema_version":"champion_state.v1"}\n'
    state.write_bytes(state_bytes)
    anchor.write_bytes(anchor_bytes)
    legacy_champion.write_bytes(legacy_champion_bytes)
    _copy_radar_verdict_validator(repo)

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    manifest_path = backup / f"ledgers_{time.strftime('%Y%m%d')}.manifest"
    manifest = dict(
        line.split("=", 1)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    )
    assert manifest["terminal_verdict_pair"] == "present"
    archive_path = backup / manifest["archive"]
    checksum_path = backup / manifest["checksum"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        manifest["sha256"]
    )
    with tarfile.open(archive_path, "r:gz") as archive_handle:
        names = set(archive_handle.getnames())
        assert {
            "output/radar_terminal_verdict.json",
            "data/radar_terminal_verdict.anchor.json",
            f"output/{legacy_champion.name}",
        } <= names
        state_member = archive_handle.extractfile(
            "output/radar_terminal_verdict.json"
        )
        anchor_member = archive_handle.extractfile(
            "data/radar_terminal_verdict.anchor.json"
        )
        assert state_member is not None
        assert anchor_member is not None
        assert state_member.read() == state_bytes
        assert anchor_member.read() == anchor_bytes
        legacy_member = archive_handle.extractfile(
            f"output/{legacy_champion.name}"
        )
        assert legacy_member is not None
        assert legacy_member.read() == legacy_champion_bytes
    checksum = subprocess.run(
        ["sha256sum", "-c", checksum_path.name],
        cwd=backup,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checksum.returncode == 0


def test_backup_rejects_incomplete_terminal_verdict_pair(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
    (output / "radar_terminal_verdict.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    log = (output / "cron_backup.log").read_text()
    assert "terminal verdict state/anchor pair incomplete" in log
    assert not list(backup.glob("ledgers_*.manifest"))


@pytest.mark.parametrize("dangling_member", ["state", "anchor"])
def test_backup_rejects_dangling_terminal_verdict_member(
    tmp_path,
    dangling_member,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
    state = output / "radar_terminal_verdict.json"
    anchor = data / "radar_terminal_verdict.anchor.json"
    verdict_bytes = _valid_terminal_verdict_bytes()
    if dangling_member == "state":
        state.symlink_to(repo / "missing-state.json")
        anchor.write_bytes(verdict_bytes)
    else:
        state.write_bytes(verdict_bytes)
        anchor.symlink_to(repo / "missing-anchor.json")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "terminal verdict state/anchor must be regular non-symlink files"
        in (output / "cron_backup.log").read_text()
    )
    assert not list(backup.glob("ledgers_*.manifest"))


def test_backup_rejects_semantically_invalid_terminal_verdict_pair(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    for path in (scripts, data, output, backup):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
    invalid_pair = b'{"state":"terminal","decision_id":"forged"}\n'
    (output / "radar_terminal_verdict.json").write_bytes(invalid_pair)
    (data / "radar_terminal_verdict.anchor.json").write_bytes(invalid_pair)
    (output / "champion_state.json").write_text("{}", encoding="utf-8")
    _copy_radar_verdict_validator(repo)

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    log = (output / "cron_backup.log").read_text()
    assert "evidence tar FAIL" in log
    assert not list(backup.glob("ledgers_*.manifest"))
    assert not list(backup.glob("ledgers_*.tar.gz"))


def test_backup_manifest_failure_preserves_previous_valid_evidence_pair(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    source_db = data / "upbit_d1.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('new')")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    date = time.strftime("%Y%m%d")
    old_archive = backup / f"ledgers_{date}_old-generation.tar.gz"
    old_payload = tmp_path / "old-ledger.csv"
    old_payload.write_text(
        "date,coin,status\n2026-07-24,KRW-BTC,closed\n",
        encoding="utf-8",
    )
    with tarfile.open(old_archive, "w:gz") as archive_handle:
        archive_handle.add(old_payload, arcname="output/old-ledger.csv")
    old_digest = hashlib.sha256(old_archive.read_bytes()).hexdigest()
    old_checksum = old_archive.with_suffix(
        old_archive.suffix + ".sha256"
    )
    old_checksum.write_text(
        f"{old_digest}  {old_archive.name}\n",
        encoding="utf-8",
    )
    manifest = backup / f"ledgers_{date}.manifest"
    manifest.write_text(
        "schema=prelude_evidence_backup.v1\n"
        f"date={date}\n"
        "generation=old-generation\n"
        f"archive={old_archive.name}\n"
        f"checksum={old_checksum.name}\n"
        f"sha256={old_digest}\n",
        encoding="utf-8",
    )
    before = {
        old_archive: old_archive.read_bytes(),
        old_checksum: old_checksum.read_bytes(),
        manifest: manifest.read_bytes(),
    }

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "$2" = "{manifest}" ]; then exit 73; fi\n'
        'exec /usr/bin/mv "$@"\n'
    )
    fake_mv.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "evidence tar FAIL" in (output / "cron_backup.log").read_text()
    for path, contents in before.items():
        assert path.read_bytes() == contents
    check = subprocess.run(
        ["sha256sum", "-c", old_checksum.name],
        cwd=backup,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0


def test_backup_evidence_generation_collision_never_overwrites_existing_pair(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('new')")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    generation = "20991231_040000_4242"
    archive = backup / f"ledgers_{generation}.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    archive.write_bytes(b"pre-existing immutable generation")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    before = {archive: archive.read_bytes(), checksum: checksum.read_bytes()}

    source = _backup_script_source(repo).replace(
        'EVIDENCE_GENERATION="${DATE}_$(date +%H%M%S)_$$"',
        f'EVIDENCE_GENERATION="{generation}"',
    )
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "evidence tar FAIL" in (output / "cron_backup.log").read_text()
    for path, contents in before.items():
        assert path.read_bytes() == contents
    assert not list(backup.glob("ledgers_*.manifest"))
    assert not list(backup.glob(".*.partial"))


def test_backup_sync_failure_never_replaces_previous_good_db_copy(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    source_db = data / "upbit_d1.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('new')")
    previous = backup / f"upbit_d1_{time.strftime('%Y%m%d')}.db"
    previous.write_bytes(b"previous-good-copy")
    before = previous.read_bytes()
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    fake_sync = fake_bin / "sync"
    fake_sync.write_text("#!/usr/bin/env bash\nexit 9\n")
    fake_sync.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (output / "cron_backup.log").read_text()

    assert result.returncode == 1
    assert previous.read_bytes() == before
    assert "durable publish FAIL" in log
    assert not list(backup.glob(".*.partial"))


def test_backup_never_skips_old_source_when_recent_copy_is_corrupt(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    for path in (scripts, data, output, backup):
        path.mkdir(parents=True)

    source_db = data / "archive.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('recoverable')")
    old_source_time = time.time() - 30 * 24 * 3600
    os.utime(source_db, (old_source_time, old_source_time))
    current = backup / f"archive_{time.strftime('%Y%m%d')}.db"
    current.write_bytes(b"corrupt recent backup")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (output / "cron_backup.log").read_text()

    assert result.returncode == 0
    assert "quick_check FAIL" in log
    assert current.read_bytes() == b"corrupt recent backup"
    immutable = list(
        backup.glob(f"archive_{time.strftime('%Y%m%d')}_*.db")
    )
    assert len(immutable) == 1
    with sqlite3.connect(immutable[0]) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == (
            "ok"
        )
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == (
            "recoverable"
        )
    checksum = immutable[0].with_suffix(".db.sha256")
    check = subprocess.run(
        ["sha256sum", "-c", checksum.name],
        cwd=backup,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0


def test_backup_rejects_integrity_ok_stdout_when_sqlite_exits_nonzero(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")
    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    fake_sqlite = fake_bin / "sqlite3"
    fake_sqlite.write_text(
        """#!/usr/bin/env bash
if [ "${2:-}" = "PRAGMA integrity_check;" ]; then
    printf 'ok\\n'
    exit 7
fi
exec /usr/bin/sqlite3 "$@"
"""
    )
    fake_sqlite.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    log = (output / "cron_backup.log").read_text()
    assert "integrity FAIL: sqlite exit 7: ok" in log
    assert not list(backup.glob("upbit_d1_*_*.db"))


def test_backup_rejects_quick_check_ok_stdout_with_nonzero_exit_in_skip_and_reuse(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    source_db = data / "archive.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('stable')")
    old_source_time = time.time() - 30 * 24 * 3600
    os.utime(source_db, (old_source_time, old_source_time))
    (output / "champion_state.json").write_text("{}", encoding="utf-8")
    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    first = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0
    generation = next(backup.glob("archive_*_*.db"))
    generation_bytes = generation.read_bytes()

    quick_check_calls = tmp_path / "quick-check-calls.txt"
    fake_sqlite = fake_bin / "sqlite3"
    fake_sqlite.write_text(
        """#!/usr/bin/env bash
if [ "${2:-}" = "PRAGMA quick_check;" ]; then
    printf '%s\\n' "$1" >> "$SQLITE_CALL_LOG"
    printf 'ok\\n'
    exit 7
fi
exec /usr/bin/sqlite3 "$@"
"""
    )
    fake_sqlite.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["SQLITE_CALL_LOG"] = str(quick_check_calls)

    second = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    latest_log = (output / "cron_backup.log").read_text().split(
        "=== prelude DB backup"
    )[-1]
    assert second.returncode == 1
    assert quick_check_calls.read_text().splitlines() == [
        str(generation),
        str(generation),
    ]
    assert "refresh archive (최근 보관본 generation/quick_check FAIL)" in latest_log
    assert "content-addressed DB collision/corruption" in latest_log
    assert "skip archive" not in latest_log
    assert generation.read_bytes() == generation_bytes


def test_backup_never_uses_valid_sqlite_with_stale_content_hash_as_skip_proof(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    for path in (scripts, data, output, backup):
        path.mkdir(parents=True)

    source_db = data / "archive.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('original')")
    old_source_time = time.time() - 30 * 24 * 3600
    os.utime(source_db, (old_source_time, old_source_time))
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    first = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0
    generation = next(backup.glob("archive_*_*.db"))
    with sqlite3.connect(generation) as connection:
        connection.execute("UPDATE evidence SET value='tampered-but-valid'")
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    second = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (output / "cron_backup.log").read_text()

    assert second.returncode != 0
    assert "generation/quick_check FAIL" in log
    assert "content-addressed DB collision/corruption" in log
    assert "skip archive" not in log.split(
        "=== prelude DB backup"
    )[-1]


def test_backup_same_day_runs_preserve_content_addressed_db_generations(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    for path in (scripts, data, output, backup):
        path.mkdir(parents=True)

    source_db = data / "upbit_d1.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('first')")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)

    first = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0
    date = time.strftime("%Y%m%d")
    first_generations = list(backup.glob(f"upbit_d1_{date}_*.db"))
    assert len(first_generations) == 1
    first_path = first_generations[0]
    first_bytes = first_path.read_bytes()

    # Identical bytes are an idempotent reuse, not a second copy.
    second = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0
    assert list(backup.glob(f"upbit_d1_{date}_*.db")) == [first_path]
    assert first_path.read_bytes() == first_bytes

    # A later same-day DB state gets a new hash-named generation. The first
    # recovery point and checksum pair remain byte-for-byte untouched.
    with sqlite3.connect(source_db) as connection:
        connection.execute("UPDATE evidence SET value='second'")
    third = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert third.returncode == 0

    generations = sorted(backup.glob(f"upbit_d1_{date}_*.db"))
    assert len(generations) == 2
    assert first_path.read_bytes() == first_bytes
    values = set()
    for generation in generations:
        assert len(generation.stem.rsplit("_", 1)[-1]) == 64
        with sqlite3.connect(generation) as connection:
            values.add(
                connection.execute(
                    "SELECT value FROM evidence"
                ).fetchone()[0]
            )
        checksum = generation.with_suffix(".db.sha256")
        checked = subprocess.run(
            ["sha256sum", "-c", checksum.name],
            cwd=backup,
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0
    assert values == {"first", "second"}


def test_backup_rejects_misdirected_existing_checksum_without_clobber(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('stable')")
    (output / "champion_state.json").write_text("{}", encoding="utf-8")

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    first = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0
    generation = next(
        backup.glob(f"upbit_d1_{time.strftime('%Y%m%d')}_*.db")
    )
    generation_bytes = generation.read_bytes()
    checksum = generation.with_suffix(".db.sha256")
    decoy = backup / "decoy.bin"
    decoy.write_bytes(b"unrelated but internally valid")
    decoy_digest = hashlib.sha256(decoy.read_bytes()).hexdigest()
    poisoned_checksum = f"{decoy_digest}  {decoy.name}\n".encode()
    checksum.write_bytes(poisoned_checksum)

    second = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert second.returncode == 1
    assert generation.read_bytes() == generation_bytes
    assert checksum.read_bytes() == poisoned_checksum
    assert "checksum/directory durable publish FAIL" in (
        output / "cron_backup.log"
    ).read_text()

    # If only the poisoned checksum pre-exists, a retry may publish the DB
    # hard-link before discovering the conflict. That new unpaired DB must be
    # removed without touching the pre-existing checksum.
    generation.unlink()
    third = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert third.returncode == 1
    assert not generation.exists()
    assert checksum.read_bytes() == poisoned_checksum


def test_backup_rejects_symlink_member_from_evidence_archive(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, backup, fake_bin):
        path.mkdir(parents=True)

    with sqlite3.connect(data / "upbit_d1.db") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
    snapshots = output / "recommend_snapshots"
    snapshots.mkdir()
    outside = repo / "outside.json"
    outside.write_text('{"secret":"must-not-enter-archive"}', encoding="utf-8")
    (snapshots / "linked.json").symlink_to(outside)

    source = _backup_script_source(repo)
    script = scripts / "backup_db.sh"
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "evidence tar FAIL" in (output / "cron_backup.log").read_text()
    assert not list(backup.glob("ledgers_*.manifest"))


@pytest.mark.parametrize(
    ("delivery_rc", "expected_rc", "expected_log"),
    [(0, 0, "[alert sent]"), (19, 1, "delivery FAIL")],
)
def test_heartbeat_uses_onfailure_only_when_detailed_alert_delivery_fails(
    tmp_path, delivery_rc, expected_rc, expected_log
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "output").mkdir()
    (repo / "output" / "policy_competition_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    shutil.copy2(Path("scripts/heartbeat.sh"), scripts / "heartbeat.sh")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [ "$1" = "-" ]; then
    content=$(cat)
    if [[ "$content" == *load_policy_artifact* ]] ||
       [[ "$content" == *"today's snapshot missing"* ]]; then
        echo HOSTILE-STDOUT-POLLUTION
    else
        echo ok
    fi
    exit 0
fi
if [ "$1" = "scripts/v2_scoreboard.py" ]; then
    echo "v2 scoreboard mock ok"
    exit 0
fi
if [ "$1" = "-c" ]; then
    exit "$DELIVERY_RC"
fi
exit 0
"""
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "DELIVERY_RC": str(delivery_rc),
        }
    )
    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (repo / "output" / "cron_heartbeat.log").read_text()

    assert result.returncode == expected_rc
    assert "paper_ledger.csv 파일 없음" in log
    assert "HOSTILE-STDOUT-POLLUTION" in log
    assert "policy_competition JSON/CSV/SQLite provenance ok" in log
    assert "recommend snapshot chain ok" in log
    assert "policy_competition provenance probe FAIL" not in log
    assert "recommend snapshot chain probe FAIL" not in log
    assert expected_log in log


def test_heartbeat_row_count_preserves_grep_failure_status():
    heartbeat = Path("scripts/heartbeat.sh").read_text()

    assert 'grep -c "^$YESTERDAY," "$ledger" 2>>"$LOG"' in heartbeat
    assert "grep_rc=$?" in heartbeat
    assert 'if [ "$grep_rc" -gt 1 ]' in heartbeat
    ledger_probe = heartbeat.split(
        "# 1) paper_ledger",
        1,
    )[1].split("# 2) DB integrity", 1)[0]
    assert "|| true" not in ledger_probe
    assert "today's snapshot missing" in heartbeat


def test_heartbeat_requires_todays_bound_backup_manifest():
    heartbeat = Path("scripts/heartbeat.sh").read_text()
    backup_probe = heartbeat.split(
        "# 3a) 오늘 04:00 evidence backup",
        1,
    )[1].split("# 3b) close no-decision", 1)[0]

    assert "BACKUP_DATE=$(date +%Y%m%d)" in backup_probe
    assert "python -m ops.backup_manifest" in backup_probe
    assert '--date "$BACKUP_DATE"' in backup_probe
    assert '--wait-seconds "$BACKUP_WAIT_SECONDS"' in backup_probe
    assert "-mtime -2" not in backup_probe


@pytest.mark.parametrize(
    ("probe_mode", "expected_exit", "expected_output"),
    (
        ("find-failure", 2, "1"),
        ("invalid-count", 0, "not-a-number"),
    ),
)
def test_heartbeat_no_decision_probe_never_reports_partial_aggregate_as_ok(
    tmp_path,
    probe_mode,
    expected_exit,
    expected_output,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (
        scripts,
        output / "close_no_decision",
        backup,
        fake_bin,
    ):
        path.mkdir(parents=True)
    shutil.copy2(Path("scripts/heartbeat.sh"), scripts / "heartbeat.sh")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-" ]; then cat >/dev/null; exit 0; fi\n'
        'if [ "${1:-}" = "-m" ]; then exit 1; fi\n'
        'if [ "${1:-}" = "scripts/v2_scoreboard.py" ]; then\n'
        "  echo 'v2 scoreboard mock ok'; exit 0\n"
        "fi\n"
        'if [ "${1:-}" = "-c" ]; then exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_find = fake_bin / "find"
    if probe_mode == "find-failure":
        fake_find.write_text(
            "#!/bin/bash\n"
            "echo '/mock/close_no_decision/2026-07-29.json'\n"
            "exit 2\n",
            encoding="utf-8",
        )
    else:
        fake_find.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        fake_wc = fake_bin / "wc"
        fake_wc.write_text(
            "#!/bin/bash\n"
            'if [ "${1:-}" = "-l" ]; then\n'
            "  cat >/dev/null\n"
            "  echo not-a-number\n"
            "  exit 0\n"
            "fi\n"
            'exec /usr/bin/wc "$@"\n',
            encoding="utf-8",
        )
        fake_wc.chmod(0o755)
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/bin/bash\n"
        "printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\\n'\n"
        "printf '/dev/mock 100 10 90 10%% /mock\\n'\n",
        encoding="utf-8",
    )
    for executable in (fake_python, fake_find, fake_df):
        executable.chmod(0o755)

    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PRELUDE_BACKUP_DIR": str(backup),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    log = (output / "cron_heartbeat.log").read_text(encoding="utf-8")

    assert result.returncode == 0
    assert "close no-decision probe FAIL (date=" in log
    assert f"exit={expected_exit}; output='{expected_output}')" in log
    assert "close no-decision (7d):" not in log
    assert "close no-decision day 감지" not in log


def test_heartbeat_rejects_numeric_output_from_failed_grep(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    output = repo / "output"
    backup = repo / "backup"
    fake_bin = tmp_path / "bin"
    for path in (scripts, output, backup, fake_bin):
        path.mkdir(parents=True)
    shutil.copy2(Path("scripts/heartbeat.sh"), scripts / "heartbeat.sh")
    (output / "paper_ledger.csv").write_text(
        "date,coin,status\n",
        encoding="utf-8",
    )

    fake_grep = fake_bin / "grep"
    fake_grep.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *paper_ledger.csv*) echo 7; exit 2 ;;\n"
        "esac\n"
        'exec /usr/bin/grep "$@"\n',
        encoding="utf-8",
    )
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-" ]; then cat >/dev/null; exit 0; fi\n'
        'if [ "${1:-}" = "-m" ]; then exit 1; fi\n'
        'if [ "${1:-}" = "scripts/v2_scoreboard.py" ]; then\n'
        "  echo 'v2 scoreboard mock ok'; exit 0\n"
        "fi\n"
        'if [ "${1:-}" = "-c" ]; then exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/bin/bash\n"
        "printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\\n'\n"
        "printf '/dev/mock 100 10 90 10%% /mock\\n'\n",
        encoding="utf-8",
    )
    for executable in (fake_grep, fake_python, fake_df):
        executable.chmod(0o755)

    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PRELUDE_BACKUP_DIR": str(backup),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    log = (output / "cron_heartbeat.log").read_text(encoding="utf-8")

    assert result.returncode == 0
    assert (
        "paper_ledger.csv row count probe FAIL "
        "(exit=2; output='7')"
    ) in log
    assert "paper_ledger.csv: 어제" not in log


def test_heartbeat_uses_canonical_policy_triplet_validator():
    heartbeat = Path("scripts/heartbeat.sh").read_text()

    assert "from ops.policy_competition import load_policy_artifact" in heartbeat
    assert "require_exact_asof=True" in heartbeat
    assert "require_current=True" in heartbeat
    assert "policy_competition provenance probe FAIL" in heartbeat


def test_heartbeat_policy_and_snapshot_probes_use_exit_code_contracts():
    heartbeat = Path("scripts/heartbeat.sh").read_text()
    policy_probe = heartbeat.split(
        "# 2b) policy competition",
        1,
    )[1].split("# 2c) ledger CSV", 1)[0]
    snapshot_probe = heartbeat.split(
        "# 2d) 단일 snapshot",
        1,
    )[1].split("# 3) disk", 1)[0]

    for probe in (policy_probe, snapshot_probe):
        assert 'if python - >>"$LOG" 2>&1 <<\'PYEOF\'' in probe
        assert "=$(python -" not in probe
        assert 'print("ok")' not in probe

    assert "raise SystemExit(1)" in policy_probe
    assert "raise SystemExit(1)" in snapshot_probe


def test_heartbeat_uses_stable_strict_json_for_v2_and_score_artifacts():
    heartbeat = Path("scripts/heartbeat.sh").read_text()
    snapshot_probe = heartbeat.split(
        "# 2d) 단일 snapshot",
        1,
    )[1].split("# 3) disk", 1)[0]

    assert "from ops.artifact_provenance import strict_json_object" in snapshot_probe
    assert "v2_payload = strict_json_object(v2_decision_path)" in snapshot_probe
    assert "report = strict_json_object(evaluation_path)" in snapshot_probe
    assert "json.loads(" not in snapshot_probe


@pytest.mark.parametrize(
    ("probe", "expected_log"),
    [
        ("sqlite", "upbit_d1.db integrity probe FAIL (exit=7): ok"),
        (
            "policy",
            "policy_competition provenance probe FAIL "
            "(exit=7; details logged)",
        ),
        ("csv", "ledger CSV 스키마 probe FAIL (exit=7): ok"),
        (
            "snapshot",
            "recommend snapshot chain probe FAIL "
            "(exit=7; details logged)",
        ),
    ],
)
def test_heartbeat_never_accepts_ok_stdout_from_failed_probe(
    tmp_path,
    probe,
    expected_log,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    data = repo / "data"
    output = repo / "output"
    fake_bin = tmp_path / "bin"
    for path in (scripts, data, output, fake_bin):
        path.mkdir(parents=True)
    source = Path("scripts/heartbeat.sh").read_text().replace(
        "PUBLISH_SCHEDULE_HHMM=1010",
        "PUBLISH_SCHEDULE_HHMM=9999",
    )
    (scripts / "heartbeat.sh").write_text(source)
    if probe == "sqlite":
        (data / "upbit_d1.db").write_bytes(b"probe")
    if probe == "policy":
        (output / "policy_competition_summary.json").write_text(
            "{}",
            encoding="utf-8",
        )

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [ "$1" = "-" ]; then
    content=$(cat)
    echo ok
    case "$FAIL_PROBE" in
        policy)
            if [[ "$content" == *load_policy_artifact* ]]; then
                echo "policy probe diagnostic" >&2
                exit 7
            fi
            ;;
        csv)
            [[ "$content" == *"REQUIRED ="* ]] && exit 7
            ;;
        snapshot)
            if [[ "$content" == *"today's snapshot missing"* ]]; then
                echo "snapshot probe diagnostic" >&2
                exit 7
            fi
            ;;
    esac
    exit 0
fi
if [ "$1" = "scripts/v2_scoreboard.py" ]; then
    echo "v2 scoreboard mock ok"
    exit 0
fi
if [ "$1" = "-c" ]; then
    exit 0
fi
exit 0
"""
    )
    fake_python.chmod(0o755)
    fake_sqlite = fake_bin / "sqlite3"
    fake_sqlite.write_text(
        "#!/usr/bin/env bash\n"
        "echo ok\n"
        '[ "$FAIL_PROBE" = "sqlite" ] && exit 7\n'
        "exit 0\n"
    )
    fake_sqlite.chmod(0o755)
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\\n'\n"
        "printf '/dev/mock 100 10 90 10%% /mock\\n'\n"
    )
    fake_df.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "FAIL_PROBE": probe,
        }
    )
    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (output / "cron_heartbeat.log").read_text()

    assert result.returncode == 0
    assert expected_log in log
    if probe in {"policy", "snapshot"}:
        assert f"{probe} probe diagnostic" in log
    assert "[alert sent]" in log


def test_heartbeat_disk_probe_failure_is_alerted_not_silently_skipped(
    tmp_path,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "output").mkdir()
    shutil.copy2(Path("scripts/heartbeat.sh"), scripts / "heartbeat.sh")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [ "$1" = "-" ]; then
    cat >/dev/null
    echo ok
    exit 0
fi
if [ "$1" = "scripts/v2_scoreboard.py" ]; then
    echo "v2 scoreboard mock ok"
    exit 0
fi
if [ "$1" = "-c" ]; then
    exit 0
fi
exit 0
"""
    )
    fake_python.chmod(0o755)
    fake_df = fake_bin / "df"
    fake_df.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake_df.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = (repo / "output" / "cron_heartbeat.log").read_text()

    # Alert delivery succeeded, so OnFailure need not run.
    assert result.returncode == 0
    assert "disk usage 검사 실행 실패" in log


def _run_heartbeat_publish_probe(
    tmp_path: Path,
    *,
    publish_log: str | None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    output = repo / "output"
    output.mkdir()
    source = Path("scripts/heartbeat.sh").read_text().replace(
        "PUBLISH_SCHEDULE_HHMM=1010",
        "PUBLISH_SCHEDULE_HHMM=0000",
    )
    (scripts / "heartbeat.sh").write_text(source)
    if publish_log is not None:
        (output / "cron_publish.log").write_text(
            publish_log,
            encoding="utf-8",
        )

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [ "$1" = "-" ]; then
    cat >/dev/null
    echo ok
    exit 0
fi
if [ "$1" = "scripts/v2_scoreboard.py" ]; then
    echo "v2 scoreboard mock ok"
    exit 0
fi
exit 0
"""
    )
    fake_python.chmod(0o755)
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\\n'\n"
        "printf '/dev/mock 100 10 90 10%% /mock\\n'\n"
    )
    fake_df.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(scripts / "heartbeat.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, (output / "cron_heartbeat.log").read_text()


def test_heartbeat_after_schedule_fails_closed_when_publish_log_missing(
    tmp_path: Path,
) -> None:
    result, log = _run_heartbeat_publish_probe(
        tmp_path,
        publish_log=None,
    )

    # Detailed heartbeat alert delivery succeeds, so generic OnFailure is not
    # needed even though the publish problem is recorded and sent.
    assert result.returncode == 0
    assert "publish 예정 시각 이후 로그 없음" in log
    assert "[alert sent]" in log


def test_heartbeat_requires_today_latest_publish_session_to_complete(
    tmp_path: Path,
) -> None:
    today = subprocess.run(
        ["date", "+%F"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result, log = _run_heartbeat_publish_probe(
        tmp_path,
        publish_log=(
            f"=== prelude publish dashboard {today} 10:10:01 ===\n"
            "[1/3] build_dashboard.py\n"
        ),
    )

    assert result.returncode == 0
    assert "publish 오늘 최신 세션 성공 완료 marker 없음" in log
    assert "[alert sent]" in log


def test_heartbeat_accepts_today_latest_publish_success_marker(
    tmp_path: Path,
) -> None:
    today = subprocess.run(
        ["date", "+%F"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _, log = _run_heartbeat_publish_probe(
        tmp_path,
        publish_log=(
            f"=== prelude publish dashboard {today} 10:10:01 ===\n"
            "[done] 10:10:09 committed + pushed\n"
        ),
    )

    assert "publish 오늘 최신 세션 완료 ok" in log
    assert "publish 오늘 최신 세션 성공 완료 marker 없음" not in log
