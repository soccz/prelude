"""Run only isolated shell copies/fake Python; never collectors or model fits."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELLS = ("daily_close_distribution.sh", "daily_close_preopen.sh")
COMMON_RESEARCH = (
    "-m ops.policy_competition",
    "scripts/train_recommendation_meta.py",
    "scripts/idea_validation_report.py",
    "scripts/build_idea_validation_html.py",
)
EVALUATOR = "scripts/evaluate_recommend_score_labels.py"
TRIAL_REVIEW = "-m ops.recommend_trial_review --refresh"
PREOPEN_RESEARCH = (EVALUATOR, TRIAL_REVIEW)
BUDGET_COMMAND = "/usr/bin/timeout --signal=TERM --kill-after=10s 300s"


def _run(
    tmp_path: Path,
    shell: str,
    *,
    fail_command: str = "",
    fail_rc: int = 43,
    hang_command: str = "",
    ignore_term: bool = False,
    via_pipeline: bool = False,
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    source = (ROOT / "scripts" / shell).read_text()
    assert source.count(BUDGET_COMMAND) == 1
    # Only the copied fixture has a shorter deadline; no production override.
    source = source.replace(BUDGET_COMMAND, BUDGET_COMMAND.replace("10s 300s", "0.2s 0.3s"))
    script = scripts / shell
    script.write_text(source)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        r'''#!/usr/bin/env bash
set -eu
printf 'CALL:%s\n' "$*" >> "$CALL_LOG"
command="$1"
if [ "$1" = "-m" ]; then command="$1 $2"; fi
if [ "$command" = "-m ops.close_input_gate" ]; then
    target=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--through-asof" ]; then target="$2"; break; fi
        shift
    done
    printf '%s\0close\0__PRELUDE_CLOSE_PLAN_V1_OK__\0' "$target"
    exit 0
fi
if [ "$command" = "$FAIL_COMMAND" ]; then exit "$FAIL_RC"; fi
if [ "$command" = "$HANG_COMMAND" ]; then
    if [ "$IGNORE_TERM" = 1 ]; then trap '' TERM; else trap 'exit 0' TERM; fi
    printf 'HANG:%s\n' "$command" >> "$CALL_LOG"
    while :; do /usr/bin/sleep 1; done
fi
if [ "$command" = "scripts/close_recommend_ledger.py" ]; then
    for arg in "$@"; do
        case "$arg" in r1-open|r1-preopen) printf 'DONE:%s\n' "$arg" >> "$CALL_LOG";; esac
    done
fi
if [ "$command" = "scripts/label_recommend_snapshots.py" ]; then
    printf 'DONE:labels\n' >> "$CALL_LOG"
fi
exit 0
'''
    )
    fake_python.chmod(0o755)
    call_log = tmp_path / "calls.log"
    env = os.environ.copy()
    # The project has neither .env nor venv; every Python workload is mocked.
    env.update(
        PATH=f"{fake_bin}:/usr/bin:/bin",
        PRELUDE_FORBID_TELEGRAM="1",
        PRELUDE_ROOT=str(repo),
        CALL_LOG=str(call_log),
        FAIL_COMMAND=fail_command,
        FAIL_RC=str(fail_rc),
        HANG_COMMAND=hang_command,
        IGNORE_TERM=str(int(ignore_term)),
    )
    command = ["/bin/bash", str(script)]
    if via_pipeline:
        stage = "distribution-close" if shell == SHELLS[0] else "preopen-close"
        command = ["/bin/bash", str(ROOT / "deploy/run_pipeline_stage.sh"), stage, *command]
    process = subprocess.Popen(
        command, cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        # A regression must not leave a hanging synthetic process group behind.
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("isolated research command exceeded fixture deadline")
    calls = call_log.read_text().splitlines()
    logs = "\n".join(path.read_text() for path in (repo / "output").glob("*.log"))
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return result, calls, logs, repo, env


def _assert_core_precedes(calls: list[str], shell: str, command: str) -> None:
    cohort = "r1-open" if shell == SHELLS[0] else "r1-preopen"
    report_index = next(i for i, row in enumerate(calls) if row == f"CALL:{command}")
    assert calls.index(f"DONE:{cohort}") < report_index
    if shell == SHELLS[1]:
        assert calls.index("DONE:labels") < report_index


@pytest.mark.parametrize("shell", SHELLS)
def test_core_finishes_before_legacy_and_research(tmp_path, shell):
    result, calls, logs, _, _ = _run(tmp_path, shell)
    assert result.returncode == 0, result.stdout + result.stderr + logs
    legacy = "scripts/close_paper_ledger.py" if shell == SHELLS[0] else "scripts/close_preopen_ledger.py"
    _assert_core_precedes(calls, shell, legacy)
    for command in COMMON_RESEARCH + (PREOPEN_RESEARCH if shell == SHELLS[1] else ()):
        _assert_core_precedes(calls, shell, command)


@pytest.mark.parametrize(
    ("shell", "command"),
    [(shell, command) for shell in SHELLS for command in COMMON_RESEARCH]
    + [(SHELLS[1], command) for command in PREOPEN_RESEARCH],
)
def test_research_error_is_loud_after_core_completion(tmp_path, shell, command):
    fail_command = "-m ops.recommend_trial_review" if command == TRIAL_REVIEW else command
    result, calls, logs, _, _ = _run(tmp_path, shell, fail_command=fail_command)
    assert result.returncode == 43
    _assert_core_precedes(calls, shell, command)
    assert "[critical]" in logs
    assert "exit=43" in logs
    if command == "scripts/idea_validation_report.py":
        assert "CALL:scripts/build_idea_validation_html.py" not in calls
        assert "stale input forbidden" in logs


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(("ignore_term", "expected_rc"), [(False, 124), (True, 137)])
def test_hung_research_is_bounded_loud_and_followups_run(tmp_path, shell, ignore_term, expected_rc):
    command = "scripts/train_recommendation_meta.py"
    result, calls, logs, _, _ = _run(
        tmp_path, shell, hang_command=command, ignore_term=ignore_term,
    )
    assert result.returncode == expected_rc
    assert f"HANG:{command}" in calls
    _assert_core_precedes(calls, shell, command)
    assert "CALL:scripts/idea_validation_report.py" in calls
    assert f"recommendation meta train failed (exit={expected_rc})" in logs


@pytest.mark.parametrize("shell", SHELLS)
def test_first_core_error_survives_later_research_timeout(tmp_path, shell):
    result, calls, logs, _, _ = _run(
        tmp_path, shell, fail_command="scripts/close_recommend_ledger.py",
        fail_rc=27, hang_command="scripts/train_recommendation_meta.py",
    )
    assert result.returncode == 27
    assert "exit=124" in logs
    assert "CALL:scripts/idea_validation_report.py" in calls
    if shell == SHELLS[1]:
        assert "DONE:labels" in calls


def test_label_error_does_not_undo_completed_r1(tmp_path):
    result, calls, logs, _, _ = _run(
        tmp_path, SHELLS[1], fail_command="scripts/label_recommend_snapshots.py",
        fail_rc=2, hang_command=EVALUATOR,
    )
    assert result.returncode == 2
    assert "DONE:r1-preopen" in calls
    assert "score label partial" in logs
    assert "exit=124" in logs


@pytest.mark.parametrize("shell", SHELLS)
def test_research_failure_still_blocks_success_marker_and_publish(tmp_path, shell):
    result, calls, _, repo, env = _run(
        tmp_path, shell, fail_command="scripts/idea_validation_report.py",
        via_pipeline=True,
    )
    assert result.returncode == 43
    _assert_core_precedes(calls, shell, "scripts/idea_validation_report.py")
    assert not list((repo / "output/pipeline_state").glob("*.ok"))
    publish_marker = repo / "publish-must-not-run"
    publish = subprocess.run(
        ["/bin/bash", str(ROOT / "deploy/run_pipeline_stage.sh"), "publish",
         "/usr/bin/touch", str(publish_marker)],
        cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=8,
    )
    assert publish.returncode != 0
    assert "success marker missing" in publish.stderr
    assert not publish_marker.exists()


@pytest.mark.parametrize("shell", SHELLS)
def test_deadline_applies_only_to_research_with_no_runtime_override(shell):
    source = (ROOT / "scripts" / shell).read_text()
    assert source.count(BUDGET_COMMAND) == 1
    bounded = [line.strip() for line in source.splitlines() if "if run_research_step" in line]
    expected = COMMON_RESEARCH + (PREOPEN_RESEARCH if shell == SHELLS[1] else ())
    if shell == SHELLS[1]:
        expected += ("-m ops.recommend_book_validation --refresh",)
    assert len(bounded) == len(expected)
    for command in expected:
        assert any(f"run_research_step python {command} >>" in line for line in bounded)
    assert "still block" in source
    assert "not complete research/publish isolation" in source


@pytest.mark.parametrize(("ignore_term", "expected_rc"), [(False, 124), (True, 137)])
def test_trial_review_timeout_is_loud_after_labels(tmp_path, ignore_term, expected_rc):
    result, calls, logs, _, _ = _run(
        tmp_path, SHELLS[1], hang_command="-m ops.recommend_trial_review",
        ignore_term=ignore_term,
    )
    assert result.returncode == expected_rc
    _assert_core_precedes(calls, SHELLS[1], TRIAL_REVIEW)
    assert "CALL:scripts/idea_validation_report.py" in calls
    assert f"post-label trial review failed (exit={expected_rc})" in logs
