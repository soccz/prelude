from __future__ import annotations

import configparser
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
PIPELINE_WRAPPER = DEPLOY / "run_pipeline_stage.sh"
INSTALLER = DEPLOY / "install_systemd.sh"
SERVICE_UNITS = (
    "prelude-distribution.service",
    "prelude-close.service",
    "prelude-preopen.service",
    "prelude-preopen-close.service",
    "prelude-publish-dashboard.service",
    "prelude-backup.service",
    "prelude-heartbeat.service",
    "prelude-selftest.service",
    "prelude-microstructure.service",
    "prelude-failure-alert@.service",
)
TIMER_UNITS = (
    "prelude-distribution.timer",
    "prelude-close.timer",
    "prelude-preopen.timer",
    "prelude-preopen-close.timer",
    "prelude-publish-dashboard.timer",
    "prelude-backup.timer",
    "prelude-heartbeat.timer",
    "prelude-selftest.timer",
    "prelude-microstructure.timer",
)
ALL_UNITS = SERVICE_UNITS + TIMER_UNITS


def _unit(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.optionxform = str
    with path.open(encoding="utf-8") as fh:
        config.read_file(fh)
    return config


def _run_wrapper(
    tmp_path: Path,
    stage: str,
    *command: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PRELUDE_ROOT"] = str(tmp_path)
    return subprocess.run(
        [str(PIPELINE_WRAPPER), stage, *command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_successful_close_logs(root: Path) -> None:
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    compact = now.strftime("%Y%m%d")
    iso = now.strftime("%Y-%m-%d")
    output = root / "output"
    state = output / "pipeline_state"
    output.mkdir(parents=True, exist_ok=True)
    state.mkdir()
    (output / f"cron_close_{compact}.log").write_text(
        f"=== prelude distribution close KST {iso} 09:30:00 ===\n"
        "[done] 09:35:00 exit=0\n",
        encoding="utf-8",
    )
    (output / f"cron_preopen_close_{compact}.log").write_text(
        f"=== prelude pre-open close KST {iso} 10:05:00 ===\n"
        "[done] 10:08:00 exit=0\n",
        encoding="utf-8",
    )
    for stage in ("distribution-close", "preopen-close"):
        (state / f"{stage}_{compact}.ok").write_text(
            f"stage={stage}\n"
            f"kst_date={iso}\n"
            "status=complete\n"
            f"completed_at={iso}T10:08:00+09:00\n",
            encoding="utf-8",
        )


def _fake_install_env(tmp_path: Path, *, systemctl_exit: int = 0) -> dict[str, str]:
    fixture_root = tmp_path / "fixture-root"
    cron_root = fixture_root
    unit_dir = fixture_root / "etc" / "systemd" / "system"
    bin_dir = fixture_root / "bin"
    unit_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    for unit in ALL_UNITS:
        target = unit_dir / unit
        target.write_bytes((DEPLOY / unit).read_bytes())
        target.chmod(0o644)

    systemctl = bin_dir / "systemctl"
    systemctl.write_text(
        "#!/bin/sh\n"
        "key=$1\n"
        "if [ \"$#\" -gt 1 ]; then key=\"$1:$2\"; fi\n"
        "case \"$1\" in\n"
        "  is-enabled|is-active)\n"
        "    command=$1\n"
        "    for argument in \"$@\"; do\n"
        "      case \"$argument\" in\n"
        "        \"$command\"|--quiet) ;;\n"
        "        *) key=\"$command:$argument\" ;;\n"
        "      esac\n"
        "    done ;;\n"
        "esac\n"
        "if [ -n \"${FAKE_SYSTEMCTL_LOG:-}\" ]; then\n"
        "  printf '%s\\n' \"$key\" >> \"$FAKE_SYSTEMCTL_LOG\"\n"
        "fi\n"
        "if [ -n \"${FAKE_SYSTEMCTL_FAIL_MATCH:-}\" ] && "
        "[ \"$key\" = \"$FAKE_SYSTEMCTL_FAIL_MATCH\" ]; then\n"
        "  count=$(/usr/bin/grep -Fxc -- \"$key\" "
        "\"$FAKE_SYSTEMCTL_LOG\" 2>/dev/null)\n"
        "  if [ \"$count\" -eq "
        "\"${FAKE_SYSTEMCTL_FAIL_OCCURRENCE:-1}\" ]; then\n"
        "    exit 70\n"
        "  fi\n"
        "fi\n"
        "if [ -n \"${FAKE_SYSTEMCTL_STATE_ROOT:-}\" ]; then\n"
        "  unit_name=${key#*:}\n"
        "  state_file=\"$FAKE_SYSTEMCTL_STATE_ROOT/$unit_name\"\n"
        "  case \"$1\" in\n"
        "    is-enabled|is-active)\n"
        "      suffix=enabled; fallback=not-found; wanted=enabled\n"
        "      if [ \"$1\" = is-active ]; then suffix=active; fallback=inactive; wanted=active; fi\n"
        "      state=$fallback\n"
        "      if [ -f \"$state_file.$suffix\" ]; then read -r state < \"$state_file.$suffix\"; fi\n"
        "      [ \"${2:-}\" = --quiet ] || printf '%s\\n' \"$state\"\n"
        "      [ \"$state\" = \"$wanted\" ]; exit $? ;;\n"
        "    enable|disable)\n"
        "      state=enabled; [ \"$1\" = enable ] || state=disabled\n"
        "      printf '%s\\n' \"$state\" > \"$state_file.enabled\"; exit 0 ;;\n"
        "    start|restart|stop)\n"
        "      state=active; [ \"$1\" != stop ] || state=inactive\n"
        "      printf '%s\\n' \"$state\" > \"$state_file.active\"; exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "case \"$1\" in\n"
        "  list-unit-files)\n"
        "    printf '%s' \"${FAKE_SYSTEMCTL_UNIT_FILES:-}\"\n"
        f"    exit {systemctl_exit} ;;\n"
        "  list-units)\n"
        "    printf '%s' \"${FAKE_SYSTEMCTL_UNITS:-}\"\n"
        f"    exit {systemctl_exit} ;;\n"
        "  show)\n"
        "    if [ -n \"${FAKE_SYSTEMCTL_FRAGMENT_OVERRIDE:-}\" ]; then\n"
        "      printf '%s\\n' \"$FAKE_SYSTEMCTL_FRAGMENT_OVERRIDE\"\n"
        "    else\n"
        "      unit_name=\"$2\"\n"
        "      case \"$unit_name\" in\n"
        "      *'@'*)\n"
        "        instance=\"${unit_name#*@}\"; instance=\"${instance%.*}\"\n"
        "        if [ -n \"$instance\" ]; then\n"
        "          unit_name=\"${unit_name%%@*}@.${unit_name##*.}\"\n"
        "        fi ;;\n"
        "      esac\n"
        "      printf '%s/%s\\n' \"$FAKE_SYSTEMCTL_FRAGMENT_ROOT\" \"$unit_name\"\n"
        "    fi ;;\n"
        "  is-enabled)\n"
        "    state=${FAKE_SYSTEMCTL_ENABLED_STATE:-enabled}\n"
        "    if [ \"$state\" = \"track-unit-files\" ]; then\n"
        "      unit_arg=\n"
        "      for argument in \"$@\"; do\n"
        "        case \"$argument\" in\n"
        "          is-enabled|--quiet) ;;\n"
        "          *) unit_arg=\"$argument\" ;;\n"
        "        esac\n"
        "      done\n"
        "      if [ ! -e \"$FAKE_SYSTEMCTL_FRAGMENT_ROOT/$unit_arg\" ]; then\n"
        "        printf 'Failed to get unit file state for %s: "
        "No such file or directory\\n' \"$unit_arg\"\n"
        "        exit 1\n"
        "      fi\n"
        "      state=enabled\n"
        "    fi\n"
        "    [ \"${2:-}\" = \"--quiet\" ] || printf '%s\\n' \"$state\"\n"
        "    [ \"$state\" = \"enabled\" ] ;;\n"
        "  is-active)\n"
        "    state=${FAKE_SYSTEMCTL_ACTIVE_STATE:-active}\n"
        "    [ \"${2:-}\" = \"--quiet\" ] || printf '%s\\n' \"$state\"\n"
        "    [ \"$state\" = \"active\" ] ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o755)

    analyze = bin_dir / "systemd-analyze"
    analyze.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_ANALYZE_LOG\"\n"
        "count=$(/usr/bin/wc -l < \"$FAKE_ANALYZE_LOG\")\n"
        "if [ \"$count\" -eq \"${FAKE_ANALYZE_FAIL_OCCURRENCE:-0}\" ]; then\n"
        "  exit 71\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    analyze.chmod(0o755)

    env_file = fixture_root / ".env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=test-token\n"
        "TELEGRAM_CHAT_ID=test-chat\n"
        "PRELUDE_DASHBOARD_PIN=test-dashboard-secret-2026\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    env = os.environ.copy()
    env.update(
        {
            "PRELUDE_INSTALL_REPO": str(ROOT),
            "PRELUDE_INSTALL_UNIT_DIR": str(unit_dir),
            "PRELUDE_INSTALL_CRON_ROOT": str(cron_root),
            "PRELUDE_INSTALL_SYSTEMCTL": str(systemctl),
            "PRELUDE_INSTALL_ANALYZE": str(analyze),
            "PRELUDE_INSTALL_ENV_FILE": str(env_file),
            "FAKE_SYSTEMCTL_FRAGMENT_ROOT": str(unit_dir),
            "FAKE_SYSTEMCTL_LOG": str(fixture_root / "systemctl.log"),
            "FAKE_ANALYZE_LOG": str(fixture_root / "analyze.log"),
        }
    )
    return env


def _run_installer(
    env: dict[str, str],
    *,
    check_only: bool = True,
    add_microstructure: bool = False,
) -> subprocess.CompletedProcess[str]:
    args = ["bash", str(INSTALLER)]
    if check_only:
        args.append("--check-only")
    else:
        env = env.copy()
        env["PRELUDE_INSTALL_FIXTURE"] = "1"
        if add_microstructure:
            args.append("--add-microstructure")
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _fixture_repo_for_installer(
    tmp_path: Path,
    env: dict[str, str],
) -> Path:
    repo = tmp_path / "repo-source"
    runtime_paths = (
        "deploy/lock_exec.py",
        "deploy/load_runtime_env.sh",
        "deploy/run_pipeline_stage.sh",
        "ops/backup_manifest.py",
        "ops/runtime_env.py",
        "data/collector_upbit_microstructure.py",
        "data/upbit_microstructure.py",
        "scripts/capture_recommend_microstructure.py",
        "signals/recommend_microstructure.py",
        "signals/recommend_microstructure_trial.py",
        "scripts/evaluate_recommend_microstructure_trial.py",
        "signals/recommend_trade_shortlist.py",
        "signals/recommend_trade_shortlist_trial.py",
        "signals/recommend_trade_shortlist_eval.py",
        "scripts/evaluate_recommend_trade_shortlist_trial.py",
        "ops/recommend_microstructure_status.py",
        "ops/recommend_trade_shortlist_status.py",
        "scripts/daily_run_distribution.sh",
        "scripts/daily_close_distribution.sh",
        "scripts/daily_run_preopen.sh",
        "scripts/daily_close_preopen.sh",
        "scripts/publish_dashboard.sh",
        "scripts/validate_dashboard_assets.py",
        "scripts/backup_db.sh",
        "scripts/heartbeat.sh",
        "scripts/systemd_failure_alert.py",
    )
    for relative in (*runtime_paths, *(f"deploy/{unit}" for unit in ALL_UNITS)):
        source = ROOT / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(source.stat().st_mode & 0o777)
    python = repo / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    env["PRELUDE_INSTALL_REPO"] = str(repo)
    return repo


def test_all_services_pin_kst_network_and_failure_contracts() -> None:
    services = sorted(DEPLOY.glob("prelude-*.service"))
    operational = [path for path in services if "failure-alert@" not in path.name]

    assert len(operational) == 9
    for path in services:
        unit = _unit(path)
        assert unit["Service"]["Type"] == "oneshot"
        assert unit["Service"]["User"] == "soccz"
        assert unit["Service"]["UMask"] == "0077"
        assert unit["Service"]["NoNewPrivileges"] == "true"
        assert unit["Service"]["WorkingDirectory"] == "/home/soccz/22tb/prelude"
        assert "TZ=Asia/Seoul" in unit["Service"]["Environment"].split()
        assert "network-online.target" in unit["Unit"]["After"].split()
        assert "network-online.target" in unit["Unit"]["Wants"].split()
        assert (
            "/home/soccz/22tb/prelude"
            in unit["Unit"]["RequiresMountsFor"].split()
        )
        assert int(unit["Service"]["TimeoutStartSec"]) >= 120

    for path in operational:
        unit = _unit(path)
        assert (
            unit["Unit"]["OnFailure"]
            == "prelude-failure-alert@%n.service"
        )
        assert unit["Service"]["StandardOutput"] == "journal"
        assert unit["Service"]["StandardError"] == "journal"


def test_selftest_runs_full_suite_without_ordering_recommendations_after_it() -> None:
    """Keep the 07:30 full-suite alarm, but never queue R1 behind a late test."""
    unit = _unit(DEPLOY / "prelude-selftest.service")
    command = unit["Service"]["ExecStart"]
    assert "/venv/bin/python -m pytest" in command
    environment = unit["Service"]["Environment"].split()
    assert "PRELUDE_FORBID_TELEGRAM=1" in environment
    assert "TMPDIR=/home/soccz/22tb/tmp" in environment
    assert int(unit["Service"]["TimeoutStartSec"]) >= 1800
    assert unit["Unit"]["OnFailure"] == "prelude-failure-alert@%n.service"
    assert "Before" not in unit["Unit"]
    assert unit["Unit"]["After"].split() == ["network-online.target"]


def test_postopen_pipeline_is_serialized_and_ordered() -> None:
    close = _unit(DEPLOY / "prelude-close.service")
    preopen_close = _unit(DEPLOY / "prelude-preopen-close.service")
    publish = _unit(DEPLOY / "prelude-publish-dashboard.service")
    heartbeat = _unit(DEPLOY / "prelude-heartbeat.service")

    for unit, stage in (
        (close, "distribution-close"),
        (preopen_close, "preopen-close"),
        (publish, "publish"),
        (heartbeat, "heartbeat"),
    ):
        command = unit["Service"]["ExecStart"]
        assert "/deploy/run_pipeline_stage.sh" in command
        assert f" {stage} " in command
        assert int(unit["Service"]["TimeoutStartSec"]) >= 3600

    assert "prelude-close.service" in preopen_close["Unit"]["After"].split()
    assert {
        "prelude-close.service",
        "prelude-preopen-close.service",
    } <= set(publish["Unit"]["After"].split())
    assert "prelude-publish-dashboard.service" in heartbeat["Unit"]["After"].split()
    assert int(heartbeat["Service"]["TimeoutStartSec"]) >= 3900


def test_timer_calendar_and_catchup_contracts_are_explicit_kst() -> None:
    expected = {
        "prelude-backup.timer": ("*-*-* 04:00:00 Asia/Seoul", "true"),
        "prelude-preopen.timer": ("*-*-* 08:50:00 Asia/Seoul", "false"),
        "prelude-distribution.timer": ("*-*-* 09:05:00 Asia/Seoul", "false"),
        "prelude-close.timer": ("*-*-* 09:30:00 Asia/Seoul", "true"),
        "prelude-preopen-close.timer": ("*-*-* 10:05:00 Asia/Seoul", "true"),
        "prelude-publish-dashboard.timer": (
            "*-*-* 10:10:00 Asia/Seoul",
            "true",
        ),
        "prelude-heartbeat.timer": ("*-*-* 10:30:00 Asia/Seoul", "true"),
        "prelude-selftest.timer": ("*-*-* 07:30:00 Asia/Seoul", "true"),
        "prelude-microstructure.timer": ("*-*-* 08:45:00 Asia/Seoul", "false"),
    }

    timers = sorted(DEPLOY.glob("prelude-*.timer"))
    assert {path.name for path in timers} == set(expected)
    for path in timers:
        timer = _unit(path)["Timer"]
        assert (timer["OnCalendar"], timer["Persistent"]) == expected[path.name]

    for name in ("prelude-preopen.timer", "prelude-distribution.timer"):
        assert _unit(DEPLOY / name)["Timer"]["AccuracySec"] == "5s"


def test_publish_fails_closed_without_both_current_successful_closes(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode != 0
    assert "success marker missing" in result.stderr


def test_publish_rejects_incomplete_latest_current_day_session(
    tmp_path: Path,
) -> None:
    _write_successful_close_logs(tmp_path)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    close_log = tmp_path / "output" / f"cron_close_{now:%Y%m%d}.log"
    with close_log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"=== prelude distribution close KST {now:%Y-%m-%d} 09:40:00 ===\n"
            "[1/3] data update\n"
        )

    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode != 0
    assert "did not finish successfully" in result.stderr


def test_publish_accepts_both_current_successful_close_sessions(
    tmp_path: Path,
) -> None:
    _write_successful_close_logs(tmp_path)

    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode == 0, result.stderr


def test_publish_rejects_marker_with_wrong_stage_or_date(tmp_path: Path) -> None:
    _write_successful_close_logs(tmp_path)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    marker = (
        tmp_path
        / "output"
        / "pipeline_state"
        / f"preopen-close_{now:%Y%m%d}.ok"
    )
    marker.write_text(
        "stage=preopen-close\nkst_date=1999-01-01\n",
        encoding="utf-8",
    )

    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode != 0
    assert "invalid current close success marker" in result.stderr


def test_publish_rejects_symlinked_success_marker(tmp_path: Path) -> None:
    _write_successful_close_logs(tmp_path)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    marker = (
        tmp_path
        / "output"
        / "pipeline_state"
        / f"preopen-close_{now:%Y%m%d}.ok"
    )
    outside = tmp_path / "outside-marker"
    outside.write_bytes(marker.read_bytes())
    marker.unlink()
    marker.symlink_to(outside)

    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode != 0
    assert "invalid current close success marker" in result.stderr


def test_publish_rejects_explicit_failed_close_session(tmp_path: Path) -> None:
    _write_successful_close_logs(tmp_path)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    close_log = tmp_path / "output" / f"cron_close_{now:%Y%m%d}.log"
    close_log.write_text(
        f"=== prelude distribution close KST {now:%Y-%m-%d} 09:30:00 ===\n"
        "[done] 09:35:00 exit=7\n",
        encoding="utf-8",
    )

    result = _run_wrapper(tmp_path, "publish", "/bin/true")

    assert result.returncode != 0
    assert "did not finish successfully" in result.stderr


def test_pipeline_wrapper_holds_one_lock_for_entire_child_process(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.txt"
    env = os.environ.copy()
    env["PRELUDE_ROOT"] = str(tmp_path)
    child = (
        "import pathlib,sys,time;"
        "p=pathlib.Path(sys.argv[1]);tag=sys.argv[2];delay=float(sys.argv[3]);"
        "f=p.open('a');f.write(tag+'-start\\n');f.flush();f.close();"
        "time.sleep(delay);"
        "f=p.open('a');f.write(tag+'-end\\n');f.flush();f.close()"
    )
    first = subprocess.Popen(
        [
            str(PIPELINE_WRAPPER),
            "distribution-close",
            sys.executable,
            "-c",
            child,
            str(events),
            "first",
            "0.3",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if events.exists() and "first-start" in events.read_text(encoding="utf-8"):
            break
        time.sleep(0.01)
    else:
        first.kill()
        raise AssertionError("first serialized child did not start")

    second = subprocess.Popen(
        [
            str(PIPELINE_WRAPPER),
            "preopen-close",
            sys.executable,
            "-c",
            child,
            str(events),
            "second",
            "0",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_stdout, first_stderr = first.communicate(timeout=5)
    second_stdout, second_stderr = second.communicate(timeout=5)

    assert first.returncode == 0, (first_stdout, first_stderr)
    assert second.returncode == 0, (second_stdout, second_stderr)
    assert events.read_text(encoding="utf-8").splitlines() == [
        "first-start",
        "first-end",
        "second-start",
        "second-end",
    ]
    today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    assert (
        tmp_path / "output" / "pipeline_state" / f"distribution-close_{today}.ok"
    ).is_file()
    assert (
        tmp_path / "output" / "pipeline_state" / f"preopen-close_{today}.ok"
    ).is_file()


def test_failed_close_propagates_status_and_leaves_no_success_marker(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(tmp_path, "distribution-close", "/bin/sh", "-c", "exit 23")

    assert result.returncode == 23
    state_dir = tmp_path / "output" / "pipeline_state"
    assert not list(state_dir.glob("distribution-close_*.ok"))


def test_close_rejects_symlinked_marker_directory_before_child(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (outside / "sentinel").write_text("untouched\n", encoding="utf-8")
    (output / "pipeline_state").symlink_to(outside, target_is_directory=True)
    child_marker = tmp_path / "child-ran"

    result = _run_wrapper(
        tmp_path,
        "distribution-close",
        "/usr/bin/touch",
        str(child_marker),
    )

    assert result.returncode != 0
    assert not child_marker.exists()
    assert (outside / "sentinel").read_text(encoding="utf-8") == "untouched\n"


def test_close_rejects_symlinked_marker_target_before_child(
    tmp_path: Path,
) -> None:
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    state = tmp_path / "output" / "pipeline_state"
    state.mkdir(parents=True)
    outside = tmp_path / "outside-marker"
    outside.write_text("untouched\n", encoding="utf-8")
    marker = state / f"distribution-close_{now:%Y%m%d}.ok"
    marker.symlink_to(outside)
    child_marker = tmp_path / "child-ran"

    result = _run_wrapper(
        tmp_path,
        "distribution-close",
        "/usr/bin/touch",
        str(child_marker),
    )

    assert result.returncode != 0
    assert not child_marker.exists()
    assert outside.read_text(encoding="utf-8") == "untouched\n"


def test_installer_check_rejects_active_cron_before_any_install(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    cron_file = Path(env["PRELUDE_INSTALL_CRON_ROOT"]) / "etc" / "cron.d" / "legacy"
    cron_file.parent.mkdir(parents=True)
    cron_file.write_text(
        "5 9 * * * soccz cd /home/soccz/22tb/prelude && "
        "bash scripts/daily_run_distribution.sh\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode != 0
    assert "active prelude cron found" in result.stderr


def test_installer_check_rejects_unexpected_prelude_timer(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    legacy = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / "prelude-recommend.timer"
    legacy.write_text("[Timer]\nOnCalendar=daily\n", encoding="utf-8")

    result = _run_installer(env)

    assert result.returncode != 0
    assert "unexpected installed prelude timer" in result.stderr


def test_installer_check_rejects_unexpected_registered_timer(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    env["FAKE_SYSTEMCTL_UNIT_FILES"] = "prelude-legacy.timer enabled\n"

    result = _run_installer(env)

    assert result.returncode != 0
    assert "unexpected registered prelude timer" in result.stderr


def test_installer_check_rejects_alternate_service_for_same_job(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    service = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / "old-daily.service"
    service.write_text(
        "[Service]\n"
        "ExecStart=/bin/bash /home/soccz/22tb/prelude/"
        "scripts/daily_run_distribution.sh\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode != 0
    assert "can duplicate prelude jobs" in result.stderr


def test_installer_check_refuses_failed_systemd_enumeration(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path, systemctl_exit=2)

    result = _run_installer(env)

    assert result.returncode != 0
    assert "refusing fail-open" in result.stderr


def test_installer_check_accepts_commented_cron_and_supported_timers(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    cron_file = Path(env["PRELUDE_INSTALL_CRON_ROOT"]) / "etc" / "crontab"
    cron_file.parent.mkdir(parents=True, exist_ok=True)
    cron_file.write_text(
        "# 5 9 * * * soccz /home/soccz/22tb/prelude/"
        "scripts/daily_run_distribution.sh\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode == 0, result.stderr
    assert "preflight: OK" in result.stdout


def test_installer_contract_lists_backup_runtime_dependencies() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    start = source.index("validate_repo_contract() {")
    end = source.index("\n}\n\nvalidate_installed_contract()", start)
    contract = source[start:end]

    assert "/usr/bin/timeout" in contract
    assert "ops/backup_manifest.py" in contract


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing", "required runtime missing: ops/backup_manifest.py"),
        ("symlink", "required runtime must not be a symlink"),
        ("directory", "required runtime is not a regular file"),
    ),
)
def test_installer_check_rejects_invalid_backup_manifest_runtime(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    env = _fake_install_env(tmp_path)
    repo = _fixture_repo_for_installer(tmp_path, env)
    module = repo / "ops" / "backup_manifest.py"
    module.unlink()
    if mutation == "symlink":
        module.symlink_to(ROOT / "ops" / "backup_manifest.py")
    elif mutation == "directory":
        module.mkdir()

    result = _run_installer(env)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_installer_check_rejects_missing_dashboard_pin(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    env_file = Path(env["PRELUDE_INSTALL_ENV_FILE"])
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=test-token\n"
        "TELEGRAM_CHAT_ID=test-chat\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode != 0
    assert "PRELUDE_DASHBOARD_PIN" in result.stderr


def test_installer_drops_root_before_repository_env_parser() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    start = source.index("validate_env_contract() {")
    end = source.index("\n}\n\nvalidate_repo_contract()", start)
    contract = source[start:end]
    root_start = contract.index('    if [ "$EUID" -eq 0 ]; then')
    else_start = contract.index("    else\n", root_start)
    branch_end = contract.rindex("    fi")
    root_branch = contract[root_start:else_start]
    non_root_branch = contract[else_start:branch_end]

    assert "/usr/bin/python3" in contract
    assert "-I" in contract
    assert '"$REPO/ops/runtime_env.py"' in contract
    assert '"$REPO/venv/bin/python" -m ops.runtime_env' not in source
    assert root_branch.count('"${parser_command[@]}"') == 1
    assert "/usr/sbin/runuser --user \"$UNIT_USER\" --" in root_branch
    assert non_root_branch.count('"${parser_command[@]}"') == 1
    assert "/usr/sbin/runuser" not in non_root_branch


def test_installer_check_treats_supported_env_values_as_literal_data(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    marker = tmp_path / "should-never-run"
    env_file = Path(env["PRELUDE_INSTALL_ENV_FILE"])
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN='$(touch "
        f"{marker})'\n"
        "TELEGRAM_CHAT_ID='-1001234567890'\n"
        "PRELUDE_DASHBOARD_PIN='literal $() `ticks`; & value'\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode == 0, result.stderr
    assert "preflight: OK" in result.stdout
    assert not marker.exists()


def test_installer_check_rejects_unknown_or_shell_control_environment_key(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    marker = tmp_path / "should-never-run"
    env_file = Path(env["PRELUDE_INSTALL_ENV_FILE"])
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=test-token\n"
        "TELEGRAM_CHAT_ID=test-chat\n"
        "PRELUDE_DASHBOARD_PIN=test-dashboard-secret-2026\n"
        f"GIT_SSH_COMMAND=$(touch {marker})\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode != 0
    assert "unsupported runtime environment key: GIT_SSH_COMMAND" in result.stderr
    assert not marker.exists()


def test_installer_check_rejects_insecure_env_mode(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    Path(env["PRELUDE_INSTALL_ENV_FILE"]).chmod(0o644)

    result = _run_installer(env)

    assert result.returncode != 0
    assert ".env mode must be 0400 or 0600" in result.stderr


def test_installer_check_rejects_insecure_installed_unit_mode(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    unit = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / ALL_UNITS[0]
    unit.chmod(0o666)

    result = _run_installer(env)

    assert result.returncode != 0
    assert "installed unit mode must be 0644" in result.stderr


@pytest.mark.parametrize("unit", ALL_UNITS)
def test_installer_check_rejects_each_missing_unit(
    tmp_path: Path,
    unit: str,
) -> None:
    env = _fake_install_env(tmp_path)
    (Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / unit).unlink()

    result = _run_installer(env)

    assert result.returncode != 0
    assert "installed unit missing" in result.stderr
    assert unit in result.stderr


def test_installer_check_rejects_stale_installed_unit(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    unit = "prelude-close.service"
    (Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / unit).write_text(
        "[Unit]\nDescription=stale fixture\n",
        encoding="utf-8",
    )

    result = _run_installer(env)

    assert result.returncode != 0
    assert f"installed unit differs from source: {unit}" in result.stderr


def test_installer_check_rejects_symlinked_installed_unit(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    unit = "prelude-close.service"
    installed = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / unit
    installed.unlink()
    installed.symlink_to(DEPLOY / unit)

    result = _run_installer(env)

    assert result.returncode != 0
    assert "installed unit must not be a symlink" in result.stderr


def test_installer_check_rejects_wrong_fragment_path(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    env["FAKE_SYSTEMCTL_FRAGMENT_OVERRIDE"] = "/wrong/unit/path"

    result = _run_installer(env)

    assert result.returncode != 0
    assert "systemd FragmentPath mismatch" in result.stderr


@pytest.mark.parametrize("state_command", ("is-enabled", "is-active"))
def test_installer_check_requires_every_timer_live_state(
    tmp_path: Path,
    state_command: str,
) -> None:
    env = _fake_install_env(tmp_path)
    timer = "prelude-heartbeat.timer"
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = f"{state_command}:{timer}"

    result = _run_installer(env)

    assert result.returncode != 0
    expected = "not enabled" if state_command == "is-enabled" else "not active"
    assert f"timer is {expected}: {timer}" in result.stderr


def _seed_old_installed_units(env: dict[str, str]) -> dict[str, bytes]:
    unit_dir = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    old: dict[str, bytes] = {}
    for unit in ALL_UNITS:
        payload = f"old installed bytes for {unit}\n".encode()
        (unit_dir / unit).write_bytes(payload)
        (unit_dir / unit).chmod(0o640)
        old[unit] = payload
    return old


def _assert_unit_bytes(
    env: dict[str, str],
    expected: dict[str, bytes],
    *,
    mode: int,
) -> None:
    unit_dir = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    for unit, payload in expected.items():
        installed = unit_dir / unit
        assert installed.read_bytes() == payload
        assert installed.stat().st_mode & 0o777 == mode
    assert not list(unit_dir.glob(".prelude-install.*"))
    assert not list(unit_dir.glob(".prelude-backup.*"))


@pytest.mark.parametrize(
    ("failure_match", "occurrence"),
    (
        ("daemon-reload", 1),
        ("enable:prelude-distribution.timer", 1),
        ("restart:prelude-distribution.timer", 1),
        ("show:prelude-close.service", 1),
        ("is-enabled:prelude-heartbeat.timer", 2),
        ("is-active:prelude-heartbeat.timer", 2),
        ("list-timers:--no-pager", 1),
    ),
)
def test_installer_rolls_back_files_and_runtime_state_on_partial_failure(
    tmp_path: Path,
    failure_match: str,
    occurrence: int,
) -> None:
    env = _fake_install_env(tmp_path)
    old = _seed_old_installed_units(env)
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = failure_match
    env["FAKE_SYSTEMCTL_FAIL_OCCURRENCE"] = str(occurrence)

    result = _run_installer(env, check_only=False)

    assert result.returncode != 0
    assert "restoring previous unit files and timer state" in result.stderr
    assert "Previous systemd configuration restored." in result.stderr
    _assert_unit_bytes(env, old, mode=0o640)
    calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text(encoding="utf-8")
    assert calls.count("daemon-reload\n") >= 2
    for timer in TIMER_UNITS:
        assert f"enable:{timer}\n" in calls
        assert f"restart:{timer}\n" in calls


def test_installer_accepts_first_install_of_brand_new_unit(tmp_path: Path) -> None:
    """신규 유닛 최초 설치: 실제 systemd 는 is-enabled 에 not-found 토큰이
    아니라 'Failed to get unit file state ...' 문구를 낸다(07-29 selftest
    설치에서 실측). 사전 상태 조회는 이를 not-found 로 관용해야 한다."""
    env = _fake_install_env(tmp_path)
    unit_dir = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    for unit in ("prelude-selftest.service", "prelude-selftest.timer"):
        (unit_dir / unit).unlink()
    env["FAKE_SYSTEMCTL_ENABLED_STATE"] = "track-unit-files"
    expected = {unit: (DEPLOY / unit).read_bytes() for unit in ALL_UNITS}

    result = _run_installer(env, check_only=False)

    assert result.returncode == 0, result.stderr
    _assert_unit_bytes(env, expected, mode=0o644)


def test_first_install_rollback_does_not_stop_nonexistent_timer(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    unit_dir = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    missing_units = {
        "prelude-selftest.service",
        "prelude-selftest.timer",
    }
    for unit in missing_units:
        (unit_dir / unit).unlink()
    env["FAKE_SYSTEMCTL_ENABLED_STATE"] = "track-unit-files"
    env["FAKE_SYSTEMCTL_ACTIVE_STATE"] = "inactive"
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = "daemon-reload"

    result = _run_installer(env, check_only=False)

    assert result.returncode != 0
    assert "Previous systemd configuration restored." in result.stderr
    for unit in missing_units:
        assert not (unit_dir / unit).exists()
    calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text(encoding="utf-8")
    assert "is-active:prelude-selftest.timer\n" not in calls
    # rollback 시작 시 새로 설치된 timer를 정지하는 1회만 허용한다. 파일을
    # 제거한 뒤 과거 inactive 상태를 "복원"하려는 두 번째 stop은 없어야 한다.
    assert calls.count("stop:prelude-selftest.timer\n") == 1


def test_installer_success_is_complete_and_idempotent(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    _seed_old_installed_units(env)
    expected = {unit: (DEPLOY / unit).read_bytes() for unit in ALL_UNITS}

    first = _run_installer(env, check_only=False)
    second = _run_installer(env, check_only=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    _assert_unit_bytes(env, expected, mode=0o644)
    calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text(encoding="utf-8")
    assert calls.count("daemon-reload\n") == 2
    assert calls.count("list-timers:--no-pager\n") == 2


def test_installer_staged_validation_failure_never_mutates_units(
    tmp_path: Path,
) -> None:
    env = _fake_install_env(tmp_path)
    old = _seed_old_installed_units(env)
    env["FAKE_ANALYZE_FAIL_OCCURRENCE"] = "2"

    result = _run_installer(env, check_only=False)

    assert result.returncode != 0
    assert "rejected one or more staged units" in result.stderr
    assert "restoring previous unit files" not in result.stderr
    _assert_unit_bytes(env, old, mode=0o640)
    systemctl_calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text(
        encoding="utf-8"
    )
    assert "daemon-reload\n" not in systemctl_calls
    assert "enable:" not in systemctl_calls
    assert "restart:" not in systemctl_calls
    analyze_calls = Path(env["FAKE_ANALYZE_LOG"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(analyze_calls) == 2
    assert all(unit in analyze_calls[1] for unit in ALL_UNITS)


def test_first_install_failure_removes_every_new_unit(tmp_path: Path) -> None:
    env = _fake_install_env(tmp_path)
    unit_dir = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    for unit in ALL_UNITS:
        (unit_dir / unit).unlink()
    env["FAKE_SYSTEMCTL_ENABLED_STATE"] = "not-found"
    env["FAKE_SYSTEMCTL_ACTIVE_STATE"] = "unknown"
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = "daemon-reload"

    result = _run_installer(env, check_only=False)

    assert result.returncode != 0
    assert "Previous systemd configuration restored." in result.stderr
    assert not any((unit_dir / unit).exists() for unit in ALL_UNITS)
    assert not list(unit_dir.glob(".prelude-install.*"))
    assert not list(unit_dir.glob(".prelude-backup.*"))


MICROSTRUCTURE_TIMER = "prelude-microstructure.timer"
MICROSTRUCTURE_UNITS = ("prelude-microstructure.service", MICROSTRUCTURE_TIMER)
ORIGINAL_TIMERS = tuple(name for name in TIMER_UNITS if name != MICROSTRUCTURE_TIMER)
ORIGINAL_UNITS = tuple(name for name in ALL_UNITS if name not in MICROSTRUCTURE_UNITS)


def _stateful_fixture(env: dict[str, str], *, omit_collector: bool = True) -> Path:
    state_root = Path(env["PRELUDE_INSTALL_CRON_ROOT"]) / "timer-state"
    state_root.mkdir()
    env["FAKE_SYSTEMCTL_STATE_ROOT"] = str(state_root)
    for timer in TIMER_UNITS:
        if omit_collector and timer == MICROSTRUCTURE_TIMER:
            continue
        (state_root / f"{timer}.enabled").write_text("enabled\n")
        (state_root / f"{timer}.active").write_text("active\n")
    if omit_collector:
        for unit in MICROSTRUCTURE_UNITS:
            (Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / unit).unlink()
    return state_root


def _unit_identities(env: dict[str, str], names=ORIGINAL_UNITS) -> dict:
    root = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    return {
        name: ((root / name).stat().st_ino, (root / name).stat().st_mode, (root / name).read_bytes())
        for name in names
    }


def _mutations(env: dict[str, str]) -> list[str]:
    return [
        line for line in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text().splitlines()
        if line.split(":", 1)[0] in {"enable", "disable", "start", "stop", "restart"}
    ]


def test_collector_service_is_bounded_public_record_only_and_independent() -> None:
    unit = _unit(DEPLOY / "prelude-microstructure.service")
    section = unit["Service"]
    assert unit["Unit"]["After"].split() == ["network-online.target"]
    assert "Before" not in unit["Unit"]
    assert "Requires" not in unit["Unit"]
    assert "BindsTo" not in unit["Unit"]
    command = shlex.split(section["ExecStart"])
    assert command[:3] == [
        "/home/soccz/22tb/prelude/venv/bin/python", "-B",
        "/home/soccz/22tb/prelude/scripts/capture_recommend_microstructure.py",
    ]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--orderbook-depth": "1",
        "--max-wait-seconds": "2400",
        "--required-warmup-seconds": "600",
        "--max-raw-bytes-per-channel": "536870912",
    }
    assert "PRELUDE_FORBID_TELEGRAM=1" in section["Environment"].split()
    assert "EnvironmentFile" not in section
    for key, expected in {
        "Restart": "no", "Nice": "15", "IOSchedulingClass": "best-effort",
        "IOSchedulingPriority": "7", "CPUQuota": "50%", "MemoryMax": "2G",
        "TimeoutStartSec": "2700", "TimeoutStopSec": "90", "KillMode": "control-group",
    }.items():
        assert section[key] == expected
    assert _unit(DEPLOY / MICROSTRUCTURE_TIMER)["Timer"]["RandomizedDelaySec"] == "0"
    assert _unit(DEPLOY / MICROSTRUCTURE_TIMER)["Timer"]["AccuracySec"] == "5s"
    for name in ORIGINAL_UNITS:
        assert "prelude-microstructure" not in (DEPLOY / name).read_text()


@pytest.mark.parametrize("enabled,active", [("enabled", "active"), ("disabled", "inactive"), ("disabled", "active"), ("enabled", "inactive")])
def test_add_only_preserves_original_eight_files_and_states(tmp_path, enabled, active):
    env = _fake_install_env(tmp_path)
    state_root = _stateful_fixture(env)
    for index, timer in enumerate(ORIGINAL_TIMERS):
        if index % 2:
            (state_root / f"{timer}.enabled").write_text(f"{enabled}\n")
            (state_root / f"{timer}.active").write_text(f"{active}\n")
    units_before = _unit_identities(env)
    states_before = {p.name: p.read_bytes() for p in state_root.iterdir()}
    result = _run_installer(env, check_only=False, add_microstructure=True)
    assert result.returncode == 0, result.stderr
    assert _unit_identities(env) == units_before
    for name, payload in states_before.items():
        assert (state_root / name).read_bytes() == payload
    assert _mutations(env) == [f"enable:{MICROSTRUCTURE_TIMER}", f"restart:{MICROSTRUCTURE_TIMER}"]
    for unit in MICROSTRUCTURE_UNITS:
        assert (Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / unit).read_bytes() == (DEPLOY / unit).read_bytes()


@pytest.mark.parametrize("failure,occurrence", [
    ("daemon-reload", 1), (f"enable:{MICROSTRUCTURE_TIMER}", 1),
    (f"restart:{MICROSTRUCTURE_TIMER}", 1),
    ("show:prelude-microstructure.service", 1),
    (f"is-enabled:{MICROSTRUCTURE_TIMER}", 2),
    (f"is-active:{MICROSTRUCTURE_TIMER}", 1),
    ("list-timers:--no-pager", 1),
])
def test_add_only_failure_rolls_back_only_collector(tmp_path, failure, occurrence):
    env = _fake_install_env(tmp_path)
    state_root = _stateful_fixture(env)
    (state_root / "prelude-distribution.timer.enabled").write_text("disabled\n")
    (state_root / "prelude-distribution.timer.active").write_text("inactive\n")
    before = _unit_identities(env)
    states_before = {p.name: p.read_bytes() for p in state_root.iterdir()}
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = failure
    env["FAKE_SYSTEMCTL_FAIL_OCCURRENCE"] = str(occurrence)
    result = _run_installer(env, check_only=False, add_microstructure=True)
    assert result.returncode != 0
    assert "Previous systemd configuration restored." in result.stderr
    assert _unit_identities(env) == before
    assert all(line.endswith(f":{MICROSTRUCTURE_TIMER}") for line in _mutations(env))
    for name, payload in states_before.items():
        assert (state_root / name).read_bytes() == payload
    unit_root = Path(env["PRELUDE_INSTALL_UNIT_DIR"])
    assert not any((unit_root / name).exists() for name in MICROSTRUCTURE_UNITS)
    assert not list(unit_root.glob(".prelude-install.*"))
    assert not list(unit_root.glob(".prelude-backup.*"))


@pytest.mark.parametrize("damage", ["changed", "missing", "symlink", "mode"])
def test_add_only_refuses_original_unit_repairs_without_mutation(tmp_path, damage):
    env = _fake_install_env(tmp_path)
    _stateful_fixture(env)
    path = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / "prelude-close.service"
    if damage == "changed":
        path.write_text("[Unit]\nDescription=old\n")
    elif damage == "missing":
        path.unlink()
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(DEPLOY / path.name)
    else:
        path.chmod(0o600)
    before = _unit_identities(env, tuple(name for name in ORIGINAL_UNITS if name != path.name))
    result = _run_installer(env, check_only=False, add_microstructure=True)
    assert result.returncode != 0
    assert not _mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert _unit_identities(env, tuple(name for name in ORIGINAL_UNITS if name != path.name)) == before
    assert not (path.parent / MICROSTRUCTURE_TIMER).exists()


@pytest.mark.parametrize("add_only", [False, True])
def test_identical_active_install_never_rearms_existing_timers(tmp_path, add_only):
    env = _fake_install_env(tmp_path)
    _stateful_fixture(env, omit_collector=False)
    before = _unit_identities(env, ALL_UNITS)
    result = _run_installer(env, check_only=False, add_microstructure=add_only)
    assert result.returncode == 0, result.stderr
    assert not _mutations(env)
    assert _unit_identities(env, ALL_UNITS) == before


@pytest.mark.parametrize("changed_unit", ["prelude-close.service", "prelude-close.timer"])
def test_full_install_rearms_only_changed_timer_pair(tmp_path, changed_unit):
    env = _fake_install_env(tmp_path)
    _stateful_fixture(env, omit_collector=False)
    path = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / changed_unit
    path.write_text(path.read_text() + "\n# old definition\n")
    untouched = tuple(name for name in ALL_UNITS if name != changed_unit)
    before = _unit_identities(env, untouched)
    result = _run_installer(env, check_only=False)
    assert result.returncode == 0, result.stderr
    assert _mutations(env) == ["enable:prelude-close.timer", "restart:prelude-close.timer"]
    assert _unit_identities(env, untouched) == before


def test_changed_collector_rollback_restores_only_its_files_and_timer_state(tmp_path):
    env = _fake_install_env(tmp_path)
    state_root = _stateful_fixture(env, omit_collector=False)
    path = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / MICROSTRUCTURE_UNITS[0]
    old = b"[Unit]\nDescription=old capture unit\n"
    path.write_bytes(old)
    path.chmod(0o640)
    (state_root / f"{MICROSTRUCTURE_TIMER}.enabled").write_text("disabled\n")
    (state_root / f"{MICROSTRUCTURE_TIMER}.active").write_text("inactive\n")
    before = _unit_identities(env)
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = f"restart:{MICROSTRUCTURE_TIMER}"
    result = _run_installer(env, check_only=False, add_microstructure=True)
    assert result.returncode != 0
    assert "Previous systemd configuration restored." in result.stderr
    assert path.read_bytes() == old
    assert path.stat().st_mode & 0o777 == 0o640
    assert _unit_identities(env) == before
    assert all(line.endswith(f":{MICROSTRUCTURE_TIMER}") for line in _mutations(env))
    assert (state_root / f"{MICROSTRUCTURE_TIMER}.enabled").read_text() == "disabled\n"
    assert (state_root / f"{MICROSTRUCTURE_TIMER}.active").read_text() == "inactive\n"


def test_nine_timer_check_only_has_no_file_or_runtime_mutations(tmp_path):
    env = _fake_install_env(tmp_path)
    state_root = _stateful_fixture(env, omit_collector=False)
    before = _unit_identities(env, ALL_UNITS)
    states_before = {p.name: p.read_bytes() for p in state_root.iterdir()}
    result = _run_installer(env)
    assert result.returncode == 0, result.stderr
    assert "prelude systemd preflight: OK" in result.stdout
    assert _unit_identities(env, ALL_UNITS) == before
    assert {p.name: p.read_bytes() for p in state_root.iterdir()} == states_before
    assert not _mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert not list(Path(env["PRELUDE_INSTALL_UNIT_DIR"]).glob(".prelude-*"))


def test_add_only_preserves_disabled_but_global_check_reports_it(tmp_path):
    env = _fake_install_env(tmp_path)
    state_root = _stateful_fixture(env)
    (state_root / "prelude-distribution.timer.enabled").write_text("disabled\n")
    first = _run_installer(env, check_only=False, add_microstructure=True)
    assert first.returncode == 0, first.stderr
    second = _run_installer(env)
    assert second.returncode != 0
    assert "timer is not enabled: prelude-distribution.timer" in second.stderr


@pytest.mark.parametrize("relative", [
    "scripts/capture_recommend_microstructure.py",
    "signals/recommend_microstructure.py",
    "signals/recommend_microstructure_trial.py",
    "scripts/evaluate_recommend_microstructure_trial.py",
    "signals/recommend_trade_shortlist.py",
    "signals/recommend_trade_shortlist_trial.py",
    "signals/recommend_trade_shortlist_eval.py",
    "scripts/evaluate_recommend_trade_shortlist_trial.py",
    "ops/recommend_microstructure_status.py",
    "ops/recommend_trade_shortlist_status.py",
])
@pytest.mark.parametrize("damage", ["missing", "symlink", "directory"])
def test_installer_requires_safe_microstructure_wrapper_and_trial_runtimes(tmp_path, relative, damage):
    env = _fake_install_env(tmp_path)
    repo = _fixture_repo_for_installer(tmp_path, env)
    path = repo / relative
    path.unlink()
    if damage == "symlink":
        path.symlink_to(ROOT / relative)
    elif damage == "directory":
        path.mkdir()
    before = _unit_identities(env, ALL_UNITS)
    result = _run_installer(env, check_only=False, add_microstructure=True)
    assert result.returncode != 0
    assert f"microstructure runtime is missing or unsafe: {relative}" in result.stderr
    assert _unit_identities(env, ALL_UNITS) == before
    assert not Path(env["FAKE_SYSTEMCTL_LOG"]).exists()
