"""One-service installer tests. All systemd/cron paths are isolated fixtures."""

from pathlib import Path
import subprocess

import pytest

import test_systemd_deploy_contract as fixtures

TARGET = "prelude-selftest.service"
UNTOUCHED = tuple(name for name in fixtures.ALL_UNITS if name != TARGET)


def _fixture(tmp_path, *, state="failed"):
    env = fixtures._fake_install_env(tmp_path)
    state_root = fixtures._stateful_fixture(env, omit_collector=False)
    command = Path(env["PRELUDE_INSTALL_SYSTEMCTL"])
    text = command.read_text()
    text = text.replace(
        'case "$1" in\n',
        'if [ "$1" = show ] && [ "${3:-}" = --property=ActiveState ]; then\n'
        '  printf "%s\\n" "$FAKE_SELFTEST_STATE"\n'
        "  exit 0\n"
        'fi\ncase "$1" in\n',
        1,
    )
    command.write_text(text)
    env["FAKE_SELFTEST_STATE"] = state
    env["PRELUDE_INSTALL_FIXTURE"] = "1"
    installed = Path(env["PRELUDE_INSTALL_UNIT_DIR"]) / TARGET
    installed.write_text(installed.read_text() + "\n# previous selftest definition\n")
    return env, state_root, installed


def _run(env):
    return subprocess.run(
        ["bash", str(fixtures.INSTALLER), "--update-selftest"],
        cwd=fixtures.ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("state", ["inactive", "failed"])
def test_only_selftest_file_changes_with_no_timer_commands(tmp_path, state):
    env, state_root, installed = _fixture(tmp_path, state=state)
    # Disabled timers are an intentional user state, not repair permission.
    (state_root / "prelude-preopen.timer.enabled").write_text("disabled\n")
    (state_root / "prelude-preopen.timer.active").write_text("inactive\n")
    before = fixtures._unit_identities(env, UNTOUCHED)
    states = {p.name: p.read_bytes() for p in state_root.iterdir()}
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert installed.read_bytes() == (fixtures.DEPLOY / TARGET).read_bytes()
    assert fixtures._unit_identities(env, UNTOUCHED) == before
    assert {p.name: p.read_bytes() for p in state_root.iterdir()} == states
    assert not fixtures._mutations(env)
    calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert calls.count("daemon-reload\n") == 1
    assert not list(installed.parent.glob(".prelude-*"))


def test_identical_selftest_keeps_file_inode(tmp_path):
    env, _, installed = _fixture(tmp_path)
    installed.write_bytes((fixtures.DEPLOY / TARGET).read_bytes())
    before = fixtures._unit_identities(env, fixtures.ALL_UNITS)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert fixtures._unit_identities(env, fixtures.ALL_UNITS) == before
    assert not fixtures._mutations(env)


@pytest.mark.parametrize(
    "state", ["active", "activating", "deactivating", "reloading", "", "unknown"]
)
def test_running_or_unknown_selftest_refused_without_reload(tmp_path, state):
    env, _, _ = _fixture(tmp_path, state=state)
    before = fixtures._unit_identities(env, fixtures.ALL_UNITS)
    result = _run(env)
    assert result.returncode != 0
    assert "must be inactive or failed" in result.stderr
    assert fixtures._unit_identities(env, fixtures.ALL_UNITS) == before
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert not fixtures._mutations(env)


@pytest.mark.parametrize("damage", ["missing", "symlink", "mode", "changed_other"])
def test_unsafe_target_or_unrelated_drift_refused(tmp_path, damage):
    env, _, installed = _fixture(tmp_path)
    if damage == "missing":
        installed.unlink()
    elif damage == "symlink":
        installed.unlink()
        installed.symlink_to(fixtures.DEPLOY / TARGET)
    elif damage == "mode":
        installed.chmod(0o600)
    else:
        path = installed.parent / "prelude-close.service"
        path.write_text(path.read_text() + "\n# unrelated drift\n")
    result = _run(env)
    assert result.returncode != 0
    assert not fixtures._mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()


@pytest.mark.parametrize(
    "failure,occurrence",
    [
        ("daemon-reload", 1),
        ("show:prelude-selftest.service", 1),
        ("list-timers:--no-pager", 1),
        ("is-active:prelude-distribution.timer", 2),
    ],
)
def test_failure_restores_target_without_touching_timer_state(
    tmp_path, failure, occurrence
):
    env, state_root, installed = _fixture(tmp_path)
    old = installed.read_bytes()
    before = fixtures._unit_identities(env, UNTOUCHED)
    states = {p.name: p.read_bytes() for p in state_root.iterdir()}
    env["FAKE_SYSTEMCTL_FAIL_MATCH"] = failure
    env["FAKE_SYSTEMCTL_FAIL_OCCURRENCE"] = str(occurrence)
    result = _run(env)
    assert result.returncode != 0
    assert "Previous systemd configuration restored" in result.stderr
    assert installed.read_bytes() == old
    assert fixtures._unit_identities(env, UNTOUCHED) == before
    assert {p.name: p.read_bytes() for p in state_root.iterdir()} == states
    assert not fixtures._mutations(env)
    assert not list(installed.parent.glob(".prelude-*"))
