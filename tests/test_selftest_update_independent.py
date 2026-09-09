"""Independent failure injection for the one-service update; fixture paths only."""
from pathlib import Path

import pytest

import test_selftest_update_install as update
import test_systemd_deploy_contract as fixtures


def _state_bytes(state_root):
    return {path.name: path.read_bytes() for path in state_root.iterdir()}


@pytest.mark.parametrize("missing", update.UNTOUCHED)
def test_every_other_installed_unit_is_required_before_transaction(tmp_path, missing):
    env, state_root, target = update._fixture(tmp_path)
    original = target.read_bytes()
    (target.parent / missing).unlink()
    states = _state_bytes(state_root)
    result = update._run(env)
    assert result.returncode != 0
    assert "installed unit missing" in result.stderr
    assert target.read_bytes() == original
    assert _state_bytes(state_root) == states
    assert not fixtures._mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert not list(target.parent.glob(".prelude-*"))


@pytest.mark.parametrize("enabled,active", [
    ("enabled", "active"), ("disabled", "inactive"),
    ("disabled", "active"), ("enabled", "inactive"),
])
def test_all_nine_timer_states_preserved_without_activation_commands(tmp_path, enabled, active):
    env, state_root, target = update._fixture(tmp_path)
    for timer in fixtures.TIMER_UNITS:
        (state_root / f"{timer}.enabled").write_text(f"{enabled}\n")
        (state_root / f"{timer}.active").write_text(f"{active}\n")
    before = fixtures._unit_identities(env, update.UNTOUCHED)
    states = _state_bytes(state_root)
    result = update._run(env)
    assert result.returncode == 0, result.stderr
    assert fixtures._unit_identities(env, update.UNTOUCHED) == before
    assert _state_bytes(state_root) == states
    assert target.read_bytes() == (fixtures.DEPLOY / update.TARGET).read_bytes()
    assert not fixtures._mutations(env)


def test_failed_state_query_cannot_hide_behind_inactive_stdout(tmp_path):
    env, state_root, target = update._fixture(tmp_path, state="inactive")
    command = Path(env["PRELUDE_INSTALL_SYSTEMCTL"])
    source = command.read_text()
    old = '  printf "%s\\n" "$FAKE_SELFTEST_STATE"\n  exit 0\n'
    assert old in source
    command.write_text(source.replace(old, old.replace("exit 0", "exit 7"), 1))
    before = fixtures._unit_identities(env, fixtures.ALL_UNITS)
    states = _state_bytes(state_root)
    result = update._run(env)
    assert result.returncode != 0
    assert "cannot determine selftest runtime state" in result.stderr
    assert fixtures._unit_identities(env, fixtures.ALL_UNITS) == before
    assert _state_bytes(state_root) == states
    assert not fixtures._mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert not list(target.parent.glob(".prelude-*"))


def test_staging_validation_failure_leaves_all_installed_files_and_states(tmp_path):
    env, state_root, target = update._fixture(tmp_path)
    env["FAKE_ANALYZE_FAIL_OCCURRENCE"] = "2"
    before = fixtures._unit_identities(env, fixtures.ALL_UNITS)
    states = _state_bytes(state_root)
    result = update._run(env)
    assert result.returncode != 0
    assert "rejected one or more staged units" in result.stderr
    assert fixtures._unit_identities(env, fixtures.ALL_UNITS) == before
    assert _state_bytes(state_root) == states
    assert not fixtures._mutations(env)
    assert "daemon-reload" not in Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert not list(target.parent.glob(".prelude-*"))


@pytest.mark.parametrize("post_state,query_rc", [("active", 0), ("activating", 0), ("inactive", 7)])
def test_post_reload_activation_or_failed_query_rolls_back_without_stopping_runtime(
    tmp_path, post_state, query_rc,
):
    env, state_root, target = update._fixture(tmp_path, state="inactive")
    command = Path(env["PRELUDE_INSTALL_SYSTEMCTL"])
    source = command.read_text()
    old = '  printf "%s\\n" "$FAKE_SELFTEST_STATE"\n  exit 0\n'
    assert old in source
    replacement = (
        '  count=0\n'
        '  if [ -f "$SELFTEST_QUERY_COUNTER" ]; then read -r count < "$SELFTEST_QUERY_COUNTER"; fi\n'
        '  count=$((count + 1))\n'
        '  printf "%s\\n" "$count" > "$SELFTEST_QUERY_COUNTER"\n'
        '  if [ "$count" -gt 1 ]; then\n'
        '    printf "%s\\n" "$POST_SELFTEST_STATE"\n'
        '    exit "$POST_SELFTEST_QUERY_RC"\n'
        '  fi\n'
        + old
    )
    command.write_text(source.replace(old, replacement, 1))
    counter = tmp_path / "selftest-state-query-count"
    env.update(
        SELFTEST_QUERY_COUNTER=str(counter),
        POST_SELFTEST_STATE=post_state,
        POST_SELFTEST_QUERY_RC=str(query_rc),
    )
    original = target.read_bytes()
    before = fixtures._unit_identities(env, update.UNTOUCHED)
    states = _state_bytes(state_root)
    result = update._run(env)
    assert result.returncode != 0
    assert counter.read_text().strip() == "2"
    assert "Previous systemd configuration restored" in result.stderr
    if query_rc:
        assert "cannot determine selftest runtime state" in result.stderr
    else:
        assert "must be inactive or failed" in result.stderr
    assert target.read_bytes() == original
    assert fixtures._unit_identities(env, update.UNTOUCHED) == before
    assert _state_bytes(state_root) == states
    assert not fixtures._mutations(env)
    calls = Path(env["FAKE_SYSTEMCTL_LOG"]).read_text()
    assert calls.count("daemon-reload\n") == 2
    assert not list(target.parent.glob(".prelude-*"))
