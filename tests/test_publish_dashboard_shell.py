from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest


EXPECTED_ASSETS = {
    "summary.json",
    "history.json",
    "accuracy.json",
    "idea_validation.json",
    "findings.json",
}


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _seed_site_remote(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    remote = tmp_path / "site-remote.git"
    seed = tmp_path / "site-seed"
    site = tmp_path / "shared-site"
    remote.mkdir()
    seed.mkdir()
    _git(remote, "init", "--bare", "--initial-branch=main")
    _git(seed, "init", "--initial-branch=main")
    _git(seed, "config", "user.name", "prelude-test")
    _git(seed, "config", "user.email", "prelude-test@example.invalid")

    data = seed / "projects" / "prelude" / "dashboard" / "data"
    data.mkdir(parents=True)
    for name in EXPECTED_ASSETS:
        (data / name).write_text(f"remote-old:{name}\n", encoding="utf-8")
    other = seed / "projects" / "other"
    other.mkdir(parents=True)
    (other / "wip.txt").write_text("remote-base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "seed site")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")
    _git(tmp_path, "clone", str(remote), str(site))
    _git(site, "config", "user.name", "prelude-test")
    _git(site, "config", "user.email", "prelude-test@example.invalid")
    return remote, site


def _write_fake_python(fake_bin: Path) -> None:
    fake_python = fake_bin / "python"
    fake_python.write_text(
        r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$PYTHON_CALL_LOG"
if [ "$1" = "-c" ] && [[ "$2" == *resolve_dashboard_passphrase* ]] &&
   [ -z "${PRELUDE_DASHBOARD_PIN:-}" ]; then
    exit 3
fi
if [ "$*" = "$FAIL_EXACT" ]; then
    exit "$FAIL_RC"
fi
if [ -n "${FAIL_PREFIX:-}" ] && [[ "$*" == "$FAIL_PREFIX"* ]]; then
    exit "$FAIL_RC"
fi
if [ "$1" = "scripts/build_dashboard.py" ]; then
    out="$3"
    mkdir -p "$out"
    for name in summary.json history.json accuracy.json idea_validation.json; do
        printf '{"encrypted":true,"asset":"%s"}\n' "$name" > "$out/$name"
    done
    if [ "${WRITE_PLAINTEXT_ASSET:-0}" = "1" ]; then
        printf 'generated:summary.json\n' > "$out/summary.json"
    fi
fi
if [ "$1" = "scripts/build_findings_dashboard.py" ]; then
    out="$3"
    mkdir -p "$out"
    printf '{"encrypted":true,"asset":"findings.json"}\n' > "$out/findings.json"
    if [ "${ADD_UNEXPECTED_ASSET:-0}" = "1" ]; then
        printf 'unexpected\n' > "$out/extra.json"
    fi
fi
if [ "$1" = "scripts/validate_dashboard_assets.py" ]; then
    out="$3"
    for name in summary.json history.json accuracy.json idea_validation.json findings.json; do
        grep -q '"encrypted":true' "$out/$name" || exit 66
    done
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)


def _write_racing_git_wrapper(fake_bin: Path) -> None:
    wrapper = fake_bin / "git"
    wrapper.write_text(
        r"""#!/usr/bin/env bash
if [ -n "${RACE_WORKTREE:-}" ] &&
   [[ " $* " == *" push origin HEAD:refs/heads/main "* ]] &&
   [ ! -e "$RACE_INJECT_MARKER" ]; then
    : > "$RACE_INJECT_MARKER"
    data="$RACE_WORKTREE/projects/prelude/dashboard/data"
    printf 'concurrent remote extra\n' > "$data/extra.json"
    /usr/bin/git -C "$RACE_WORKTREE" add -- \
        projects/prelude/dashboard/data/extra.json
    /usr/bin/git -C "$RACE_WORKTREE" commit -m "concurrent dashboard update"
    /usr/bin/git -C "$RACE_WORKTREE" push origin main
fi
exec /usr/bin/git "$@"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def _seed_remote_with_generated_assets(tmp_path: Path, remote: Path) -> None:
    writer = tmp_path / "generated-seed"
    _git(tmp_path, "clone", str(remote), str(writer))
    _git(writer, "config", "user.name", "prelude-generated-test")
    _git(writer, "config", "user.email", "generated@example.invalid")
    data = writer / "projects" / "prelude" / "dashboard" / "data"
    for name in EXPECTED_ASSETS:
        (data / name).write_text(
            f'{{"encrypted":true,"asset":"{name}"}}\n',
            encoding="utf-8",
        )
    _git(writer, "add", "-A", "--", "projects/prelude/dashboard/data")
    _git(writer, "commit", "-m", "seed generated dashboard")
    _git(writer, "push", "origin", "main")


def _install_publish_runner(
    repo: Path,
    *,
    site: Path,
    scratch: Path,
) -> Path:
    scripts = repo / "scripts"
    deploy = repo / "deploy"
    scripts.mkdir(parents=True)
    deploy.mkdir()

    source = Path("scripts/publish_dashboard.sh").read_text(encoding="utf-8")
    source = source.replace(
        'SITE_ROOT="/home/soccz/22tb/soccz.github.io"',
        f'SITE_ROOT="{site}"',
    ).replace(
        'TMP_ROOT="/home/soccz/22tb/tmp"',
        f'TMP_ROOT="{scratch}"',
    )
    script = scripts / "publish_dashboard.sh"
    script.write_text(source, encoding="utf-8")
    script.chmod(0o755)
    shutil.copy2("deploy/lock_exec.py", deploy / "lock_exec.py")
    return script


def _commit_remote_path_attack(
    tmp_path: Path,
    remote: Path,
    relative: str,
    kind: str,
) -> None:
    attacker = tmp_path / "remote-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _git(attacker, "config", "user.name", "prelude-attacker-test")
    _git(attacker, "config", "user.email", "attacker@example.invalid")
    _git(attacker, "rm", "-r", "--", relative)

    target = attacker / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        outside = tmp_path / "outside-clone"
        full_data = Path("projects/prelude/dashboard/data")
        remainder = full_data.relative_to(Path(relative))
        outside_data = outside / remainder
        outside_data.mkdir(parents=True)
        (outside_data / "do-not-delete.txt").write_text(
            "outside sentinel\n",
            encoding="utf-8",
        )
        target.symlink_to(outside, target_is_directory=True)
    elif kind == "file":
        target.write_text("remote non-directory attack\n", encoding="utf-8")
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported attack kind: {kind}")

    _git(attacker, "add", "-A", "--", relative)
    _git(attacker, "commit", "-m", f"attack {relative} as {kind}")
    _git(attacker, "push", "origin", "main")


def _run_publish(
    tmp_path: Path,
    *,
    pin: str | None = "test-dashboard-secret-2026",
    fail_exact: str = "never",
    fail_prefix: str = "",
    fail_rc: int = 9,
    add_unexpected_asset: bool = False,
    write_plaintext_asset: bool = False,
    under_systemd: bool = True,
    remote_attack: tuple[str, str] | None = None,
    remote_url_override: str | None = None,
    hold_lock: bool = False,
    concurrent_remote_extra: bool = False,
    remote_already_generated: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    remote, site = _seed_site_remote(tmp_path)
    if remote_already_generated:
        _seed_remote_with_generated_assets(tmp_path, remote)
    if remote_attack is not None:
        _commit_remote_path_attack(
            tmp_path,
            remote,
            remote_attack[0],
            remote_attack[1],
        )
    if remote_url_override is not None:
        _git(site, "config", "remote.origin.url", remote_url_override)

    repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    scratch = tmp_path / "scratch"
    fake_bin.mkdir()

    script = _install_publish_runner(
        repo,
        site=site,
        scratch=scratch,
    )
    _write_fake_python(fake_bin)
    race_worktree = ""
    if concurrent_remote_extra:
        _write_racing_git_wrapper(fake_bin)
        race_path = tmp_path / "race-writer"
        _git(tmp_path, "clone", str(remote), str(race_path))
        _git(race_path, "config", "user.name", "prelude-race-test")
        _git(race_path, "config", "user.email", "race@example.invalid")
        race_worktree = str(race_path)

    python_log = tmp_path / "python.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHON_CALL_LOG": str(python_log),
            "FAIL_EXACT": fail_exact,
            "FAIL_PREFIX": fail_prefix,
            "FAIL_RC": str(fail_rc),
            "ADD_UNEXPECTED_ASSET": "1" if add_unexpected_asset else "0",
            "WRITE_PLAINTEXT_ASSET": (
                "1" if write_plaintext_asset else "0"
            ),
            "GIT_AUTHOR_NAME": "prelude-test",
            "GIT_AUTHOR_EMAIL": "prelude-test@example.invalid",
            "GIT_COMMITTER_NAME": "prelude-test",
            "GIT_COMMITTER_EMAIL": "prelude-test@example.invalid",
            "RACE_WORKTREE": race_worktree,
            "RACE_INJECT_MARKER": str(tmp_path / "race-injected"),
        }
    )
    if pin is None:
        env.pop("PRELUDE_DASHBOARD_PIN", None)
    else:
        env["PRELUDE_DASHBOARD_PIN"] = pin
    if under_systemd:
        env["INVOCATION_ID"] = "test-systemd-run"
    else:
        env.pop("INVOCATION_ID", None)

    lock_handle = None
    if hold_lock:
        output_dir = repo / "output"
        output_dir.mkdir()
        lock_handle = (output_dir / ".publish_dashboard.lock").open("w")
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if lock_handle is not None:
            lock_handle.close()
    return result, remote, site, scratch


def _shared_state(site: Path) -> tuple[str, str, dict[str, bytes]]:
    head = _git(site, "rev-parse", "HEAD")
    status = _git(site, "status", "--porcelain=v1")
    files = {
        str(path.relative_to(site)): path.read_bytes()
        for path in site.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    return head, status, files


def test_publish_uses_isolated_clone_and_preserves_shared_wip(tmp_path):
    # Create staged and unstaged user work, including a dashboard-data edit.
    remote, site = _seed_site_remote(tmp_path / "seed-only")
    (site / "projects" / "other" / "wip.txt").write_text(
        "user staged work\n",
        encoding="utf-8",
    )
    (site / "projects" / "prelude" / "dashboard" / "data" / "summary.json").write_text(
        "user dashboard work\n",
        encoding="utf-8",
    )
    _git(
        site,
        "add",
        "projects/other/wip.txt",
        "projects/prelude/dashboard/data/summary.json",
    )
    (site / "user-untracked.txt").write_text("do not touch\n", encoding="utf-8")
    before = _shared_state(site)

    # Reuse the prepared shared repo while keeping the helper's isolated runner.
    runner_root = tmp_path / "run"
    runner_root.mkdir()
    repo = runner_root / "repo"
    fake_bin = runner_root / "bin"
    scratch = runner_root / "scratch"
    fake_bin.mkdir()
    script = _install_publish_runner(
        repo,
        site=site,
        scratch=scratch,
    )
    _write_fake_python(fake_bin)
    python_log = runner_root / "python.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHON_CALL_LOG": str(python_log),
            "FAIL_EXACT": "never",
            "FAIL_PREFIX": "",
            "FAIL_RC": "9",
            "WRITE_PLAINTEXT_ASSET": "0",
            "PRELUDE_DASHBOARD_PIN": "test-dashboard-secret-2026",
            "INVOCATION_ID": "test-systemd-run",
            "GIT_AUTHOR_NAME": "prelude-test",
            "GIT_AUTHOR_EMAIL": "prelude-test@example.invalid",
            "GIT_COMMITTER_NAME": "prelude-test",
            "GIT_COMMITTER_EMAIL": "prelude-test@example.invalid",
        }
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _shared_state(site) == before
    for name in EXPECTED_ASSETS:
        published = _git(
            remote,
            "show",
            f"main:projects/prelude/dashboard/data/{name}",
        )
        assert '"encrypted":true' in published
        assert f'"asset":"{name}"' in published
    assert not scratch.exists() or not any(scratch.iterdir())
    assert "committed + pushed" in (
        repo / "output" / "cron_publish.log"
    ).read_text(encoding="utf-8")


def test_push_retry_reapplies_exact_generation_after_concurrent_remote_update(
    tmp_path,
):
    result, remote, _site, scratch = _run_publish(
        tmp_path,
        concurrent_remote_extra=True,
        remote_already_generated=True,
    )

    assert result.returncode == 0, result.stderr
    tree = set(
        _git(
            remote,
            "ls-tree",
            "--name-only",
            "main:projects/prelude/dashboard/data",
        ).splitlines()
    )
    assert tree == EXPECTED_ASSETS
    assert not scratch.exists() or not any(scratch.iterdir())


def test_build_failure_never_pushes_and_cleans_isolated_tree(tmp_path):
    result, remote, site, scratch = _run_publish(
        tmp_path,
        fail_prefix="scripts/build_findings_dashboard.py --out-dir ",
        fail_rc=8,
    )

    assert result.returncode == 8
    assert not scratch.exists() or not any(scratch.iterdir())
    assert _git(remote, "rev-parse", "main") == _git(site, "rev-parse", "HEAD")


def test_missing_pin_fails_before_clone_or_push(tmp_path):
    result, remote, site, scratch = _run_publish(tmp_path, pin=None)

    assert result.returncode == 2
    assert not scratch.exists()
    assert _git(remote, "rev-parse", "main") == _git(site, "rev-parse", "HEAD")
    log = (tmp_path / "repo" / "output" / "cron_publish.log").read_text(
        encoding="utf-8"
    )
    assert "PRELUDE_DASHBOARD_PIN missing or invalid" in log


def test_plaintext_generated_asset_fails_validation_before_push(tmp_path):
    result, remote, site, scratch = _run_publish(
        tmp_path,
        write_plaintext_asset=True,
    )

    assert result.returncode == 66
    assert not scratch.exists() or not any(scratch.iterdir())
    assert _git(remote, "rev-parse", "main") == _git(
        site,
        "rev-parse",
        "HEAD",
    )
    log = (tmp_path / "repo" / "output" / "cron_publish.log").read_text(
        encoding="utf-8"
    )
    assert "[fail] validate:" in log


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        ("projects", "symlink"),
        ("projects/prelude", "file"),
        ("projects/prelude/dashboard", "symlink"),
        ("projects/prelude/dashboard/data", "symlink"),
    ],
)
def test_remote_path_escape_fails_without_touching_outside_clone(
    tmp_path,
    relative,
    kind,
):
    result, remote, _site, scratch = _run_publish(
        tmp_path,
        remote_attack=(relative, kind),
    )

    assert result.returncode == 2
    assert _git(remote, "rev-list", "--count", "main") == "2\n"
    sentinel = tmp_path / "outside-clone"
    if sentinel.exists():
        sentinels = list(sentinel.rglob("do-not-delete.txt"))
        assert len(sentinels) == 1
        assert sentinels[0].read_text(encoding="utf-8") == "outside sentinel\n"
    assert not scratch.exists() or not any(scratch.iterdir())
    log = (tmp_path / "repo" / "output" / "cron_publish.log").read_text(
        encoding="utf-8"
    )
    assert "symlink or non-directory" in log


def test_leading_dash_remote_is_rejected_before_git_repository_lookup(tmp_path):
    result, remote, site, scratch = _run_publish(
        tmp_path,
        remote_url_override="--upload-pack=/bin/false",
    )

    assert result.returncode == 2
    assert not scratch.exists()
    assert _git(remote, "rev-parse", "main") == _git(site, "rev-parse", "HEAD")
    log = (tmp_path / "repo" / "output" / "cron_publish.log").read_text(
        encoding="utf-8"
    )
    assert "origin remote scheme is not allowed" in log


def test_lock_contention_is_reported_as_retryable_failure(tmp_path):
    result, remote, site, scratch = _run_publish(tmp_path, hold_lock=True)

    assert result.returncode == 75
    assert not scratch.exists()
    assert _git(remote, "rev-parse", "main") == _git(site, "rev-parse", "HEAD")
    log = (tmp_path / "repo" / "output" / "cron_publish.log").read_text(
        encoding="utf-8"
    )
    assert "[fail] lock:" in log
    assert "retry required" in log


def test_publish_source_has_no_shared_worktree_mutation_commands():
    source = Path("scripts/publish_dashboard.sh").read_text(encoding="utf-8")

    assert "git restore" not in source
    assert "git clean" not in source
    assert "--autostash" not in source
    assert 'TMP_ROOT="/home/soccz/22tb/tmp"' in source
    assert "git clone --quiet --single-branch" in source
    assert 'git ls-remote --exit-code --heads -- "$REMOTE_URL"' in source
    assert 'prepare_dashboard_data_directory "rebase"' in source
    assert 'assert_clone_directory "$DATA_REL" "rebase"' in source
    assert "trap cleanup_isolated_tree EXIT" in source
    assert "terminate_publish TERM 143" in source
    assert "EXIT HUP INT TERM" not in source
