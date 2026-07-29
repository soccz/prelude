from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOOK = PROJECT_ROOT / "deploy" / "git-hooks" / "pre-push"
ZERO_SHA = "0" * 40


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        check=check,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        timeout=20,
    )


def _fake_repo(tmp_path: Path) -> tuple[Path, str, dict[str, str], Path]:
    repo = tmp_path / "repo"
    hook = repo / "deploy" / "git-hooks" / "pre-push"
    python = repo / "venv" / "bin" / "python"
    ruff = repo / "fake-ruff"
    hook.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    shutil.copy2(HOOK, hook)
    (repo / "guarded.py").write_text("VALUE = 1\n", encoding="utf-8")
    python.write_text(
        "#!/bin/sh\n"
        "printf 'pytest:%s\\n' \"$*\" >> \"$PREPUSH_CALLS\"\n"
        "exit \"${FAKE_PYTEST_RC:-0}\"\n",
        encoding="utf-8",
    )
    ruff.write_text(
        "#!/bin/sh\n"
        "printf 'ruff:%s\\n' \"$*\" >> \"$PREPUSH_CALLS\"\n"
        "exit \"${FAKE_RUFF_RC:-0}\"\n",
        encoding="utf-8",
    )
    for executable in (hook, python, ruff):
        executable.chmod(0o755)

    _run(["git", "init", "-q"], cwd=repo, check=True)
    _run(["git", "config", "user.name", "Prelude Test"], cwd=repo, check=True)
    _run(
        ["git", "config", "user.email", "prelude-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    _run(["git", "add", "."], cwd=repo, check=True)
    _run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    head = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
    ).stdout.strip()
    calls = tmp_path / "prepush-calls.log"
    env = {
        **os.environ,
        "PRELUDE_RUFF_BIN": str(ruff),
        "PREPUSH_CALLS": str(calls),
    }
    return repo, head, env, calls


def _push_update(head: str) -> str:
    return f"refs/heads/main {head} refs/heads/main {ZERO_SHA}\n"


def test_pre_push_hook_is_executable_and_has_both_gates() -> None:
    text = HOOK.read_text(encoding="utf-8")

    assert os.access(HOOK, os.X_OK)
    assert "git diff --name-only --diff-filter=ACMR" in text
    assert '"$RUFF_BIN" check "${python_files[@]}"' in text
    assert "venv/bin/python -m pytest -q -p no:cacheprovider" in text
    assert text.index('"$RUFF_BIN" check') < text.index(
        "venv/bin/python -m pytest"
    )


def test_pre_push_hook_has_only_explicit_emergency_bypass() -> None:
    proc = subprocess.run(
        [str(HOOK), "origin", "unused"],
        input="",
        capture_output=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PRELUDE_SKIP_PREPUSH": "1"},
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0
    assert "PRELUDE_SKIP_PREPUSH=1" in proc.stderr


def test_pre_push_hook_parses_as_bash() -> None:
    proc = subprocess.run(
        ["/bin/bash", "-n", str(HOOK)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr


def test_pre_push_hook_runs_lint_then_full_suite_on_pushed_commit(
    tmp_path,
) -> None:
    repo, head, env, calls = _fake_repo(tmp_path)

    proc = _run(
        [str(repo / "deploy/git-hooks/pre-push"), "origin", "unused"],
        cwd=repo,
        env=env,
        input_text=_push_update(head),
    )

    assert proc.returncode == 0, proc.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "ruff:check guarded.py",
        "pytest:-m pytest -q -p no:cacheprovider",
    ]


@pytest.mark.parametrize(
    ("env_key", "expected"),
    (
        ("FAKE_RUFF_RC", "Ruff 실패"),
        ("FAKE_PYTEST_RC", "전수 스위트 실패"),
    ),
)
def test_pre_push_hook_blocks_failed_gate(
    tmp_path,
    env_key,
    expected,
) -> None:
    repo, head, env, _calls = _fake_repo(tmp_path)
    env[env_key] = "9"

    proc = _run(
        [str(repo / "deploy/git-hooks/pre-push"), "origin", "unused"],
        cwd=repo,
        env=env,
        input_text=_push_update(head),
    )

    assert proc.returncode != 0
    assert expected in proc.stderr


def test_pre_push_hook_blocks_dirty_tree_instead_of_testing_other_bytes(
    tmp_path,
) -> None:
    repo, head, env, calls = _fake_repo(tmp_path)
    (repo / "guarded.py").write_text("VALUE = broken\n", encoding="utf-8")

    proc = _run(
        [str(repo / "deploy/git-hooks/pre-push"), "origin", "unused"],
        cwd=repo,
        env=env,
        input_text=_push_update(head),
    )

    assert proc.returncode != 0
    assert "worktree/index/untracked 변경 존재" in proc.stderr
    assert not calls.exists()


def test_pre_push_hook_blocks_unavailable_remote_base(tmp_path) -> None:
    repo, head, env, calls = _fake_repo(tmp_path)
    unavailable = "f" * 40
    update = (
        f"refs/heads/main {head} refs/heads/main {unavailable}\n"
    )

    proc = _run(
        [str(repo / "deploy/git-hooks/pre-push"), "origin", "unused"],
        cwd=repo,
        env=env,
        input_text=update,
    )

    assert proc.returncode != 0
    assert "remote base 객체가 로컬에 없어" in proc.stderr
    assert not calls.exists()
