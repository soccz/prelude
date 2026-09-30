from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ops.runtime_env import (
    MAX_RUNTIME_ENV_BYTES,
    RuntimeEnvError,
    load_runtime_env,
    parse_runtime_env,
)


ROOT = Path(__file__).resolve().parent.parent


def _private_env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_parse_runtime_env_accepts_only_supported_literal_values(
    tmp_path: Path,
) -> None:
    path = _private_env(
        tmp_path / ".env",
        "\n".join(
            [
                "# comment",
                "export TELEGRAM_BOT_TOKEN=123456:abc_DEF-ghi",
                "TELEGRAM_CHAT_ID='-1001234567890'",
                'PRELUDE_DASHBOARD_PIN="literal $() `ticks`; & value"',
            ]
        )
        + "\n",
    )

    values = parse_runtime_env(
        path,
        required_keys=(
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "PRELUDE_DASHBOARD_PIN",
        ),
    )

    assert values == {
        "PRELUDE_DASHBOARD_PIN": "literal $() `ticks`; & value",
        "TELEGRAM_BOT_TOKEN": "123456:abc_DEF-ghi",
        "TELEGRAM_CHAT_ID": "-1001234567890",
    }


@pytest.mark.parametrize(
    "content",
    [
        "PATH=/home/soccz/22tb/tmp/attacker\n",
        "PYTHONPATH=/home/soccz/22tb/tmp/attacker\n",
        "GIT_SSH_COMMAND=touch-owned\n",
        "TELEGRAM_CHAT_ID=one\nTELEGRAM_CHAT_ID=two\n",
        "TELEGRAM_CHAT_ID=has whitespace\n",
        "TELEGRAM_CHAT_ID='unterminated\n",
    ],
)
def test_parse_runtime_env_rejects_unknown_duplicate_or_ambiguous_syntax(
    tmp_path: Path,
    content: str,
) -> None:
    path = _private_env(tmp_path / ".env", content)

    with pytest.raises(RuntimeEnvError):
        parse_runtime_env(path)


def test_parse_runtime_env_rejects_symlink_and_public_mode(
    tmp_path: Path,
) -> None:
    target = _private_env(tmp_path / "target", "TELEGRAM_CHAT_ID=1\n")
    symlink = tmp_path / ".env"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeEnvError):
        parse_runtime_env(symlink)

    target.chmod(0o640)
    with pytest.raises(RuntimeEnvError):
        parse_runtime_env(target)


def test_parse_runtime_env_rejects_embedded_nul(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"TELEGRAM_CHAT_ID='one\x00two'\n")
    path.chmod(0o600)

    with pytest.raises(RuntimeEnvError, match="control character"):
        parse_runtime_env(path)


def test_parse_runtime_env_rejects_symlinked_direct_parent(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    path = _private_env(
        real_parent / ".env",
        "TELEGRAM_CHAT_ID=1\n",
    )
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeEnvError, match="parent must be a real directory"):
        parse_runtime_env(linked_parent / path.name)


def test_parse_runtime_env_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(
        b"#" + b"x" * MAX_RUNTIME_ENV_BYTES + b"\n"
    )
    path.chmod(0o600)

    with pytest.raises(RuntimeEnvError, match="exceeds"):
        parse_runtime_env(path)


def test_load_runtime_env_does_not_override_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _private_env(tmp_path / ".env", "TELEGRAM_CHAT_ID=file-value\n")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "explicit-value")

    load_runtime_env(path)

    assert os.environ["TELEGRAM_CHAT_ID"] == "explicit-value"


def test_shell_loader_never_executes_env_value(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    env_path = _private_env(
        tmp_path / ".env",
        (
            "TELEGRAM_BOT_TOKEN='$(touch "
            + str(marker)
            + ")'\nTELEGRAM_CHAT_ID=-1001\n"
        ),
    )
    command = (
        "set -euo pipefail; "
        f"cd {ROOT!s}; "
        "source deploy/load_runtime_env.sh; "
        f"load_prelude_runtime_env {env_path!s} {ROOT / 'venv/bin/python'!s}; "
        "test \"$TELEGRAM_BOT_TOKEN\" = '$(touch "
        + str(marker)
        + ")'"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_shell_loader_fails_closed_without_success_sentinel(
    tmp_path: Path,
) -> None:
    env_path = _private_env(
        tmp_path / ".env",
        "PATH=/home/soccz/22tb/tmp/attacker\n",
    )
    command = (
        "set -euo pipefail; "
        f"cd {ROOT!s}; "
        "source deploy/load_runtime_env.sh; "
        f"load_prelude_runtime_env {env_path!s} {ROOT / 'venv/bin/python'!s}"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "validation/export failed" in result.stderr


def test_shell_loader_rejects_valid_records_from_failed_parser(
    tmp_path: Path,
) -> None:
    env_path = _private_env(tmp_path / ".env", "TELEGRAM_CHAT_ID=ignored\n")
    parser = tmp_path / "failed-parser"
    parser.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'TELEGRAM_CHAT_ID\\0forged\\0"
        "__PRELUDE_RUNTIME_ENV_V1_OK__\\0'\n"
        "exit 7\n",
        encoding="utf-8",
    )
    parser.chmod(0o700)
    command = (
        "set -euo pipefail; "
        f"cd {ROOT!s}; "
        "source deploy/load_runtime_env.sh; "
        f"load_prelude_runtime_env {env_path!s} {parser!s}"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "validation/export failed" in result.stderr


def test_production_shell_scripts_never_source_dotenv_as_code() -> None:
    pattern = re.compile(r"(?m)^\s*(?:source|\.)\s+[^\n]*\.env(?:\s|$)")

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("*.sh")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def _run_shell_protocol(
    tmp_path: Path,
    payload: bytes,
    *,
    producer_rc: int = 0,
    finish_before_capture: bool = False,
    expected_value: str = "unchanged",
) -> subprocess.CompletedProcess[str]:
    """Exercise the production loader with synthetic, never live secrets."""
    env_path = _private_env(tmp_path / ".env", "# synthetic protocol test\n")
    parser = tmp_path / "parser"
    parser.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"os.write(1, {payload!r})\n"
        f"sys.exit({producer_rc})\n",
        encoding="utf-8",
    )
    parser.chmod(0o700)
    loader = ROOT / "deploy/load_runtime_env.sh"
    if finish_before_capture:
        source = loader.read_text()
        capture = "    parser_pid=$!"
        assert source.count(capture) == 1
        loader = tmp_path / "loader.sh"
        # Tiny fixture output fits in the pipe: complete the producer first,
        # deterministically reproducing a parent descheduled before capture.
        loader.write_text(source.replace(capture, '    wait "$!" || :\n' + capture))
    env = os.environ.copy()
    env.update(TELEGRAM_CHAT_ID="unchanged", EXPECTED_TEST_VALUE=expected_value)
    return subprocess.run(
        [
            "/bin/bash", "-c",
            'set -euo pipefail; source "$1"; '
            'if load_prelude_runtime_env "$2" "$3"; then '
            'test "$TELEGRAM_CHAT_ID" = "$EXPECTED_TEST_VALUE"; '
            'else rc=$?; test "$TELEGRAM_CHAT_ID" = unchanged || exit 99; '
            'exit "$rc"; fi',
            "runtime-env-protocol-test", str(loader), str(env_path), str(parser),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("finish_before_capture", [False, True])
def test_shell_loader_preserves_literal_newlines_and_metacharacters(
    tmp_path: Path, finish_before_capture: bool,
) -> None:
    marker = tmp_path / "must-not-execute"
    value = f"first\n$(touch {marker})\\literal\nlast\n"
    result = _run_shell_protocol(
        tmp_path,
        b"TELEGRAM_CHAT_ID\0" + value.encode() + b"\0__PRELUDE_RUNTIME_ENV_V1_OK__\0",
        finish_before_capture=finish_before_capture,
        expected_value=value,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not marker.exists()


@pytest.mark.parametrize("finish_before_capture", [False, True])
@pytest.mark.parametrize(
    ("payload", "producer_rc"),
    [
        (b"", 0),
        (b"TELEGRAM_CHAT_ID\0changed\0", 0),
        (b"TELEGRAM_CHAT_ID\0changed\0__PRELUDE_RUNTIME_ENV_V1_OK__", 0),
        (b"TELEGRAM_CHAT_ID\0changed\0__PRELUDE_RUNTIME_ENV_V1_OK__\0tail", 0),
        (b"TELEGRAM_CHAT_ID\0__PRELUDE_RUNTIME_ENV_V1_OK__\0", 0),
        (b"TELEGRAM_CHAT_ID\0changed\0PATH\0unsafe\0__PRELUDE_RUNTIME_ENV_V1_OK__\0", 0),
        (b"TELEGRAM_CHAT_ID\0changed\0__PRELUDE_RUNTIME_ENV_V1_OK__\0", 7),
    ],
)
def test_shell_loader_rejects_partial_failed_or_unsafe_protocol_atomically(
    tmp_path: Path, payload: bytes, producer_rc: int, finish_before_capture: bool,
) -> None:
    result = _run_shell_protocol(
        tmp_path,
        payload,
        producer_rc=producer_rc,
        finish_before_capture=finish_before_capture,
    )
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "changed" not in result.stderr
    assert "unbound variable" not in result.stderr
