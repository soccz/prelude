from __future__ import annotations

import os
import re
import subprocess
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
