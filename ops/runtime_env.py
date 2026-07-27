"""Strict, non-executing loader for the small production runtime environment.

The project ``.env`` file is configuration data, not shell source.  Only the
three secrets used by the supported runtime are accepted; values are never
passed through ``eval`` or interpolation.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path
from typing import Iterable, Mapping


ALLOWED_RUNTIME_KEYS = frozenset(
    {
        "PRELUDE_DASHBOARD_PIN",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    }
)
_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPORT_SENTINEL = "__PRELUDE_RUNTIME_ENV_V1_OK__"
MAX_RUNTIME_ENV_BYTES = 64 * 1024


class RuntimeEnvError(ValueError):
    """The runtime environment file is unsafe or violates its contract."""


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _validate_private_file_stat(
    value: os.stat_result,
    *,
    path: Path,
) -> os.stat_result:
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeEnvError(
            f"runtime environment must be a regular non-symlink file: {path}"
        )
    if stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeEnvError(
            f"runtime environment must not grant group/other access: {path}"
        )
    effective_uid = os.geteuid()
    if effective_uid != 0 and value.st_uid != effective_uid:
        raise RuntimeEnvError(
            f"runtime environment must be owned by the runtime user: {path}"
        )
    if value.st_size > MAX_RUNTIME_ENV_BYTES:
        raise RuntimeEnvError(
            f"runtime environment exceeds {MAX_RUNTIME_ENV_BYTES} bytes: {path}"
        )
    return value


def _private_file_stat(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise RuntimeEnvError(f"cannot inspect runtime environment: {path}") from exc
    return _validate_private_file_stat(value, path=path)


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_RUNTIME_ENV_BYTES:
            raise RuntimeEnvError(
                f"runtime environment exceeds {MAX_RUNTIME_ENV_BYTES} bytes"
            )
    return b"".join(chunks)


def _stable_private_bytes(path: Path) -> bytes:
    try:
        directory_before = path.parent.lstat()
    except OSError as exc:
        raise RuntimeEnvError(
            f"cannot inspect runtime environment parent: {path.parent}"
        ) from exc
    if not stat.S_ISDIR(directory_before.st_mode):
        raise RuntimeEnvError(
            f"runtime environment parent must be a real directory: {path.parent}"
        )
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        directory = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise RuntimeEnvError(
            f"cannot open runtime environment parent: {path.parent}"
        ) from exc
    try:
        directory_opened = os.fstat(directory)
        if (
            int(directory_before.st_dev),
            int(directory_before.st_ino),
        ) != (
            int(directory_opened.st_dev),
            int(directory_opened.st_ino),
        ):
            raise RuntimeEnvError(
                f"runtime environment parent changed before read: {path.parent}"
            )
        try:
            path_before = _validate_private_file_stat(
                os.stat(
                    path.name,
                    dir_fd=directory,
                    follow_symlinks=False,
                ),
                path=path,
            )
        except OSError as exc:
            raise RuntimeEnvError(
                f"cannot inspect runtime environment: {path}"
            ) from exc

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory)
        except OSError as exc:
            raise RuntimeEnvError(
                f"cannot open runtime environment: {path}"
            ) from exc
        try:
            fd_before = _validate_private_file_stat(
                os.fstat(descriptor),
                path=path,
            )
            if _stat_token(fd_before) != _stat_token(path_before):
                raise RuntimeEnvError(
                    f"runtime environment changed before read: {path}"
                )
            content = _read_all(descriptor)
            fd_after = _validate_private_file_stat(
                os.fstat(descriptor),
                path=path,
            )
        finally:
            os.close(descriptor)

        try:
            path_after = _validate_private_file_stat(
                os.stat(
                    path.name,
                    dir_fd=directory,
                    follow_symlinks=False,
                ),
                path=path,
            )
        except OSError as exc:
            raise RuntimeEnvError(
                f"runtime environment changed or disappeared during read: {path}"
            ) from exc
        if (
            _stat_token(path_before) != _stat_token(fd_before)
            or _stat_token(fd_before) != _stat_token(fd_after)
            or _stat_token(fd_after) != _stat_token(path_after)
            or len(content) != fd_after.st_size
        ):
            raise RuntimeEnvError(
                f"runtime environment changed during read: {path}"
            )
        return content
    finally:
        os.close(directory)


def _decode_value(raw_value: str, *, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in {"'", '"'}:
        if any(character.isspace() for character in value):
            raise RuntimeEnvError(
                f"unquoted whitespace is forbidden at line {line_number}"
            )
        decoded = value
    else:
        if len(value) < 2 or value[-1] != value[0]:
            raise RuntimeEnvError(f"unmatched quote at line {line_number}")
        # Quotes are delimiters only.  No shell expansion, interpolation, or
        # backslash processing is performed.
        decoded = value[1:-1]
    if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
        raise RuntimeEnvError(
            f"control character is forbidden at line {line_number}"
        )
    return decoded


def validate_runtime_values(
    values: Mapping[str, str],
    *,
    required_keys: Iterable[str] = (),
) -> None:
    required = tuple(required_keys)
    unsupported_required = sorted(set(required) - ALLOWED_RUNTIME_KEYS)
    if unsupported_required:
        raise RuntimeEnvError(
            "unsupported required runtime key(s): "
            + ", ".join(unsupported_required)
        )
    for key in required:
        if not values.get(key):
            raise RuntimeEnvError(f"required runtime key missing or empty: {key}")

    pin = values.get("PRELUDE_DASHBOARD_PIN")
    if pin is not None:
        if pin != pin.strip():
            raise RuntimeEnvError(
                "PRELUDE_DASHBOARD_PIN must not have edge whitespace"
            )
        # 2026-07-28 사용자 명시 승인으로 12 → 4 완화 (개인용 대시보드).
        if len(pin) < 4:
            raise RuntimeEnvError(
                "PRELUDE_DASHBOARD_PIN must contain at least 4 characters"
            )


def parse_runtime_env(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
) -> dict[str, str]:
    source = Path(path)
    content = _stable_private_bytes(source)
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeEnvError(
            f"runtime environment is not strict UTF-8: {source}"
        ) from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not _KEY_PATTERN.fullmatch(name):
            raise RuntimeEnvError(
                f"unsupported runtime environment syntax at line {line_number}"
            )
        if name not in ALLOWED_RUNTIME_KEYS:
            raise RuntimeEnvError(f"unsupported runtime environment key: {name}")
        if name in values:
            raise RuntimeEnvError(f"duplicate runtime environment key: {name}")
        values[name] = _decode_value(raw_value, line_number=line_number)

    validate_runtime_values(values, required_keys=required_keys)
    return values


def load_runtime_env(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
    override: bool = False,
) -> dict[str, str]:
    """Parse ``path`` and copy only approved keys into ``os.environ``."""
    values = parse_runtime_env(path, required_keys=required_keys)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or export the strict prelude runtime environment"
    )
    parser.add_argument("path")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=sorted(ALLOWED_RUNTIME_KEYS),
    )
    parser.add_argument(
        "--export-nul",
        action="store_true",
        help="emit key/value NUL records plus a success sentinel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        values = parse_runtime_env(args.path, required_keys=args.require)
    except RuntimeEnvError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.export_nul:
        records: list[str] = []
        for key in sorted(values):
            records.extend((key, values[key]))
        records.append(_EXPORT_SENTINEL)
        sys.stdout.buffer.write(
            b"".join(record.encode("utf-8") + b"\0" for record in records)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
