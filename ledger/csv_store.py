"""Process-safe, crash-atomic primitives for CSV ledgers and their artifacts.

All read-modify-write users of the same ledger must hold :func:`ledger_lock`
for the complete transaction.  Readers that only need a snapshot don't need
the lock: :func:`atomic_write_csv` exposes either the complete old file or the
complete new file through a single same-filesystem ``os.replace``.

The lock file is intentionally persistent.  Removing it after unlock would
allow two processes to lock different inodes during an unlink/open race.
"""

from __future__ import annotations

import json
import math
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

import pandas as pd

from ops.file_lock import file_lock


def lock_path_for(ledger_path: str | Path) -> Path:
    """Return the stable sibling lock path used by every ledger writer."""
    path = Path(ledger_path)
    return path.with_name(f".{path.name}.lock")


@contextmanager
def ledger_lock(ledger_path: str | Path) -> Iterator[Path]:
    """Hold an exclusive process lock for one ledger transaction.

    The yielded path is the normalized ``Path`` supplied by the caller.  This
    lock isn't re-entrant; a helper that owns it must not call another helper
    that tries to acquire the same ledger lock.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = lock_path_for(path)
    with file_lock(lock_path):
        try:
            target = path.lstat()
        except FileNotFoundError:
            target = None
        if target is not None and not stat.S_ISREG(target.st_mode):
            raise OSError(f"ledger target must be a regular file: {path}")
        yield path


def _inode(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _atomic_write_text(
    artifact_path: str | Path,
    writer: Callable[[TextIO], None],
) -> None:
    """Write text to a private sibling and durably replace the target."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for safe atomic writes")

    directory_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    flags |= os.O_NOFOLLOW

    directory_before = path.parent.lstat()
    if not stat.S_ISDIR(directory_before.st_mode):
        raise OSError(f"artifact parent must be a real directory: {path.parent}")
    directory_fd = os.open(path.parent, directory_flags)
    fd = -1
    temp_created = False
    replaced = False
    try:
        directory_opened = os.fstat(directory_fd)
        if _inode(directory_before) != _inode(directory_opened):
            raise OSError(f"artifact parent changed while opening: {path.parent}")

        original: os.stat_result | None
        try:
            original = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            original = None
        if original is not None and not stat.S_ISREG(original.st_mode):
            raise OSError(f"artifact target must be a regular file: {path}")

        fd = os.open(temp_name, flags, 0o666, dir_fd=directory_fd)
        temp_created = True
        if original is not None:
            os.fchmod(fd, original.st_mode & 0o7777)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd = -1
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            directory_current = path.parent.lstat()
        except FileNotFoundError as exc:
            raise OSError(
                f"artifact parent disappeared during write: {path.parent}"
            ) from exc
        if (
            not stat.S_ISDIR(directory_current.st_mode)
            or _inode(directory_current) != _inode(directory_opened)
        ):
            raise OSError(
                f"artifact parent changed during write: {path.parent}"
            )

        # A writer that ignores the transaction lock must not be silently
        # overwritten after the target was validated.
        try:
            current = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current = None
        if original is None:
            if current is not None:
                raise OSError(f"artifact target appeared during write: {path}")
        elif current is None or _inode(current) != _inode(original):
            raise OSError(f"artifact target changed during write: {path}")
        elif not stat.S_ISREG(current.st_mode):
            raise OSError(f"artifact target became unsafe: {path}")

        os.replace(
            temp_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        os.fsync(directory_fd)
        try:
            directory_after = path.parent.lstat()
        except FileNotFoundError as exc:
            raise OSError(
                f"artifact parent disappeared during replace: {path.parent}"
            ) from exc
        if (
            not stat.S_ISDIR(directory_after.st_mode)
            or _inode(directory_after) != _inode(directory_opened)
        ):
            raise OSError(
                f"artifact parent changed during replace: {path.parent}"
            )
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_created and not replaced:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def atomic_write_csv(
    frame: pd.DataFrame,
    ledger_path: str | Path,
    *,
    index: bool = False,
) -> None:
    """Durably replace a CSV without ever exposing a partial target.

    Callers are responsible for holding :func:`ledger_lock` across the read,
    mutation, and this write.  Any exception before ``os.replace`` leaves the
    existing ledger untouched and removes the private temporary file.
    """
    _atomic_write_text(
        ledger_path,
        lambda handle: frame.to_csv(handle, index=index),
    )


def atomic_write_json(
    payload: Any,
    artifact_path: str | Path,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default: Callable[[Any], Any] | None = str,
) -> None:
    """Durably replace a strict-JSON diagnostic artifact."""
    _atomic_write_text(
        artifact_path,
        lambda handle: json.dump(
            _json_safe(payload),
            handle,
            indent=indent,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            default=default,
        ),
    )


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/non-finite values to interoperable JSON values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
