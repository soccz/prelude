"""Fail-closed advisory lock files for process-safe local transactions.

The named lock inode is persistent.  Callers must never unlink it: replacing
an active lock path would let two processes lock different inodes.
"""
from __future__ import annotations

import errno
import fcntl
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class FileLockError(RuntimeError):
    """A lock path is unsafe or cannot be acquired reliably."""


class FileLockBusyError(FileLockError):
    """A non-blocking lock is already held by another open file description."""


def _inode(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


@contextmanager
def file_lock(
    path_value: str | Path,
    *,
    shared: bool = False,
    blocking: bool = True,
) -> Iterator[int]:
    """Hold one no-follow, owner-bound, persistent regular-file lock."""
    path = Path(path_value)
    if not hasattr(os, "O_NOFOLLOW"):
        raise FileLockError("O_NOFOLLOW is required for safe lock files")

    directory_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    lock_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        lock_flags |= os.O_CLOEXEC

    directory_fd = -1
    lock_fd = -1
    acquired = False
    try:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            directory_before = path.parent.lstat()
            if not stat.S_ISDIR(directory_before.st_mode):
                raise FileLockError(
                    f"lock parent must be a real directory: {path.parent}"
                )
            directory_fd = os.open(path.parent, directory_flags)
            directory_opened = os.fstat(directory_fd)
            if _inode(directory_before) != _inode(directory_opened):
                raise FileLockError(
                    f"lock parent changed while opening: {path.parent}"
                )

            lock_fd = os.open(
                path.name,
                lock_flags,
                0o600,
                dir_fd=directory_fd,
            )
            opened = os.fstat(lock_fd)
            named = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or _inode(opened) != _inode(named)
            ):
                raise FileLockError(f"lock path is unsafe: {path}")
            os.fchmod(lock_fd, 0o600)

            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(lock_fd, operation)
            except OSError as exc:
                if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise FileLockBusyError(
                        f"lock is already held: {path}"
                    ) from exc
                raise
            acquired = True

            named_after = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            opened_after = os.fstat(lock_fd)
            if (
                _inode(named_after) != _inode(opened_after)
                or not stat.S_ISREG(opened_after.st_mode)
                or opened_after.st_uid != os.geteuid()
                or opened_after.st_nlink != 1
            ):
                raise FileLockError(
                    f"lock path changed while acquiring: {path}"
                )
        except FileLockError:
            raise
        except OSError as exc:
            raise FileLockError(
                f"cannot safely acquire lock: {path}"
            ) from exc

        # Keep the caller's exception type and traceback intact.  In
        # particular, an OSError raised by the protected transaction is not a
        # lock-acquisition failure and must not be wrapped as FileLockError.
        yield lock_fd
    finally:
        if acquired and lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
