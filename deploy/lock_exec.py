#!/usr/bin/env python3
"""Run a command while holding a securely opened, persistent file lock.

Shell redirections such as ``exec 9>lock`` truncate an existing file and
follow a final-component symlink before ``flock`` runs.  This helper opens the
lock with ``O_NOFOLLOW`` and no truncation, validates the pathname and file
descriptor refer to the same private regular file, then retains the descriptor
for the complete lifetime of the child process.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import re
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence


_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_LOCK_MODE = 0o600
_DEFAULT_ERROR_EXIT = 73


class LockContractError(RuntimeError):
    """The lock path or inherited descriptor violates the safety contract."""


class LockBusyError(RuntimeError):
    """The canonical lock is already held by another process."""


@dataclass(frozen=True)
class _LockPath:
    absolute: Path
    parent_fd: int
    basename: str


def _exit_code(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("must be between 0 and 255")
    return parsed


def _fd_number(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be an inherited descriptor >= 3")
    return parsed


def _open_lock_parent(path: str) -> _LockPath:
    absolute = Path(path).absolute()
    basename = absolute.name
    if basename in {"", ".", ".."}:
        raise LockContractError(f"invalid lock filename: {path!r}")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(absolute.parent, flags)
    except OSError as exc:
        raise LockContractError(
            f"cannot securely open lock directory {absolute.parent}: {exc}"
        ) from exc

    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise LockContractError(
                f"lock parent is not a directory: {absolute.parent}"
            )
        if parent_stat.st_uid != os.geteuid():
            raise LockContractError(
                "lock directory owner does not match the effective user: "
                f"{absolute.parent}"
            )
    except BaseException:
        os.close(parent_fd)
        raise
    return _LockPath(absolute, parent_fd, basename)


def _validate_regular_lock(
    file_stat: os.stat_result,
    *,
    label: str,
    require_mode: bool,
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise LockContractError(f"{label} is not a regular file")
    if file_stat.st_uid != os.geteuid():
        raise LockContractError(
            f"{label} owner does not match the effective user"
        )
    if file_stat.st_nlink != 1:
        raise LockContractError(f"{label} must have exactly one hard link")
    if require_mode and stat.S_IMODE(file_stat.st_mode) != _LOCK_MODE:
        raise LockContractError(f"{label} mode is not {_LOCK_MODE:#05o}")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_fd_path_binding(
    fd: int,
    lock_path: _LockPath,
    *,
    require_mode: bool,
) -> None:
    try:
        fd_stat = os.fstat(fd)
        path_stat = os.stat(
            lock_path.basename,
            dir_fd=lock_path.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LockContractError(
            f"cannot validate lock inode {lock_path.absolute}: {exc}"
        ) from exc

    _validate_regular_lock(
        fd_stat,
        label=f"lock descriptor for {lock_path.absolute}",
        require_mode=require_mode,
    )
    _validate_regular_lock(
        path_stat,
        label=f"lock path {lock_path.absolute}",
        require_mode=require_mode,
    )
    if not _same_inode(fd_stat, path_stat):
        raise LockContractError(
            f"lock pathname changed during acquisition: {lock_path.absolute}"
        )
    try:
        access_mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as exc:
        raise LockContractError(f"cannot inspect lock descriptor: {exc}") from exc
    if access_mode != os.O_RDWR:
        raise LockContractError("lock descriptor is not open read/write")


def _acquire(
    lock_file: str,
    *,
    blocking: bool,
) -> tuple[int, _LockPath]:
    lock_path = _open_lock_parent(lock_file)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            fd = os.open(
                lock_path.basename,
                flags,
                _LOCK_MODE,
                dir_fd=lock_path.parent_fd,
            )
        except OSError as exc:
            raise LockContractError(
                f"cannot securely open lock {lock_path.absolute}: {exc}"
            ) from exc

        try:
            _validate_fd_path_binding(fd, lock_path, require_mode=False)
            try:
                operation = fcntl.LOCK_EX
                if not blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(fd, operation)
            except OSError as exc:
                if (
                    not blocking
                    and exc.errno in {errno.EACCES, errno.EAGAIN}
                ):
                    raise LockBusyError(
                        f"lock is already held: {lock_path.absolute}"
                    ) from exc
                raise LockContractError(
                    f"cannot acquire lock {lock_path.absolute}: {exc}"
                ) from exc

            os.fchmod(fd, _LOCK_MODE)
            _validate_fd_path_binding(fd, lock_path, require_mode=True)
            os.set_inheritable(fd, True)
            return fd, lock_path
        except BaseException:
            os.close(fd)
            raise
    except BaseException:
        os.close(lock_path.parent_fd)
        raise


def _verify_held(fd: int, lock_file: str) -> None:
    lock_path = _open_lock_parent(lock_file)
    try:
        _validate_fd_path_binding(fd, lock_path, require_mode=True)
        try:
            # An inherited descriptor shares the same open-file description,
            # so this is idempotent.  An injected, unlocked descriptor acquires
            # the lock here and remains held by the calling shell's copy.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise LockContractError(
                    f"inherited descriptor does not hold {lock_path.absolute}"
                ) from exc
            raise LockContractError(
                f"cannot verify inherited lock: {exc}"
            ) from exc
        _validate_fd_path_binding(fd, lock_path, require_mode=True)
    finally:
        os.close(lock_path.parent_fd)


def _safe_append(path: str, message: str) -> None:
    output_path = _open_lock_parent(path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            fd = os.open(
                output_path.basename,
                flags,
                _LOCK_MODE,
                dir_fd=output_path.parent_fd,
            )
        except OSError as exc:
            raise LockContractError(
                f"cannot securely open busy log {output_path.absolute}: {exc}"
            ) from exc
        try:
            _validate_fd_path_binding_for_append(fd, output_path)
            payload = (message.rstrip("\n") + "\n").encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise LockContractError("short write to busy log")
                view = view[written:]
            os.fsync(fd)
            _validate_fd_path_binding_for_append(fd, output_path)
        finally:
            os.close(fd)
    finally:
        os.close(output_path.parent_fd)


def _validate_fd_path_binding_for_append(
    fd: int,
    output_path: _LockPath,
) -> None:
    try:
        fd_stat = os.fstat(fd)
        path_stat = os.stat(
            output_path.basename,
            dir_fd=output_path.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise LockContractError(
            f"cannot validate busy log {output_path.absolute}: {exc}"
        ) from exc
    _validate_regular_lock(
        fd_stat,
        label=f"busy log descriptor for {output_path.absolute}",
        require_mode=False,
    )
    _validate_regular_lock(
        path_stat,
        label=f"busy log path {output_path.absolute}",
        require_mode=False,
    )
    if not _same_inode(fd_stat, path_stat):
        raise LockContractError(
            f"busy log pathname changed during append: {output_path.absolute}"
        )


def _restore_signal_and_raise(signum: int) -> NoReturn:
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    raise AssertionError("unreachable after signal delivery")


def _run_child(command: Sequence[str], fd: int, fd_env: str) -> int:
    if not command:
        raise LockContractError("missing child command")
    if not _ENV_NAME.fullmatch(fd_env):
        raise LockContractError(f"invalid descriptor environment name: {fd_env}")
    if fd_env in os.environ:
        raise LockContractError(
            f"descriptor environment variable already exists: {fd_env}"
        )

    child_env = os.environ.copy()
    child_env[fd_env] = str(fd)
    try:
        completed = subprocess.run(
            list(command),
            env=child_env,
            close_fds=True,
            pass_fds=(fd,),
            check=False,
        )
    except OSError as exc:
        raise LockContractError(f"cannot execute child command: {exc}") from exc
    if completed.returncode < 0:
        _restore_signal_and_raise(-completed.returncode)
    return completed.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--lock-file", required=True)
    run.add_argument("--fd-env", required=True)
    run.add_argument("--busy-exit", type=_exit_code, required=True)
    run.add_argument("--error-exit", type=_exit_code, default=_DEFAULT_ERROR_EXIT)
    run.add_argument("--busy-message")
    run.add_argument("--busy-log")
    run.add_argument(
        "--wait",
        action="store_true",
        help="wait for the lock instead of returning the busy exit code",
    )
    run.add_argument("command", nargs=argparse.REMAINDER)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--lock-file", required=True)
    verify.add_argument("--fd", type=_fd_number, required=True)
    verify.add_argument(
        "--error-exit",
        type=_exit_code,
        default=_DEFAULT_ERROR_EXIT,
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise LockContractError("missing child command after --")

    try:
        fd, lock_path = _acquire(
            args.lock_file,
            blocking=args.wait,
        )
    except LockBusyError as exc:
        message = args.busy_message or str(exc)
        if args.busy_log:
            _safe_append(args.busy_log, message)
        print(message, file=sys.stderr)
        return args.busy_exit

    try:
        return _run_child(command, fd, args.fd_env)
    finally:
        os.close(fd)
        os.close(lock_path.parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    error_exit = args.error_exit
    try:
        if args.action == "verify":
            _verify_held(args.fd, args.lock_file)
            return 0
        return _run(args)
    except LockContractError as exc:
        print(f"lock_exec: {exc}", file=sys.stderr)
        return error_exit


if __name__ == "__main__":
    raise SystemExit(main())
