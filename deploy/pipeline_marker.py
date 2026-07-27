#!/usr/bin/env python3
"""Safely clear or publish one post-open pipeline success marker."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.artifact_provenance import atomic_write_bytes  # noqa: E402


STAGES = ("distribution-close", "preopen-close")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


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


def clear_marker(path: Path) -> None:
    """Remove only a safely bound regular marker, never a link target."""
    parent_before = path.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise OSError(f"marker parent must be a real directory: {path.parent}")
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path.parent, flags)
    try:
        parent_opened = os.fstat(directory_fd)
        if not _same_inode(parent_before, parent_opened):
            raise OSError(f"marker parent changed while opening: {path.parent}")
        try:
            existing = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or existing.st_nlink != 1
        ):
            raise OSError(f"marker target is unsafe: {path}")
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _canonical_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("KST date is not canonical")
    return value


def _canonical_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.isoformat(timespec="seconds") != value:
        raise ValueError("completion timestamp is not canonical and aware")
    return value


def write_marker(
    path: Path,
    *,
    stage: str,
    kst_date: str,
    completed_at: str,
) -> None:
    payload = (
        f"stage={stage}\n"
        f"kst_date={_canonical_date(kst_date)}\n"
        "status=complete\n"
        f"completed_at={_canonical_timestamp(completed_at)}\n"
    ).encode("utf-8")
    os.umask(0o077)
    atomic_write_bytes(path, payload)


def verify_marker(path: Path, *, stage: str, kst_date: str) -> None:
    """Validate one complete marker from a stable no-follow descriptor."""
    expected_date = _canonical_date(kst_date)
    parent_before = path.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise OSError(f"marker parent must be a real directory: {path.parent}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path.parent, directory_flags)
    marker_fd = -1
    try:
        if not _same_inode(parent_before, os.fstat(directory_fd)):
            raise OSError(f"marker parent changed while opening: {path.parent}")
        marker_flags = os.O_RDONLY
        marker_flags |= getattr(os, "O_NOFOLLOW", 0)
        marker_flags |= getattr(os, "O_CLOEXEC", 0)
        marker_fd = os.open(path.name, marker_flags, dir_fd=directory_fd)
        opened_before = os.fstat(marker_fd)
        named_before = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_uid != os.geteuid()
            or opened_before.st_nlink != 1
            or _stat_token(opened_before) != _stat_token(named_before)
        ):
            raise OSError(f"marker target is unsafe: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(marker_fd, 4096):
            total += len(chunk)
            if total > 4096:
                raise ValueError("marker payload is too large")
            chunks.append(chunk)
        opened_after = os.fstat(marker_fd)
        named_after = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _stat_token(opened_before) != _stat_token(opened_after)
            or _stat_token(opened_after) != _stat_token(named_after)
        ):
            raise OSError(f"marker changed during read: {path}")
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        os.close(directory_fd)

    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("marker is not strict UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("marker is not newline-terminated")
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ValueError("marker records are malformed or duplicated")
        values[key] = value
    if set(values) != {"stage", "kst_date", "status", "completed_at"}:
        raise ValueError("marker schema is invalid")
    if (
        values["stage"] != stage
        or values["kst_date"] != expected_date
        or values["status"] != "complete"
    ):
        raise ValueError("marker stage/date/status is invalid")
    completed_at = datetime.fromisoformat(
        _canonical_timestamp(values["completed_at"])
    )
    if completed_at.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat() != (
        expected_date
    ):
        raise ValueError("marker completion date is inconsistent")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    clear = subparsers.add_parser("clear")
    clear.add_argument("--marker", type=Path, required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--marker", type=Path, required=True)
    write.add_argument("--stage", choices=STAGES, required=True)
    write.add_argument("--kst-date", required=True)
    write.add_argument("--completed-at", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--marker", type=Path, required=True)
    verify.add_argument("--stage", choices=STAGES, required=True)
    verify.add_argument("--kst-date", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "clear":
            clear_marker(args.marker)
        elif args.action == "write":
            write_marker(
                args.marker,
                stage=args.stage,
                kst_date=args.kst_date,
                completed_at=args.completed_at,
            )
        else:
            verify_marker(
                args.marker,
                stage=args.stage,
                kst_date=args.kst_date,
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
