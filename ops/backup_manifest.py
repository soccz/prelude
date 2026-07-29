"""Validate the terminal manifest for one daily evidence backup generation."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Callable

SCHEMA = "prelude_evidence_backup.v1"
MANIFEST_KEYS = {
    "schema",
    "date",
    "generation",
    "archive",
    "checksum",
    "sha256",
    "terminal_verdict_pair",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BackupManifestError(RuntimeError):
    """The daily backup terminal manifest or its bound files are invalid."""


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        if path.is_symlink():
            raise BackupManifestError(f"{label} must not be a symlink: {path}")
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise BackupManifestError(
                    f"{label} must be a regular file: {path}"
                )
            return handle.read()
    except BackupManifestError:
        raise
    except OSError as exc:
        raise BackupManifestError(f"cannot read {label} {path}: {exc}") from exc


def _sha256_regular_file(path: Path, *, label: str) -> str:
    try:
        if path.is_symlink():
            raise BackupManifestError(f"{label} must not be a symlink: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise BackupManifestError(
                    f"{label} must be a regular file: {path}"
                )
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except BackupManifestError:
        raise
    except OSError as exc:
        raise BackupManifestError(f"cannot read {label} {path}: {exc}") from exc


def _parse_manifest(raw: bytes, source: Path) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackupManifestError(
            f"manifest is not UTF-8: {source}"
        ) from exc
    if not text.endswith("\n"):
        raise BackupManifestError("manifest must end with newline")

    payload: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise BackupManifestError(f"invalid manifest line: {line!r}")
        if key in payload:
            raise BackupManifestError(f"duplicate manifest key: {key}")
        payload[key] = value

    missing = sorted(MANIFEST_KEYS - payload.keys())
    unknown = sorted(payload.keys() - MANIFEST_KEYS)
    if missing or unknown:
        raise BackupManifestError(
            f"manifest key mismatch: missing={missing}, unknown={unknown}"
        )
    return payload


def validate_daily_evidence_backup(backup_dir: Path, day: str) -> str:
    """Validate today's terminal manifest, checksum, and archive hash."""
    if not re.fullmatch(r"[0-9]{8}", day):
        raise BackupManifestError(f"invalid backup date: {day!r}")
    if backup_dir.is_symlink() or not backup_dir.is_dir():
        raise BackupManifestError(
            f"backup directory must be a real directory: {backup_dir}"
        )

    manifest_path = backup_dir / f"ledgers_{day}.manifest"
    payload = _parse_manifest(
        _read_regular_file(manifest_path, label="manifest"),
        manifest_path,
    )
    if payload["schema"] != SCHEMA:
        raise BackupManifestError(
            f"unexpected manifest schema: {payload['schema']!r}"
        )
    if payload["date"] != day:
        raise BackupManifestError(
            f"manifest date mismatch: {payload['date']!r} != {day!r}"
        )
    if not re.fullmatch(rf"{day}_[0-9]{{6}}_[0-9]+", payload["generation"]):
        raise BackupManifestError(
            f"invalid evidence generation: {payload['generation']!r}"
        )
    expected_archive = f"ledgers_{payload['generation']}.tar.gz"
    if payload["archive"] != expected_archive:
        raise BackupManifestError(
            f"manifest archive mismatch: {payload['archive']!r}"
        )
    expected_checksum = f"{expected_archive}.sha256"
    if payload["checksum"] != expected_checksum:
        raise BackupManifestError(
            f"manifest checksum mismatch: {payload['checksum']!r}"
        )
    expected_hash = payload["sha256"]
    if not SHA256_RE.fullmatch(expected_hash):
        raise BackupManifestError("manifest sha256 is invalid")
    if payload["terminal_verdict_pair"] not in {"absent", "present"}:
        raise BackupManifestError(
            "terminal_verdict_pair must be absent or present"
        )

    archive_path = backup_dir / expected_archive
    checksum_path = backup_dir / expected_checksum
    checksum_bytes = _read_regular_file(checksum_path, label="checksum")
    expected_checksum_bytes = (
        f"{expected_hash}  {expected_archive}\n".encode("ascii")
    )
    if checksum_bytes != expected_checksum_bytes:
        raise BackupManifestError("checksum file does not match manifest")
    actual_hash = _sha256_regular_file(archive_path, label="archive")
    if actual_hash != expected_hash:
        raise BackupManifestError(
            f"archive sha256 mismatch: {actual_hash} != {expected_hash}"
        )
    return expected_archive


def wait_for_daily_evidence_backup(
    backup_dir: Path,
    day: str,
    *,
    wait_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Wait for a concurrent Persistent-timer backup, then validate it."""
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise BackupManifestError("wait_seconds must be nonnegative")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise BackupManifestError("poll_seconds must be positive")
    if backup_dir.is_symlink() or not backup_dir.is_dir():
        # Directory contract failures cannot be repaired by an in-flight
        # backup publish and must never consume the catch-up grace window.
        return validate_daily_evidence_backup(backup_dir, day)

    deadline = monotonic() + wait_seconds
    manifest_path = backup_dir / f"ledgers_{day}.manifest"
    while True:
        try:
            return validate_daily_evidence_backup(backup_dir, day)
        except BackupManifestError as exc:
            # backup_db publishes the manifest last, after archive/checksum
            # validation and durable sync. Once today's manifest exists it is
            # terminal evidence: corruption must alert immediately, not sleep.
            if manifest_path.is_symlink() or manifest_path.exists():
                return validate_daily_evidence_backup(backup_dir, day)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise BackupManifestError(
                    f"daily backup did not become valid within "
                    f"{wait_seconds:g}s: {exc}"
                ) from exc
            sleep(min(poll_seconds, remaining))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        archive = wait_for_daily_evidence_backup(
            args.backup_dir,
            args.date,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
        )
    except BackupManifestError as exc:
        print(f"backup manifest validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"backup manifest/hash ok: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
