from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import ops.backup_manifest as backup_manifest
from ops.backup_manifest import (
    BackupManifestError,
    validate_daily_evidence_backup,
    wait_for_daily_evidence_backup,
)

DAY = "20260729"
GENERATION = f"{DAY}_040100_123"


def _write_valid_backup(backup_dir: Path) -> tuple[Path, Path, Path]:
    backup_dir.mkdir()
    archive = backup_dir / f"ledgers_{GENERATION}.tar.gz"
    archive.write_bytes(b"immutable-forward-evidence")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = backup_dir / f"{archive.name}.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    manifest = backup_dir / f"ledgers_{DAY}.manifest"
    manifest.write_text(
        "\n".join(
            (
                "schema=prelude_evidence_backup.v1",
                f"date={DAY}",
                f"generation={GENERATION}",
                f"archive={archive.name}",
                f"checksum={checksum.name}",
                f"sha256={digest}",
                "terminal_verdict_pair=absent",
                "",
            )
        ),
        encoding="utf-8",
    )
    return manifest, archive, checksum


def test_daily_backup_manifest_binds_archive_and_checksum(tmp_path):
    backup_dir = tmp_path / "backup"
    _manifest, archive, _checksum = _write_valid_backup(backup_dir)

    assert validate_daily_evidence_backup(backup_dir, DAY) == archive.name


def test_yesterday_manifest_does_not_satisfy_today(tmp_path):
    backup_dir = tmp_path / "backup"
    _write_valid_backup(backup_dir)

    with pytest.raises(BackupManifestError, match="cannot read manifest"):
        validate_daily_evidence_backup(backup_dir, "20260730")


@pytest.mark.parametrize("target", ("manifest", "archive", "checksum"))
def test_backup_manifest_rejects_symlinked_contract_file(tmp_path, target):
    backup_dir = tmp_path / "backup"
    manifest, archive, checksum = _write_valid_backup(backup_dir)
    path = {
        "manifest": manifest,
        "archive": archive,
        "checksum": checksum,
    }[target]
    outside = tmp_path / f"outside-{target}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(BackupManifestError, match="must not be a symlink"):
        validate_daily_evidence_backup(backup_dir, DAY)


def test_backup_manifest_rejects_archive_tamper(tmp_path):
    backup_dir = tmp_path / "backup"
    _manifest, archive, _checksum = _write_valid_backup(backup_dir)
    archive.write_bytes(b"tampered")

    with pytest.raises(BackupManifestError, match="archive sha256 mismatch"):
        validate_daily_evidence_backup(backup_dir, DAY)


def test_backup_manifest_rejects_duplicate_key(tmp_path):
    backup_dir = tmp_path / "backup"
    manifest, _archive, _checksum = _write_valid_backup(backup_dir)
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"date={DAY}\n",
        encoding="utf-8",
    )

    with pytest.raises(BackupManifestError, match="duplicate manifest key"):
        validate_daily_evidence_backup(backup_dir, DAY)


def test_wait_retries_concurrent_backup_until_manifest_becomes_valid(
    tmp_path,
    monkeypatch,
):
    attempts = 0
    clock = 0.0

    def eventually_valid(_backup_dir, _day):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise BackupManifestError("manifest not published yet")
        return "ledgers-ready.tar.gz"

    def fake_sleep(seconds):
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(
        backup_manifest,
        "validate_daily_evidence_backup",
        eventually_valid,
    )

    archive = wait_for_daily_evidence_backup(
        tmp_path,
        DAY,
        wait_seconds=30,
        poll_seconds=10,
        sleep=fake_sleep,
        monotonic=lambda: clock,
    )

    assert archive == "ledgers-ready.tar.gz"
    assert attempts == 3
    assert clock == 20


def test_wait_fails_after_bounded_deadline(tmp_path, monkeypatch):
    clock = 0.0

    def never_valid(_backup_dir, _day):
        raise BackupManifestError("still missing")

    def fake_sleep(seconds):
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(
        backup_manifest,
        "validate_daily_evidence_backup",
        never_valid,
    )

    with pytest.raises(
        BackupManifestError,
        match=r"did not become valid within 25s: still missing",
    ):
        wait_for_daily_evidence_backup(
            tmp_path,
            DAY,
            wait_seconds=25,
            poll_seconds=10,
            sleep=fake_sleep,
            monotonic=lambda: clock,
        )

    assert clock == 25


def test_wait_does_not_delay_existing_terminal_manifest_corruption(
    tmp_path,
):
    backup_dir = tmp_path / "backup"
    _manifest, archive, _checksum = _write_valid_backup(backup_dir)
    archive.write_bytes(b"tampered")
    sleeps: list[float] = []

    with pytest.raises(BackupManifestError, match="archive sha256 mismatch"):
        wait_for_daily_evidence_backup(
            backup_dir,
            DAY,
            wait_seconds=3300,
            poll_seconds=10,
            sleep=sleeps.append,
        )

    assert sleeps == []


def test_wait_does_not_delay_symlinked_backup_directory(tmp_path):
    real_backup = tmp_path / "real-backup"
    real_backup.mkdir()
    linked_backup = tmp_path / "linked-backup"
    linked_backup.symlink_to(real_backup, target_is_directory=True)
    sleeps: list[float] = []

    with pytest.raises(
        BackupManifestError,
        match="backup directory must be a real directory",
    ):
        wait_for_daily_evidence_backup(
            linked_backup,
            DAY,
            wait_seconds=3300,
            poll_seconds=10,
            sleep=sleeps.append,
        )

    assert sleeps == []
